// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019
//           Roberto Paredes Palacios, <rparedes@dsic.upv.es>
//           Jon Ander Gómez, <jon@dsic.upv.es>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>
#include <stdio.h>

#ifndef _LAYER_
#define _LAYER_

#include <string>

#include "../tensor.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;


class Layer {
public:
    string name;
    Tensor *input;
    Tensor *output;
    Tensor *target;
    Tensor *delta;
    Layer *orig;

    vector<Tensor *> params;
    vector<Tensor *> gradients;

    vector<Layer *> parent;
    vector<Layer *> child;

    int mode;
    int dev;
    int lin, lout;
    int delta_bp;

    Layer(string name, int dev);

    void initialize();

    void reset();

    void info();

    void setmode(int m);

    Tensor getWeights();
    Tensor setWeights(Tensor bias);

    Tensor getBias();
    Tensor setBias(Tensor bias);

    //virtual
    virtual string plot(int c) { return ""; }

    virtual void addchild(Layer *l) {}

    virtual void addparent(Layer *l) {}

    virtual void forward() {}

    virtual void backward() {}

    virtual Layer *share(int c, int bs, vector<Layer *>) { return NULL; }

    virtual Layer *clone(int c, int bs, vector<Layer *>, int todev) { return NULL; }

};


/////////////////////////////////////////
/////////////////////////////////////////
// Layers with only one input
class LinLayer : public Layer {
public:

    LinLayer(string name, int dev);

    void addchild(Layer *l);

    void addparent(Layer *l);
};

/// Tensor Layer
class LTensor : public LinLayer {
public:
    static int total_layers;

    LTensor(string fname);

    LTensor(const initializer_list<int> &init, int dev);

    LTensor(const vector<int> shape, int dev);

    LTensor(Layer *l);

    Layer *share(int c, int bs, vector<Layer *> p) { return NULL; }

    Layer *clone(int c, int bs, vector<Layer *>, int todev) { return NULL; }

    void info() {}

    void forward() {}

    void backward() {}

    string plot(int c) { return ""; }

    LTensor operator+(LTensor L);


};

/// INPUT Layer
class LInput : public LinLayer {
public:
    static int total_layers;

    LInput(Tensor *in, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p);

    Layer *clone(int c, int bs, vector<Layer *>, int todev);

    void forward();

    void backward();

    string plot(int c);

};

/// EMBEDDING Layer
class LEmbedding : public LinLayer {
public:
    int input_dim;
    int output_dim;
    static int total_layers;

    LEmbedding(int input_dim, int output_dim, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p);

    Layer *clone(int c, int bs, vector<Layer *>, int todev);

    void forward();

    void backward();

    string plot(int c);

};

/// Dense Layer
class LDense : public LinLayer {
public:
    int ndim;
    bool use_bias;  // TODO: Implement
    static int total_layers;

    LDense(Layer *parent, int ndim, bool use_bias, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p);

    Layer *clone(int c, int bs, vector<Layer *>, int todev);

    // Params
    Tensor *W;
    Tensor *gW;
    Tensor *bias;
    Tensor *gbias;

    void forward();

    void backward();

    string plot(int c);

};

/// Activation Layer
class LActivation : public LinLayer {
public:
    string act;
    static int total_layers;

    LActivation(Layer *parent, string act, string name, int d);

    Layer *share(int c, int bs, vector<Layer *> p);

    Layer *clone(int c, int bs, vector<Layer *>, int todev);

    void forward();

    void backward();

    string plot(int c);

};

/// Reshape Layer
class LReshape : public LinLayer {
public:
    static int total_layers;
    vector<int> ls;

    // constructors and clones
    LReshape(Layer *parent, const initializer_list<int> &init, string name, int d);

    LReshape(Layer *parent, vector<int> shape, string name, int d);

    Layer *share(int c, int bs, vector<Layer *> p);

    Layer *clone(int c, int bs, vector<Layer *> p, int todev);


    // implementation
    void forward();

    void backward();

    string plot(int c);

};

/// Transpose Layer
class LTranspose : public LinLayer {
public:
    static int total_layers;
    vector<int> dims;

    // constructors and clones
    LTranspose(Layer *parent, const initializer_list<int> &dims, string name, int dev);

//    Layer *share(int c, int bs, vector<Layer *> p);
//
//    Layer *clone(int c, int bs, vector<Layer *> p, int todev);
//
//
//    // implementation
//    void forward();
//
//    void backward();
//
//    string plot(int c);

};

/// Conv2D Layer
class LConv : public LinLayer {
public:
    static int total_layers;

    ConvolDescriptor *cd;

    // constructors and clones
    LConv(Layer *parent, const initializer_list<int> &ks, const initializer_list<int> &st, string p, string name, int d);

    LConv(Layer *parent, const initializer_list<int> &ks, const initializer_list<int> &st,
          const initializer_list<int> &p, string name, int d);

    LConv(Layer *parent, int filters, const initializer_list<int> &kernel_size, const initializer_list<int> &strides, string padding,
            int groups, const initializer_list<int> &dilation_rate, bool use_bias, string name, int dev);

    LConv(Layer *parent, const vector<int> &ks, const vector<int> &st, string p, string name, int d);

    LConv(Layer *parent, ConvolDescriptor *cd, string name, int d);

    Layer *share(int c, int bs, vector<Layer *> p);

    Layer *clone(int c, int bs, vector<Layer *> p, int todev);

    // Params are in ConvolDescriptor

    // implementation
    void forward();

    void backward();

    string plot(int c);

};

/// ConvT2D Layer
class LConvT : public LinLayer {
public:
    static int total_layers;

    ConvolDescriptor *cd;

    // constructors and clones
    LConvT(Layer *parent, int filters, const initializer_list<int> &kernel_size,
        const initializer_list<int> &output_padding, string padding, const initializer_list<int> &dilation_rate,
        const initializer_list<int> &strides, bool use_bias, string name, int dev);

    LConvT(Layer *parent, ConvolDescriptor *cd, string name, int dev);

//    Layer *share(int c, int bs, vector<Layer *> p);
//
//    Layer *clone(int c, int bs, vector<Layer *> p, int todev);
//
//    // Params are in ConvolDescriptor
//
//    // implementation
//    void forward();
//
//    void backward();
//
//    string plot(int c);

};

/// UpSampling2D Layer
class LUpSampling : public LinLayer {
public:
    vector<int> size;
    string interpolation;
    static int total_layers;

    // constructors and clones
    LUpSampling(Layer *parent, const initializer_list<int> &size, string interpolation, string name, int dev);

//    Layer *share(int c, int bs, vector<Layer *> p);
//
//    Layer *clone(int c, int bs, vector<Layer *> p, int todev);
//
//    // Params are in ConvolDescriptor
//
//    // implementation
//    void forward();
//
//    void backward();
//
//    string plot(int c);

};

/// Pool2D Layer
class LPool : public LinLayer {
public:
    static int total_layers;
    PoolDescriptor *pd;

    // constructors
    LPool(Layer *parent, PoolDescriptor *cd, string name, int d);
};

/// MaxPool2D Layer
class LMaxPool : public LPool {
public:

    // constructors and clones
    LMaxPool(Layer *parent, const initializer_list<int> &ks, const initializer_list<int> &st, string p, string name,
           int d);

    LMaxPool(Layer *parent, const initializer_list<int> &ks, const initializer_list<int> &st,
           const initializer_list<int> &p, string name, int d);

    LMaxPool(Layer *parent, const vector<int> &ks, const vector<int> &st, string p, string name, int d);

    LMaxPool(Layer *parent, PoolDescriptor *cd, string name, int d);

    // Params
    Tensor *indX, *indY;

    // implementation
    void forward();

    void backward();

    Layer *share(int c, int bs, vector<Layer *> p);

    Layer *clone(int c, int bs, vector<Layer *> p, int todev);

    string plot(int c);

};


/// AveragePool2D Layer
class LAveragePool : public LPool {
public:

    // constructors and clones
    LAveragePool(Layer *parent, const initializer_list<int> &pool_size, const initializer_list<int> &strides, string padding, string name, int dev);
    LAveragePool(Layer *parent, PoolDescriptor *D, string name, int dev);

//    // Params
//    Tensor *indX, *indY;
//
//    // implementation
//    void forward();
//
//    void backward();
//
//    Layer *share(int c, int bs, vector<Layer *> p);
//
//    Layer *clone(int c, int bs, vector<Layer *> p, int todev);
//
//    string plot(int c);

};

/// GlobalMaxPool2D Layer
class LGlobalMaxPool : public LPool {
public:

    // constructors and clones
    LGlobalMaxPool(Layer *parent, PoolDescriptor *D, string name, int dev);

//    // Params
//    Tensor *indX, *indY;
//
//    // implementation
//    void forward();
//
//    void backward();
//
//    Layer *share(int c, int bs, vector<Layer *> p);
//
//    Layer *clone(int c, int bs, vector<Layer *> p, int todev);
//
//    string plot(int c);

};

/// GlobalAveragePool2D Layer
class LGlobalAveragePool : public LPool {
public:

    // constructors and clones
    LGlobalAveragePool(Layer *parent, PoolDescriptor *D, string name, int dev);

//    // Params
//    Tensor *indX, *indY;
//
//    // implementation
//    void forward();
//
//    void backward();
//
//    Layer *share(int c, int bs, vector<Layer *> p);
//
//    Layer *clone(int c, int bs, vector<Layer *> p, int todev);
//
//    string plot(int c);

};

/// Drop-out Layer
class LDropout : public LinLayer {
public:
    int ndim;
    static int total_layers;

    // constructors and clones
    LDropout(Layer *parent, float df, string name, int d);

    Layer *share(int c, int bs, vector<Layer *> p);

    Layer *clone(int c, int bs, vector<Layer *> p, int todev);

    float df;
    Tensor *mask;

    // implementation
    void forward();

    void backward();

    string plot(int c);

};


/////////////////////////////////////////
/////////////////////////////////////////
// Layers with several inputs (ADD, CAT,...)
class MLayer : public Layer {
public:

    MLayer(string name, int dev);

    void addchild(Layer *l);

    void addparent(Layer *l);

    //virtual

    virtual string plot(int c) { return ""; }

    virtual void forward() {}

    virtual void backward() {}

    virtual Layer *share(int c, int bs, vector<Layer *> p) { return NULL; }

    virtual Layer *clone(int c, int bs, vector<Layer *>, int todev) { return NULL; }

};

/// Add Layer
class LAdd : public MLayer {
public:
    static int total_layers;


    LAdd(vector<Layer *> in, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p);

    Layer *clone(int c, int bs, vector<Layer *>, int todev);

    void forward();

    void backward();

    string plot(int c);

};

/// Subtract Layer
class LSubtract : public MLayer {
public:
    static int total_layers;


    LSubtract(vector<Layer *> in, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p);

    Layer *clone(int c, int bs, vector<Layer *>, int todev);

    void forward();

    void backward();

    string plot(int c);

};


/// MatMul Layer
class LMatMul : public MLayer {
public:
    static int total_layers;


    LMatMul(vector<Layer *> in, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p);

    Layer *clone(int c, int bs, vector<Layer *>, int todev);

    void forward();

    void backward();

    string plot(int c);

};


/// Average Layer
class LAverage : public MLayer {
public:
    static int total_layers;


    LAverage(vector<Layer *> in, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p);

    Layer *clone(int c, int bs, vector<Layer *>, int todev);

    void forward();

    void backward();

    string plot(int c);

};

/// Maximum Layer
class LMaximum : public MLayer {
public:
    static int total_layers;


    LMaximum(vector<Layer *> in, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p);

    Layer *clone(int c, int bs, vector<Layer *>, int todev);

    void forward();

    void backward();

    string plot(int c);

};

/// Maximum Layer
class LMinimum : public MLayer {
public:
    static int total_layers;


    LMinimum(vector<Layer *> in, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p);

    Layer *clone(int c, int bs, vector<Layer *>, int todev);

    void forward();

    void backward();

    string plot(int c);

};


/// Concat Layer
class LConcat : public MLayer {
public:
    int ndim;
    vector<int> index;
    static int total_layers;

    // constructors and clones
    LConcat(vector<Layer *> in, string name, int d);

    Layer *share(int c, int bs, vector<Layer *> p);

    Layer *clone(int c, int bs, vector<Layer *> p, int todev);

    // Params


    // implementation
    void forward();

    void backward();

    string plot(int c);

};


/// BatchNormalization Layer
class LBatchNorm : public LinLayer {
public:
    float momentum;
    float epsilon;
    bool affine;
    static int total_layers;

    LBatchNorm(Layer *parent, float momentum, float epsilon, bool affine, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p);

    Layer *clone(int c, int bs, vector<Layer *>, int todev);

    void forward();

    void backward();

    string plot(int c);
};


/// GaussianNoise Layer
class LGaussianNoise : public LinLayer {
public:
    float stdev;
    static int total_layers;

    LGaussianNoise(Layer *parent, float stdev, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p);

    Layer *clone(int c, int bs, vector<Layer *>, int todev);

    void forward();

    void backward();

    string plot(int c);
};


/// RNN Layer
class LRNN : public LinLayer {
public:
    int units;
    int num_layers;
    bool use_bias;
    float dropout;
    bool bidirectional;
    static int total_layers;

    LRNN(Layer *parent, int units, int num_layers, bool use_bias, float dropout, bool bidirectional, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p);

    Layer *clone(int c, int bs, vector<Layer *>, int todev);

    void forward();

    void backward();

    string plot(int c);
};


/// LSTM Layer
class LLSTM : public LinLayer {
public:
    int units;
    int num_layers;
    bool use_bias;
    float dropout;
    bool bidirectional;
    static int total_layers;

    LLSTM(Layer *parent, int units, int num_layers, bool use_bias, float dropout, bool bidirectional, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p);

    Layer *clone(int c, int bs, vector<Layer *>, int todev);

    void forward();

    void backward();

    string plot(int c);
};

#endif
