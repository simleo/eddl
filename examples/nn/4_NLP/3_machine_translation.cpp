/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"


using namespace eddl;

Tensor *onehot(Tensor *in, int vocs)
{
  int n=in->shape[0];
  int l=in->shape[1];

  Tensor *out=new Tensor({n,l,vocs});
  out->fill_(0.0);

  int p=0;
  for(int i=0;i<n*l;i++,p+=vocs) {
    int w=in->ptr[i];
    out->ptr[p+w]=1.0;
  }

  return out;
}


//////////////////////////////////
// MT
// using EuTrans
//////////////////////////////////
int main(int argc, char **argv) {
    // Download EuTrans
    //download_eutrans();

    // Settings
    int epochs = 10;
    int batch_size = 32;

    int ilength=30;
    int olength=30;
    int invs=687;
    int outvs=514;
    int embedding=64;

    // Define network
    layer in = Input({1}); //1 word
    layer l = in;

    layer lE = Dropout(RandomUniform(Embedding(l, invs, 1,embedding),-0.05,0.05),0.2);

    l = LSTM(lE,128);

    // Decoder
    l = Decoder(LSTM(l,128),outvs);

    layer out = Softmax(Dense(l, outvs));

    model net = Model({in}, {out});

    // dot from graphviz should be installed:
    plot(net, "model.pdf");



    optimizer opt=adam(0.001);
    //opt->set_clip_val(0.01);

    // Build model
    build(net,
          opt, // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"accuracy"}, // Metrics
          //CS_GPU({1}) // one GPU
          //CS_GPU({1,1},100) // two GPU with weight sync every 100 batches
          CS_CPU()
    );

    // View model
    summary(net);


    // Load dataset
    Tensor *x_train=Tensor::load("EuTransCod/tres.bin","bin");
    Tensor *y_train=Tensor::load("EuTransCod/tren.bin","bin");
    y_train=onehot(y_train,outvs);

    Tensor *x_test=Tensor::load("EuTransCod/tses.bin","bin");
    Tensor *y_test=Tensor::load("EuTransCod/tsen.bin","bin");
    y_test=onehot(y_test,outvs);

    x_train->reshape_({x_train->shape[0],ilength,1}); //batch x timesteps x input_dim
    y_train->reshape_({y_train->shape[0],olength,outvs}); //batch x timesteps x ouput_dim

    x_test->reshape_({x_test->shape[0],ilength,1}); //batch x timesteps x input_dim
    y_test->reshape_({y_test->shape[0],olength,outvs}); //batch x timesteps x ouput_dim

    // Train model
    for(int i=0;i<epochs;i++) {
      fit(net, {x_train}, {y_train}, batch_size, 1);
      evaluate(net,{x_test},{y_test});
    }

}
