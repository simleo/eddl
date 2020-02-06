/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* copyright (c) 2020, CRS4
* Date: February 2020
* Author: PRHLT Research Centre, UPV; CRS4 (rparedes@prhlt.upv.es), (jon@prhlt.upv.es); (simone.leo@crs4.it)
* All rights reserved
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>

#include "apis/eddl.h"
#include "apis/eddlT.h"
#include "metrics/metric.h"

using namespace eddl;

//////////////////////////////////
// mnist_custom_metric.cpp:
// same as mnist_auto_encoder,
// but with a custom metric
//////////////////////////////////

class MRootMeanSquaredError : public Metric {
public:
    MRootMeanSquaredError();
    float value(Tensor *T, Tensor *Y) override;
};

MRootMeanSquaredError::MRootMeanSquaredError() : Metric("root_mean_squared_error") {}

float MRootMeanSquaredError::value(Tensor *T, Tensor *Y) {
    float f;
    int size=T->size/T->shape[0];
    Tensor *aux1 = new Tensor(T->getShape(), T->device);
    Tensor::add(1.0, T, -1.0, Y, aux1, 0);
    Tensor::el_mult(aux1, aux1, aux1, 0);
    f = std::sqrt(aux1->sum()/size);
    delete aux1;
    return f;
}


int main(int argc, char **argv) {

    // Download dataset
    download_mnist();

    // Settings
    int epochs = 10;
    int batch_size = 100;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = Activation(Dense(l, 256), "relu");
    l = Activation(Dense(l, 128), "relu");
    l = Activation(Dense(l, 64), "relu");
    l = Activation(Dense(l, 128), "relu");
    l = Activation(Dense(l, 256), "relu");

    layer out = Dense(l, 784);

    model net = Model({in}, {out});

    // View model
    summary(net);
    plot(net, "model.pdf");

    Metric* rmse = new MRootMeanSquaredError();

    // Build model
    build(net,
          sgd(0.001, 0.9), // Optimizer
          {getLoss("mean_squared_error")}, // Losses
          {rmse}, // Metrics
          CS_CPU()
          //CS_GPU({1})
    );

    // Load dataset
    tensor x_train = eddlT::load("trX.bin");
    // Preprocessing
    eddlT::div_(x_train, 255.0);

    // Train model
    fit(net, {x_train}, {x_train}, batch_size, epochs);

}


///////////
