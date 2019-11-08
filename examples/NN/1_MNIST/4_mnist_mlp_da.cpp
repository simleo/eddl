/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "apis/eddl.h"
#include "apis/eddlT.h"

using namespace eddl;

//////////////////////////////////
// mnist_mlp.cpp:
// A very basic MLP for mnist
// Playing with Data Augmentation
// Using fit for training
//////////////////////////////////

int main(int argc, char **argv) {

    // Download mnist
    download_mnist();

    // Settings
    int epochs = 25;
    int batch_size = 100;
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    // Data augmentation
    layer lda = ShiftRandom(l, {-0.1f, +0.1f}, {-0.1f, +0.1f});
    // Note that Data Augmentation is performed in the same
    // Computing Service, e.g. GPU

    l = ReLu(Dense(lda, 1024));
    l = ReLu(Dense(l, 1024));
    l = ReLu(Dense(l, 1024));

    layer out = Activation(Dense(l, num_classes), "softmax");
    model net = Model({in}, {out});

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    // Build model
    build(net,
          sgd(0.01, 0.9), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          //CS_GPU({1}) // one GPU
          CS_CPU() // CPU with maximum threads availables
    );

    // View model
    summary(net);

    // Load dataset
    tensor x_train = eddlT::load("trX.bin");
    tensor y_train = eddlT::load("trY.bin");
    tensor x_test = eddlT::load("tsX.bin");
    tensor y_test = eddlT::load("tsY.bin");

    // Preprocessing
    eddlT::div_(x_train, 255.0);
    eddlT::div_(x_test, 255.0);



    tensor t=eddlT::select(x_train,0);
    eddlT::reshape_(t,{1,1,28,28});
    eddlT::save_png(t,"m.png");

    tensor t2=t->clone();

    Tensor::shift_random(t,t2,{-0.1,0.1},{-0.1,0.1});
    eddlT::save_png(t2,"m2.png");


    vector<int> indices = random_indices(batch_size, 100);

    train_batch(net, {x_train}, {y_train}, indices);

    t=eddlT::select(lda->input,0);
    eddlT::reshape_(t,{1,1,28,28});
    eddlT::save_png(t,"l.png");

    t2=eddlT::select(lda->output,0);
    t2->print();
    eddlT::reshape_(t2,{1,1,28,28});
    eddlT::save_png(t2,"l2.png");



}


///////////                               
