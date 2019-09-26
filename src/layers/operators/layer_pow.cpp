
/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer_operators.h"


using namespace std;

int LPow::total_layers = 0;

/**
  @brief Computes the power of one layer to another

  @param l1 a Layer.
  @param l2 a Layer.
  @param name a name for the operation (predefined as 'pow+TotalPowLayers')
  @param dev which computing service utilize

  @returns the result of l1^l2

  Example:
  \verbatim
      # x contains [[2, 2], [3, 3]]
      # y contains [[8, 10], [2, 3]]
      eddl.Pow(x, y)  # [[256, 1024], [9, 27]]
   \endverbatim

  */
LPow::LPow(Layer *l1, Layer *l2, string name, int dev): OperatorLayer(name, dev) {
    if(name.empty()) this->name = "pow_" + to_string(++total_layers);
    //TODO: Implement
}

/**
  @brief Computes the power of one layer to a value

  @param l a Layer.
  @param k a float.
  @param name a name for the operation (predefined as 'diff+TotalDiffLayers')
  @param dev which computing service utilize

  @returns the result of l^k

  Example:
  \verbatim
      # x contains [[1, 2], [3, 4]]
      eddl.Pow(x, 2.0)  # [[1, 4], [9, 16]]
   \endverbatim

  */
LPow::LPow(Layer *l, float k, string name, int dev): OperatorLayer(name, dev) {
    if(name.empty()) this->name = "pow_" + to_string(++total_layers);
    //TODO: Implement
}

void LPow::forward(){
    //TODO: Implement
}

void LPow::backward(){
    //TODO: Implement
}

Layer *LPow::share(int c, int bs, vector<Layer *> p) {

    return nullptr;
}

Layer *LPow::clone(int c, int bs, vector<Layer *> p, int todev) {

    return nullptr;
}
