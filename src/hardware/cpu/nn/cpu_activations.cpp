/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */
#include <iostream>

#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"

void cpu_relu(Tensor *A, Tensor *B){ 
  _profile(_CPU_RELU, 0);
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++) {
    if (A->ptr[i] > 0.0) B->ptr[i] = A->ptr[i];
    else B->ptr[i] = 0.0;
  }
    _profile(_CPU_RELU, 1);
}

void cpu_d_relu(Tensor *D, Tensor *I, Tensor *PD){
 _profile(_CPU_D_RELU, 0);
  #pragma omp parallel for
  for (int i = 0; i < D->size; i++) {
    if (I->ptr[i] > 0.0) PD->ptr[i] += D->ptr[i];
    else PD->ptr[i] += 0.0;
  }
    _profile(_CPU_D_RELU, 1);
}

void cpu_thresholded_relu(Tensor *A, Tensor *B,float param){
  _profile(_CPU_THRESHOLDED_RELU, 0);
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++) {
    if (A->ptr[i] > param) B->ptr[i] = A->ptr[i];
    else B->ptr[i] = 0.0;
  }
    _profile(_CPU_THRESHOLDED_RELU, 1);
}

void cpu_d_thresholded_relu(Tensor *D, Tensor *I, Tensor *PD,float param){
  _profile(_CPU_D_THRESHOLDED_RELU, 0);
  #pragma omp parallel for
  for (int i = 0; i < D->size; i++) {
    if (I->ptr[i] > param) PD->ptr[i] += D->ptr[i];
    else PD->ptr[i] += 0.0;
  }
    _profile(_CPU_D_THRESHOLDED_RELU, 1);
}

void cpu_leaky_relu(Tensor *A, Tensor *B,float param){
  _profile(_CPU_LEAKY_RELU, 0);
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++) {
    if (A->ptr[i] > 0.0) B->ptr[i] = A->ptr[i];
    else B->ptr[i] = param*A->ptr[i];;
  }
    _profile(_CPU_LEAKY_RELU, 1);
}

void cpu_d_leaky_relu(Tensor *D, Tensor *I, Tensor *PD,float param){
  _profile(_CPU_D_LEAKY_RELU, 0);
  #pragma omp parallel for
  for (int i = 0; i < D->size; i++) {
    if (I->ptr[i] > 0.0) PD->ptr[i] += D->ptr[i];
    else PD->ptr[i] += param*D->ptr[i];
  }
    _profile(_CPU_D_LEAKY_RELU, 1);
}

void cpu_elu(Tensor *A, Tensor *B, float param){
  _profile(_CPU_ELU, 0);
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++) {
    if (A->ptr[i] > 0.0) B->ptr[i] = A->ptr[i];
    else B->ptr[i] = param * (::expf(A->ptr[i]) - 1.0);
  }
    _profile(_CPU_ELU, 1);
}

void cpu_d_elu(Tensor *D, Tensor *I, Tensor *PD, float param){
  _profile(_CPU_D_ELU, 0);
  #pragma omp parallel for
  for (int i = 0; i < D->size; i++) {
    if (I->ptr[i] > 0.0) PD->ptr[i] += D->ptr[i];
    else PD->ptr[i] += D->ptr[i] * (param * ::expf(I->ptr[i]));
  }
    _profile(_CPU_D_ELU, 1);
}

void cpu_softplus(Tensor *A, Tensor *B){
    _profile(_CPU_SOFTPLUS, 0);
    #pragma omp parallel for
    for (int i = 0; i < A->size; i++) {
        B->ptr[i] = ::logf(1 + ::expf(A->ptr[i]));
    }
    _profile(_CPU_SOFTPLUS, 1);
}

void cpu_d_softplus(Tensor *D, Tensor *I, Tensor *PD){
    _profile(_CPU_D_SOFTPLUS, 0);
    #pragma omp parallel for
    for (int i = 0; i < D->size; i++) {
        PD->ptr[i] += D->ptr[i] * 1/(1 + ::expf(-I->ptr[i]));
    }
    _profile(_CPU_D_SOFTPLUS, 1);
}

void cpu_softsign(Tensor *A, Tensor *B){
    _profile(_CPU_SOFTSIGN, 0);
    #pragma omp parallel for
    for (int i = 0; i < A->size; i++) {
        B->ptr[i] = A->ptr[i] / (1 + ::fabs(A->ptr[i]));
    }
    _profile(_CPU_SOFTSIGN, 1);
}

void cpu_d_softsign(Tensor *D, Tensor *I, Tensor *PD){
    _profile(_CPU_D_SOFTSIGN, 0);
    #pragma omp parallel for
    for (int i = 0; i < D->size; i++) {
        float denom = 1 + ::fabs(I->ptr[i]);
        PD->ptr[i] += D->ptr[i] * 1/(denom*denom);
    }
    _profile(_CPU_D_SOFTSIGN, 1);
}

void cpu_linear(Tensor *A, Tensor *B, float param){
  _profile(_CPU_LINEAR, 0);
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++) {
    B->ptr[i] = param * A->ptr[i];
  }
    _profile(_CPU_LINEAR, 1);
}

void cpu_d_linear(Tensor *D, Tensor *I, Tensor *PD, float param){
  _profile(_CPU_D_LINEAR, 0);
  #pragma omp parallel for
  for (int i = 0; i < D->size; i++) {
    PD->ptr[i] += D->ptr[i] * param;
  }
    _profile(_CPU_D_LINEAR, 1);
}

//void cpu_sigmoid(Tensor *A, Tensor *B){
//  _profile(_CPU_SIGMOID, 0);
//  #pragma omp parallel for
//  for (int i = 0; i < A->size; i++)
//    B->ptr[i] = 1/(1+std::exp(-A->ptr[i]));
//    _profile(_CPU_SIGMOID, 1);
//}

void cpu_d_sigmoid(Tensor *D, Tensor *I, Tensor *PD){
    _profile(_CPU_D_SIGMOID, 0);
  #pragma omp parallel for
  for (int i = 0; i < D->size; i++)
    PD->ptr[i] += D->ptr[i]*((1-I->ptr[i])*I->ptr[i]);
    _profile(_CPU_D_SIGMOID, 1);
}

void cpu_hard_sigmoid(Tensor *A, Tensor *B){
  _profile(_CPU_HARD_SIGMOID, 0);
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++) {
    if (A->ptr[i] > 2.5) B->ptr[i] = 1.0;
    else if (A->ptr[i] < -2.5) B->ptr[i] = 0.0;
    else B->ptr[i] = (0.2 * A->ptr[i]) + 0.5;
  }
    _profile(_CPU_HARD_SIGMOID, 1);
}

void cpu_d_hard_sigmoid(Tensor *D, Tensor *I, Tensor *PD){
  _profile(_CPU_D_HARD_SIGMOID, 0);
  #pragma omp parallel for
  for (int i = 0; i < D->size; i++)
    if (I->ptr[i] < -2.5 || I->ptr[i] > 2.5) PD->ptr[i] += 0;
    else PD->ptr[i] += D->ptr[i] * 0.2;
    _profile(_CPU_D_HARD_SIGMOID, 1);
}

//void cpu_exp(Tensor *A, Tensor *B){
//  _profile(_CPU_EXP, 0);
//  #pragma omp parallel for
//  for (int i = 0; i < A->size; i++) {
//    B->ptr[i] = std::exp(A->ptr[i]);
//  }
//    _profile(_CPU_EXP, 1);
//}

void cpu_d_exp(Tensor *D, Tensor *I, Tensor *PD){
  _profile(_CPU_D_EXP, 0);
  #pragma omp parallel for
  for (int i = 0; i < D->size; i++)
    PD->ptr[i] += D->ptr[i] * I->ptr[i];
    _profile(_CPU_D_EXP, 1);
}

//void cpu_tanh(Tensor *A, Tensor *B){
//  _profile(_CPU_TANH, 0);
//  #pragma omp parallel for
//  for (int i = 0; i < A->size; i++) {
//    float p=std::exp(A->ptr[i]);
//    float n=std::exp(-A->ptr[i]);
//    B->ptr[i] = (p-n)/(p+n);
//  }
//    _profile(_CPU_TANH, 1);
//}

void cpu_d_tanh(Tensor *D, Tensor *I, Tensor *PD){
  _profile(_CPU_D_TANH, 0);
  #pragma omp parallel for
  for (int i = 0; i < D->size; i++)
    PD->ptr[i] += D->ptr[i]*(1-(I->ptr[i]*I->ptr[i]));
    _profile(_CPU_D_TANH, 1);
}


void cpu_softmax(Tensor *A, Tensor *B) {
  _profile(_CPU_SOFTMAX, 0);
  float max, sum;

  //#pragma omp parallel for
  for (int i = 0; i < A->shape[0]; i++) {
    max = (*A->ptr2).col(i).maxCoeff();
    for (int j = 0; j < A->shape[1]; j++)
    (*B->ptr2)(j, i) = std::exp((*A->ptr2)(j, i) - max);

    sum = (*B->ptr2).col(i).sum();
    for (int j = 0; j < B->shape[1]; j++)
    (*B->ptr2)(j, i) = (*B->ptr2)(j, i) / sum;
  }
    _profile(_CPU_SOFTMAX, 1);
}

void cpu_d_softmax(Tensor *D, Tensor *I, Tensor *PD) {
    _profile(_CPU_D_SOFTMAX, 0);
  PD->tsem->lock();

  #pragma omp parallel for
  for (int i = 0; i < D->size; i++)
    PD->ptr[i] += D->ptr[i] * (I->ptr[i] * (1.0 - I->ptr[i]));

  PD->tsem->unlock();
    _profile(_CPU_D_SOFTMAX, 1);
}
