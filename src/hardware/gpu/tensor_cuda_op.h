// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019 Roberto Paredes Palacios, <rparedes@dsic.upv.es>

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

#ifndef _TENSOR_CUDA_OP_
#define _TENSOR_CUDA_OP_

#include <cuda.h>
#include "../tensor/tensor.h"


void gpu_set(Tensor *A,float v);
void gpu_mult(Tensor *A,float v);
void gpu_sum(Tensor *A, float v);
void gpu_log(Tensor *A);
void gpu_exp(Tensor *A);
void gpu_sqrt(Tensor *A);
void gpu_sqr(Tensor *A);
void gpu_mask(Tensor *A,float v);
void gpu_total_sum(Tensor *A,float *tot);

void gpu_copy_to_gpu(float *nptr,Tensor *B);
void gpu_copy_from_gpu(Tensor *A,float *nptr);
void gpu_copy_gpu(Tensor *A,Tensor *B);
void gpu_fill(Tensor *A,int aini,int aend,Tensor *B,int bini,int bend,int inc);

void gpu_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C,int incC);
void gpu_sum2D(float scA,Tensor *A, float scB,Tensor *B, Tensor *C,int incC);
void gpu_sum(float scA,Tensor *A, float scB,Tensor *B, Tensor *C,int incC);

void gpu_el_mult(Tensor *A,Tensor *B,Tensor *C,int incC);
void gpu_el_div(Tensor *A,Tensor *B,Tensor *C,int incC);

void gpu_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C);
void gpu_sum2D_colwise(Tensor *A, Tensor *B, Tensor *C);
void gpu_reduce_sum2D(Tensor *A,Tensor *B,int axis,int incB);

void gpu_rand_uniform(Tensor *A, float v);
void gpu_rand_suniform(Tensor *A, float v);
void gpu_rand_gaussian(Tensor *A, float m,float s);
void gpu_rand_binary(Tensor *A, float v);

void gpu_cent(Tensor *A,Tensor *B,Tensor *C);
void gpu_accuracy(Tensor *A,Tensor *B,int *acc);


void gpu_relu(Tensor *A,Tensor *B);
void gpu_d_relu(Tensor *D,Tensor *I,Tensor *PD);
void gpu_softmax(Tensor *A,Tensor *B);

#endif