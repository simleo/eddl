/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "gpu_tensor.h"
#include "gpu_kernels.h"
#include "gpu_hw.h"

#include "../../tensor/tensor.h"
#include "../../descriptors/descriptors.h"

void gpu_shift(Tensor *A, Tensor *B, vector<int> t_shift, int mode, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(B);

    // Copy vector from host to device
    int *d_shift; cudaMalloc((int**)&d_shift, 2*sizeof(int));
    cudaMemcpy(d_shift, t_shift.data(), 2*sizeof(int), cudaMemcpyHostToDevice);

    shift<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], A->shape[1], A->shape[2], A->shape[3], d_shift, mode, constant);
    check_cuda(cudaDeviceSynchronize(),"shift");
}

void gpu_rotate(Tensor *A, Tensor *B, float angle, vector<int> axis, int mode, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(B);

//    rotate<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(),"rotate");

}

void gpu_scale(Tensor *A, Tensor *B, vector<int> new_shape, int mode, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

 // Copy vector from host to device
    int *d_new_shape; cudaMalloc((int**)&d_new_shape, new_shape.size()*sizeof(int));
    cudaMemcpy(d_new_shape, new_shape.data(), new_shape.size()*sizeof(int), cudaMemcpyHostToDevice);

    setDims(B);
    scale<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], A->shape[1], A->shape[2], A->shape[3], B->shape[2], B->shape[3], d_new_shape, mode, constant);
    check_cuda(cudaDeviceSynchronize(), "scale");
}

void gpu_flip(Tensor *A, Tensor *B, int axis){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(B);

    flip<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], A->shape[1], A->shape[2], A->shape[3], axis, true);
    check_cuda(cudaDeviceSynchronize(), "flip");
}

void gpu_crop(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant, bool inverse){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(B);

    // Copy vector from host to device
    int *d_coords_from; cudaMalloc((int**)&d_coords_from, coords_from.size()*sizeof(int));
    cudaMemcpy(d_coords_from, coords_from.data(), coords_from.size()*sizeof(int), cudaMemcpyHostToDevice);

    // Copy vector from host to device
    int *d_coords_to; cudaMalloc((int**)&d_coords_to, coords_to.size()*sizeof(int));
    cudaMemcpy(d_coords_to, coords_to.data(), coords_to.size()*sizeof(int), cudaMemcpyHostToDevice);

    crop<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], A->shape[1], A->shape[2], A->shape[3], B->shape[2], B->shape[3], d_coords_from, d_coords_to, constant, inverse);
    check_cuda(cudaDeviceSynchronize(),"crop");
}

void gpu_crop_scale(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, int mode, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(B);

    // Copy vector from host to device
    int *d_coords_from; cudaMalloc((int**)&d_coords_from, coords_from.size()*sizeof(int));
    cudaMemcpy(d_coords_from, coords_from.data(), coords_from.size()*sizeof(int), cudaMemcpyHostToDevice);

    // Copy vector from host to device
    int *d_coords_to; cudaMalloc((int**)&d_coords_to, coords_to.size()*sizeof(int));
    cudaMemcpy(d_coords_to, coords_to.data(), coords_to.size()*sizeof(int), cudaMemcpyHostToDevice);

    crop_scale<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], A->shape[1], A->shape[2], A->shape[3], B->shape[2], B->shape[3], d_coords_from, d_coords_to, mode, constant);
    check_cuda(cudaDeviceSynchronize(), "crop_scale");
}


void gpu_shift_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, int mode, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Copy vector from host to device
    float *d_factor_x; cudaMalloc((float**)&d_factor_x, 2*sizeof(float));
    cudaMemcpy(d_factor_x, factor_x.data(), 2*sizeof(float), cudaMemcpyHostToDevice);

    // Copy vector from host to device
    float *d_factor_y; cudaMalloc((float**)&d_factor_y, 2*sizeof(float));
    cudaMemcpy(d_factor_y, factor_y.data(), 2*sizeof(float), cudaMemcpyHostToDevice);

    // Generate random numbers
    int rnd_size = A->shape[0] * 2;  // Batch x dims (x,y,...)
    float *d_rnd; cudaMalloc((float**)&d_rnd, rnd_size*sizeof(float));
    int *t_bdim = get_block_dim(rnd_size, MAX_TPB);
    uniform_array<<<t_bdim[0], t_bdim[1]>>>(d_rnd, rnd_size, 42);

    setDims(B);
    //shift_random<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], A->shape[1], A->shape[2], A->shape[3], d_factor_x, d_factor_y, mode, constant, d_rnd);
    check_cuda(cudaDeviceSynchronize(),"shift_random");
}

void gpu_rotate_random(Tensor *A, Tensor *B, vector<float> factor, vector<int> axis, int mode, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Copy vector from host to device
    float *d_factor; cudaMalloc((float**)&d_factor, 2*sizeof(float));
    cudaMemcpy(d_factor, factor.data(), 2*sizeof(float), cudaMemcpyHostToDevice);

    // Generate random numbers
    int rnd_size = A->shape[0] * 1;  // Batch x dims (x,y,...)
    float *d_rnd; cudaMalloc((float**)&d_rnd, rnd_size*sizeof(float));
    int *t_bdim = get_block_dim(rnd_size, MAX_TPB);
    uniform_array<<<t_bdim[0], t_bdim[1]>>>(d_rnd, rnd_size, 42);

    //setDims(B);
    //rotate_random<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(),"rotate_random");

}

void gpu_scale_random(Tensor *A, Tensor *B, vector<float> factor, int mode, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Copy vector from host to device
    float *d_factor; cudaMalloc((float**)&d_factor, 2*sizeof(float));
    cudaMemcpy(d_factor, factor.data(), 2*sizeof(float), cudaMemcpyHostToDevice);

    // Generate random numbers
    int rnd_size = A->shape[0] * 1;  // Batch x dims (x,y,...)
    float *d_rnd; cudaMalloc((float**)&d_rnd, rnd_size*sizeof(float));
    int *t_bdim = get_block_dim(rnd_size, MAX_TPB);
    uniform_array<<<t_bdim[0], t_bdim[1]>>>(d_rnd, rnd_size, 42);

    setDims(B);
    //scale_random<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], A->shape[1], A->shape[2], A->shape[3], B->shape[2], B->shape[3], d_factor, mode, constant, d_rnd);
    check_cuda(cudaDeviceSynchronize(),"scale_random");
}

void gpu_flip_random(Tensor *A, Tensor *B, int axis){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Generate random numbers
    int rnd_size = A->shape[0] * 1;  // Batch x dims (x,y,...)
    float *d_rnd; cudaMalloc((float**)&d_rnd, rnd_size*sizeof(float));
    int *t_bdim = get_block_dim(rnd_size, MAX_TPB);
    uniform_array<<<t_bdim[0], t_bdim[1]>>>(d_rnd, rnd_size, 42);

    setDims(B);
    //flip_random<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], A->shape[1], A->shape[2], A->shape[3], axis, d_rnd);
    check_cuda(cudaDeviceSynchronize(),"flip_random");
}

void gpu_crop_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant, bool inverse){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Copy vector from host to device
    float *d_factor_x; cudaMalloc((float**)&d_factor_x, 2*sizeof(float));
    cudaMemcpy(d_factor_x, factor_x.data(), 2*sizeof(float), cudaMemcpyHostToDevice);

    // Copy vector from host to device
    float *d_factor_y; cudaMalloc((float**)&d_factor_y, 2*sizeof(float));
    cudaMemcpy(d_factor_y, factor_y.data(), 2*sizeof(float), cudaMemcpyHostToDevice);

    // Generate random numbers
    int rnd_size = A->shape[0] * 4;  // Batch x dims (x,y,...)
    float *d_rnd; cudaMalloc((float**)&d_rnd, rnd_size*sizeof(float));
    int *t_bdim = get_block_dim(rnd_size, MAX_TPB);
    uniform_array<<<t_bdim[0], t_bdim[1]>>>(d_rnd, rnd_size, 42);

    setDims(B);
    //crop_random<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], A->shape[1], A->shape[2], A->shape[3], B->shape[2], B->shape[3], d_factor_x, d_factor_y, constant, d_rnd);
    check_cuda(cudaDeviceSynchronize(),"crop_random");
}


void gpu_crop_scale_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, int mode, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Copy vector from host to device
    float *d_factor_x; cudaMalloc((float**)&d_factor_x, 2*sizeof(float));
    cudaMemcpy(d_factor_x, factor_x.data(), 2*sizeof(float), cudaMemcpyHostToDevice);

    // Copy vector from host to device
    float *d_factor_y; cudaMalloc((float**)&d_factor_y, 2*sizeof(float));
    cudaMemcpy(d_factor_y, factor_y.data(), 2*sizeof(float), cudaMemcpyHostToDevice);

    // Generate random numbers
    int rnd_size = A->shape[0] * 4;  // Batch x dims (x,y,...)
    float *d_rnd; cudaMalloc((float**)&d_rnd, rnd_size*sizeof(float));
    int *t_bdim = get_block_dim(rnd_size, MAX_TPB);
    uniform_array<<<t_bdim[0], t_bdim[1]>>>(d_rnd, rnd_size, 42);
    
    setDims(B);
    //crop_scale_random<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], A->shape[1], A->shape[2], A->shape[3], B->shape[2], B->shape[3], d_factor_x, d_factor_y, constant, d_rnd);
    check_cuda(cudaDeviceSynchronize(),"crop_scale_random");
}
