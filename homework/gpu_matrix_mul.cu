#include"cuda_runtime.h"
#include"../common/book.h"
#include<stdio.h>
#include<stdlib.h>

#define BLOCK_SIZE 32

__global__ void matrix_mul(int* M, int* N, int* P, int m, int k, int n) {
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    int sum = 0;

    // TAG: 这种写法是因为启动的 32x32 的 thread 在某些位置可能超过 100x100 原始矩阵的计算范围
    // valid range check
    if(tx < n && ty < m) {
        for (int i=0; i<k; ++i) {
            sum += M[i + ty*k] * N[tx + i*n];
        }
        // TAG: 虽然计算时是分块计算的，但是回填时仍然按照原始大矩阵的位置
        P[tx + ty*n] = sum;
    }
}

int main(int argc, char const* argv[]) {
    // 1. specify size the matrix `M`, `N`, `P` and allocate memory for them
    int m = 100;
    int n = 100;
    int k = 100;

    int *h_M, *h_N, *h_P, *h_cc;

    // TODO: what is diffrence between `cudaMallocHost` and `malloc`
    // TAG: memory which is applicated by `cudaMallocHost` need `cudaFree` to get free
    HANDLE_ERROR(cudaMallocHost((void**) &h_M, sizeof(int)*m*k));
    HANDLE_ERROR(cudaMallocHost((void**) &h_N, sizeof(int)*k*n));
    HANDLE_ERROR(cudaMallocHost((void**) &h_P, sizeof(int)*m*n));
    // cudaMallocHost((void**) &h_cc, sizeof(int)*m*n);

    // 2. initialize `M`, `N`
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            h_M[j + i*k] = rand() % 32; 
        }
    }

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            h_N[j + i*n] = rand() % 32; 
        }
    }

    // 3. allocate memory of GPU for `M`, `N`, `P` 
    // and copy only `M` and `N` to GPU
    int *d_M, *d_N, *d_P;
    HANDLE_ERROR(cudaMalloc((void**) &d_M, sizeof(int)*m*k));
    HANDLE_ERROR(cudaMalloc((void**) &d_N, sizeof(int)*k*n));
    HANDLE_ERROR(cudaMalloc((void**) &d_P, sizeof(int)*m*n));

    HANDLE_ERROR(cudaMemcpy(d_M, h_M, sizeof(int)*m*k, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_N, h_N, sizeof(int)*k*n, cudaMemcpyHostToDevice));

    // 4. set size of block and thread
    // TODO: 为什么采用这种 block 的数值设置风格？
    // TAG: block 内能申请的 thread 个数是有限制的，最多启动 1024 个 thread，所以只能将问题分块
    // 当 BLOCK_SIZE 设置成 32x32 时，4x4 个 block 能覆盖住本例中 100x100 的矩阵运算
    // 每个 block 负责 100x100 原始矩阵的*一部分*的计算
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // TAG: 这里 `x` 轴的方向是列、`y` 轴方向是行，不要弄混
    dim3 gridDim(grid_cols, grid_rows);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    // 5. execute the kernel
    matrix_mul<<<gridDim, blockDim>>>(d_M, d_N, d_P, m, k, n);

    // 6. copy the result back to CPU
    HANDLE_ERROR(cudaMemcpy(h_P, d_P, sizeof(int)*m*n, cudaMemcpyDeviceToHost));

    // 7. free memory on both GPU and CPU
    cudaThreadSynchronize();
    HANDLE_ERROR(cudaFree(d_M));
    HANDLE_ERROR(cudaFree(d_N));
    HANDLE_ERROR(cudaFree(d_P));
    HANDLE_ERROR(cudaFree(h_M));
    HANDLE_ERROR(cudaFree(h_N));
    HANDLE_ERROR(cudaFree(h_P));
    
    return 0;
}