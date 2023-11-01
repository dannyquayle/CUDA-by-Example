#include"cuda_runtime.h"
#include"../common/book.h"
#include<stdio.h>

__global__ void printThreadIdx() {
    // block 外偏移 + block 内部偏移
    // 并按两个方向分解
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    // 线性化计算行主序整体 tid
    unsigned tid = ty * gridDim.x * blockDim.x + tx;
    printf("--thread_id (%d, %d)--block_id (%d, %d)--coordinate (%d, %d)--global idx %d\n", 
        threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, tx, ty, tid);
}

int main() {
    // multi block thread location
    dim3 gridDim(2,2);
    dim3 blockDim(4,2);
    // single block thread location
    // dim3 gridDim(1);
    // dim3 blockDim(8,4);
    printThreadIdx<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
}