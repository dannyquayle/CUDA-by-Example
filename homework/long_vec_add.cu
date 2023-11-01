#include"cuda_runtime.h"
#include"../common/book.h"
#include<stdio.h>

#define N (1024 * 1024)

__global__ void kernel(int* a, int* b, int* c) {
    int tid = blockIdx.x;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += gridDim.x;
    }
}

int main() {
    // 1. claim variables of cpu and gpu
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;

    // 2. cpu memory application of cpu variables
    a = (int*)malloc(N * sizeof(int));
    b = (int*)malloc(N * sizeof(int));
    c = (int*)malloc(N * sizeof(int));

    // 3. initialize cpu varibales
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = 2 * i;
    }

    // 4. gpu memory application of gpu variables
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));

    // 5. memory copy for gpu variables from cpu to gpu
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

    // 6. exec the kernel
    dim3 gridDim(128);
    dim3 blockDim(1);
    kernel<<<gridDim, blockDim>>>(dev_a, dev_b, dev_c);

    // 7. memory copy for cpu variables from gpu to cpu
    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    // 8. result check
    bool result = true;
    for (int i = 0; i < N; i++)
    {
        if (a[i] + b[i] != c[i]) {
            printf("Error:  %d + %d != %d\n", a[i], b[i], c[i]);
            result = false;
            break;
        }
    }
    if (result) printf("WE DID IT!\n");

    // 9. free both the memory of cpu and gpu variables
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));
    free(a);
    free(b);
    free(c);

    return 0;
}