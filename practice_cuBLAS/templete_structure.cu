#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>

int main()
{
    // cublas の宣言
    cublasHandle_t handle;
    cublasCreate(&handle);

    // GPUメモリ確保
    double *d_A;
    cudaMalloc((void **)&d_A, N * sizeof(double));

    // 使用後の解放
    cudaFree(d_A);
    cublasDestroy(handle);
    
    return 1;
}
