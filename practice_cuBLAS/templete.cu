#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>

//4次元の複素ベクトルのスカラー倍を cuBLAS にて実装
int main()
{
    // STEP1. CPUメモリでの変数宣言
    int N = 4;
    cuDoubleComplex h_vec[N] = {
        make_cuDoubleComplex(1.0,1.0),
        make_cuDoubleComplex(1.0,1.0),
        make_cuDoubleComplex(1.0,1.0),
        make_cuDoubleComplex(1.0,1.0)
    };

    // STEP2. GPUメモリの確保
    cuDoubleComplex *d_vec;
    cudaMalloc((void**) &d_vec, N * sizeof(cuDoubleComplex) );

    // STEP3. ハンドルの宣言 + Create
    cublasHandle_t handle;
    cublasCreate(&handle);

    // STEP4. CPUメモリ -> GPUメモリの転送
    cudaMemcpy(d_vec, h_vec, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  
    // STEP5. 計算
    double alpha = 3.0;
    cublasZdscal(handle, N, &alpha, d_vec, 1);

    // STEP6. GPUメモリ => CPUメモリの転送
    cudaMemcpy(h_vec, d_vec, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // STEP7. メモリ開放 + Destory
    cudaFree(d_vec);
    cublasDestroy(handle);
}
