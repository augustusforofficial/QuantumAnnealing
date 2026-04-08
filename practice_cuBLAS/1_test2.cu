#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>

#define IDX2C(i,j,ld) ( ( (j) * (ld)) + (i) )

int main(){
    int N = 2;
    
    //行列宣言。列優先に注意
    cuDoubleComplex h_U[4] = {
        make_cuDoubleComplex(0.0,0.0),
        make_cuDoubleComplex(0.0,1.0),
        make_cuDoubleComplex(0.0,-1.0),
        make_cuDoubleComplex(0.0,0.0)
    };

    cuDoubleComplex h_vec[2] = {
        make_cuDoubleComplex(1.0,0.0),
        make_cuDoubleComplex(0.0,0.0)
    };

    //GPUメモリ確保
    cuDoubleComplex *d_U, *d_vec, *d_out;
    cudaMalloc((void**)&d_vec, N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_U,   N * N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_out, N * sizeof(cuDoubleComplex));

    //ハンドルの宣言
    cublasHandle_t handle;
    cublasCreate(&handle);

    // CPUメモリ => GPUメモリへ
    cublasSetVector(N, sizeof(cuDoubleComplex), h_vec, 1, d_vec, 1);
    cublasSetVector(N * N, sizeof(cuDoubleComplex), h_U, 1, d_U, 1);

    // 行列とベクトルの積のためのパラメータ宣言
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

    //行列積 d_out = U * d_vec
    // d_out = alpha * op(d_U) *d_vec + beta * d_out
    cublasZgemv(handle, CUBLAS_OP_N, N, N, &alpha, d_U, N, d_vec, 1, &beta, d_out, 1);

    // GPU => CPU
    cuDoubleComplex result[N];
    cublasGetVector(N, sizeof(cuDoubleComplex), d_out, 1, result, 1);

    printf("result = (%f,%f) , (%f,%f)\n", cuCreal(result[0]), cuCimag(result[0]), cuCreal(result[1]), cuCimag(result[1]));

    // メモリ開放 + Destroy
    cudaFree(d_U);
    cudaFree(d_vec);
    cudaFree(d_out);
    cublasDestroy(handle);

    return 0;
}