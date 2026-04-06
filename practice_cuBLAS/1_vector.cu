#include <stdio.h>
#include <cuda_runtime.h>   //GPUメモリ管理など
#include <cublas_v2.h>  //cuBLASの本体
#include <cuComplex.h>  //複素数型の利用

int main(){
    int N = 4;

    //  CPUメモリ上に宣言するベクトル (host)
    cuDoubleComplex h_vec[4] = {
        make_cuDoubleComplex(1.0, 0.0),
        make_cuDoubleComplex(1.0, 0.0),
        make_cuDoubleComplex(0.0, 0.5),
        make_cuDoubleComplex(0.0, 0.5)
    };

    //　GPUメモリの確保
    cuDoubleComplex *d_vec;
    cudaMalloc((void**)&d_vec, N * sizeof(cuDoubleComplex));    // (型) 変数 の宣言が (void**)&d_vec

    // 環境設定クラスのインスタンス handle (使用GPU,テンソルコア設定,ポインタモード, ...)
    cublasHandle_t handle;
    cublasCreate(&handle);

    // CPUメモリ => GPUメモリ への転送
    cublasSetVector(N, sizeof(cuDoubleComplex), h_vec, 1, d_vec, 1);

    // 内積の計算
    cuDoubleComplex result; //CPUの変数
    cublasZdotc(handle, N, d_vec, 1, d_vec, 1, &result);    //incx, incy = 1 : 何個飛ばしか. k = 1 + (i-1) * incx

    // double
    double norm = cuCreal(result);
    printf("norm = %f\n", norm);
    
    // GPUメモリの d_vecのメモリ開放 + cuBLASコンテキスト終了
    cudaFree(d_vec);
    cublasDestroy(handle);
}