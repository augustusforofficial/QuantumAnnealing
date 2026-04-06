#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>

cublasStatus_t normalization_cuDoubleComplex(cublasHandle_t handle, int n, cuDoubleComplex *vec, int incx){
    // norm の計算
    double norm;
    cublasStatus_t status;

    // norm calculation
    status = cublasDznrm2(handle, n, vec, incx, &norm);
    if (status != CUBLAS_STATUS_SUCCESS){
        return status;
    }

    // 正規化
    if (norm == 0.0){
        return CUBLAS_STATUS_SUCCESS;
    }
    double normalization_factor = 1 / norm;
    status = cublasZdscal(handle, n, &normalization_factor, vec, incx);
    return status;
}

int main() {
    int  N = 4;

    cuDoubleComplex h_vec[N] = {
        make_cuDoubleComplex(1.0 , 0.0),
        make_cuDoubleComplex(0.5, 0.5),
        make_cuDoubleComplex(0.5, 0.0),
        make_cuDoubleComplex(0.0, 0.5)
    };

    cuDoubleComplex *d_vec;
    cudaMalloc((void**) &d_vec, N * sizeof(cuDoubleComplex));

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSetVector(N, sizeof(cuDoubleComplex), h_vec, 1, d_vec, 1);

    normalization_cuDoubleComplex(handle, N, d_vec, 1);

    // GPU => CPU　のコピー
    cuDoubleComplex h_result[N];
    cublasGetVector(N, sizeof(cuDoubleComplex), d_vec, 1, h_result, 1);

    for(int i=0; i<N;i++){
        double real = cuCreal(h_result[i]);
        double image = cuCimag(h_result[i]);
        printf(" %f + %f * I \n",real,image);
    }

    cudaFree(d_vec);
    cublasDestroy(handle);
}