#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <math.h>

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

// calculate tensor product A * B
void kron2_2(cuDoubleComplex *A, cuDoubleComplex *B, cuDoubleComplex *C)
{
    int i, j, k, l;
    int m = 2; // Aのサイズ m*m
    int n = 2; // Bのサイズ n*n

    for (i = 0; i < m; i++)
    {
        for (j = 0; j < m; j++)
        {
            cuDoubleComplex a = A[IDX2C(i, j, m)];

            for (k = 0; k < n; k++)
            {
                for (l = 0; l < n; l++)
                {
                    int row = i * n + k;
                    int col = j * n + l;

                    C[IDX2C(row, col, 4)] = cuCmul(a, B[IDX2C(k, l, n)]);
                }
            }
        }
    }
}

void show_matrix(cuDoubleComplex *A, int m, int n, char label[]){
    printf("%s\n", label);
    int i,j;
    for (i=0; i<m; i++){
        for (j=0; j<n; j++){
            cuDoubleComplex a = A[IDX2C(i,j,m)];
            printf("(%f,%f) ", cuCreal(a), cuCimag(a));
        }
        printf("\n");
    }
}

void show_vector(cuDoubleComplex *vec, int dim, char label[]){
    printf("%s\n", label);
    for(int i=0;i<dim;i++){
        printf("(%f, %f)\n", cuCreal(vec[i]), cuCimag(vec[i]));
    }
    printf("\n");
}

int main()
{
    // 1qubit + 1ancilla のアダマールテストを実装してみる

    int num_qubit = 2;
    int dim = 1; // dim = 4
    for (int i = 0; i < num_qubit; i++)
    {
        dim *= 2;
    }

    // cuDoubleComplex h_vec[num_qubit] = {
    //     make_cuDoubleComplex(1.0, 0.0),
    //     make_cuDoubleComplex(0.0, 0.0),
    //     make_cuDoubleComplex(0.0, 0.0),
    //     make_cuDoubleComplex(0.0, 0.0)};

// STEP1. CPUメモリでの変数定義
    cuDoubleComplex h_vec[dim] = {
        make_cuDoubleComplex(1.0,0.0),
        make_cuDoubleComplex(0.0,0.0),
        make_cuDoubleComplex(0.0,0.0),
        make_cuDoubleComplex(0.0,0.0)
    };

    cuDoubleComplex h_vec_2[dim] = {
        make_cuDoubleComplex(0.0, 0.0),
        make_cuDoubleComplex(0.0, 0.0),
        make_cuDoubleComplex(0.0, 0.0),
        make_cuDoubleComplex(0.0, 0.0)
    };

    double s = 1.0 / sqrt(2.0);
    cuDoubleComplex h_Hadamard[4] = {
        make_cuDoubleComplex(s, 0.0),
        make_cuDoubleComplex(s, 0.0),
        make_cuDoubleComplex(s, 0.0),
        make_cuDoubleComplex(-s, 0.0)};

    cuDoubleComplex h_Identity[4] = {
        make_cuDoubleComplex(1.0, 0.0),
        make_cuDoubleComplex(0.0, 0.0),
        make_cuDoubleComplex(0.0, 0.0),
        make_cuDoubleComplex(1.0, 0.0)};

    cuDoubleComplex h_Projection_0[4] = {
        make_cuDoubleComplex(1.0, 0.0),
        make_cuDoubleComplex(0.0, 0.0),
        make_cuDoubleComplex(0.0, 0.0),
        make_cuDoubleComplex(0.0, 0.0)};

    cuDoubleComplex h_Projection_1[4] = {
        make_cuDoubleComplex(0.0, 0.0),
        make_cuDoubleComplex(0.0, 0.0),
        make_cuDoubleComplex(0.0, 0.0),
        make_cuDoubleComplex(1.0, 0.0)
    };

    cuDoubleComplex h_X[4] = {
        make_cuDoubleComplex(0.0, 0.0),
        make_cuDoubleComplex(1.0, 0.0),
        make_cuDoubleComplex(1.0,0.0),
        make_cuDoubleComplex(0.0,0.0)
    };

    // I * H の宣言
    cuDoubleComplex h_Hadamard_qubit[dim * dim];
    kron2_2(h_Identity, h_Hadamard, h_Hadamard_qubit);  // I * H

    cuDoubleComplex h_Identity_right_0_projection[dim*dim]; // I * |0><0|
    cuDoubleComplex h_X_right_1_projection[dim*dim];    // X * |1><1|
    kron2_2(h_Identity, h_Projection_0, h_Identity_right_0_projection);
    kron2_2(h_X, h_Projection_1, h_X_right_1_projection);
    cuDoubleComplex h_CNOT_to_left[dim*dim];
    
// STEP2. GPUメモリの確保
    cuDoubleComplex *d_vec, *d_vec_2, *d_Hadamard_qubit, *d_CNOT_to_left, *d_Identity_right_0_projection, *d_X_right_1_projection;
    cudaMalloc((void**) &d_vec, dim * sizeof(cuDoubleComplex));
    cudaMalloc((void**) &d_vec_2, dim * sizeof(cuDoubleComplex));
    cudaMalloc((void**) &d_Hadamard_qubit, dim * dim * sizeof(cuDoubleComplex));
    cudaMalloc((void**) &d_CNOT_to_left, dim * dim * sizeof(cuDoubleComplex));
    cudaMalloc((void**) &d_Identity_right_0_projection, dim * dim * sizeof(cuDoubleComplex));
    cudaMalloc((void**) &d_X_right_1_projection, dim * dim * sizeof(cuDoubleComplex));

// STEP3. GPUの状況記録の handle　宣言
    cublasHandle_t handle;
    cublasCreate(&handle);

// STEP4. CPUメモリ => GPUメモリ へのデータ移行
    cublasSetVector(dim, sizeof(cuDoubleComplex), h_vec, 1, d_vec, 1);
    cublasSetVector(dim, sizeof(cuDoubleComplex), h_vec_2, 1, d_vec_2, 1);
    cublasSetVector(dim * dim, sizeof(cuDoubleComplex), h_Hadamard_qubit, 1, d_Hadamard_qubit, 1);
    cublasSetVector(dim * dim, sizeof(cuDoubleComplex), h_CNOT_to_left, 1, d_CNOT_to_left, 1);
    cublasSetVector(dim * dim, sizeof(cuDoubleComplex), h_Identity_right_0_projection, 1, d_Identity_right_0_projection, 1);
    cublasSetVector(dim * dim, sizeof(cuDoubleComplex), h_X_right_1_projection, 1, d_X_right_1_projection, 1);

// STEP5. 計算
    // GPU上にて CNOT を作成
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(1.0, 0.0);
    cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, &alpha, d_Identity_right_0_projection, dim, &beta, d_X_right_1_projection, dim, d_CNOT_to_left, dim);
    
    cuDoubleComplex vector[dim];
    char label[] = "test";
    // Hadamardゲートを作用させる
    alpha = make_cuDoubleComplex(1.0, 0.0);
    beta = make_cuDoubleComplex(0.0, 0.0);
    cublasZgemv(handle, CUBLAS_OP_N, dim, dim, &alpha, d_Hadamard_qubit, dim, d_vec, 1, &beta, d_vec_2, 1);
    // リセット
    cublasZcopy(handle, dim, d_vec_2, 1, d_vec, 1);
    cublasSetVector(dim, sizeof(cuDoubleComplex), h_vec_2, 1, d_vec_2, 1);
    cublasGetVector(dim, sizeof(cuDoubleComplex), d_vec, 1, vector, 1);
    show_vector(vector, dim, label);
    // CNOTゲート作用
    alpha = make_cuDoubleComplex(1.0, 0.0);
    beta = make_cuDoubleComplex(0.0, 0.0);
    cublasZgemv(handle, CUBLAS_OP_N, dim, dim, &alpha, d_CNOT_to_left, dim, d_vec, 1, &beta, d_vec_2, 1);
    // リセット
    cublasZcopy(handle, dim, d_vec_2, 1, d_vec, 1);
    cublasSetVector(dim, sizeof(cuDoubleComplex), h_vec_2, 1, d_vec_2, 1);
    cublasGetVector(dim, sizeof(cuDoubleComplex), d_vec, 1, vector, 1);
    show_vector(vector, dim, label);
    // Hadamardゲートの作用
    alpha = make_cuDoubleComplex(1.0, 0.0);
    beta = make_cuDoubleComplex(0.0, 0.0);
    cublasZgemv(handle, CUBLAS_OP_N, dim, dim, &alpha, d_Hadamard_qubit, dim, d_vec, 1, &beta, d_vec_2, 1);
    // リセット(d_vec_2 はリセットしない)
    cublasZcopy(handle, dim, d_vec_2, 1, d_vec, 1);
    cublasGetVector(dim, sizeof(cuDoubleComplex), d_vec, 1, vector, 1);
    show_vector(vector, dim, label);

// STEP6.  GPU => CPU
    cuDoubleComplex result[dim];
    cublasGetVector(dim, sizeof(cuDoubleComplex), d_vec, 1, result, 1);
    char str2[] = "result vector";
    show_vector(result, dim, str2);

// STEP7. メモリ開放 + Destoroy
    cudaFree(d_vec);
    cudaFree(d_vec_2);
    cudaFree(d_Hadamard_qubit);
    cudaFree(d_CNOT_to_left);
    cudaFree(d_Identity_right_0_projection);
    cudaFree(d_X_right_1_projection);
    cublasDestroy(handle);
}