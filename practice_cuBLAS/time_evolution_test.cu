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

    cuDoubleComplex h_Hadamard_qubit[dim * dim];
    kron2_2(h_Identity, h_Hadamard, h_Hadamard_qubit);

    cuDoubleComplex h_Identity_right_0_projection[dim*dim];
    cuDoubleComplex h_X_right_1_projection[dim*dim];
    kron2_2(h_Identity, h_Projection_0, h_Identity_right_0_projection);
    kron2_2(h_X, h_Projection_1, h_X_right_1_projection);
    char str1[] = "h_Hadamard_qubit";
    char str2[] = "h_Identity_right_0_projection";
    char str3[] = "h_X_right_1_projection";
    show_matrix(h_Hadamard_qubit,dim,dim,str1);
    show_matrix(h_Identity_right_0_projection,dim,dim,str2);
    show_matrix(h_X_right_1_projection, dim, dim, str3);

    // cuDoubleComplex h_CNOT_left_to_right[dim*dim];
}