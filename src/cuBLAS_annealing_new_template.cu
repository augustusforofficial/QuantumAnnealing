#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <math.h>

#define N 3
#define Nums 8

cublasStatus_t cublas_normalize(cublasHandle_t handle, cuDoubleComplex *d_vec, int dim){
    double norm;
    cublasDznrm2(handle, dim, d_vec, 1, &norm);

    double normalization_factor = 1 / norm;
    cublasZdscal(handle, dim, &normalization_factor, d_vec, 1);
}

cublasStatus_t cublas_time_evolution(cublasHandle_t handle, cuDoubleComplex* f0 ,cuDoubleComplex* f1 , double J[N][N], int Time, double B0, double tau){
    double dt = tau / (double)Time;
    int time;
    double t;
    cuDoubleComplex *tmp; // (f0,f1入れ替え用) 
    cuDoubleComplex *T;
    cudaMalloc((void**) &T, Nums * Nums * sizeof(cuDoubleComplex));

    /*時間発展関数化 (f0,f1)を入れたら、それを変更したい。*/
    for (time = 0; time < Time; time++)
    {
        t = (double)time * dt;
        /*A(t)はハミルトニアンの係数.tに単調増加*/
        double At = t / tau;
        /*B(t)は横磁場の大きさ.tに単調減少*/
        double Bt = B0 * (1 - t / tau);

        /*ついに時間発展 f1=T・f0*/
        

        /* f1 の正規化*/
        cublas_normalize(handle,f1,Nums);

        /* f0 <- f1 のためにポインタを入れ替える */
        /* f1 <- f0 になるが、どうせ f1 は次の時間発展で入れ替えになる。 */
        tmp = f0;
        f0 = f1;
        f1 = tmp;
    }
}

void show_vector(cuDoubleComplex *vec, int dim, char label[]){
    printf("%s\n", label);
    for(int i=0;i<dim;i++){
        printf("(%f, %f)\n", cuCreal(vec[i]), cuCimag(vec[i]));
    }
    printf("\n");
}

int main(){
// 1. CPUメモリにおける変数宣言

    int i,j,k;
    // 定数宣言
    int ni[N] = {2,3,5};
    double J[N][N] = {0.0};
    double H[Nums] = {0.0};
    cuDoubleComplex h_f1[Nums] = {make_cuDoubleComplex(0.0, 0.0)};
    char label[] = "test";
    show_vector(h_f1, Nums, label);

    // 時間発展定数宣言
    double B0 = 1.0;
    int Time = 1000;
    double tau = 1.0;

    //J[i][j]の定義
    for(i=0;i<N;i++){
        for(j=0;j<N;j++){
            J[i][j] = -1 * ni[i] * ni[j];
        }
    }

// 2. GPUハンドルの宣言
    cublasHandle_t handle;
    cublasCreate(&handle);

// 3. GPUメモリにおける変数宣言
    cuDoubleComplex *d_f0,*d_f1;
    cudaMalloc((void**) &d_f0, Nums * sizeof(cuDoubleComplex));
    cudaMalloc((void**) &d_f1, Nums * sizeof(cuDoubleComplex));

    cublas_time_evolution(handle,d_f0,d_f1,J,Time,B0,tau);
}