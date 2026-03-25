#include "cuBLAS_annealing.h"

/*整数の累乗*/
int int_pow(int base, int exp)
{
    int res = 1;
    for (int i = 0; i < exp; i++)
    {
        res *= base;
    }
    return res;
}

/*σ=+-1 => q=0,1*/
int StoQ(int s)
{
    return (1 + s) / 2;
}

/*q=0,1 => σ=+-1*/
int QtoS(int q)
{
    return 2 * q - 1;
}

/*ビット演算子による実装*/
int iBitNumLeft(int d, int i)
{
    return (d >> (N - i - 1)) & 1;
}

/*ビット演算子による実装*/
int iBitNumRight(int d, int i)
{
    return (d >> i) & 1;
}

/*iとjの2進数でのハミング距離を返却*/
/*Brian Kernighanのアルゴリズム*/
int hamDistance(int i, int j)
{
    int count = 0;
    int k = i ^ j; /*XOR*/
    while (k > 0)
    {
        k &= k - 1;
        count++;
    }
    return count;
}

/*ハミルトニアン対角埋め込み*/
/*iBitNumの切り替えで読み方を変更可能*/
void embed_diagonal_H(double *H, double J[N][N])
{
    int i, j, k;
    /*対角成分の初期化*/
    /*問題によって異なる*/
    for (i = 0; i < Nums; i++)
    {
        for (j = 0; j < N; j++)
        {
            for (k = j + 1; k < N; k++)
            {
                H[i] += (2 * iBitNumRight(i, j) - 1) * (2 * iBitNumRight(i, k) - 1) * J[j][k];
            }
        }
    }
}

/* CUDA kernels */
__global__ void diagonal_kernel(cuDoubleComplex *f1, cuDoubleComplex *f0, double *H, double At, double dt, int num_states)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_states)
    {
        cuDoubleComplex coeff = make_cuDoubleComplex(1.0, -0.5 * H[i] * At * dt);
        f1[i] = cuCmul(coeff, f0[i]);
    }
}

__global__ void off_diagonal_kernel(cuDoubleComplex *f1, cuDoubleComplex *f0, double Bt, double dt, int num_bits, int num_states)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_states)
    {
        cuDoubleComplex coeff = make_cuDoubleComplex(0.0, -Bt * dt);
        for (int bit = 0; bit < num_bits; bit++)
        {
            int j = i ^ (1 << bit);
            f1[i] = cuCadd(f1[i], cuCmul(coeff, f0[j]));
        }
    }
}

__global__ void normalize_kernel(cuDoubleComplex *f, double norm, int num_states)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Nums)
    {
        f[i] = cuCdiv(f[i], make_cuDoubleComplex(norm, 0.0));
    }
}

void time_evolution_Hamiltonian_cu(cuDoubleComplex *h_f1, double *H, int Time, double B0, double tau)
{
    double dt = tau / (double)Time;

    // Device memory
    cuDoubleComplex *d_f0, *d_f1;
    double *d_H;
    cudaMalloc(&d_f0, Nums * sizeof(cuDoubleComplex));
    cudaMalloc(&d_f1, Nums * sizeof(cuDoubleComplex));
    cudaMalloc(&d_H, Nums * sizeof(double));

    // Initialize f0
    cuDoubleComplex *h_f0 = (cuDoubleComplex *)malloc(Nums * sizeof(cuDoubleComplex));
    for (int i = 0; i < Nums; i++)
    {
        h_f0[i] = make_cuDoubleComplex(1.0 / sqrt(Nums), 0.0);
    }
    cudaMemcpy(d_f0, h_f0, Nums * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_H, H, Nums * sizeof(double), cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    int threads = 256;
    int blocks = (Nums + threads - 1) / threads;

    for (int time = 0; time < Time; time++)
    {
        double t = (double)time * dt;
        double At = t / tau;
        double Bt = B0 * (1 - t / tau);

        // f1 = 0
        cudaMemset(d_f1, 0, Nums * sizeof(cuDoubleComplex));

        // Diagonal term
        diagonal_kernel<<<blocks, threads>>>(d_f1, d_f0, d_H, At, dt, Nums);

        // Off-diagonal term
        off_diagonal_kernel<<<blocks, threads>>>(d_f1, d_f0, Bt, dt, N, Nums);

        // Normalize
        double norm;
        cublasDznrm2(handle, Nums, d_f1, 1, &norm);
        normalize_kernel<<<blocks, threads>>>(d_f1, norm, Nums);

        // f0 = f1
        cudaMemcpy(d_f0, d_f1, Nums * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    }

    // Copy result back
    cudaMemcpy(h_f1, d_f1, Nums * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_f0);
    cudaFree(d_f1);
    cudaFree(d_H);
    cublasDestroy(handle);
    free(h_f0);
}

/*時間発展*/
void time_evolution_cu(cuDoubleComplex *f1, double J[N][N], int Time, double B0, double tau)
{
    /*ハミルトニアン対角成分の定義*/
    double *H = (double *)malloc(Nums * sizeof(double));
    memset(H, 0, Nums * sizeof(double));
    embed_diagonal_H(H, J);
    
    /*ハミルトニアンの対角項を渡して計算させる*/
    time_evolution_Hamiltonian_cu(f1, H, Time, B0, tau);
    free(H);
}

/*正規化 - for host data*/
void normalize(cuDoubleComplex psi[Nums])
{
    double abs_f1 = 0.0;
    for (int i = 0; i < Nums; i++)
    {
        double real = cuCreal(psi[i]);
        double imag = cuCimag(psi[i]);
        abs_f1 += real * real + imag * imag;
    }
    abs_f1 = sqrt(abs_f1);
    for (int i = 0; i < Nums; i++)
    {
        psi[i] = cuCdiv(psi[i], make_cuDoubleComplex(abs_f1, 0.0));
    }
}

/*結果の出力*/
void print_state(cuDoubleComplex psi[Nums])
{
    double p;
    for (int i = 0; i < Nums; i++)
    {
        printf("%f + %f * I\n", cuCreal(psi[i]), cuCimag(psi[i]));
    }
    for (int i = 0; i < Nums; i++)
    {
        double abs_val = cuCabs(psi[i]);
        p = abs_val * abs_val;
        printf("%d : %f\n", i, p);
    }
}