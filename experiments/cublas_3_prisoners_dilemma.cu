#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <math.h>

#define N 21
#define Nums 2097152

int payoff_of_3_prisoners_dilemma(int p, int i, int j, int k)
{
    /*1. 全員自白*/
    if (i && j && k)
        return -5;
    /*2. 全員黙秘*/
    if (!i && !j && !k)
        return -3;

    /*pで戦略を参照できるように strategy配列としてまとめる*/
    int strategy[3] = {i, j, k};

    /*3. 自分が自白、ほか黙秘あり*/
    if (strategy[p] == 1)
        return -1;
    /*4. 自分が黙秘、ほか自白あり*/
    return -10;
}

int iBitNumRight(int d, int i)
{
    return (d >> i) & 1;
}

void embed_diagonal_H(double H[Nums], double J[N][N])
{
    int candidate_num, j, k;
    /*対角成分の初期化*/
    /*問題によって異なる*/
    for (candidate_num = 0; candidate_num < Nums; candidate_num++)
    {
        for (j = 0; j < N;
            j++)
        {
            for (k = j + 1; k < N; k++)
            {
                H[candidate_num] += (2 * iBitNumRight(candidate_num, j) - 1) * (2 * iBitNumRight(candidate_num, k) - 1) * J[j][k];
            }
        }
    }
}

/*iとjの2進数でのハミング距離を返却*/
/*Brian Kernighanのアルゴリズム*/
/* O(logN) */
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

/* improve : Tの埋め込みは CUDA にて並列化したい */
/* improve : そもそも T を行列として保持したくない。 H, f0, f1　のみの保持で行いたい*/
void embed_T(cuDoubleComplex h_T[Nums][Nums], double diagonal_H[Nums], double At, double Bt, double dt){
    int i,j;
    for (i=0;i<Nums;i++){
        for(j=0; j<Nums; j++){
            h_T[i][j] = make_cuDoubleComplex(0.0,0.0);
            if(i==j){
                h_T[i][j] = make_cuDoubleComplex(1.0, -0.5 * diagonal_H[i] * At * dt);
            }else if(hamDistance(i,j) == 1){
                h_T[i][j] = make_cuDoubleComplex(0.0, -1 * Bt * dt);
            }
        }
    }
}

cublasStatus_t cublas_normalize(cublasHandle_t handle, int dim, cuDoubleComplex *d_vec){
    cublasStatus_t status;

    // improve : norm が CPUメモリ側の変数だから通信コストある
    double norm;
    status = cublasDznrm2(handle, dim, d_vec, 1, &norm);
    if(status != CUBLAS_STATUS_SUCCESS){
        return status;
    }

    if(norm == 0.0){
        return CUBLAS_STATUS_SUCCESS;
    }
    double normalization_factor = 1 / norm;
    status = cublasZdscal(handle, dim, &normalization_factor, d_vec, 1);
    if(status != CUBLAS_STATUS_SUCCESS){
        return status;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublas_time_evolution_Hamiltonian(cublasHandle_t handle,cuDoubleComplex *d_T, cuDoubleComplex *f0, cuDoubleComplex *f1, double diagonal_H[Nums], int Time, double B0, double tau){
    double dt = tau / (double)Time;
    int time;
    double t;
    /*時間発展での行列積のためのハイパーパラメータ*/
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0,0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0,0.0);
    cublasStatus_t status;
    cuDoubleComplex *tmp; // (f0,f1入れ替え用) 

    /* improve : Tを行列として保持しているが、これを都度計算にしてスレッド並列化するほうが絶対に良い。*/
    cuDoubleComplex h_T[Nums][Nums];

    /*時間発展関数化 (f0,f1)を入れたら、それを変更したい。*/
    for (time = 0; time < Time; time++)
    {
        t = (double)time * dt;
        /*A(t)はハミルトニアンの係数.tに単調増加*/
        double At = t / tau;
        /*B(t)は横磁場の大きさ.tに単調減少*/
        double Bt = B0 * (1 - t / tau);

        /* ついに時間発展 f1=T・f0 */
        // improve : Tを古典的に埋め込み
        embed_T(h_T, diagonal_H, At, Bt, dt);
        cublasSetVector(Nums * Nums, sizeof(cuDoubleComplex), h_T, 1, d_T, 1);
        status = cublasZgemv(handle, CUBLAS_OP_N, Nums, Nums, &alpha, d_T, Nums, f0, 1, &beta, f1, 1);

        /* f1 の正規化*/
        status =cublas_normalize(handle,Nums,f1);

        /* f0 <- f1 のためにポインタを入れ替える */
        /* f1 <- f0 になるが、どうせ f1 は次の時間発展で入れ替えになる。 */
        tmp = f0;
        f0 = f1;
        f1 = tmp;
    }

    return status;
}

cublasStatus_t cublas_time_evolution(cublasHandle_t handle, cuDoubleComplex *d_T, cuDoubleComplex* f0 ,cuDoubleComplex* f1 , double J[N][N], int Time, double B0, double tau){
    /* improve : J[N][N] => H[Nums] の埋め込みを並列化 + H[Nums]をGPU上で保持 */
    double diagonal_H[Nums];
    embed_diagonal_H(diagonal_H, J);
    
    cublas_time_evolution_Hamiltonian(handle, d_T, f0, f1, diagonal_H, Time, B0, tau);
}

void show_vector(cuDoubleComplex *vec, int dim, char label[]){
    printf("%s\n", label);
    double probablicity = 0.0;
    for(int i=0;i<dim;i++){
        probablicity = cuCreal(vec[i]) * cuCreal(vec[i]) + cuCimag(vec[i]) * cuCimag(vec[i]);
        printf("%d : %f\n", i, probablicity);
    }
}

int main(){
// 1. CPUメモリにおける変数宣言
    int i,j,k;
    // 定数宣言
    int ni[N] = {2,3,5};
    double J[N][N] = {0.0};
    double H[Nums] = {0.0};
    cuDoubleComplex h_f0[Nums];
    for (i=0;i<Nums;i++){
        h_f0[i] = make_cuDoubleComplex(1.0 / sqrt(Nums), 0.0);
    }

    /*improve : */
    /*定数宣言*/
    /*0 : 黙秘 , 1 : 自白　とする*/
    const int num_player = 3;
    int strategy_0[] = {0, 1};
    int strategy_1[] = {0, 1};
    int strategy_2[] = {0, 1};

    /*各プレイヤーの戦略数を取得*/
    int num_stg[] = {
        sizeof(strategy_0) / sizeof(strategy_0[0]),
        sizeof(strategy_1) / sizeof(strategy_1[0]),
        sizeof(strategy_2) / sizeof(strategy_2[0])};
    int sum_stg = 0;
    for (i = 0; i < num_player; i++)
    {
        sum_stg += num_stg[i];
    }

    /*利得行列.今回は 3人分×各戦略数*/
    int payoff_sum[num_stg[0]][num_stg[1]][num_stg[2]];
    memset(payoff_sum, 0, sizeof(payoff_sum));

    for (int p = 0; p < num_player; p++)
    {
        for (i = 0; i < num_stg[0]; i++)
        {
            for (j = 0; j < num_stg[1]; j++)
            {
                for (k = 0; k < num_stg[2]; k++)
                {
                    payoff_sum[i][j][k] += payoff_of_3_prisoners_dilemma(p, i, j, k);
                }
            }
        }
    }

    // 時間発展定数宣言
    double B0 = 1.0;
    int Time = 1000;
    double tau = 1.0;

    int x, y, z;                           /*戦略のバイナリ変数*/
    int s1, s2, s3;                        /*ペナルティ項毎のスラック変数用*/
    int H0 = 0;                            /*目的関数部分のハミルトニアン関数 H0(q0,q1,q2)（演算子ではない。）*/
    int alpha = -5, beta = -5, gamma = -5; /*戦略の期待値を抑えるハイパーパラメータ*/
    const int num_pen = 6;                 /*ペナルティ項の個数.*/
    const int num_slack = 3;               /*各ペナルティ項におけるスラック変数の個数*/
    const int start_slack = 3;             /*スラックが始まるインデックス番号*/
    int Pen[num_pen];                      /*各ペナルティ項*/

    //J[i][j]の定義
    for(i=0;i<N;i++){
        for(j=0;j<N;j++){
            J[i][j] = -1 * ni[i] * ni[j];
        }
    }

     /*qubit数 : 戦略3つ + スラック3個*6行=18個 の計21個*/
    for (i = 0; i < Nums; i++)
    {
        for (j = 0; j < num_pen; j++)
        {
            Pen[j] = 0;
        }

        /*x,y,z は　iの２進数表記における左から 0,1,2個目のキュビット*/
        x = (i >> (N - 1)) & 1;
        y = (i >> (N - 2)) & 1;
        z = (i >> (N - 3)) & 1;

        /*ハミルトニアンの目的関数値*/
        H[i] += (-1) * payoff_sum[x][y][z];

        /*ペナルティ項の作成*/
        /*improve : Pen[index] の indexの形に拡張性がない。*/
        Pen[0] += payoff_of_3_prisoners_dilemma(0, 0, y, z);
        Pen[1] += payoff_of_3_prisoners_dilemma(0, 1, y, z);
        Pen[2] += payoff_of_3_prisoners_dilemma(1, x, 0, z);
        Pen[3] += payoff_of_3_prisoners_dilemma(1, x, 1, z);
        Pen[4] += payoff_of_3_prisoners_dilemma(2, x, y, 0);
        Pen[5] += payoff_of_3_prisoners_dilemma(2, x, y, 1);

        for (j = 0; j < num_pen; j++)
        {
            s1 = (i >> (N - 1 - j * num_slack - start_slack)) & 1;
            s2 = (i >> (N - 2 - j * num_slack - start_slack)) & 1;
            s3 = (i >> (N - 3 - j * num_slack - start_slack)) & 1;

            Pen[j] += s1 + 2 * s2 + 4 * s3;
        }
        /*ペナルティ項にスラックと alpha,beta,gammaを加える*/
        Pen[0] -= alpha;
        Pen[1] -= alpha;
        Pen[2] -= beta;
        Pen[3] -= beta;
        Pen[4] -= gamma;
        Pen[5] -= gamma;

        for (j = 0; j < num_pen; j++)
        {
            Pen[j] = Pen[j] * Pen[j];
            H[i] += Pen[j];
        }

        H[i] = H[i] + alpha + beta + gamma;

        if (H[i] == 0)
        {
            printf("i = %d\n", i);
            printf("H0 = %d\n", (-1) * payoff_sum[x][y][z]);
            for (j = 0; j < num_pen; j++)
            {
                printf("Pen[%d] = %d\n", j, Pen[j]);
            }
            printf("H[%d] = %f\n", i, H[i]);
        }
    }

    printf("finished");

// 2. GPUハンドルの宣言
    cublasHandle_t handle;
    cublasCreate(&handle);

// 3. GPUメモリにおける変数宣言
    cuDoubleComplex *d_f0,*d_f1,*d_T;
    cudaMalloc((void**) &d_f0, Nums * sizeof(cuDoubleComplex));
    cudaMalloc((void**) &d_f1, Nums * sizeof(cuDoubleComplex));
    cudaMalloc((void**) &d_T, Nums * Nums * sizeof(cuDoubleComplex));

// 4. CPU => GPU
    cublasSetVector(Nums, sizeof(cuDoubleComplex), h_f0, 1, d_f0, 1);

// 5. GPU計算
    cublas_time_evolution_Hamiltonian(handle,d_T,d_f0,d_f1,H,Time,B0,tau);

// 6. GPUメモリ => CPUメモリ
    cublasGetVector(Nums, sizeof(cuDoubleComplex), d_f0, 1, h_f0, 1);
    char label[] = "f0";
    show_vector(h_f0, Nums, label);

// 7. 開放
    cudaFree(d_f0);
    cudaFree(d_f1);
    cublasDestroy(handle);

    return 0;
}