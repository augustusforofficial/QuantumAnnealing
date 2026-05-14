#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <complex.h>
#include "../src/annealing.h"

#define NUM_AP 3
#define NUM_USER 3

/* N=18, Nums = 262144*/
/* optimal solution : 218624*/

/* 意味 : (Σx_i - a)^2 の展開した時の係数を Q[i][j] に足し合わせる*/
/* Q[N][N] : 埋め込む先
    a : 式内のa
    index[N] : Σx に入る場合1,入らない場合0の配列*/
/* 返り値として定数項を返す.*/
double embed_pow_in_Jij(double Q[N][N], int a, double coef[N], double hyper_parameter)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = i; j < N; j++)
        {
            if (i == j)
            {
                Q[i][j] += (coef[i] * coef[i] - 2 * a * coef[i]) * hyper_parameter;
            }
            else
            {
                Q[i][j] += 2 * coef[i] * coef[j] * hyper_parameter;
            }
        }
    }

    return (double) hyper_parameter * a * a;
}

void print_matrix(double A[N][N], double num_row, double num_collum)
{
    printf("matrix\n");
    for (int i = 0; i < num_row; i++)
    {
        for (int j = 0; j < num_collum; j++)
        {
            printf("%.4f, ", A[i][j]);
        }
        printf("\n");
    }
}

void Initialization_array_double(double *A, int length){

    #pragma omp schedule for
    for(int i=0; i<length; i++){
        A[i] = 0.0;
    }
}

/* 定数項も追加するように*/
/* note : H[i] = -1 * f(i) であることに注意*/
void Add_Energy_QUBO_to_Hamiltonian(double H[Nums], double Q[N][N], double term_const){
    for (int c = 0; c < Nums; c++)
    {
        double sum = 0.0;
        for (int i = 0; i < N; i++)
        {
            for (int j = i; j < N; j++)
            {
                sum += Q[i][j] * iBitNumLeft(c, i) * iBitNumLeft(c, j);
            }
        }
        H[c] = (sum + term_const);
    }
}

void Show_Hamiltonian_max_min(double H[Nums]){
    /* H[i] の最大値確認*/
    double max = -99999.9999;
    int max_index = 0;
    for (int i = 0; i < Nums; i++)
    {
        if (max <= H[i])
        {
            max = H[i];
            max_index = i;
        }
    }
    printf("max H[i] is H[%d] = %f\n", max_index, H[max_index]);

    /* H[i] の最小値確認*/
    double min = 99999.9999;
    int min_index = 0;
    for (int i = 0; i < Nums; i++)
    {
        if (min >= H[i])
        {
            min = H[i];
            min_index = i;
        }
    }
    printf("min H[i] is H[%d] = %f\n", min_index, H[min_index]);
}

void Show_matrix_NN(double A[N][N], double row, double column){
    for(int i=0; i<row; i++){
        for(int j=0; j<column; j++){
            printf("%.5f, ", A[i][j]);
        }
        printf("\n");
    }
}

void Show_vector_N(double A[N]){
    for(int i=0; i<N; i++){
        printf("%f, ", A[i]);
    }
}

void Show_vector_Nums(double A[Nums]){
    for(int i=0; i<Nums; i++){
        printf("%d : %f\n",i , A[i]);
    }
}

void make_prob_vec(double complex f1[Nums], double prob[Nums]){
    for(int i=0; i<Nums; i++){
        double re = creal(f1[i]);
        double im = cimag(f1[i]);
        prob[i] = re * re + im * im;
    }
}

int main()
{
    const double sigma = 1;     /*ノイズ*/
    double g[NUM_USER][NUM_AP]; /* improve : 距離の逆数で定義*/
    const int U = 2;            /*各 AP での接続上限数*/
    const int L = 2;            /*各 user の接続下限数*/

    // 各APの座標定義
    double coordinate_AP[NUM_AP][2] = {{30, 30}, {80, 50}, {30, 80}};

    // 各userの座標定義
    double coordinate_user[NUM_USER][2] = {{90, 30}, {20, 50}, {60, 70}};

    // 各AP,user 間の距離の逆数を g へ代入
    double distance;
    double sub_x;
    double sub_y;
    double hyper_parameter_a = 0.5;
    double hyper_parameter_b = 0.5;
    for (int index_user = 0; index_user < NUM_USER; index_user++)
    { // k : ユーザー番号
        for (int index_ap = 0; index_ap < NUM_AP; index_ap++)
        { // l : AP番号
            /* 距離を計算する */
            sub_x = coordinate_user[index_user][0] - coordinate_AP[index_ap][0];
            sub_y = coordinate_user[index_user][1] - coordinate_AP[index_ap][1];
            distance = sub_x * sub_x + sub_y * sub_y;
            distance = sqrt(distance);
            if (distance > 1e-12)
            {
                g[index_user][index_ap] = 1.0 / distance;
            }
            else
            {
                g[index_user][index_ap] = 0.0;
            }
        }
    }

    // for(int i=0; i<NUM_USER; i++){
    //     for(int j=0; j<NUM_AP; j++){
    //         printf("%.5f, ", g[i][j]);
    //     }
    //     printf("\n");
    // }
    /* test : 2026/05/14, g[i][j] はOK */

    // Q[i][j] を g[k][l] から作成する

    /* qubit は以下のように配列するものとする */
    /* s[0][0] s[0][1] s[0][2] s[1][0] ... s[2][2] y[0][0] y[0][1] ... y[2][1] z[0] z[1] z[2] の計 18 qubit*/
    /* bit_index : 0~8　は s[k][l] , 9~14 は y[i][j], 15~17 は z[i]*/
    double Q[N][N] = {0.0}; // Q[i][j] から時間発展させる
    double J[N][N] = {0.0};
    double H[Nums] = {0.0};
    double term_const = 0.0; // ハミルトニアンの定数項用の変数
    double complex f1[Nums] = {0.0 + 0.0 * I};

/*1. 目的関数部分. Q[i][i] += g / sigma^2, Q[i][j] = g*g / sigma^2 */
/* 目的関数部分は max なので -1 をかけて min にする*/
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NUM_USER * NUM_AP; i++)
    { /* 0<=i<= 8 の形. jも同様*/
        for (int j = 0; j < NUM_USER * NUM_AP; j++)


        {
            /* i,jから各々のユーザー番号, AP番号を逆算*/
            int index_user_i = i / NUM_AP;
            int index_user_j = j / NUM_AP;
            int index_AP_i = i % NUM_AP;
            int index_AP_j = j % NUM_AP;
            if (i == j)
            {
                Q[i][j] += -1 * g[index_user_i][index_AP_i] / (sigma * sigma);
            }
            else if (index_user_i != index_user_j)
            {
                Q[i][j] += -1 * g[index_user_i][index_AP_i] * g[index_user_j][index_AP_j] / (sigma * sigma * sigma * sigma);
            }
        }
    }

    /* test : Q[i][j] および ここまでのH[i]も OK.*/
    // Initialization_array_double(H, Nums);
    // Show_matrix_NN(Q, N, N);
    // Add_Energy_QUBO_to_Hamiltonian(H, Q);
    // Show_Hamiltonian_max_min(H);
    

    /*2. >=L の各user下限制約.*/
    int num_bit_slack_of_y = ((int)(log(NUM_AP - L) / log(2)) + 1); /*各行でのスラック変数数. log2(U)*/
    printf("num_bit_slack_of_y : %d\n", num_bit_slack_of_y);
    for (int index_user = 0; index_user < NUM_USER; index_user++)
    {

        /* 関係するqubitは、
         s[index_user][] : 0~2, 3~5, 6~8
         y[index_user][] : 9~10, 11~12, 13~14 */
        int index_start_slack = NUM_USER * NUM_AP;
        double pickup[N]; // 制約式の ()^2 内での係数配列.もし()内に出なければ 0.0 をとる
        Initialization_array_double(pickup, N);
        /* s[][] 部分のピックアップ*/
        for (int i = 0; i < NUM_AP; i++)
        {
            pickup[index_user * NUM_AP + i] = 1.0;
        }
        /* y[][] 部分のピックアップ. y[index_user][i] = 2^i の係数*/
        for (int i = 0; i < num_bit_slack_of_y; i++)
        {
            pickup[index_start_slack + num_bit_slack_of_y * index_user + i] = -1 * pow(2.0, (double)i);
        }
        
        term_const +=  embed_pow_in_Jij(Q, L, pickup, hyper_parameter_a);
    }

    // Show_matrix_NN(Q, N, N);
    // Initialization_array_double(H, Nums);
    // Add_Energy_QUBO_to_Hamiltonian(H, Q);
    // Show_Hamiltonian_max_min(H);

    /*3. <= U の各AP上限制約.*/
    int num_bit_slack_of_z = ((int)(log(U) / log(2)) + 1);
    printf("num_bit_slack_of_z : %d\n", num_bit_slack_of_z);
    for (int index_AP = 0; index_AP < NUM_AP; index_AP++)
    {
        /*今回の slack は、前回の終わりから NUM_USER * num_bit_slack_of_y */
        int index_start_slack = NUM_USER * NUM_AP + NUM_USER * num_bit_slack_of_y;
        double pickup[N] = {0.0};
        Initialization_array_double(pickup, N);
        for (int j = 0; j < NUM_USER; j++)
        {
            pickup[index_AP + j * NUM_AP] = 1.0;
        }
        for (int j = 0; j < num_bit_slack_of_z; j++)
        {
            pickup[index_start_slack + index_AP * num_bit_slack_of_z + j] = pow(2.0, (double)j);
        }
        term_const +=  embed_pow_in_Jij(Q, U, pickup, hyper_parameter_b);
    }

    double B0 = 1.0;
    int Time = 10000;
    double tau = 1.0;

    // Hの初期化
    Initialization_array_double(H, Nums);
    /* improve : Q[i][j] => J[i][j] => H[i] のどこかでミスっている*/
    Add_Energy_QUBO_to_Hamiltonian(H, Q, term_const);
    Show_Hamiltonian_max_min(H);

    time_evolution_Hamiltonian(f1,H,Time,B0,tau);

    double prob[Nums] = {0.0};
    make_prob_vec(f1, prob);
    Show_Hamiltonian_max_min(prob);

    double p;
    FILE *fp = fopen("./bin/CFmMIMO_result.bin", "wb");
    fwrite(prob, sizeof(double), Nums, fp);
    fclose(fp);
}