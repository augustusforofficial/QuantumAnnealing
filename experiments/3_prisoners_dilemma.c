#include "../src/annealing.h"
#include "string.h"

/*p:利得プレイヤー番号*/
/*i,j,k : それぞれプレイヤー1,2,3 の戦略番号. 今回は　i,j,k = 0,1*/
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

int main()
{
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
    for (int i = 0; i < num_player; i++)
    {
        sum_stg += num_stg[i];
    }

    /*利得行列.今回は 3人分×各戦略数*/
    int payoff_sum[num_stg[0]][num_stg[1]][num_stg[2]];
    memset(payoff_sum, 0, sizeof(payoff_sum));

    for (int p = 0; p < num_player; p++)
    {
        for (int i = 0; i < num_stg[0]; i++)
        {
            for (int j = 0; j < num_stg[1]; j++)
            {
                for (int k = 0; k < num_stg[2]; k++)
                {
                    payoff_sum[i][j][k] += payoff_of_3_prisoners_dilemma(p, i, j, k);
                }
            }
        }
    }

    /* 2026/3/9,13 : 利得までOK*/
    double *H = calloc(Nums, sizeof(double)); /*ハミルトニアンの対角項*/
    double complex *f2 = calloc(Nums, sizeof(double complex));

    double B0 = 1.0;
    int Time = 10000;
    double tau = 1.0;

    /*数値シミュレーション上では J[i][j]はいらない。関数式そのものにすべての場合を代入すれば対角成分は計算可能である*/
    /* <0010|H^|0010> = H(0,0,1,0) ここで H^ は横磁場イジングモデルにおける作用素であることに注意。*/
    /* 従って、演算子行列 H の対角項は　H(0,0,0) ~ H(1,1,1) までの計算で可能である。*/

    int alpha = -5, beta = -5, gamma = -5; /*戦略の期待値を抑えるハイパーパラメータ*/
    const double hypers = 20;      /*制約項のハイパーパラメータ*/
    const int num_pen = 6;                 /*ペナルティ項の個数.*/
    const int num_slack = 3;               /*各ペナルティ項におけるスラック変数の個数*/
    const int start_slack = 3;             /*スラックが始まるインデックス番号*/

    /*qubit数 : 戦略3つ + スラック3個*6行=18個 の計21個*/
    /* ここに #pragma omp parallel for でスタック*/
    for (int i = 0; i < Nums; i++)
    {
        int x, y, z;                           /*戦略のバイナリ変数*/
        int s1, s2, s3;                        /*ペナルティ項毎のスラック変数用*/
        int Pen[num_pen];                      /*各ペナルティ項*/
        for (int j = 0; j < num_pen; j++)
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

        for (int j = 0; j < num_pen; j++)
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

        for (int j = 0; j < num_pen; j++)
        {
            Pen[j] = Pen[j] * Pen[j];
            H[i] += hypers * Pen[j];
        }

        H[i] = H[i] + alpha + beta + gamma;
    }

    // #pragma omp parallel for
    // for(int i=0; i<Nums; i++){
    //     H[i] = -1 * H[i];
    // }

    Show_Hamiltonian_max_min(H);

    /*時間発展*/
    time_evolution_Hamiltonian_Iidaka(f2,H,Time,B0,tau);

    /*確率ベクトルの作成*/
    double *prob = calloc(Nums, sizeof(double));
    make_prob_vec(f2, prob);
    Show_top_X(prob, 10);

    /*最終出力*/
    FILE *fp = fopen("./3_prisoners_dilemma_Hamiltonian_0522_02.bin", "wb");
    fwrite(H, sizeof(double), Nums, fp);
    fclose(fp);

    free(H);
    free(f2);
    free(prob);
}
