#include "annealing.h"

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
    int i, j, k;
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

    /*利得行列.今回は 3人分×各戦略数*/
    int payoff[num_player][num_stg[0]][num_stg[1]][num_stg[2]];

    for (int p = 0; p < num_player; p++)
    {
        for (i = 0; i < num_stg[0]; i++)
        {
            for (j = 0; j < num_stg[1]; j++)
            {
                for (k = 0; k < num_stg[2]; k++)
                {
                    payoff[p][i][j][k] = payoff_of_3_prisoners_dilemma(p, i, j, k);
                }
            }
        }
    }

    for (int p = 0; p < num_player; p++)
    {
        printf("player%d\n", p);
        for (i = 0; i < num_stg[0]; i++)
        {
            for (j = 0; j < num_stg[1]; j++)
            {
                for (k = 0; k < num_stg[2]; k++)
                {
                    printf("%d, ", payoff[p][i][j][k]);
                }
            }
        }
        printf("\n");
    }

    /* 2026/3/9 : 利得までOK*/

    double J[N][N] = {0.0};
    double H[Nums] = {0.0};
    double complex f1[Nums] = {0.0 + 0.0 * I};

    double B0 = 1.0;
    int Time = 1000000;
    double tau = 1.0;

    /*J[i][j]の定義.これは問題によって定義する*/
    /*H = - Σ J[i][j]q[i]q[j], -ついていることに注意*/
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            if (i != j)
                J[i][j];
        }
    }

    /*時間発展*/
    time_evolution(f1, J, Time, B0, tau);

    /*正規化.時間発展中で毎回行うのが実際だが、計算上は最後にまとめて行っても良い。*/
    normalize(f1);

    /*最終出力*/
    double p;
    for (int i = 0; i < Nums; i++)
    {
        p = cabs(f1[i]) * cabs(f1[i]);
        printf("%d : %f\n", i, p);
    }
}
