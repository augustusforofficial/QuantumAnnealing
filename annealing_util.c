#include "annealing.h"

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
void embed_diagonal_H(double H[Nums], double J[Nums][Nums])
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

/*時間発展*/
void time_evolution(double complex f1[Nums], double J[Nums][Nums], int Time, double B0, double tau)
{
    double dt = tau / (double)Time;
    int time;
    double t;

    /*f0の定義*/
    double complex f0[Nums];
    for (int i = 0; i < Nums; i++)
    {
        f0[i] = (1.0 / sqrt(Nums)) + 0.0 * I;
    }

    /*ハミルトニアン対角成分の定義*/
    double H[Nums] = {0.0};
    embed_diagonal_H(H,J);

    /*時間発展関数化 (f0,f1)を入れたら、それを変更したい。*/
    for (time = 0; time < Time; time++)
    {
        t = (double)time * dt;
        /*A(t)はハミルトニアンの係数.tに単調増加*/
        double At = t / tau;
        /*B(t)は横磁場の大きさ.tに単調減少*/
        double Bt = B0 * (1 - t / tau);

        printf("time_evolution%d\n",time);

        /*時間発展演算子Tを作成*/
        double complex T[Nums][Nums] = {0.0 + 0.0 * I};

        /*対角,磁場,0すべてやる*/
        for(int i=0; i<Nums; i++){
            for(int j=0; j<Nums; j++){
                if(i==j){
                    T[i][j] = 1.0 - ((H[i]*At*dt*0.5) * I);
                }else if(hamDistance(i,j)==1){
                    T[i][j] = -1 * Bt * dt * I;
                }else{
                    T[i][j] = 0.0;
                }
            }
        }

        // /*非対角成分*/
        // double Ht[Nums][Nums] = {0.0};
        // for (int i = 0; i < Nums; i++)
        // {
        //     for (int j = 0; j < Nums; j++)
        //     {
        //         if (hamDistance(i, j) == 1)
        //         {
        //             Ht[i][j] = -1 * Bt;
        //         }
        //         else
        //         {
        //             Ht[i][j] = 0;
        //         }
        //         /*improve : 3.時間発展exp(-iHt)の近似法*/
        //         T[i][j] = (Ht[i][j] * dt * -0.5) * I;
        //     }
        // }
        // /*対角成分.非対角から対角の順番じゃ無いと対角成分0になるので注意*/
        // for (int i = 0; i < Nums; i++)
        // {
        //     Ht[i][i] = At * H[i];
        //     T[i][i] = 1.0 - ((Ht[i][i] * dt * 0.5) * I); /* T = I - iHdt*/
        // }

        /*ついに時間発展 f1=T・f0*/
        /*improve : ここ行列ライブラリ使うのあり*/
        for (int i = 0; i < Nums; i++)
        {
            f1[i] = 0;
            for (int j = 0; j < Nums; j++)
            {
                f1[i] += T[i][j] * f0[j];
            }
        }
        for (int i = 0; i < Nums; i++)
        {
            f0[i] = f1[i];
        }

        // 時間発展出力用
        //  printf("%dth\n f0",time);
        //  for(i=0;i<Nums;i++){
        //      printf("%f + %f * I\n",creal(f1[i]),cimag(f1[i]));
        //  }
        //  printf("\n");
    }
}

/*正規化*/
void normalize(double complex psi[Nums])
{
    double abs_f1 = 0.0;
    for (int i = 0; i < Nums; i++)
    {
        abs_f1 += cabs(psi[i]) * cabs(psi[i]);
    }
    abs_f1 = sqrt(abs_f1);
    for (int i = 0; i < Nums; i++)
    {
        psi[i] = psi[i] / abs_f1;
    }
}

/*結果の出力*/
void print_state(double complex psi[Nums])
{
    double p;
    for (int i = 0; i < Nums; i++)
    {
        printf("%f + %f * I\n", creal(psi[i]), cimag(psi[i]));
    }
    for (int i = 0; i < Nums; i++)
    {
        p = cabs(psi[i]) * cabs(psi[i]);
        printf("%d : %f\n", i, p);
    }
}