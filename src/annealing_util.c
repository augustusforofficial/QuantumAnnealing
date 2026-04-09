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

void time_evolution_Hamiltonian(double complex f1[Nums], double H[Nums], int Time, double B0, double tau){
    double dt = tau / (double)Time;
    int time;
    double t;

    /*f0の定義*/
    double complex f0[Nums];
    for (int i = 0; i < Nums; i++)
    {
        f0[i] = (1.0 / sqrt(Nums)) + 0.0 * I;
    }

    /*時間発展関数化 (f0,f1)を入れたら、それを変更したい。*/
    for (time = 0; time < Time; time++)
    {
        t = (double)time * dt;
        /*A(t)はハミルトニアンの係数.tに単調増加*/
        double At = t / tau;
        /*B(t)は横磁場の大きさ.tに単調減少*/
        double Bt = B0 * (1 - t / tau);

        /*ついに時間発展 f1=T・f0*/
        /*improve : ここ行列ライブラリ使うのあり*/
        /*行列の代わりに変数を用意。これに都度代入を行う*/
        double complex T_ij = 0.0 + 0.0 * I;
        for (int i = 0; i < Nums; i++)
        {
            f1[i] = 0;

            /*先に対角成分だけ足しこんで、そのあとに非対角成分も足しこむ*/
            /*まずは対角成分*/
            T_ij = 1.0 - ((0.5 * H[i] * At * dt) * I);
            f1[i] += T_ij * f0[i];
            for (int bit = 0; bit < N; bit++)
            {
                /*次に非対角成分*/
                /*i = 7の時は、 1<<bit で 001,010,100 とXORして　j=110,101,011*/
                int j = i ^ (1 << bit);
                T_ij = -1 * Bt * dt * I;
                f1[i] += T_ij * f0[j];
            }
        }
        for (int i = 0; i < Nums; i++)
        {
            f0[i] = f1[i];
        }
        
        /*時間毎に正規化しないとオーバーフローして -nan になってしまった*/
        normalize(f0);

        // 時間発展出力用
        //  printf("%dth\n f0",time);
        //  for(i=0;i<Nums;i++){
        //      printf("%f + %f * I\n",creal(f1[i]),cimag(f1[i]));
        //  }
        //  printf("\n");
    }
}

/*時間発展*/
void time_evolution(double complex f1[Nums], double J[N][N], int Time, double B0, double tau)
{
    /*ハミルトニアン対角成分の定義*/
    double H[Nums] = {0.0};
    embed_diagonal_H(H, J);
    
    /*ハミルトニアンの対角項を渡して計算させる*/
    time_evolution_Hamiltonian(f1,H,Time,B0,tau);
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