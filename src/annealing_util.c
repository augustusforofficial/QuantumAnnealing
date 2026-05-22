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

/* QUBO形式 Q[i][j](対象行列) => Ising形式 J[i][j](上三角行列) への変換*/
void transform_QUBO_to_Ising(double Q[N][N], double J[N][N])
{

#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (i < j)
            {
                J[i][j] = Q[i][j] / 2.0;
            }
            else if (i == j)
            {
                double qi_sum = 0.0;
                for (int k = 0; k < N; k++)
                {
                    qi_sum += Q[i][k];
                }
                J[i][j] = qi_sum / 2.0;
            }
            else
            {
                J[i][j] = 0.0;
            }
        }
    }
}

/* !! : J[i][j] は上三角の形式で入力すること　*/
/* !! : J[i][j] は "最小化目的関数"となるように設定. これにより、H = -ΣJ[i][j] まで埋め込めるので、マイナスを気にしなくてOK */
void embed_diagonal_H(double H[Nums], double J[N][N])
{
#pragma omp parallel for
    for (int candidate_num = 0; candidate_num < Nums; candidate_num++)
    {
        for (int j = 0; j < N; j++)
        {
            /* 対角項 */
            H[candidate_num] += (1 - 2 * iBitNumLeft(candidate_num, j)) * J[j][j];

            /* 相互作用項（上三角のみ）*/
            for (int k = j + 1; k < N; k++)
            {
                H[candidate_num] += (1 - 2 * iBitNumLeft(candidate_num, j)) * (1 - 2 * iBitNumLeft(candidate_num, k)) * J[j][k];
            }
        }
    }
}

void time_evolution_Hamiltonian(double complex *f1, double *H, int Time, double B0, double tau)
{
    double dt = tau / (double)Time;
    int time;
    double t;

    /*f0の定義*/
    double complex *f0 = malloc(sizeof(double complex) * Nums);
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
        for (int i = 0; i < Nums; i++)
        {
            double complex T_ij = 0.0 + 0.0 * I;
            f1[i] = 0.0 + 0.0 * I;

            /*先に対角成分だけ足しこんで、そのあとに非対角成分も足しこむ*/
            /*まずは対角成分*/
            T_ij = 1.0 - (0.5 * H[i] * At * dt * I);
            f1[i] += T_ij * f0[i];
            for (int bit = 0; bit < N; bit++)
            {
                /*次に非対角成分*/
                /*i = 7の時は、 1<<bit で 001,010,100 とXORして　j=110,101,011*/
                int j = i ^ (1 << bit);
                T_ij = 0.5 * Bt * dt * I;
                f1[i] += T_ij * f0[j];
            }
        }

        /* 近似によりノルムが保存されないため毎ステップで正規化*/
        normalize(f1);

#pragma omp parallel for
        for (int i = 0; i < Nums; i++)
        {
            f0[i] = f1[i];
        }

        if (time % 10000 == 0)
        {
            printf("#%d th \n", time);
        }
    }

    free(f0);
}

void time_evolution_Hamiltonian_Iidaka(double complex *f2, double *H, int Time, double B0, double tau)
{
    double dt = tau / (double)Time;
    int time;
    double t;

    /*f0,f1の定義*/
    double complex *f0 = malloc(sizeof(double complex) * Nums);
    double complex *f1 = malloc(sizeof(double complex) * Nums);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < Nums; i++)
    {
        f0[i] = (1.0 / sqrt(Nums)) + 0.0 * I;
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < Nums; i++)
    {
        f1[i] = 0.0 + 0.0 * I;
    }

    /*時間発展関数化 (f0,f1)を入れたら、それを変更したい。*/
    for (time = 0; time < Time; time++)
    {
        t = (double)time * dt;
        /*A(t)はハミルトニアンの係数.tに単調増加*/
        double At = t / tau;
        /*B(t)は横磁場の大きさ.tに単調減少*/
        double Bt = B0 * (1 - t / tau);

        if (time == 0)
        {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < Nums; i++)
            {
                double complex T_ij = 0.0 + 0.0 * I;
                f1[i] = 0.0 + 0.0 * I;

                /*先に対角成分だけ足しこんで、そのあとに非対角成分も足しこむ*/
                /*まずは対角成分*/
                T_ij = 1.0 - (0.5 * H[i] * At * dt * I);
                f1[i] += T_ij * f0[i];
                for (int bit = 0; bit < N; bit++)
                {
                    /*次に非対角成分*/
                    /*i = 7の時は、 1<<bit で 001,010,100 とXORして　j=110,101,011*/
                    int j = i ^ (1 << bit);
                    T_ij = 0.5 * Bt * dt * I;
                    f1[i] += T_ij * f0[j];
                }
            }

            normalize(f1);
        }
        else
        {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < Nums; i++)
            {
                double complex T_ij = 0.0 + 0.0 * I;
                f2[i] = f0[i]; /* f2 += f0 を実質行う */

                /*まずは対角成分*/
                T_ij = -1 * At * H[i] * dt * I;
                f2[i] += T_ij * f1[i];
                for (int bit = 0; bit < N; bit++)
                {
                    /*次に非対角成分*/
                    /*i = 7の時は、 1<<bit で 001,010,100 とXORして　j=110,101,011*/
                    int j = i ^ (1 << bit);
                    T_ij = Bt * dt * I;
                    f2[i] += T_ij * f1[j];
                }
            }
            if(time % 10 == 0){
                normalize(f1);
                normalize(f2);
            }
            
#pragma omp parallel for
            for (int i = 0; i < Nums; i++)
            {
                f0[i] = f1[i];
            }
#pragma omp parallel for
            for (int i = 0; i < Nums; i++)
            {
                f1[i] = f2[i];
            }

            if (time % 10000 == 0)
            {
                printf("#%d th \n", time);
            }
        }
    }
    free(f0);
    free(f1);
}

/*時間発展*/
void time_evolution(double complex f1[Nums], double J[N][N], int Time, double B0, double tau)
{
    /*ハミルトニアン対角成分の定義*/
    double H[Nums] = {0.0};
    embed_diagonal_H(H, J);

    /*ハミルトニアンの対角項を渡して計算させる*/
    time_evolution_Hamiltonian(f1, H, Time, B0, tau);
}

/*正規化*/
void normalize(double complex psi[Nums])
{
    double abs_f1 = 0.0;

#pragma omp parallel for reduction(+ : abs_f1)
    for (int i = 0; i < Nums; i++)
    {
        double re = creal(psi[i]);
        double im = cimag(psi[i]);
        abs_f1 += re * re + im * im;
    }
    double inv_norm = 1.0 / sqrt(abs_f1);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < Nums; i++)
    {
        psi[i] *= inv_norm;
    }
}

double embed_pow_in_Qij(double Q[N][N], int a, double coef[N], double hyper_parameter)
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

    return (double)hyper_parameter * a * a;
}

void print_matrix(double A[N][N], double num_row, double num_column)
{
    printf("matrix\n");
    for (int i = 0; i < num_row; i++)
    {
        for (int j = 0; j < num_column; j++)
        {
            printf("%.4f, ", A[i][j]);
        }
        printf("\n");
    }
}

void Initialization_array_double(double *A, int length)
{

#pragma omp schedule for
    for (int i = 0; i < length; i++)
    {
        A[i] = 0.0;
    }
}

void Initialization_array_int(int *A, int length, int a)
{

#pragma omp parallel for
    for (int i = 0; i < length; i++)
    {
        A[i] = a;
    }
}

/* 定数項も追加するように*/
/* note : H[i] は min 探索形式でOK*/
void Add_Energy_QUBO_to_Hamiltonian(double H[Nums], double Q[N][N], double term_const)
{
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

void Show_Hamiltonian_max_min(double H[Nums])
{
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

void Show_matrix_NN(double A[N][N], double row, double column)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            printf("%.5f, ", A[i][j]);
        }
        printf("\n");
    }
}

void Show_vector_N(double A[N])
{
    for (int i = 0; i < N; i++)
    {
        printf("%f, ", A[i]);
    }
}

void Show_vector_Nums(double A[Nums])
{
    for (int i = 0; i < Nums; i++)
    {
        printf("%d : %f\n", i, A[i]);
    }
}

void make_prob_vec(double complex f1[Nums], double prob[Nums])
{
    for (int i = 0; i < Nums; i++)
    {
        double re = creal(f1[i]);
        double im = cimag(f1[i]);
        prob[i] = re * re + im * im;
    }
}

void Show_top_X(double prob[Nums], int X)
{
    int topX_index[X]; // [1位, 2, 3,..., X位]
    Initialization_array_int(topX_index, X, -1);

    for (int i = 0; i < Nums; i++)
    {
        int k = 0; // 何個ずらすのかの変数
        for(int j = 0; j < X; j++){
            // 暫定top10のものと比較して
            if(topX_index[j] == -1){
                k++;
            }else{
                if(prob[i] > prob[topX_index[X - k - 1]]){
                    k++;
                }else{
                    break;
                }
            }
        }
        /* 挿入することが確定したら */
        if (k > 0)
        {
            /* 右からk-1個は左のものを代入する*/
            /* [... ,k, a, b, c] => [... , k, k, a, b]*/
            for (int j = 0; j < k - 1; j++)
            {
                topX_index[X - j - 1] = topX_index[X - (j + 1) - 1];
            }
            topX_index[X - k] = i;
        }
    }

    printf("x th : (index, val)\n");
    for (int i = 0; i < X; i++)
    {
        printf("%d th : (%d, %.17g)\n", i + 1, topX_index[i], prob[topX_index[i]]);
    }
}