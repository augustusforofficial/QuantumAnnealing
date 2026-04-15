#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define N 21
#define Nums 2097152

/* =========================
   Utility Functions
   ========================= */
int iBitNumRight(int d, int i) { return (d >> i) & 1; }

void normalize(double complex psi[Nums])
{
    double norm2 = 0.0;
    #pragma omp parallel for reduction(+ : norm2)
    for (int i = 0; i < Nums; i++)
    {
        double re = creal(psi[i]);
        double im = cimag(psi[i]);
        norm2 += re * re + im * im;
    }

    double inv_norm = 1.0 / sqrt(norm2);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < Nums; i++)
    {
        psi[i] *= inv_norm;
    }
}

/* =========================
   Time Evolution
   ========================= */
void time_evolution_Hamiltonian(double complex f1[Nums], double H[Nums], int Time, double B0, double tau)
{
    const double dt = tau / (double)Time;
    double complex *cur = malloc(sizeof(double complex) * Nums);
    double complex *next = malloc(sizeof(double complex) * Nums);

    if (!cur || !next)
    {
        printf("memory allocation failed\n");
        free(cur);
        free(next);
        return;
    }

    const double init_amp = 1.0 / sqrt((double)Nums);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < Nums; i++)
    {
        cur[i] = init_amp + 0.0 * I;
    }

    for (int time = 0; time < Time; time++)
    {
        const double t = (double)time * dt;
        const double At = t / tau;
        const double Bt = B0 * (1.0 - t / tau);
        const double complex offdiag = -Bt * dt * I;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nums; i++)
        {
            double complex val = (1.0 - 0.5 * H[i] * At * dt * I) * cur[i];
            for (int bit = 0; bit < N; bit++)
            {
                int j = i ^ (1 << bit);
                val += offdiag * cur[j];
            }
            next[i] = val;
        }

        if (time % 100 == 0) normalize(next);

        double complex *tmp = cur;
        cur = next;
        next = tmp;

        if (time % 1000 == 0) printf("%dth\n", time);
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < Nums; i++) f1[i] = cur[i];

    free(cur);
    free(next);
}

/* =========================
   3 Prisoners' Dilemma
   ========================= */
static inline int payoff_of_3_prisoners_dilemma(int p, int i, int j, int k)
{
    if (i && j && k) return -5;
    if (!i && !j && !k) return -3;

    int strategy[3] = {i, j, k};
    if (strategy[p] == 1) return -1;
    return -10;
}

/* =========================
   Main (all-in-one)
   ========================= */
int main(void)
{
    int i, j, k;
    const int num_player = 3;
    const int num_stg[3] = {2, 2, 2};

    int payoff_sum[2][2][2];
    memset(payoff_sum, 0, sizeof(payoff_sum));

    for (int p = 0; p < num_player; p++)
        for (i = 0; i < 2; i++)
            for (j = 0; j < 2; j++)
                for (k = 0; k < 2; k++)
                    payoff_sum[i][j][k] += payoff_of_3_prisoners_dilemma(p, i, j, k);

    double H[Nums] = {0.0};
    double complex f1[Nums] = {0.0 + 0.0 * I};

    double B0 = 1.0;
    int Time = 1000000;
    double tau = 1.0;

    int alpha = -5, beta = -5, gamma = -5;
    const double hypers[3] = {3.0, 3.0, 3.0};
    const int num_pen = 6;
    const int num_slack = 3;
    const int start_slack = 3;

    #pragma omp parallel for schedule(static)
    for (i = 0; i < Nums; i++)
    {
        int Pen[6] = {0};
        int x = (i >> (N - 1)) & 1;
        int y = (i >> (N - 2)) & 1;
        int z = (i >> (N - 3)) & 1;

        H[i] = -payoff_sum[x][y][z];

        Pen[0] = payoff_of_3_prisoners_dilemma(0, 0, y, z);
        Pen[1] = payoff_of_3_prisoners_dilemma(0, 1, y, z);
        Pen[2] = payoff_of_3_prisoners_dilemma(1, x, 0, z);
        Pen[3] = payoff_of_3_prisoners_dilemma(1, x, 1, z);
        Pen[4] = payoff_of_3_prisoners_dilemma(2, x, y, 0);
        Pen[5] = payoff_of_3_prisoners_dilemma(2, x, y, 1);

        for (j = 0; j < num_pen; j++)
        {
            int base = N - 1 - j * num_slack - start_slack;
            int s1 = (i >> base) & 1;
            int s2 = (i >> (base - 1)) & 1;
            int s3 = (i >> (base - 2)) & 1;
            Pen[j] += s1 + 2 * s2 + 4 * s3;
        }

        Pen[0] -= alpha; Pen[1] -= alpha;
        Pen[2] -= beta;  Pen[3] -= beta;
        Pen[4] -= gamma; Pen[5] -= gamma;

        for (j = 0; j < num_pen; j++)
        {
            Pen[j] *= Pen[j];
            H[i] += hypers[j / 2] * Pen[j];
        }

        H[i] += alpha + beta + gamma;
    }

    time_evolution_Hamiltonian(f1, H, Time, B0, tau);

    printf("finished\n");
    FILE *fp = fopen("./results/result.bin", "wb");
    fwrite(f1, sizeof(double complex), Nums, fp);
    fclose(fp);

    return 0;
}
