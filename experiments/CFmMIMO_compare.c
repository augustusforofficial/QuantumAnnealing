
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <float.h>

typedef struct coordinates {
    double x;
    double y;
} Coordinates;

typedef struct result_info {
    double value;
    int mask;
} ResultInfo;

#define NUM_USERS 3
#define NUM_APS   3
#define NUM_BITS  (NUM_USERS * NUM_APS)
#define NUM_TRIALS 10000

/* 固定AP配置 */
static const Coordinates AP_POS[NUM_APS] = {
    {30.0, 30.0},
    {80.0, 50.0},
    {30.0, 80.0}
};

/* 制約条件 */
static const int L = 2;   /* 各ユーザの接続下限 */
static const int U = 2;   /* 各APの接続上限 */

/* 伝搬モデルのパラメータ */
static const double ALPHA = 1.7;
static const double BETA  = 64.0;
static const double POW_SIGMA = 2.07e-12; /* user code の値を踏襲 */

/* 数値安定化 */
static const double EPS = 1e-12;

static double rand_uniform(double min, double max)
{
    return min + (max - min) * ((double)rand() / (double)RAND_MAX);
}

static Coordinates gene_random_coordinate(double width_x, double width_y)
{
    Coordinates c;
    c.x = rand_uniform(0.0, width_x);
    c.y = rand_uniform(0.0, width_y);
    return c;
}

static double euclidean_distance(Coordinates a, Coordinates b)
{
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double d = sqrt(dx * dx + dy * dy);
    return (d < EPS) ? EPS : d;
}

static double capacity(double d, double alpha, double beta)
{
    return -10.0 * alpha * log10(d) - beta;
}

/* 元コードの形を保った係数生成関数
   ※ ここは必要なら論文の定義に合わせて差し替え可能 */
double make_g(double d, double alpha, double beta, double pow_sigma){
    // 1000で除すのは、dBm => dB　の変換か？
    return pow((double) 10.0, capacity(d,alpha,beta) / 10.0) / pow_sigma * 1e-3;
};

static void build_g(double g[NUM_USERS][NUM_APS], const Coordinates users[NUM_USERS])
{
    for (int k = 0; k < NUM_USERS; ++k) {
        for (int n = 0; n < NUM_APS; ++n) {
            double d = euclidean_distance(users[k], AP_POS[n]);
            g[k][n] = make_g(d, ALPHA, BETA, POW_SIGMA);
        }
    }
}

static bool is_feasible(int s[NUM_USERS][NUM_APS])
{
    for (int n = 0; n < NUM_APS; ++n) {
        int cnt = 0;
        for (int k = 0; k < NUM_USERS; ++k) {
            cnt += s[k][n];
        }
        if (cnt > U) return false;
    }

    for (int k = 0; k < NUM_USERS; ++k) {
        int cnt = 0;
        for (int n = 0; n < NUM_APS; ++n) {
            cnt += s[k][n];
        }
        if (cnt < L) return false;
    }

    return true;
}

/* 2次打ち切り：
   Π_k (1 + a_k) = 1 + Σ a_k + Σ_{i<j} a_i a_j + a_1 a_2 a_3
   のうち三次項を落としたもの */
static double truncated_objective(double g[NUM_USERS][NUM_APS],
                                  int s[NUM_USERS][NUM_APS])
{
    double a[NUM_USERS] = {0.0, 0.0, 0.0};

    for (int k = 0; k < NUM_USERS; ++k) {
        for (int n = 0; n < NUM_APS; ++n) {
            a[k] += g[k][n] * (double)s[k][n];
        }
    }

    double val = 1.0;
    for (int k = 0; k < NUM_USERS; ++k) {
        val += a[k];
    }

    for (int i = 0; i < NUM_USERS; ++i) {
        for (int j = i + 1; j < NUM_USERS; ++j) {
            val += a[i] * a[j];
        }
    }

    return val;
}

/* 厳密式：
   Π_k (1 + a_k) */
static double exact_objective(double g[NUM_USERS][NUM_APS],
                              int s[NUM_USERS][NUM_APS])
{
    double prod = 1.0;

    for (int k = 0; k < NUM_USERS; ++k) {
        double a_k = 0.0;
        for (int n = 0; n < NUM_APS; ++n) {
            a_k += g[k][n] * (double)s[k][n];
        }
        prod *= (1.0 + a_k);
    }

    return prod;
}

static int mask_to_assignment(int mask, int s[NUM_USERS][NUM_APS])
{
    for (int k = 0; k < NUM_USERS; ++k) {
        for (int n = 0; n < NUM_APS; ++n) {
            int bit = k * NUM_APS + n;
            s[k][n] = (mask >> bit) & 1;
        }
    }
    return 0;
}

static void print_assignment(int s[NUM_USERS][NUM_APS])
{
    for (int k = 0; k < NUM_USERS; ++k) {
        for (int n = 0; n < NUM_APS; ++n) {
            printf("%d ", s[k][n]);
        }
        printf("\n");
    }
}

static void print_users(Coordinates users[NUM_USERS])
{
    for (int k = 0; k < NUM_USERS; ++k) {
        printf("  User %d: (%.6f, %.6f)\n", k, users[k].x, users[k].y);
    }
}

static void print_g(double g[NUM_USERS][NUM_APS])
{
    for (int k = 0; k < NUM_USERS; ++k) {
        printf("  g[%d][*] = ", k);
        for (int n = 0; n < NUM_APS; ++n) {
            printf("%.6e ", g[k][n]);
        }
        printf("\n");
    }
}

static void print_solution_block(const char *title, ResultInfo *res)
{
    int s[NUM_USERS][NUM_APS];
    mask_to_assignment(res->mask, s);

    printf("%s\n", title);
    printf("  value = %.15e\n", res->value);
    printf("  mask  = %d\n", res->mask);
    printf("  assignment (users x APs):\n");
    print_assignment(s);
}

int main(void)
{
    srand((unsigned int)time(NULL));

    const double width_x = 100.0;
    const double width_y = 100.0;

    int changed_count = 0;
    FILE *fp = fopen("changed_configurations.txt", "w");
    if (!fp) {
        perror("fopen");
        return 1;
    }

    fprintf(fp, "Random-trial results where exact and truncated optima differ\n");
    fprintf(fp, "NUM_USERS=%d, NUM_APS=%d, L=%d, U=%d\n\n", NUM_USERS, NUM_APS, L, U);

    for (int trial = 0; trial < NUM_TRIALS; ++trial) {
        Coordinates users[NUM_USERS];
        double g[NUM_USERS][NUM_APS];

        for (int k = 0; k < NUM_USERS; ++k) {
            users[k] = gene_random_coordinate(width_x, width_y);
        }

        build_g(g, users);

        ResultInfo best_exact = { .value = -DBL_MAX, .mask = -1 };
        ResultInfo best_trunc = { .value = -DBL_MAX, .mask = -1 };

        int s[NUM_USERS][NUM_APS];

        for (int mask = 0; mask < (1 << NUM_BITS); ++mask) {
            mask_to_assignment(mask, s);
            if (!is_feasible(s)) continue;

            double v_exact = exact_objective(g, s);
            double v_trunc = truncated_objective(g, s);

            if ((v_exact > best_exact.value + EPS) ||
                (fabs(v_exact - best_exact.value) <= EPS && mask < best_exact.mask)) {
                best_exact.value = v_exact;
                best_exact.mask = mask;
            }

            if ((v_trunc > best_trunc.value + EPS) ||
                (fabs(v_trunc - best_trunc.value) <= EPS && mask < best_trunc.mask)) {
                best_trunc.value = v_trunc;
                best_trunc.mask = mask;
            }
        }

        int exact_s[NUM_USERS][NUM_APS];
        int trunc_s[NUM_USERS][NUM_APS];
        mask_to_assignment(best_exact.mask, exact_s);
        mask_to_assignment(best_trunc.mask, trunc_s);

        int different = (best_exact.mask != best_trunc.mask);

        printf("============================================================\n");
        printf("Trial %d\n", trial + 1);
        print_users(users);
        print_g(g);

        print_solution_block("Exact optimum", &best_exact);
        print_solution_block("Truncated optimum", &best_trunc);

        if (different) {
            changed_count++;
            printf(">>> DIFFERENT OPTIMUM FOUND <<<\n");

            fprintf(fp, "Trial %d\n", trial + 1);
            fprintf(fp, "Users:\n");
            for (int k = 0; k < NUM_USERS; ++k) {
                fprintf(fp, "  User %d: %.6f %.6f\n", k, users[k].x, users[k].y);
            }
            fprintf(fp, "g matrix:\n");
            for (int k = 0; k < NUM_USERS; ++k) {
                fprintf(fp, "  ");
                for (int n = 0; n < NUM_APS; ++n) {
                    fprintf(fp, "%.6e ", g[k][n]);
                }
                fprintf(fp, "\n");
            }

            fprintf(fp, "Exact optimum:\n");
            fprintf(fp, "  value = %.15e\n", best_exact.value);
            fprintf(fp, "  mask  = %d\n", best_exact.mask);
            for (int k = 0; k < NUM_USERS; ++k) {
                fprintf(fp, "  ");
                for (int n = 0; n < NUM_APS; ++n) {
                    fprintf(fp, "%d ", exact_s[k][n]);
                }
                fprintf(fp, "\n");
            }

            fprintf(fp, "Truncated optimum:\n");
            fprintf(fp, "  value = %.15e\n", best_trunc.value);
            fprintf(fp, "  mask  = %d\n", best_trunc.mask);
            for (int k = 0; k < NUM_USERS; ++k) {
                fprintf(fp, "  ");
                for (int n = 0; n < NUM_APS; ++n) {
                    fprintf(fp, "%d ", trunc_s[k][n]);
                }
                fprintf(fp, "\n");
            }
            fprintf(fp, "\n");
        }
    }

    printf("============================================================\n");
    printf("Total changed trials = %d / %d\n", changed_count, NUM_TRIALS);
    printf("Detailed changed cases were written to changed_configurations.txt\n");

    fprintf(fp, "Total changed trials = %d / %d\n", changed_count, NUM_TRIALS);
    fclose(fp);

    return 0;
}
