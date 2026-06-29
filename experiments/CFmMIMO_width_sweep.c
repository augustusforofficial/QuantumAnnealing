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

#define DELTA_WIDTH 5.0
#define MAX_WIDTH   1000.0
#define NUM_WIDTHS  ((int)(MAX_WIDTH / DELTA_WIDTH))

static const Coordinates AP_POS[NUM_APS] = {
    {30.0, 30.0},
    {80.0, 50.0},
    {30.0, 80.0}
};

static const int L = 2;
static const int U = 2;

static const double ALPHA = 1.7;
static const double BETA  = 64.0;
static const double POW_SIGMA = 2.07e-12;

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

static double make_g(double d, double alpha, double beta, double pow_sigma)
{
    return pow(10.0, capacity(d, alpha, beta) / 10.0) * 1e-3 / pow_sigma;
}

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
        for (int k = 0; k < NUM_USERS; ++k) cnt += s[k][n];
        if (cnt > U) return false;
    }

    for (int k = 0; k < NUM_USERS; ++k) {
        int cnt = 0;
        for (int n = 0; n < NUM_APS; ++n) cnt += s[k][n];
        if (cnt < L) return false;
    }

    return true;
}

static double truncated_objective(double g[NUM_USERS][NUM_APS], int s[NUM_USERS][NUM_APS])
{
    double a[NUM_USERS] = {0.0};

    for (int k = 0; k < NUM_USERS; ++k) {
        for (int n = 0; n < NUM_APS; ++n) {
            a[k] += g[k][n] * (double)s[k][n];
        }
    }

    double val = 1.0;
    for (int k = 0; k < NUM_USERS; ++k) val += a[k];

    for (int i = 0; i < NUM_USERS; ++i) {
        for (int j = i + 1; j < NUM_USERS; ++j) {
            val += a[i] * a[j];
        }
    }

    return val;
}

static double exact_objective(double g[NUM_USERS][NUM_APS], int s[NUM_USERS][NUM_APS])
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

static void mask_to_assignment(int mask, int s[NUM_USERS][NUM_APS])
{
    for (int k = 0; k < NUM_USERS; ++k) {
        for (int n = 0; n < NUM_APS; ++n) {
            int bit = k * NUM_APS + n;
            s[k][n] = (mask >> bit) & 1;
        }
    }
}

int main(void)
{
    srand((unsigned int)time(NULL));

    FILE *summary_fp = fopen("width_summary.csv", "w");
    if (!summary_fp) {
        perror("width_summary.csv");
        return 1;
    }

    fprintf(summary_fp, "width,changed_count,total_trials,changed_rate\n");

    for (int wi = 0; wi < NUM_WIDTHS; ++wi) {
        const double width = DELTA_WIDTH * (wi + 1);
        const double width_x = width;
        const double width_y = width;

        int changed_count = 0;

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

            if (best_exact.mask != best_trunc.mask) {
                changed_count++;
            }
        }

        double changed_rate = (double)changed_count / (double)NUM_TRIALS;

        printf("WIDTH = %.6f\n", width);
        printf("  changed_count = %d / %d\n", changed_count, NUM_TRIALS);
        printf("  changed_rate   = %.6f\n\n", changed_rate);

        fprintf(summary_fp, "%.6f,%d,%d,%.10f\n",
                width, changed_count, NUM_TRIALS, changed_rate);
    }

    fclose(summary_fp);
    printf("Summary written to width_summary.csv\n");
    return 0;
}