#ifndef CUBLAS_ANNEALING_H
#define CUBLAS_ANNEALING_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <stdio.h>
#include <math.h>

#define N 16
/*21*/
#define Nums 65536
/*2097152*/

/*utility*/
int int_pow(int base, int exp);
int StoQ(int s);
int QtoS(int q);
int iBitNumLeft(int d, int i);
int iBitNumRight(int d, int i);
int hamDistance(int i, int j);

/*Make Hamiltonian*/
void embed_diagonal_H(double *H, double J[N][N]);

/*Time Evolution Hamiltonian*/
void time_evolution_Hamiltonian_cu(cuDoubleComplex *f1, double *H, int Time, double B0, double tau);

/*Time Evolution*/
void time_evolution_cu(cuDoubleComplex f1[Nums], double J[N][N], int Time, double B0, double tau);

/*normalize - CPU version for host data*/
void normalize(cuDoubleComplex psi[Nums]);

/*output*/
void print_state(cuDoubleComplex psi[Nums]);

#endif // CUBLAS_ANNEALING_H