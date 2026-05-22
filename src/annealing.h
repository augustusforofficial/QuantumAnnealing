#ifndef ANNEALING_H
#define ANNEALING_H

#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* 3_prisoners_dilemma : N=21, Nums = 2097152*/
/* CFmMIMO : N=18, Nums= 262144*/

#define N 21
#define Nums 2097152

/*utility*/
int int_pow(int base, int exp);
int StoQ(int s);

int QtoS(int q);
int iBitNumLeft(int d,int i);
int iBitNumRight(int d,int i);
int hamDistance(int i,int j);

/* transform QUBO matrix to Ising matrix (uptriangle) */
void transform_QUBO_to_Ising(double Q[N][N], double J[N][N]);

/*Make Hamiltonian*/
void embed_diagonal_H(double H[Nums], double J[N][N]);

/*Time Evolution Hamiltonian*/
void time_evolution_Hamiltonian(double complex f1[Nums],double H[Nums],int Time,double B0, double tau);

/*second-order approximation of Hamiltonian time evolution*/
void time_evolution_Hamiltonian_Iidaka(double complex f1[Nums],double H[Nums],int Time,double B0, double tau);

/*Time Evolution*/
void time_evolution(double complex f1[Nums],double J[N][N],int Time,double B0,double tau);

/*normalize*/
void normalize(double complex psi[Nums]);

/* (x0+x1+...+xn - a)^2 を展開する*/
double embed_pow_in_Qij(double Q[N][N], int a, double coef[N], double hyper_parameter);

/* N * N　行列の num_row * num_column を表示 */
void print_matrix(double A[N][N], double num_row, double num_column);

void Initialization_array_double(double *A, int length);

void Initialization_array_int(int *A, int length, int a);

/* 定数項も追加するように*/
/* note : H[i] = -1 * f(i) であることに注意*/
void Add_Energy_QUBO_to_Hamiltonian(double H[Nums], double Q[N][N], double term_const);

void Show_Hamiltonian_max_min(double H[Nums]);

void Show_matrix_NN(double A[N][N], double row, double column);

void Show_vector_N(double A[N]);

void Show_vector_Nums(double A[Nums]);

void make_prob_vec(double complex f1[Nums], double prob[Nums]);

void Show_top_X(double prob[Nums], int X);

#endif // ANNEALING_H