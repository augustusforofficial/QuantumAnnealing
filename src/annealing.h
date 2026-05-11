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

#define N 18
#define Nums 262144

/*utility*/
int int_pow(int base, int exp);
int StoQ(int s);

int QtoS(int q);
int iBitNumLeft(int d,int i);
int iBitNumRight(int d,int i);
int hamDistance(int i,int j);

/*Make Hamiltonian*/
void embed_diagonal_H(double H[Nums], double J[N][N]);

/*Time Evolution Hamiltonian*/
void time_evolution_Hamiltonian(double complex f1[Nums],double H[Nums],int Time,double B0, double tau);

/*Time Evolution*/
void time_evolution(double complex f1[Nums],double J[N][N],int Time,double B0,double tau);

/*normalize*/
void normalize(double complex psi[Nums]);

/*output*/
void print_state(double complex psi[Nums]);

#endif // ANNEALING_H