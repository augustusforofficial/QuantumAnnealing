#ifndef ANNEALING_H
#define ANNEALING_H

#include <stdio.h>
#include <complex.h>
#include <math.h>

#define N 9
#define Nums 512

/*utility*/
int int_pow(int base, int exp);
int StoQ(int s);

int QtoS(int q);
int iBitNumLeft(int d,int i);
int iBitNumsRight(int d,int i);
int hamDistance(int i,int j);

/*Make Hamiltonian*/
void embed_diagonal_H(double H[Nums], double J[Nums][Nums]);

/*Time Evolution*/
void time_evolution(double complex f1[Nums],double J[Nums][Nums],int Time,double B0,double tau);

/*normalize*/
void normalize(double complex psi[Nums]);

/*output*/
void print_state(double complex psi[Nums]);

#endif // ANNEALING_H