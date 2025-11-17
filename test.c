#include <stdio.h>
#include "annealing.h"

void vector_plus_one(double* f){
    for(int i=0;i<Nums;i++){
        f[i] += 1.0;
    }
}

int main(){
    double f1[Nums] = {0.0};
    vector_plus_one(f1);
    for(int i=0;i<Nums;i++){
        printf("%f * I\n",f1[i]);
    }
}