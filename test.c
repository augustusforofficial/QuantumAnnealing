#include <stdio.h>

void print_binary(unsigned int x){
    if(x == 0){
        printf("0");
        return;
    }

    int started = 0;

    for(int i = 31; i >= 0; i--){
        if((x >> i) & 1){
            started = 1;
        }

        if(started){
            printf("%d", (x >> i) & 1);
        }
    }
}

int main(){
    unsigned int n;

    printf("10進数を入力してください: ");
    scanf("%u", &n);

    printf("2進数: ");
    print_binary(n);
    printf("\n");

    return 0;
}