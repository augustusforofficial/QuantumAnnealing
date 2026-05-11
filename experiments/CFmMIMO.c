#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include "annealing.h"

#define NUM_AP 3
#define NUM_USER 3

/* N=18, Nums = 262144*/

/* 意味 : (Σx_i - a)^2 の展開した時の係数を J[i][j] に足し合わせる*/
/* J[N][N] : 埋め込む先
    a : 式内のa
    index[N] : Σx に入る場合1,入らない場合0の配列*/
void embed_pow_in_Jij (double J[N][N], int a, double coef[N]){
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            if(i==j){
                J[i][j] += coef[i] * coef[i] - 2 * a * coef[i];
            }else{
                J[i][j] += coef[i] * coef[j];
            }
        }
    }
}

int main(){
    const double sigma = 1; /*ノイズ*/
    double g[NUM_USER][NUM_AP];    /* improve : 距離の逆数で定義*/
    const int U = 2; /*各 AP での接続上限数*/
    const int L = 2; /*各 user の接続下限数*/

    // 各APの座標定義
    double coordinate_AP[NUM_AP][2] = {{30,30}, {30,80}, {80,50}};

    // 各userの座標定義
    double coordinate_user[NUM_USER][2] = {{20,50}, {60,70}, {90,30}} ;

    // 各AP,user 間の距離の逆数を g へ代入
    double distance;
    double sub_x;
    double sub_y;
    for(int index_user=0;index_user<NUM_USER;index_user++){  // k : ユーザー番号
        for(int index_ap=0;index_ap<NUM_AP;index_ap++){  // l : AP番号
            /* 距離を計算する */
            sub_x = coordinate_user[index_user][0] - coordinate_AP[index_ap][0];
            sub_y = coordinate_user[index_user][1] - coordinate_AP[index_ap][1];
            distance = sub_x * sub_x + sub_y * sub_y;
            distance = sqrt(distance);
            if(distance > 1e-12){
                g[index_user][index_ap] = 1.0 / distance;
            }
            else{
                g[index_user][index_ap] = 0.0;
            }
        }
    }
    /* 2026/05/11 : distanceまでは完了 */

    // J[i][j] を g[k][l] から作成する

    /* qubit は以下のように配列するものとする */
    /* s[0][0] s[0][1] s[0][2] s[1][0] ... s[2][2] y[0][0] y[0][1] ... y[2][1] z[0] z[1] z[2] の計 18 qubit*/
    /* bit_index : 0~8　は s[k][l] , 9~14 は y[i][j], 15~17 は z[i]*/
    double J[N][N] = {0.0};
    
        /*1. 目的関数部分. J[i][i] += g / sigma^2, J[i][j] = g*g / sigma^2 */
        #pragma omp parallel for collapse(2)
        for(int i=0; i<NUM_USER*NUM_AP; i++){   /* 0<=i<= 8 の形. jも同様*/
            for(int j=0; j<NUM_USER*NUM_AP; j++){
                /* i,jから各々のユーザー番号, AP番号を逆算*/
                int index_user_i = i / NUM_AP;
                int index_user_j = j / NUM_AP;
                int index_AP_i = i % NUM_AP;
                int index_AP_j = j % NUM_AP; 
                if(i==j){
                    J[i][j] += g[index_user_i][index_AP_i] / (sigma * sigma);
                }else if(index_user_i != index_user_j){
                    J[i][j] += g[index_user_i][index_AP_i] * g[index_user_j][index_AP_j] / (sigma*sigma*sigma*sigma);
                }
            }
        }

        /*2. <=U の各AP上限制約.*/
        int num_bit_slack_of_y = ((int)(log(U) / log(2)) + 1);   /*各行でのスラック変数数. log2(U)*/
        printf("num_bit_slack_of_y : %d\n",num_bit_slack_of_y);
        for(int index_user=0; index_user<NUM_USER; index_user++){

            /* 関係するqubitは、
             s[index_user][] : 0~2, 3~5, 6~8
             y[index_user][] : 9~10, 11~12, 13~14 */
            int index_start_slack = NUM_USER * NUM_AP;
            double pickup[N] = {0.0};    // 制約式の ()^2 内での係数配列.もし()内に出なければ 0.0 をとる
            /* s[][] 部分のピックアップ*/
            for(int i=0; i<NUM_AP; i++){
                pickup[index_user * NUM_AP + i] = 1.0;
            }
            /* y[][] 部分のピックアップ. y[index_user][i] = 2^i の係数*/
            for(int i=0; i<num_bit_slack_of_y; i++){
                pickup[index_start_slack + num_bit_slack_of_y * index_user + i] = pow(2.0, (double) i);
            }

            /*テスト出力用*/
            // printf("index_user = %d : ",index_user);
            // for(int i=0;i<N;i++){
            //     printf("%f, ",pickup[i]);
            // }
            // printf("\n");

            embed_pow_in_Jij(J, U, pickup);
        }

        /*3. >=L の各user下限制約.*/
        int num_bit_slack_of_z = ((int)(log(NUM_AP-L)/log(2)) + 1);
        printf("num_bit_slack_of_z : %d\n",num_bit_slack_of_z);
        for(int index_AP=0; index_AP<NUM_AP; index_AP++){
            /*今回の slack は、前回の終わりから NUM_USER * num_bit_slack_of_y */
            int index_start_slack = NUM_USER * NUM_AP + NUM_USER * num_bit_slack_of_y;
            double pickup[N] = {0.0};
            for(int j=0; j<NUM_USER; j++){
                pickup[index_AP + j * NUM_AP] = 1.0;
            }
            for(int j=0; j<num_bit_slack_of_z; j++){
                pickup[index_start_slack + index_AP * num_bit_slack_of_z + j] = -1 * pow(2.0, (double) j);
            }
            embed_pow_in_Jij(J, L, pickup);
        }

        /* J[i][j] のテスト*/
        printf("J[%d][%d]\n",N,N);
        for(int i=0; i<N; i++){
            for(int j=0; j<N; j++){
                printf("%1.4f, ",J[i][j]);
            }
            printf("\n");
        }

    // J[i][j] から時間発展させる
}