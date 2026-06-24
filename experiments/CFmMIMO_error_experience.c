#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <complex.h>
#include "../src/annealing.h"

// 位置座標
typedef struct coordinates {
    double x;
    double y;
} Coordinates;

// 最適解のデータ
typedef struct optimal_solution{
    double opt_val;
    double opt_idx;
} Opt_sol;

double distance(struct coordinates a, struct coordinates b){
    double sub_x = a.x - b.x;
    double sub_y = a.y - b.y;
    return sub_x * sub_x + sub_y * sub_y;
};

// capacity関数.
double capacity(double d, double alpha, double beta){
    return -10 * alpha * log10(d) - beta;
};

// 実際の係数
double make_g(double d, double alpha, double beta, double pow_sigma){
    // 1000で除すのは、dBm => dB　の変換か？
    return pow((double) 10.0, capacity(d,alpha,beta) / 10.0 / pow_sigma / 1000);
};

// (0,0) ~ (max_x, max_y) のランダム Coordinates 型を返却
Coordinates gene_random_coordinate(double maxs[2]){
    Coordinates c;

    for(int i=0; i<2; i++) {    // x,y座標両方において
        srand((unsigned int)time(NULL));    // 時刻をシード値として利用
        double rand_double = (double) rand() / (double) maxs[i];
        if(i==0){
            c.x = rand_double;
        }
        else if(i==1){
            c.y = rand_double;
        }
    }

    return c;
};

// H0 の部分.
double obj_function(int i, int j ,double **g, int num_Users, int num_APs){
    int range_explanatory_var = num_Users * num_APs;    //スラック変数でなく、説明変数部分の範囲
    if ((i < range_explanatory_var) && (j < range_explanatory_var)){
         /* i,jから各々のユーザー番号, AP番号を逆算*/
            int index_user_i = i / num_APs;
            int index_user_j = j / num_APs;
            int index_AP_i = i % num_APs;
            int index_AP_j = j % num_APs;
            if (i == j) //上参画ではなく対象行列になることに注意
            {
                // 一次項 は g 1個
                return -1 * g[index_user_i][index_AP_i];    
            }
            else if (index_user_i != index_user_j)
            {
                // 二次項は g * g
                return -1 * g[index_user_i][index_AP_i] * g[index_user_j][index_AP_j];
            }else {
                return 0;   // 同一ユーザの変数の積の係数は 0
            }
    } 

    return 0;   // i,jのどちらかがスラック変数の場合も 0
};

// <= sup_AP の AP制約項
double sub_AP(int i, int j, int num_Users, int num_APs, int sup_AP){
    int range_explanatory_var = num_Users * num_APs;    //スラック変数でなく、説明変数部分の範囲
    int num_slack_AP = ((int)(log(sup_AP) / log(2)) + 1);   // AP制約に必要なスラック変数の数

    // スラック変数に関係なかったら return 0
    if ( ((i < range_explanatory_var) || (range_explanatory_var + num_APs <= i))
        &&((j < range_explanatory_var) || (range_explanatory_var + num_APs <= j))){
            return 0;
        }
    
    // i,jのどちらかがスラック変数に関係があれば
    
};

// L <= の User制約項
double sub_User(int i, int j, int num_Users, int num_APs, int inf_User, int sup_AP){
    // AP制約とUser制約両方のスラック変数の数をカウントする
    // そうしないと　User制約のスラック変数が何番目から始まるのかがわからない
    int num_slack_AP = ((int)(log(sup_AP) / log(2)) + 1);
    int num_slack_Users = ((int) (log(num_APs - inf_User) / log(2)) + 1);
};

// インデックス番号 [i][j]　にて　Q[i][j] の要素を return する
// sup_AP : 各APの接続上限
// inf_User : 各Userの接続下限
double make_QUBO_coef(int i, int j, double **g, int num_Users, int num_APs, int sup_AP, int inf_User){
    // 目的関数部分
    double H0 = obj_function(i, j, g, num_Users, num_APs);

    // 制約項1 : 各APの上限制約
    double H1 = sub_AP(i, j, num_Users, num_Users, sup_AP);

    // 制約項2 : 各Userの下限制約
    double H2 = sub_User(i, j, num_APs,num_Users, inf_User, sup_AP);

    // すべてを足し合わせて返却する
    return H0 + H1 + H2;
};

int main(){
    // AP数,User数の指定
    const int num_APs = 3;
    const int num_Users = 3;
    // AP座標の指定
    Coordinates location_APs[] = {{30, 30}, {80, 50}, {30, 80}};
    Coordinates location_Users[num_Users];
    // User配置のランダム生成数
    const int num_random_gene = 10;
    // 配置座標の広さ
    const double width_x = 100.0;
    const double width_y = 100.0;
    // 係数行列
    double g[num_Users][num_APs];
    // capacityの alpha, beta および ノイズ pow_sigma の定義
    const double alpha = 1.7;
    const double beta = 80.0;
    const double pow_sigma = 2.07e-12;

    // bit数の定義
    const int num_bits = 21;

    // cost関数のための QUBO係数行列 Q[i][j] , 目的関数の最適解および最適値データの opt の宣言
    double Q[num_bits][num_bits];
    Opt_sol opt;

    // 以下の手順を for num_random_gene
    for(int index_random_gene=0; index_random_gene<num_random_gene; index_random_gene++){
        
        for(int index_Users=0; index_Users < num_Users; index_Users++){
            // 乱数によるUser生成
            location_Users[index_Users] = gene_random_coordinate([width_x,width_y]);

            // AP,User座標から係数行列 g[num_Users][num_APs]を作成
            for(int index_APs = 0; index_APs < num_APs; index_APs++){
                double dis = distance(location_Users[index_Users], location_APs[index_APs]);
                g[index_Users][index_APs] = make_g(dis, alpha, beta, pow_sigma);
            }
        }
        
        // g[][]から目的関数 f(s) を作成
        

        // 各sについて全探索し、厳密最適解と2次打ち切り最適解を探索
        // もし最適解が違ったら,問題設定出力 + diff_counter++;
    }
        
};