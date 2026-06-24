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

// 座標間の距離を与える関数
double distance(struct coordinates a, struct coordinates b){
    double sub_x = a.x - b.x;
    double sub_y = a.y - b.y;
    return sub_x * sub_x + sub_y * sub_y;
};

// capacity関数.
double capacity(double d, double alpha, double beta){
    return -10 * alpha * log10(d) - beta;
};

// 実際の係数を計算する関数
double make_g(double d, double alpha, double beta, double pow_sigma){
    // 1000で除すのは、dBm => dB　の変換か？
    return pow((double) 10.0, capacity(d,alpha,beta) / 10.0 / pow_sigma / 1000);
};

// ペナルティ項を展開して与える関数
double penalty_coef(int i, int j, double RH_const, double *coef, double hyper_parameter){
    // 一次項
    if( i == j ){
        return (coef[i] * coef[i] - 2 * RH_const * coef[i]) * hyper_parameter;
    }

    // 二次項
    return 2 * coef[i] * coef[j] * hyper_parameter;
}

// ペナルティ項の定数部分
double const_penalty(double RH_const, double hyper_parameter){
    return hyper_parameter * RH_const * RH_const;
}

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

// H0 を Q[i][j] に埋め込み
void obj_function(double **Q ,double **g, int num_Users, int num_APs){
    int range_explanatory_var = num_Users * num_APs;    //スラック変数でなく、説明変数部分の範囲
    for (int i=0; i < range_explanatory_var; i++){
        for ( int j=0; j < range_explanatory_var; j++){
            /* i,jから各々のユーザー番号, AP番号を逆算*/
            int index_user_i = i / num_APs;
            int index_user_j = j / num_APs;
            int index_AP_i = i % num_APs;
            int index_AP_j = j % num_APs;
            if (i == j) //上参画ではなく対象行列になることに注意
            {
                // 一次項 は g 1個
                Q[i][j] += -1 * g[index_user_i][index_AP_i];    
            }
            else if (index_user_i != index_user_j)
            {
                // 二次項は g * g
                Q[i][j] += -1 * g[index_user_i][index_AP_i] * g[index_user_j][index_AP_j];
            }
        }
    }
};

// <= sup_AP の AP制約項. return 値は const
double sub_AP(double *Q , int num_Users, int num_APs, int sup_AP, double hyper_param){
    double term_const = 0.0;
    int range_explanatory_var = num_Users * num_APs;    //スラック変数でなく、説明変数部分の範囲
    int num_slack_AP = ((int)(log(sup_AP) / log(2)) + 1);   // AP制約に必要なスラック変数の数
    int index_start_slack = num_Users * num_APs;    // スラック変数の開始位置

    for (int index_AP=0;index_AP<num_APs;index_AP++){
        double coef_pen[range_explanatory_var + num_slack_AP];  // 係数を記録
        Initialization_array_double(coef_pen, range_explanatory_var + num_slack_AP);
    
        for(int j=0; j<num_Users; j++){
            coef_pen[index_AP + j * num_APs] = 1.0;
        }
        for(int j=0; j<num_slack_AP; j++){
            // []内は, start位置 + index_AP個目までののAPについての制約で使ったスラック変数の個数 + 今の制約項での j 個目
            coef_pen[index_start_slack + index_AP * num_slack_AP +  j] = -1 * pow(2.0, (double) j);
        }
        term_const += embed_pow_in_Qij(Q, sup_AP, coef_pen, hyper_param);
    }
    
    return term_const;
};

// L <= の User制約項
double sub_User(double *Q , int num_Users, int num_APs, int inf_User, int sup_AP, double hyper_param){
    // AP制約とUser制約両方のスラック変数の数をカウントする
    // そうしないと　User制約のスラック変数が何番目から始まるのかがわからない
    int num_slack_AP = ((int)(log(sup_AP) / log(2)) + 1);
    int num_slack_Users = ((int) (log(num_APs - inf_User) / log(2)) + 1);

    
};

// インデックス番号 [i][j]　にて　Q[i][j] の要素を return する
// sup_AP : 各APの接続上限
// inf_User : 各Userの接続下限
double make_QUBO_coef(double **Q, double **g, int num_Users, int num_APs, int sup_AP, int inf_User, double hyper_param){
    double term_const = 0.0;

    // 目的関数部分
    obj_function(Q, g, num_Users, num_APs);

    // 制約項1 : 各APの上限制約
    term_const += sub_AP(Q, num_Users, num_Users, sup_AP, hyper_param);

    // 制約項2 : 各Userの下限制約
    term_const += sub_User(Q , num_APs,num_Users, inf_User, sup_AP, hyper_param);

    return term_const;
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
            location_Users[index_Users] = gene_random_coordinate([width_x, width_y]);

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