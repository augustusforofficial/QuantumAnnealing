0.108280 + -0.263405 * I
0.207783 + -0.382537 * I
-0.368317 + -0.051479 * I
0.225344 + 0.200742 * I
0.225344 + 0.200742 * I
-0.368317 + -0.051479 * I
0.207783 + -0.382537 * I
0.108280 + -0.263405 * I
0 : 0.081107
1 : 0.189508
2 : 0.138307
3 : 0.091078
4 : 0.091078
5 : 0.138307
6 : 0.189508
7 : 0.081107

# gitの使い方メモ
## ローカル
```
# リポジトリ作成
git init

# ステージング（ローカルに一時的保存）
git add [ファイル,フォルダ]
#現在ディレクトリ全部を指定
git add .

# コミット (ローカルリポジトリ保存)
git commit -m "message"
```

## リモート
```
#GitHubリポジトリ登録
git remote add origin [URL]

# pull
git pull origin main

# 自分の作業保存 ~ アップロード
git add .   #ステージ保存
git commit -m "message" #コミット
git push origin main    #プッシュ（アップロード
```

## グラフ出力
フィルタリング機能 (2 つのオプション):

--threshold 0.01 : 確率が 0.01 以上の状態のみ表示
--top 20 : 確率が高い順に上位 20 个の状態を表示
ソート表示: フィルタリング後の状態を確率でソートして、見やすく表示。

▶ 使い方の例
確率が高い上位 20 個の状態を表示：
```shell
python3 scripts/plot_annealing.py new_template --top 20
```

確率が 0.01 以上の状態のみ表示：
```shell
python3 scripts/plot_annealing.py new_template --threshold 0.01
```

出力ファイルを指定する場合：
```shell
python3 scripts/plot_annealing.py new_template --top 20 -o result.png
```

Claude Haiku 4.5 • 1x