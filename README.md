# ✈️ Travel Package Purchase Prediction — SIGNATE Cup（SOTA Challenge）

> **旅行パッケージを顧客が購入するかどうかを予測する二値分類タスク**  
> 高度な特徴量エンジニアリング（AutoFeat + UMAP密度埋め込み）と LightGBM を用いて構築。

> **注記:**  
> このリポジトリは実験プロセスを記録したものです。  
> 記載されているスコアはコンペ期間中のベスト提出結果であり、環境や乱数シードの違いにより完全再現できない場合があります。

---

# コンペ概要

| 項目 | 内容 |
|---|---|
| プラットフォーム | SIGNATE Cup – SOTA Challenge |
| タスク | 二値分類（ProdTaken: 購入 / 非購入） |
| 評価指標 | AUC-ROC |

---

# 技術的ハイライト

## 問題概要

顧客属性データおよび営業接触データを用いて、旅行商品の購入有無を予測します。  

主な課題:

- カテゴリ変数と数値変数が混在
- 欠損値の存在
- 中程度のクラス不均衡

---

## ソリューションパイプライン

```text
Raw Data
   ↓
[Preprocessing]  ── notebooks/01_preprocessing.ipynb
   ├── カテゴリデータのクリーニング（Unicode正規化・誤字修正）
   ├── 数値特徴量へのKNN補完
   └── Label Encoding + OneHotEncoding
   ↓
[Feature Engineering]  ── notebooks/03_feature_engineering_and_training.ipynb
   ├── AutoFeat による多項式交差特徴量生成
   ├── 相関ベース特徴量選択
   │      ├── 目的変数との相関 > 0.01
   │      └── 特徴量間相関 < 0.9
   └── UMAP 密度埋め込み
          (n_neighbors = 15 / 20 / 25 / 30)
          ── notebooks/02_umap.ipynb
   ↓
[Modeling]
   ├── LightGBM + Optuna
   │      └── 5-fold KFold / AUC最適化
   └── CatBoost（比較用）
   ↓
[Threshold Optimization]
   └── 検証AUC上で最適な閾値を探索
```

---

## 主な実験結果

| 手法 | CV AUC |
|---|---|
| LightGBM ベースライン | 0.698 |
| + AutoFeat 特徴量 | 0.721 |
| + UMAP 密度埋め込み | 0.736 |
| + Optuna チューニング | 0.743 |

---

## 注目ポイント: UMAP Density Embeddings

UMAP を単なる可視化用途として使うのではなく、  
UMAP から得られる **密度マップ座標** を追加の数値特徴量として利用しました。

これにより、入力空間における低次元多様体構造（manifold structure）をモデルへ学習させることができました。

```python
# n_neighbors を 15, 20, 25, 30 で探索
# densMAP=True により局所密度情報を保持
reducer = umap.UMAP(n_neighbors=n, densmap=True, random_state=42)
embedding = reducer.fit_transform(X_scaled)
```

---

# リポジトリ構成

```text
signate-cup-travel-prediction/
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   │      ← データクリーニング & エンコーディング
│   ├── 02_umap.ipynb
│   │      ← UMAP 密度埋め込み生成
│   ├── 03_feature_engineering_and_training.ipynb
│   │      ← AutoFeat + LightGBM + Optuna
│   └── 04_training_simple.ipynb
│          ← シンプルな LightGBM ベースライン
│
├── data/
│   ├── train.csv
│   │      ← 学習データ（SIGNATE提供）
│   ├── test.csv
│   │      ← テストデータ
│   └── sample_submit.csv
│          ← 提出フォーマット
│
├── outputs/
│   ├── submission_final.csv
│   │      ← 最終提出ファイル
│   ├── feature_importance.csv
│   │      ← LightGBM 特徴量重要度
│   └── eda_*.png
│          ← EDA可視化（特徴量ごとの目的変数比率）
│
├── requirements.txt
└── .gitignore
```

---

# クイックスタート

```bash
git clone https://github.com/MasahiroTatsuta/signate-cup-travel-prediction
cd signate-cup-travel-prediction

pip install -r requirements.txt

# SIGNATEから取得したデータを data/ に配置

# Notebook を順番に実行
jupyter notebook notebooks/01_preprocessing.ipynb
jupyter notebook notebooks/02_umap.ipynb
jupyter notebook notebooks/03_feature_engineering_and_training.ipynb
```

---

# 実行環境

| ライブラリ | バージョン |
|---|---|
| Python | 3.10+ |
| LightGBM | ≥ 4.0 |
| Optuna | ≥ 3.5 |
| scikit-learn | ≥ 1.3 |
| umap-learn | ≥ 0.5 |
| autofeat | ≥ 2.1 |

完全な一覧は `requirements.txt` を参照してください。

---

# 学んだこと

- **UMAP density embeddings** を追加特徴量として利用することで、通常の特徴量エンジニアリング単体と比較して AUC が約 `+0.015` 向上した  
  → 多様体構造が、通常の表形式特徴量だけでは表現できない顧客セグメント情報を捉えていた

- `n_neighbors` を 15 → 30 で探索した結果、このデータセットでは `n=20` が最適だった  
  → 大きすぎる値では局所構造が過度に平滑化された

- `AutoFeat` の `feateng_steps=2` により、有用な交互作用特徴量（例: `Age × NumberOfTrips`）が生成され、目的変数との高い相関を示した
