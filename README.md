# ✈️ Travel Package Purchase Prediction — SIGNATE Cup (SOTA Challenge)

> **Binary classification** to predict whether a customer will purchase a travel package,
> using advanced feature engineering (AutoFeat + UMAP density embeddings) and LightGBM.

> **Note:** This repository documents the experimentation process.
> Scores cited reflect the best submission during the competition period;
> exact reproducibility may vary due to environment and seed differences.

---

## Competition Info

| Item | Detail |
|---|---|
| Platform | [SIGNATE Cup – SOTA Challenge](https://signate.jp/) |
| Task | Binary classification (ProdTaken: purchased / not purchased) |
| Metric | AUC-ROC |

---

## Technical Highlights

### Problem Overview
Predict travel product purchase from customer demographic and sales interaction data.  
Key challenges: mixed categorical/numeric features, missing values, and moderate class imbalance.

### Solution Pipeline

```
Raw Data
   ↓
[Preprocessing]  ── notebooks/01_preprocessing.ipynb
   ├── Categorical cleaning (unicode normalization, typo correction)
   ├── KNN imputation for numeric features
   └── Label encoding + OneHotEncoding for categoricals
   ↓
[Feature Engineering]  ── notebooks/03_feature_engineering_and_training.ipynb
   ├── AutoFeat polynomial cross-features
   ├── Correlation-based selection (target corr > 0.01, inter-feature < 0.9)
   └── UMAP density embeddings (n_neighbors = 15/20/25/30)  ── notebooks/02_umap.ipynb
   ↓
[Modeling]
   ├── LightGBM + Optuna (5-fold KFold, AUC objective)
   └── CatBoost (comparison)
   ↓
[Threshold Optimization]
   └── Best decision threshold search on validation AUC
```

### Key Experiments

| Method | CV AUC |
|---|---|
| LightGBM baseline | 0.698 |
| + AutoFeat features | 0.721 |
| + UMAP density embeddings | 0.736 |
| + Optuna tuning | 0.743 |

### Notable: UMAP Density Embeddings

Rather than using UMAP purely for visualization, this project incorporated UMAP-derived **density map coordinates** as additional numeric features — providing the model with a learned low-dimensional manifold structure of the input space.

```python
# n_neighbors swept: 15, 20, 25, 30
# densMAP=True to capture local density in embedding
reducer = umap.UMAP(n_neighbors=n, densmap=True, random_state=42)
embedding = reducer.fit_transform(X_scaled)
```

---

## Repository Structure

```
signate-cup-travel-prediction/
├── notebooks/
│   ├── 01_preprocessing.ipynb                    ← Data cleaning & encoding
│   ├── 02_umap.ipynb                             ← UMAP density embedding generation
│   ├── 03_feature_engineering_and_training.ipynb ← AutoFeat + LightGBM + Optuna
│   └── 04_training_simple.ipynb                  ← Simple LightGBM baseline
├── data/
│   ├── train.csv          ← raw training data (from SIGNATE)
│   ├── test.csv           ← raw test data
│   └── sample_submit.csv  ← submission format
├── outputs/
│   ├── submission_final.csv   ← final submission
│   ├── feature_importance.csv ← LightGBM feature importance
│   └── eda_*.png              ← EDA visualizations (target rate by feature)
├── requirements.txt
└── .gitignore
```

---

## Quickstart

```bash
git clone https://github.com/MasahiroTatsuta/signate-cup-travel-prediction
cd signate-cup-travel-prediction
pip install -r requirements.txt

# Place competition data in data/ (download from SIGNATE)
# Run notebooks in order:
jupyter notebook notebooks/01_preprocessing.ipynb
jupyter notebook notebooks/02_umap.ipynb
jupyter notebook notebooks/03_feature_engineering_and_training.ipynb
```

---

## Environment

| Library | Version |
|---|---|
| Python | 3.10+ |
| LightGBM | ≥ 4.0 |
| Optuna | ≥ 3.5 |
| scikit-learn | ≥ 1.3 |
| umap-learn | ≥ 0.5 |
| autofeat | ≥ 2.1 |

See [`requirements.txt`](requirements.txt) for the full list.

---

## What I Learned

- **UMAP density embeddings** as additional features improved AUC by ~0.015 over standard feature engineering alone — manifold structure captured meaningful customer segmentation that tabular features alone couldn't express
- Sweeping `n_neighbors` (15→30) showed n=20 optimal for this dataset; larger values over-smoothed local structure
- `AutoFeat` with `feateng_steps=2` generated useful interaction terms (e.g. `Age × NumberOfTrips`) that had high correlation with the target
