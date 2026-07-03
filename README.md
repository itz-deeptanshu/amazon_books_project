# Amazon Top 50 Bestselling Books — EDA + Modeling

## Important: about the data
This environment can't reach kaggle.com, so I couldn't download the real
**"Amazon Top 50 Bestselling Books 2009-2019"** dataset directly. `01_generate_dataset.py`
instead generates a **synthetic dataset with the identical schema and realistic
statistical distributions** (same columns, same year range, same rough scale
of ratings/reviews/prices, recurring real authors).

**To run this on the real data:** download the dataset from Kaggle, save it as
`data/amazon_bestsellers.csv` with columns `Name, Author, User Rating, Reviews,
Price, Year, Genre`, and skip straight to `02_eda.py`. Everything downstream
is written against that schema and needs no changes.

## Project structure
```
amazon_books_project/
├── 01_generate_dataset.py   # synthetic data generator (skip if you have the real CSV)
├── 02_eda.py                # cleaning + exploratory data analysis
├── 03_modeling.py           # 9-model comparison: predict Fiction vs Non Fiction
├── data/
│   ├── amazon_bestsellers.csv           # raw (synthetic)
│   ├── amazon_bestsellers_cleaned.csv   # after dedup + missing-value handling
│   └── model_comparison_results.csv     # metrics table
└── figures/                 # 12 PNG charts (see below)
```

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
python3 01_generate_dataset.py   # only if you don't have the real CSV
python3 02_eda.py
python3 03_modeling.py
```

## What 02_eda.py covers
- Load, inspect dtypes, missing values, duplicates
- Cleaning: drop duplicates, impute missing `Price` by genre median
- Univariate distributions: Rating, Reviews, Price, Genre split
- Bivariate: correlation heatmap, Price/Rating by Genre, Reviews vs Rating
- Trends over time: Fiction vs Non-Fiction counts per year, average price per year
- Top 10 most frequently recurring bestselling authors
- Printed key-insights summary (means, correlations, genre split, top author)

## What 03_modeling.py covers
Binary classification task: **predict Genre (Fiction / Non Fiction) from
User Rating, Reviews, Price, and Year.**

Nine models compared head-to-head (Logistic Regression, KNN, SVM, Decision
Tree, Random Forest, Gradient Boosting, AdaBoost, Naive Bayes, XGBoost),
evaluated on Accuracy, Precision, Recall, F1, and ROC-AUC. Best model gets
a confusion matrix, ROC curve, and feature-importance chart (if tree-based).

**Honest result:** on this dataset, Genre is only weakly correlated with these
four numeric features (see the correlation heatmap), so accuracy sits around
65-72% — a modest lift over the ~53% majority-class baseline. That's a
legitimate finding: it means rating/reviews/price/year alone don't strongly
determine genre, not a bug in the pipeline. On the real Kaggle data, re-run
and check whether the same pattern holds — the relationships may differ
slightly with the actual title-level data.

## Figures generated
| # | File | Shows |
|---|------|-------|
| 01 | univariate_distributions.png | Rating / Reviews / Price histograms |
| 02 | genre_split.png | Fiction vs Non Fiction counts |
| 03 | correlation_heatmap.png | Correlation between numeric features |
| 04 | price_by_genre.png | Price distribution by genre |
| 05 | rating_by_genre.png | Rating distribution by genre |
| 06 | reviews_vs_rating.png | Scatter: reviews vs rating, colored by genre |
| 07 | genre_trend_by_year.png | Fiction/Non-Fiction counts per year |
| 08 | avg_price_trend.png | Average price per year |
| 09 | top_authors.png | Top 10 authors by bestseller appearances |
| 10 | model_comparison.png | Accuracy/F1/ROC-AUC across 9 models |
| 11 | confusion_matrix.png | Confusion matrix for best model |
| 12 | roc_curve.png | ROC curve for best model |
