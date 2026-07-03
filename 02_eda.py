"""
02_eda.py
---------
Exploratory Data Analysis on the Amazon Top 50 Bestselling Books dataset.

Mirrors the structure of the Netflix EDA project:
  1. Load & inspect
  2. Clean (duplicates, missing values, dtypes)
  3. Univariate analysis
  4. Bivariate / multivariate analysis
  5. Trends over time
  6. Key insights summary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA_PATH = "data/amazon_bestsellers.csv"
FIG_DIR = "figures"

# ---------------------------------------------------------------
# 1. LOAD & INSPECT
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("SHAPE:", df.shape)
print("=" * 60)
print(df.info())
print("\nFirst rows:\n", df.head())
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# ---------------------------------------------------------------
# 2. CLEAN
# ---------------------------------------------------------------
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"\nDropped {before - len(df)} duplicate rows.")

# Impute missing Price with the median price of the same Genre
df["Price"] = df.groupby("Genre")["Price"].transform(lambda s: s.fillna(s.median()))

df["Genre"] = df["Genre"].astype("category")
df["Year"] = df["Year"].astype(int)

print("\nPost-cleaning missing values:\n", df.isna().sum())
print("\nDescribe (numeric):\n", df.describe())

# ---------------------------------------------------------------
# 3. UNIVARIATE ANALYSIS
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df["User Rating"], bins=15, kde=True, ax=axes[0], color="#2b6cb0")
axes[0].set_title("Distribution of User Rating")
sns.histplot(df["Reviews"], bins=30, kde=True, ax=axes[1], color="#c05621")
axes[1].set_title("Distribution of Reviews")
sns.histplot(df["Price"], bins=20, kde=True, ax=axes[2], color="#2f855a")
axes[2].set_title("Distribution of Price")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_univariate_distributions.png")
plt.close()

fig, ax = plt.subplots(figsize=(5, 4))
df["Genre"].value_counts().plot(kind="bar", color=["#2b6cb0", "#c05621"], ax=ax)
ax.set_title("Genre Split (all years combined)")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_genre_split.png")
plt.close()

# ---------------------------------------------------------------
# 4. BIVARIATE / MULTIVARIATE ANALYSIS
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    df[["User Rating", "Reviews", "Price", "Year"]].corr(),
    annot=True, cmap="coolwarm", center=0, ax=ax
)
ax.set_title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_correlation_heatmap.png")
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(data=df, x="Genre", y="Price", ax=ax)
ax.set_title("Price by Genre")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_price_by_genre.png")
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(data=df, x="Genre", y="User Rating", ax=ax)
ax.set_title("User Rating by Genre")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_rating_by_genre.png")
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
sns.scatterplot(data=df, x="Reviews", y="User Rating", hue="Genre", alpha=0.6, ax=ax)
ax.set_title("Reviews vs User Rating")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_reviews_vs_rating.png")
plt.close()

# ---------------------------------------------------------------
# 5. TRENDS OVER TIME
# ---------------------------------------------------------------
genre_trend = df.groupby(["Year", "Genre"], observed=True).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(8, 4.5))
genre_trend.plot(kind="line", marker="o", ax=ax)
ax.set_title("Fiction vs Non-Fiction Bestsellers per Year")
ax.set_ylabel("Count of bestselling titles")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_genre_trend_by_year.png")
plt.close()

price_trend = df.groupby("Year")["Price"].mean()
fig, ax = plt.subplots(figsize=(8, 4.5))
price_trend.plot(kind="line", marker="o", color="#2f855a", ax=ax)
ax.set_title("Average Book Price by Year")
ax.set_ylabel("Average Price ($)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/08_avg_price_trend.png")
plt.close()

# Top recurring authors
top_authors = df["Author"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(7, 5))
top_authors.sort_values().plot(kind="barh", color="#805ad5", ax=ax)
ax.set_title("Top 10 Authors by Bestseller Appearances")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/09_top_authors.png")
plt.close()

# ---------------------------------------------------------------
# 6. KEY INSIGHTS SUMMARY
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print(f"Total unique titles: {df['Name'].nunique()}")
print(f"Total unique authors: {df['Author'].nunique()}")
print(f"Average User Rating: {df['User Rating'].mean():.2f}")
print(f"Average Price: ${df['Price'].mean():.2f}")
print(f"Genre split:\n{df['Genre'].value_counts(normalize=True).round(2)}")
print(f"\nCorrelation of Reviews with Rating: {df['Reviews'].corr(df['User Rating']):.3f}")
print(f"Correlation of Price with Rating: {df['Price'].corr(df['User Rating']):.3f}")
print(f"\nMost frequent bestselling author: {df['Author'].value_counts().idxmax()} "
      f"({df['Author'].value_counts().max()} appearances)")

df.to_csv("data/amazon_bestsellers_cleaned.csv", index=False)
print("\nCleaned dataset saved to data/amazon_bestsellers_cleaned.csv")
print("All figures saved to figures/")
