"""
01_generate_dataset.py
-----------------------
Generates a SYNTHETIC version of the "Amazon Top 50 Bestselling Books
2009-2019" dataset.

WHY SYNTHETIC:
This environment cannot reach kaggle.com to download the real file. This
script reproduces the exact schema (Name, Author, User Rating, Reviews,
Price, Year, Genre) and realistic statistical distributions of the real
dataset, so 02_eda.py and 03_modeling.py run unchanged on it.

TO USE REAL DATA INSTEAD:
Download "Amazon Top 50 Bestselling Books 2009-2019" from Kaggle, save it
as data/amazon_bestsellers.csv with the same column names, and skip this
script entirely.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

YEARS = list(range(2009, 2020))          # 11 years
BOOKS_PER_YEAR = 50

# A pool of recurring bestselling authors (real, publicly known names —
# these authors genuinely appear repeatedly on Amazon's yearly bestseller
# lists across this period), split by the genre they're mostly known for.
FICTION_AUTHORS = [
    "Suzanne Collins", "Jeff Kinney", "Rick Riordan", "John Green",
    "Paula Hawkins", "Gillian Flynn", "E L James", "Dr. Seuss",
    "J.K. Rowling", "George R.R. Martin", "Veronica Roth", "Stephenie Meyer",
    "Delia Owens", "Celeste Ng", "Colleen Hoover",
]
NONFICTION_AUTHORS = [
    "American Psychological Association", "Gary Chapman", "Rachel Hollis",
    "Jordan B. Peterson", "Tara Westover", "Michelle Obama", "Bessel van der Kolk",
    "Malcolm Gladwell", "Bill O'Reilly", "Angie Thomas", "National Geographic Kids",
    "Adam Gasiewski", "William P. Young", "Dav Pilkey",
]

FICTION_TITLES = [
    "The Silent Patient", "Where the Crawdads Sing", "It Ends with Us",
    "The Hunger Games", "Diary of a Wimpy Kid", "The Girl on the Train",
    "Gone Girl", "Percy Jackson and the Lightning Thief", "The Fault in Our Stars",
    "Fifty Shades of Grey", "A Game of Thrones", "Divergent", "Twilight",
    "Little Fires Everywhere", "Verity", "The Da Vinci Code", "Educated: A Memoir",
]
NONFICTION_TITLES = [
    "The 5 Love Languages", "12 Rules for Life", "Becoming",
    "The Body Keeps the Score", "Outliers", "Killing the Rising Sun",
    "The Girl Who Kicked the Hornet's Nest", "Publication Manual of the APA",
    "Girl, Wash Your Face", "The Wonderful Things You Will Be",
    "Strengths Finder 2.0", "Oh, the Places You'll Go!", "The Shack",
    "Dog Man", "How to Win Friends and Influence People",
]

rows = []
for year in YEARS:
    n_fiction = rng.integers(18, 30)          # fiction share varies by year
    n_nonfiction = BOOKS_PER_YEAR - n_fiction

    for _ in range(n_fiction):
        author = rng.choice(FICTION_AUTHORS)
        title = rng.choice(FICTION_TITLES)
        rating = np.clip(rng.normal(4.55, 0.25), 3.3, 4.9).round(1)
        reviews = int(np.clip(rng.lognormal(8.5, 1.1), 30, 90000))
        price = int(np.clip(rng.normal(9, 5), 0, 55))
        rows.append([title, author, rating, reviews, price, year, "Fiction"])

    for _ in range(n_nonfiction):
        author = rng.choice(NONFICTION_AUTHORS)
        title = rng.choice(NONFICTION_TITLES)
        rating = np.clip(rng.normal(4.65, 0.2), 3.3, 4.9).round(1)
        reviews = int(np.clip(rng.lognormal(8.0, 1.2), 30, 90000))
        price = int(np.clip(rng.normal(12, 7), 0, 105))
        rows.append([title, author, rating, reviews, price, year, "Non Fiction"])

df = pd.DataFrame(
    rows,
    columns=["Name", "Author", "User Rating", "Reviews", "Price", "Year", "Genre"],
)

# Introduce a small amount of realistic messiness (Kaggle's real file is
# clean, but real-world Amazon exports often aren't) — a few duplicate
# rows and a couple of missing prices — so the EDA script has something
# genuine to clean, matching the style of the Netflix/stroke projects.
dupe_idx = rng.choice(df.index, size=5, replace=False)
df = pd.concat([df, df.loc[dupe_idx]], ignore_index=True)
missing_idx = rng.choice(df.index, size=6, replace=False)
df.loc[missing_idx, "Price"] = np.nan

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

out_path = "data/amazon_bestsellers.csv"
df.to_csv(out_path, index=False)
print(f"Synthetic dataset written to {out_path}")
print(f"Shape: {df.shape}")
print(df.head())
