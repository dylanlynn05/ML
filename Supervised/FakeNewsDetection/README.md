# Fake News Detection

A binary text classifier that labels news articles as **REAL** or **FAKE** using a classic
**TF-IDF + linear-model** pipeline.

## Files

| File | Purpose |
|---|---|
| [`fake_news_detection.ipynb`](fake_news_detection.ipynb) | End-to-end, fully-executed notebook (data → cleaning → TF-IDF → models → evaluation → predictions on new headlines) |
| [`Project_Report.docx`](Project_Report.docx) | Combined Word write-up covering this project and the companion Customer-Segmentation project |

## What the notebook does

1. **Build the dataset** — generate a balanced corpus of 600 articles (300 REAL, 300 FAKE) from
   two distinct stylistic distributions:
   - **REAL** — Reuters / AP-style: named institutions, dated attributions, neutral verbs.
   - **FAKE** — clickbait / conspiracy-style: ALL-CAPS shock openers, vague subjects,
     hidden-agenda verbs, sensational closers.
2. **Explore** — class balance, word-count distributions, sample inspection.
3. **Clean** — lower-case, strip URLs / HTML / punctuation / digits, collapse whitespace.
4. **Vectorise** — `TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.9, min_df=2)`
   → 924 features.
5. **Train** two models and compare:
   - Multinomial Naive Bayes
   - Logistic Regression
6. **Evaluate** — accuracy, precision/recall/F1, confusion matrix, top informative words from
   the LR coefficients.
7. **Predict** on five brand-new headlines as a sanity check.

## Results

| Model | Test accuracy |
|---|---|
| Multinomial Naive Bayes | **1.0000** |
| Logistic Regression     | **1.0000** |

Confusion matrix on the held-out 120-sample test set is perfectly diagonal
(60 TN, 60 TP, 0 FP, 0 FN).

The 100 % accuracy is expected on this clean, stylistically-distinct synthetic corpus — on a
noisier real-world dataset (e.g. the Kaggle "Fake and Real News" set, ~45 k articles) the same
pipeline typically lands in the **92–97 %** range. The notebook can be retargeted to that
dataset by replacing only the data-generation cell with `pd.read_csv(...)`.

## How to run

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook Supervised/FakeNewsDetection/fake_news_detection.ipynb
```

`random_state = 42` is fixed throughout, so re-running top-to-bottom reproduces every metric
and plot exactly.

## Why this is "supervised"

Each training article comes with its true label (`REAL` / `FAKE`). The model learns the
mapping `text → label` from those `(x, y)` pairs — the textbook definition of supervised
classification.
