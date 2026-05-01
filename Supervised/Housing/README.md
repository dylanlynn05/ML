# Housing — Regression

Supervised regression project: predict housing prices from a set of numerical and categorical features.

## File

- [`Housing.ipynb`](Housing.ipynb) — end-to-end notebook: data loading, exploratory analysis, preprocessing, model training, and evaluation.

## How to run

```bash
# from the repo root, with your venv active
pip install jupyter numpy pandas scikit-learn matplotlib seaborn
jupyter notebook Supervised/Housing/Housing.ipynb
```

Then run the cells top-to-bottom.

## What this is an example of

A textbook supervised-learning regression pipeline:

1. **Load** the housing dataset (features + price label).
2. **Explore** distributions and correlations.
3. **Preprocess** — handle missing values, encode categoricals, scale numerics.
4. **Split** into train / test.
5. **Train** a regression model (e.g. linear regression / random forest / gradient boosting).
6. **Evaluate** with MSE / MAE / R² on the held-out test set.
