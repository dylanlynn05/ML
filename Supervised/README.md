# Supervised Learning

Projects in this folder train models on **labeled** data — every training example has both an input `x` and a known target `y`. The model's job is to learn the mapping `x → y` (regression for continuous `y`, classification for categorical `y`).

## Projects

| Folder | Task | Type |
|---|---|---|
| [`Housing/`](Housing/) | Predict housing prices from features | Regression |
| [`Assignment2-LossFunctions/`](Assignment2-LossFunctions/) | Theory: loss functions and CNN shape math | Theory / write-up |

## Common loss functions used here

- **MSE / MAE / Huber** — regression losses (used by `Housing/`).
- **Binary / Categorical Cross-Entropy** — classification losses (covered in `Assignment2-LossFunctions/`).

See [`Assignment2-LossFunctions/`](Assignment2-LossFunctions/) for the full reference of each loss and when to pick which.
