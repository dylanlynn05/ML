# Semi-Supervised Learning

Projects in this folder use a **mix** of a small amount of labeled data and a large amount of unlabeled data. Semi-supervised methods aim to do better than purely supervised training when labels are expensive to collect, by exploiting the structure of the unlabeled set.

## Projects

| File | Description |
|---|---|
| [`assignment1_doe_john.ipynb`](assignment1_doe_john.ipynb) | Assignment 1 — semi-supervised classification template/exercise |

## Common semi-supervised techniques (for reference)

- **Self-training / pseudo-labelling** — train on labeled data, predict on unlabeled, add high-confidence predictions back to the training set.
- **Consistency regularisation** (e.g. FixMatch, MixMatch) — penalise the model for changing its predictions under input perturbations.
- **Graph-based label propagation** — spread labels along a similarity graph of all examples.
- **Generative semi-supervised models** — combine a generative model `p(x)` with a discriminative head `p(y \| x)`.
