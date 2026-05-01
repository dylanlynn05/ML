# Assignment 2 — Loss Functions & CNN Shape Math

**Course:** Dr. Banerjee
**Author:** Dylan Lynn
**Submitted:** March 29, 2026

The full write-up is in [`assignment2_dylanlynn.docx`](assignment2_dylanlynn.docx). This README is a quick reference of the contents.

---

## Part 1 — Loss functions

| # | Name | Equation | Used for |
|---|---|---|---|
| 1 | **MSE** (Mean Squared Error) | `(1/N) Σ (y_i − ŷ_i)²` | Regression — large errors penalised heavily (squared) |
| 2 | **MAE** (Mean Absolute Error) | `(1/N) Σ \|y_i − ŷ_i\|` | Regression — robust to outliers (linear penalty) |
| 3 | **BCE** (Binary Cross-Entropy) | `−(1/N) Σ [y_i log p_i + (1 − y_i) log(1 − p_i)]` | Binary classification (e.g. normal vs abnormal) |
| 4 | **CCE** (Categorical Cross-Entropy) | `−(1/N) Σ_i Σ_c [y_{i,c} log p_{i,c}]` | Multi-class classification with softmax outputs |
| 5 | **Huber** | Quadratic for `\|y − ŷ\| ≤ δ`, linear for `\|y − ŷ\| > δ` | Regression with outliers — best of MSE + MAE |

Pick **MSE** when you want to punish big errors more, **MAE/Huber** when outliers shouldn't dominate, **BCE/CCE** for classification.

---

## Part 2 — CNN shape calculations

**Given input:** matrix shape `4 × 5`, channels = 1 → tensor `(4, 5, 1)`

**Layer settings:**
- `Conv2D`: kernel `2×2`, stride `1`, padding `0`
- `MaxPool`: kernel `2×2`, stride `2`, padding `0`

**Conv2D output formula:**
```
H_out = ((H_in − K_h + 2P) / S) + 1
W_out = ((W_in − K_w + 2P) / S) + 1
```
→ `H_out = (4 − 2)/1 + 1 = 3`, `W_out = (5 − 2)/1 + 1 = 4` → `(3, 4, F)` where `F` = #filters.

**MaxPool** on `(3, 4, F)` with `2×2` stride 2:
→ `H_out = floor((3 − 2)/2) + 1 = 1`, `W_out = floor((4 − 2)/2) + 1 = 2` → `(1, 2, F)`.

**Flatten:** `1 · 2 · F = 2F` → `(2F,)`.

**Dense head:** `(2F,) → (64,) → (32,) → (1,)` for binary output.

**Full flow (general F):**
```
(4, 5, 1) → (3, 4, F) → (1, 2, F) → (2F,) → (64,) → (32,) → (1,)
```

**Full flow (F = 1):**
```
(4, 5, 1) → (3, 4, 1) → (1, 2, 1) → (2,) → (64,) → (32,) → (1,)
```
