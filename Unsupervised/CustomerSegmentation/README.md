# Customer Segmentation for Targeted Advertising

Group shoppers into a small number of behaviourally distinct segments using **K-Means**, then
turn each segment into a concrete advertising persona.

## Files

| File | Purpose |
|---|---|
| [`customer_segmentation.ipynb`](customer_segmentation.ipynb) | End-to-end, fully-executed notebook (data → EDA → scaling → k-selection → K-Means → cluster profiling → personas → predict-for-new-customer) |
| [`Project_Report.docx`](Project_Report.docx) | Combined Word write-up covering this project and the companion Fake-News-Detection project |

## What the notebook does

1. **Build the dataset** — synthesise a 200-row Mall-Customers-style table
   (`Age`, `Annual Income (k$)`, `Spending Score (1-100)`, `Gender`) as a mixture of **five**
   latent customer archetypes plus light Gaussian noise.
2. **Explore** — distributions, scatter of income vs. spending, correlation heat-map.
3. **Standardise** numeric features with `StandardScaler` (essential because K-Means uses
   Euclidean distance).
4. **Choose `k`** — sweep `k = 2..10` and plot both:
   - **Inertia** (elbow method)
   - **Average silhouette score** (peaks at the chosen `k`)
5. **Fit K-Means** with the chosen `k` and visualise the clusters in the original
   income / spending plane *and* in PCA space.
6. **Profile** each cluster (mean age / income / spend, gender mix) and **label** it with a
   marketing persona + a tailored campaign idea.
7. **Predict** the segment for five brand-new customers as a sanity check.

## Results

The silhouette curve peaks unambiguously at **`k = 5`** with an average silhouette of
**0.486** — comfortably above the 0.4 rule-of-thumb threshold for structurally meaningful
clusters, and a clean recovery of the five latent archetypes that generated the data.

| Cluster | Persona | Avg age | Avg income (k$) | Avg spend | Recommended campaign |
|---|---|---|---|---|---|
| 2 | Premium loyalists   | 27.8 | 83.9 | 78.4 | VIP previews, premium loyalty tier, early access |
| 0 | Aspirational youth  | 24.4 | 30.6 | 72.4 | Trend-driven social ads, BNPL financing, influencers |
| 4 | Mainstream regulars | 38.0 | 55.1 | 50.2 | Broad seasonal campaigns, mid-tier cross-sell |
| 1 | Cautious affluents  | 44.7 | 89.4 | 18.7 | Quality- and durability-focused, premium-but-rational |
| 3 | Budget-conscious    | 49.3 | 26.9 | 20.1 | Discount-led, coupons, low-ticket loyalty points |

Cluster sizes: 47 / 41 / 43 / 40 / 29.

## How to run

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook Unsupervised/CustomerSegmentation/customer_segmentation.ipynb
```

`random_state = 42` is fixed throughout, so re-running top-to-bottom reproduces every metric
and plot exactly. To run on the real Kaggle "Mall Customers" CSV, replace the data-generation
cell with `pd.read_csv("Mall_Customers.csv")` — the rest of the pipeline is unchanged.

## Why this is "unsupervised"

The model never sees a "correct cluster" label during training — K-Means only sees the raw
feature vectors and discovers groupings on its own by minimising within-cluster squared
distance. The personas are attached *after* the fact by inspecting each cluster's centroid.
