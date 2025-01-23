# Findings

Measured results from this repository. Each number below comes from re-running the repository's own code offline
(`python etl/pipeline.py`, then `python models/segmentation.py`); the transactions are synthetic and seeded.

## Do RFM features separate customers into distinct groups?

**How it was tested:** recency, frequency and monetary value are computed for each of the 2,000 customers, standardised, and
clustered with K-Means (k = 4, `random_state=42`). The generator draws transactions so that 20% of customers make about 80% of
purchases (`etl/pipeline.py`), so a good segmentation should isolate that group.

| Measure | Value |
|---|---|
| Variance in the scaled RFM features explained by the 4 clusters | 93.9% |
| Variance kept by the 2-D PCA view used for the chart | 99.6% |

| Segment | Customers | Mean recency (days) | Mean frequency | Mean monetary |
|---|---|---|---|---|
| VIP Champions | 400 | 3.985 | 212.245 | 71,759.55 |
| Loyal Customers | 937 | 20.727855 | 13.913554 | 4,813.462647 |
| New Customers | 504 | 77.767857 | 12.918651 | 4,260.455357 |
| At Risk / Hibernating | 159 | 175.075472 | 11.207547 | 3,550.386792 |

**What it shows:** one cluster stands far apart on frequency and spend (the VIP Champions); the other three have similar frequency
and spend and differ mainly in recency.

**Decision:** segment names are written back to the customer dimension in the star schema for the Power BI dashboard.

**Skills shown:** RFM analysis, scikit-learn (StandardScaler, KMeans, PCA), pandas, SQLite star schema, Power BI.

**Source:** `models/segmentation.py` (printed variance, cluster means and segment counts; segment counts confirmed in the
`dim_customers` table it writes), `etl/pipeline.py`.
