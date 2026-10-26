# Clustering & anomaly metrics

Without true labels you can't count right and wrong answers, so clusterings are judged by how tight and well separated the groups are, and anomaly detection is judged with stability checks and common sense.

## Why these are different

> **What you need to know**
> - [supervised](#/t/supintro) metrics (RMSE, precision, recall, F1) compare predictions with **true answers**
> - unsupervised models have **no answers** ([What unsupervised means](#/t/unsupintro)), so their metrics measure the **structure** the model found
> - you can't compute accuracy for a K-means result, and a silhouette score means nothing for a regression
>
> **Skip:** other cluster scores (Davies–Bouldin, Calinski–Harabasz, ARI); not needed at junior level

## The running example: seven customers

Let's say a shop describes seven customers by two numbers, **visits per month** (x) and **spend per visit** (y, in £10s), and a clustering has put them into two groups:

| Customer | Visits (x) | Spend (y) | Group |
|---|---|---|---|
| P1 | 1 | 1 | A (occasional, low spend) |
| P2 | 2 | 1 | A |
| P3 | 1 | 2 | A |
| P4 | 6 | 5 | B (regular, high spend) |
| P5 | 7 | 5 | B |
| P6 | 6 | 6 | B |
| P7 | 4 | 3 | B |

P7 sits between the two groups. A group's **centroid** is its centre point (the average of its members): A's is (1.33, 1.33) and B's is (5.75, 4.75).

## Inertia (WCSS) and the elbow

> **What you need to know**
> - **In plain words:** inertia, also called **WCSS** (within-cluster sum of squares), adds up each point's squared distance to its own group's centre. **Lower = tighter groups**
> - **In context:** the squared distances are 0.22, 0.56, 0.56 (group A) and 0.13, 1.63, 1.63, 6.13 (group B), total **10.83**. P7 alone gives more than half, because it sits far from B's centre
> - **Best with:** [K-means](#/t/kmeans), which works by making inertia as small as possible (sklearn's `inertia_`)
> - **Use it when:** choosing the number of clusters $k$ with the **elbow method**; customer segments, grouping sensor readings
> - **Watch out:** inertia **always** falls as $k$ grows (one customer per group gives 0), so never pick $k$ by the lowest value
>
> **Skip:** the formula

**The elbow method:** plot inertia against $k$ and look for the bend after which extra clusters barely help.

![Inertia (WCSS) against k for data with four true groups. It drops steeply up to k = 4 and then flattens.](figures/unsupmetrics-elbow.svg)

In the figure, going from 3 to 4 clusters cuts inertia from 831 to 246, but going from 4 to 5 only takes it to 218. The bend at $k = 4$ is the elbow. Real elbows are often blurry, so check with the silhouette.

## Silhouette score

> **What you need to know**
> - **In plain words:** for each point, "am I closer to my own group than to the next-nearest group?" Scored from **−1 to +1**
> - near **+1**: deep inside its own group. Near **0**: on the border. **Below 0**: probably in the wrong group
> - the **silhouette score** is the average over all points; higher is better
> - **Best with:** K-means and other distance-based clustering ([hierarchical](#/t/hierarchical) too)
> - **Use it when:** choosing $k$ alongside the elbow; checking whether a customer segmentation is clean or muddled
> - **Watch out:** it favours round, evenly sized, well-separated groups; a high score means "tight and separated", not "useful"
>
> **Skip:** the formula; `silhouette_score` does it

**In context, P1 by hand:**

1. Average distance to its own group-mates (P2, P3): both are 1 away → **1.0**. Small is good.
2. Average distance to the other group's customers (P4–P7): 6.40, 7.21, 7.07 and 3.61 → **6.07**. Big is good.
3. P1 is much nearer its own group, so its score is high: **0.835**.

P1 to P6 all score between 0.70 and 0.84. **P7 scores −0.04**: on average it is slightly closer to group A, so it's probably in the wrong group. The mean silhouette is **0.64**.

![Left: the seven customers, with lines from P7 to every other point. Right: each customer's silhouette; P7 is the only negative bar.](figures/unsupmetrics-silworked.svg)

In the figure, P7's lines to group A are, on average, a little shorter than its lines to group B. That's why its bar dips below zero.

![Silhouette plots for k = 3, 4 and 5 on the four-blob data. k = 4 gives the highest mean and the most even bands.](figures/unsupmetrics-silplot.svg)

Each band is one cluster's points, sorted by score. At $k = 3$ one band is very wide (two real groups merged); at $k = 5$ a real group is split into thin, low-scoring bands. $k = 4$ scores best (0.72) and agrees with the elbow.

## Anomaly detection: hard to score

> **What you need to know**
> - **anomalies are rare and usually unlabelled**, so there is nothing to count right or wrong
> - models give a **score**, not a yes/no; where you cut (top 1%? top 20?) is your decision. In sklearn, `contamination` is that guess
> - so you check **stability** (are the same points flagged with different seeds or settings?) and **sense** (do the flags match known events?)
> - if someone has confirmed a few real anomalies, use **precision@k**: of your top $k$ flags, how many are real?
> - **Best with:** [anomaly detectors](#/t/anomaly) such as Isolation Forest
> - **Watch out:** unusual isn't always interesting; a flag may be a data error, not a real event

![Isolation Forest scores on 1,000 synthetic points. Normal points form the bulk, the 12 true anomalies sit in the right tail, and the dashed line marks the top 2%.](figures/unsupmetrics-anomscores.svg)

In the figure, the true anomalies (wine) sit in the right tail. In real life you wouldn't know which those are. The 2% cut catches all 12, plus some normal points from the edge of the bulk.

**In context:** for the [GB electricity demand project](#/p/elec), a sense check is whether flagged half-hours line up with bank holidays, cold snaps or big TV events. If 10 known events are listed and 7 of your top-10 flags match them, precision@10 = 0.7.

## Which one?

| Situation | Use |
|---|---|
| Choosing $k$ for K-means | elbow (inertia) **and** silhouette together |
| Is this clustering clean or muddled? | mean silhouette; look for negative points |
| Which customer is in the wrong group? | per-point silhouette (below 0) |
| Anomalies, no labels at all | stability across reruns + sense against known events |
| Anomalies, a few confirmed cases | precision@k |

## Code

```python
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score

X = np.array([[1, 1], [2, 1], [1, 2], [6, 5], [7, 5], [6, 6], [4, 3]], float)  # 7 customers
groups = np.array([0, 0, 0, 1, 1, 1, 1])                 # A = 0, B = 1
print(silhouette_samples(X, groups).round(2))            # one score per customer; P7 is -0.04
print(silhouette_score(X, groups).round(2))              # the average: 0.64
```

> **Trap:** distances depend on each feature's units. Scale first ([Scaling & encoding](#/t/scaling)), or one large-unit feature (say, spend in pounds) dominates every distance.

## Interview one-liners

- **How do you evaluate K-means?** "No labels, so inertia with the elbow method and the silhouette score, plus whether the groups make business sense."
- **Why not pick the k with the lowest inertia?** "Inertia always falls as k grows; you look for the elbow instead."
- **What does a negative silhouette mean?** "That point is closer to another cluster than its own, so it's probably misassigned."
- **How do you evaluate anomaly detection?** "Stability and domain sense; precision@k if a few anomalies are confirmed."

## Quick reference

| Term | Meaning | Better |
|---|---|---|
| Centroid | centre (average) of a cluster | |
| Inertia / WCSS | total squared distance from each point to its centroid | lower, but always falls with $k$ |
| Elbow | the $k$ after which inertia stops falling steeply | |
| Silhouette | own group vs nearest other group, −1 to +1 | higher |
| Contamination | your guess of the share of anomalies | (an assumption) |
| Precision@k | share of your top $k$ flags that are real | higher |
