# Unsupervised metrics

Without labels you can't count right and wrong answers, so unsupervised models are judged with their own metrics, which measure how compact and well separated clusters are, how much information a reduction keeps, or how sensible and stable the flagged anomalies are.

## Different families, different metrics

- **Supervised** models have true labels, so their metrics compare predictions with answers: MAE and RMSE for [regression](#/t/regmetrics); precision, recall, F1 and PR-AUC for [classification](#/t/clsmetrics).
- **Unsupervised** models have no labels (see [What unsupervised means](#/t/unsupintro)). Their metrics measure the **structure** the model found.

These two sets of metrics are not interchangeable. You can't compute accuracy for a K-means result, and silhouette means nothing for a regression.

| Family | What we measure | Metrics in this chapter |
|---|---|---|
| Clustering, internal (no labels at all) | Compact, well-separated clusters | Inertia/WCSS, silhouette, Davies–Bouldin, Calinski–Harabasz |
| Clustering, external (some labels exist) | Agreement with known groups | Adjusted Rand Index, NMI |
| Dimensionality reduction | Information kept | Explained variance ratio, reconstruction error |
| Anomaly detection | Sensible, stable flags | Contamination, score thresholds, precision@k, stability |

**Notation.** $n$ points $x_i$, $k$ clusters $C_1, \dots, C_k$, centroid $\mu_j$ (the mean of the points in $C_j$), and $d(\cdot,\cdot)$ is the Euclidean distance.

## The worked example

Seven 2-D points, already put into two clusters:

| Point | x | y | Cluster |
|---|---|---|---|
| P1 | 1 | 1 | A |
| P2 | 2 | 1 | A |
| P3 | 1 | 2 | A |
| P4 | 6 | 5 | B |
| P5 | 7 | 5 | B |
| P6 | 6 | 6 | B |
| P7 | 4 | 3 | B |

P7 sits between the two groups. Centroids: $\mu_A = (1.33, 1.33)$, $\mu_B = (5.75, 4.75)$. All numbers below were computed in Python.

## Clustering: internal metrics

### Inertia / WCSS

**Inertia**, also called **WCSS** (within-cluster sum of squares), is the total squared distance from each point to its own cluster's centroid:

$$
\text{WCSS} = \sum_{j=1}^{k} \sum_{x_i \in C_j} \lVert x_i - \mu_j \rVert^2
$$

- It measures how **tight** the clusters are. Lower = tighter.
- K-means minimises exactly this quantity.

**Worked example.** The squared distances to the centroids are P1 0.22, P2 0.56, P3 0.56 (cluster A) and P4 0.13, P5 1.63, P6 1.63, P7 6.13 (cluster B). Total WCSS = **10.83**. P7 alone contributes more than half.

> **Trap:** WCSS always goes down when you add clusters. With $k = n$, every point is its own centroid and WCSS = 0. So you can't pick $k$ by minimising it.

**Elbow method.** Plot WCSS against $k$ and look for the "elbow": the point after which extra clusters give only a small improvement.

![WCSS against k for data with four true groups. It drops steeply up to k = 4 and then flattens.](figures/unsupmetrics-elbow.svg)

In the figure, going from 3 to 4 clusters cuts WCSS from 831 to 246. From 4 to 5, it falls only to 218. The bend at $k = 4$ is the elbow. On real data the elbow is often blurry, so check it against the silhouette score.

### Silhouette score

The silhouette asks, for each point: *am I closer to my own cluster than to the nearest other cluster?*

1. $a(i)$ = mean distance from point $i$ to the **other points in its own cluster** (cohesion).
2. $b(i)$ = mean distance from point $i$ to the points of the **nearest other cluster** (separation). "Nearest" = the cluster with the smallest mean distance.
3. Combine them:

$$
s(i) = \frac{b(i) - a(i)}{\max\big(a(i),\, b(i)\big)}
$$

4. The **silhouette score** is the mean of $s(i)$ over all points.

**Where it comes from.** $b - a$ is how much closer the point is to its own cluster. Dividing by the larger of the two scales it to the range −1 to +1, so scores are comparable across datasets.

- $s \approx 1$: the point is deep inside its cluster, far from the others.
- $s \approx 0$: the point sits on the border between two clusters.
- $s < 0$: the point is closer to another cluster. It is probably in the wrong one.

**Worked example.**

| Point | Cluster | $a(i)$ | $b(i)$ | $s(i) = (b-a)/\max(a,b)$ |
|---|---|---|---|---|
| P1 | A | 1.000 | 6.073 | 0.835 |
| P2 | A | 1.207 | 5.323 | 0.773 |
| P3 | A | 1.207 | 5.526 | 0.782 |
| P4 | B | 1.609 | 5.964 | 0.730 |
| P5 | B | 2.007 | 6.774 | 0.704 |
| P6 | B | 2.007 | 6.626 | 0.697 |
| P7 | B | 3.347 | 3.199 | **−0.044** |
| **Mean** | | | | **0.640** |

Step by step for P1: its neighbours in A are P2 and P3, each 1 away, so $a = (1 + 1)/2 = 1.000$. Its distances to cluster B are 6.403, 7.211, 7.071 and 3.606, so $b = 24.291 / 4 = 6.073$. Then $s = (6.073 - 1.000)/6.073 = 0.835$.

For P7: $a = 3.347$ is larger than $b = 3.199$, so P7 is slightly closer to cluster A on average and gets a negative score.

![Left: the seven points, with lines from P7 to every other point. Right: each point's silhouette; P7 is the only negative bar.](figures/unsupmetrics-silworked.svg)

In the figure, P7's dotted lines to cluster A are, on average, slightly shorter than its solid lines to cluster B. That's why its bar is below zero.

**Silhouette plot.** For real data, draw every point's $s(i)$, grouped by cluster and sorted. Good clusterings have thick bands of similar width, mostly above the mean line, and few negative values.

![Silhouette plots for k = 3, 4 and 5 on the four-blob data. k = 4 gives the highest mean and the most even bands.](figures/unsupmetrics-silplot.svg)

In the figure, at $k = 3$ one band is very wide: two true groups were merged. At $k = 5$ a true group was split, so two thin bands appear with low scores. $k = 4$ scores best (0.72).

### Davies–Bouldin index

For each cluster, find its "worst partner": the other cluster it is most similar to. Then average:

$$
\text{DB} = \frac{1}{k} \sum_{i=1}^{k} \max_{j \ne i} \frac{s_i + s_j}{d(\mu_i, \mu_j)}
$$

- $s_i$ = average distance from the points in cluster $i$ to its centroid (the cluster's spread).
- $d(\mu_i, \mu_j)$ = the distance between the two centroids.
- The ratio is large when two clusters are spread out **and** close together, which means they overlap.
- **Lower is better.** The minimum is 0.

**Worked example.** $s_A = 0.654$, $s_B = 1.344$, $d(\mu_A, \mu_B) = 5.584$. With only two clusters, each one's worst partner is the other, so $\text{DB} = (0.654 + 1.344)/5.584 = $ **0.358**.

### Calinski–Harabasz index

This is the ratio of **between-cluster** spread to **within-cluster** spread, each adjusted for how many clusters and points there are:

$$
\text{CH} = \frac{B / (k - 1)}{W / (n - k)}, \qquad B = \sum_{j=1}^{k} n_j \lVert \mu_j - \mu \rVert^2, \quad W = \text{WCSS}
$$

- $\mu$ = the overall mean of all points. $n_j$ = the number of points in cluster $j$.
- It has the same structure as the F-statistic in ANOVA: variation between groups over variation within groups.
- **Higher is better.** There is no upper limit, so compare it only across $k$ on the same data.

**Worked example.** $B = 53.45$, $W = 10.83$, $k = 2$, $n = 7$:

$$
\text{CH} = \frac{53.45 / 1}{10.83 / 5} = 24.67
$$

### Comparing k with all three

On the four-blob data (K-means):

| k | Silhouette (higher better) | Davies–Bouldin (lower better) | Calinski–Harabasz (higher better) |
|---|---|---|---|
| 2 | 0.480 | 0.971 | 194.9 |
| 3 | 0.591 | 0.612 | 305.0 |
| **4** | **0.720** | **0.373** | **840.3** |
| 5 | 0.632 | 0.656 | 712.4 |
| 6 | 0.539 | 0.818 | 642.0 |

All three agree on $k = 4$. On real data they often disagree, and that disagreement tells you the structure is not clear-cut.

> **Trap:** All of these internal metrics favour round, evenly sized, well-separated clusters, which is the shape K-means produces. They can score a correct DBSCAN result (say, two interlocking crescents) lower than a wrong K-means one. A high score means "compact and separated", not "correct".

## Clustering: external metrics (when some labels exist)

Sometimes you do have true groups for some or all of the data, for example to check a method on a benchmark. Cluster numbers are arbitrary (calling a cluster "0" or "1" means nothing), so you can't use accuracy directly. Use metrics that ignore the names:

- **Adjusted Rand Index (ARI)**: looks at every **pair** of points and asks whether the clustering and the true labels agree on "same group" or "different group". It is adjusted so that random labelling scores about 0. The range is about −0.5 to 1; 1 = perfect agreement. Example: true `[0,0,0,1,1,1]` vs predicted `[1,1,1,0,0,0]` gives ARI = 1.0, because only the names are swapped. Predicted `[0,0,1,1,1,1]` (one point wrong) gives 0.32.
- **Normalised Mutual Information (NMI)**: how much knowing the cluster tells you about the true label (from information theory, related to the [entropy](#/t/trees) used in decision trees), scaled to 0–1. The one-point-wrong example scores 0.48.

## Dimensionality reduction metrics

### Explained variance ratio

For PCA, each component $j$ captures variance $\lambda_j$ (its eigenvalue). The **explained variance ratio** is its share of the total:

$$
\text{EVR}_j = \frac{\lambda_j}{\sum_{m=1}^{p} \lambda_m}
$$

The **cumulative** EVR of the first $m$ components tells you how much of the information you keep if you keep only those. A common rule is to keep enough components for 90–95%.

![Scree plot for 8 standardised features built from 3 hidden factors. The first three bars are tall, then they drop sharply; the cumulative line passes 90% at component 3.](figures/unsupmetrics-scree.svg)

In the figure, components 1–3 explain 44%, 28% and 22% (cumulative 93.5%), and component 4 adds only 3%. The sharp drop after the third bar (the scree "elbow") says 3 components are enough. That fits how the data was built, from 3 hidden factors plus noise.

### Reconstruction error

Compress the data to $m$ components, map it back to the original features ($\hat{x}_i$), and measure what was lost:

$$
\text{Reconstruction error} = \frac{1}{n} \sum_{i=1}^{n} \lVert x_i - \hat{x}_i \rVert^2
$$

- **Lower is better.** For PCA on standardised data, the error as a fraction of total variance is exactly $1 -$ cumulative EVR. In the example: 0.56 with 1 component, 0.28 with 2, 0.065 with 3.
- It also works for methods without eigenvalues, such as autoencoders.
- Points with an unusually high reconstruction error don't fit the main pattern, which makes it an anomaly score too.

## Anomaly detection metrics

### Why it's hard

- Anomalies are **rare** and usually **unlabelled**. You often don't know how many there are, or what they look like.
- Most methods output a **score**, not a yes/no answer. Where you draw the line is a decision you make, not something the model learns.
- "Unusual" is not the same as "interesting". A model can rightly flag a half-hour as unusual because of a data error, not a real event.

### Contamination rate

The **contamination rate** is the fraction of the data you expect (or choose) to be anomalous. In sklearn (`IsolationForest(contamination=0.02)`) it sets the threshold so that exactly that fraction is flagged. It is an **assumption you supply**, not something measured, so try several values and report it.

### Score distributions and thresholds

Plot a histogram of the anomaly scores. Most points form a bulk; anomalies sit in a long tail. Ways to set the threshold:

- a **percentile** (flag the top 1% or 2%)
- a **statistical rule** (scores more than 3 robust standard deviations above the median, or above $Q_3 + 1.5 \times \text{IQR}$; see [Quartiles](#/t/quartiles))
- a **capacity rule** (we can investigate 20 cases a week, so flag the top 20)

![Isolation Forest scores on 1,000 synthetic points. Normal points form the bulk, the 12 true anomalies sit in the right tail, and the dashed line marks the top 2%.](figures/unsupmetrics-anomscores.svg)

In the figure, the wine bars are true anomalies, which in real life you wouldn't see. The threshold at the 98th percentile catches all of them. It also flags some normal points from the edge of the bulk, because 2% of 1,000 is 20 flags for only 12 real anomalies.

### Precision@k and recall (when a few anomalies are labelled)

If someone has confirmed a handful of anomalies (from known events or a manual review), you can score the ranking:

$$
\text{precision@}k = \frac{\text{confirmed anomalies in the top } k}{k}
$$

$$
\text{recall@}k = \frac{\text{confirmed anomalies in the top } k}{\text{all confirmed anomalies}}
$$

In the figure's data, the top 20 contain all 12 anomalies, so precision@20 = 12/20 = **0.60** and recall@20 = 12/12 = **1.00**. Use PR-AUC over the whole ranking if you have enough labels ([Classification metrics](#/t/clsmetrics)).

### Stability checks

When there are no labels at all, check that the flags are **robust**:

- **Re-run with different random seeds** or on bootstrap samples. Do the same points stay in the top $k$? Measure the overlap, e.g. as the fraction of shared flags.
- **Vary the settings** (contamination, number of neighbours, window length). Real anomalies should survive small changes.
- **Compare methods.** Points flagged by both Isolation Forest and a residual-based method are more convincing.
- **Sense-check against the domain.** For the [GB demand project](#/p/elec): do flagged half-hours line up with bank holidays, cold snaps or big televised events?

## Code: sklearn.metrics

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             davies_bouldin_score, calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score)

# the seven worked-example points and their clusters
X = np.array([[1, 1], [2, 1], [1, 2], [6, 5], [7, 5], [6, 6], [4, 3]], float)
labels = np.array([0, 0, 0, 1, 1, 1, 1])

print(silhouette_samples(X, labels).round(3))       # s(i) for each point
print(silhouette_score(X, labels))                  # mean s(i) = 0.640
print(davies_bouldin_score(X, labels))              # 0.358, lower is better
print(calinski_harabasz_score(X, labels))           # 24.67, higher is better

# elbow + silhouette over k on your own scaled data X_big
# (K-means stores WCSS as .inertia_)
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_big)
    print(k, km.inertia_, silhouette_score(X_big, km.labels_))

# external metrics: only when true groups are known
print(adjusted_rand_score([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1]))          # 0.32
print(normalized_mutual_info_score([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1])) # 0.48

# PCA: explained variance and reconstruction error
pca = PCA(n_components=3).fit(Z)                    # Z: standardised data
print(pca.explained_variance_ratio_.cumsum())       # cumulative share kept
Z_hat = pca.inverse_transform(pca.transform(Z))     # compress, then expand back
print(((Z - Z_hat) ** 2).sum(axis=1).mean())        # reconstruction error

# anomaly scores + precision@k against a few confirmed anomalies
iso = IsolationForest(contamination=0.02, random_state=0).fit(D)
score = -iso.score_samples(D)                       # flip sign: higher = more unusual
top_k = np.argsort(-score)[:20]                     # the 20 most unusual points
print(np.isin(top_k, confirmed_idx).mean())         # precision@20
```

`X_big`, `Z`, `D` and `confirmed_idx` are placeholders for your own data.

> **Trap:** Distance-based metrics (and K-means itself) depend on feature scale. Standardise first ([Scaling & encoding](#/t/scaling)), or one large-unit feature will dominate every distance.

## Summary table

| Metric | Family | Formula (short) | Better when |
|---|---|---|---|
| Inertia / WCSS | Clustering, internal | $\sum_j \sum_{x \in C_j} \lVert x - \mu_j \rVert^2$ | Lower (but always falls with k: use the elbow) |
| Silhouette | Clustering, internal | mean of $(b - a)/\max(a, b)$ | Higher (range −1 to 1) |
| Davies–Bouldin | Clustering, internal | mean of $\max_j (s_i + s_j)/d(\mu_i, \mu_j)$ | Lower (min 0) |
| Calinski–Harabasz | Clustering, internal | $[B/(k-1)] / [W/(n-k)]$ | Higher |
| Adjusted Rand Index | Clustering, external | Pair agreement, chance-adjusted | Higher (1 = perfect, about 0 = random) |
| NMI | Clustering, external | Mutual information scaled to 0–1 | Higher |
| Explained variance ratio | Dimensionality reduction | $\lambda_j / \sum \lambda$ | Higher cumulative share with fewer components |
| Reconstruction error | Dimensionality reduction | $\frac{1}{n}\sum \lVert x - \hat{x} \rVert^2$ | Lower |
| Contamination | Anomaly detection | Assumed fraction flagged | (an assumption, not a score) |
| Precision@k | Anomaly detection | confirmed in top k / k | Higher |
| Recall@k | Anomaly detection | confirmed in top k / all confirmed | Higher |
| Stability (overlap) | Anomaly detection, clustering | Shared flags across reruns | Higher |

## Quick reference

| Term | Meaning |
|---|---|
| $a(i)$ | Mean distance to the other points in your own cluster |
| $b(i)$ | Mean distance to the points of the nearest other cluster |
| Elbow | The $k$ after which WCSS stops falling steeply |
| Scree plot | Explained variance per component, used to choose how many to keep |
| Contamination | Expected share of anomalies; sets the threshold in sklearn |
| Internal vs external | Uses only the data vs compares with known labels |
| Key warning | Supervised metrics need labels; unsupervised metrics measure structure, not correctness |
