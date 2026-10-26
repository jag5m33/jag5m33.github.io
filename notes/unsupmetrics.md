# Unsupervised metrics

Without labels you can't count right and wrong answers, so clusterings are judged by how tight and well separated the groups are, and anomaly detection is judged with care and common sense.

## Different families, different metrics

- **Supervised** models have true labels, so their metrics compare predictions with answers: MAE and RMSE for [regression](#/t/regmetrics); precision, recall and F1 for [classification](#/t/clsmetrics).
- **Unsupervised** models have no labels (see [What unsupervised means](#/t/unsupintro)). Their metrics measure the **structure** the model found.

You can't compute accuracy for a K-means result, and a silhouette score means nothing for a regression.

## The worked example

Let's say we have seven points, already put into two clusters:

| Point | x | y | Cluster |
|---|---|---|---|
| P1 | 1 | 1 | A |
| P2 | 2 | 1 | A |
| P3 | 1 | 2 | A |
| P4 | 6 | 5 | B |
| P5 | 7 | 5 | B |
| P6 | 6 | 6 | B |
| P7 | 4 | 3 | B |

P7 sits between the two groups. The **centroid** of a cluster is its centre point (the average of its points): A's is (1.33, 1.33) and B's is (5.75, 4.75).

## 1. Inertia (WCSS) and the elbow

**Inertia**, also called **WCSS** (within-cluster sum of squares), measures how **tight** the clusters are:

- For each point, take its distance to its own cluster's centroid and square it.
- Add these up over all points.
- **Lower = tighter clusters.** K-means works by making this as small as possible.

**Example:** the squared distances are P1 0.22, P2 0.56, P3 0.56 (cluster A) and P4 0.13, P5 1.63, P6 1.63, P7 6.13 (cluster B). Total WCSS = **10.83**. P7 alone gives more than half of it, because it sits far from B's centre.

> **Trap:** WCSS **always** goes down when you add more clusters. If every point is its own cluster, WCSS = 0. So you can't choose $k$ by just picking the lowest WCSS.

### The elbow method

Plot WCSS against $k$ (the number of clusters) and look for the **elbow**: the point after which extra clusters barely help.

![WCSS against k for data with four true groups. It drops steeply up to k = 4 and then flattens.](figures/unsupmetrics-elbow.svg)

In the figure, going from 3 to 4 clusters cuts WCSS from 831 to 246, but going from 4 to 5 only takes it to 218. The bend at $k = 4$ is the elbow. On real data the elbow is often blurry, so check it with the silhouette score.

## 2. Silhouette score

The silhouette asks, for each point: **am I closer to my own cluster than to the next-nearest cluster?**

1. $a$ = the average distance from the point to the **other points in its own cluster**. Small is good.
2. $b$ = the average distance from the point to the points of the **nearest other cluster**. Big is good.
3. Combine them:

$$
s = \frac{b - a}{\max(a,\, b)}
$$

- $\max(a, b)$ is just the bigger of the two; dividing by it keeps $s$ between −1 and +1.
- $s$ close to **+1**: the point sits deep inside its own cluster.
- $s$ close to **0**: the point is on the border between two clusters.
- $s$ **below 0**: the point is closer to another cluster, so it's probably in the wrong one.

The **silhouette score** of the whole clustering is the average $s$ over all points. Higher is better.

**Example, P1 by hand:**

- Its cluster-mates P2 and P3 are each 1 away, so $a = (1 + 1)/2 = 1.000$.
- Its distances to cluster B's points are 6.403, 7.211, 7.071 and 3.606, so $b = 6.073$.
- $s = (6.073 - 1.000)/6.073 = $ **0.835**. P1 is comfortably in the right cluster.

Doing the same for every point gives scores between 0.70 and 0.84 for P1 to P6, but **−0.04 for P7**: it is slightly closer to cluster A on average. The mean silhouette is **0.64**.

![Left: the seven points, with lines from P7 to every other point. Right: each point's silhouette; P7 is the only negative bar.](figures/unsupmetrics-silworked.svg)

In the figure, P7's lines to cluster A are, on average, a little shorter than its lines to cluster B. That's why its bar dips below zero.

### Using the silhouette to choose k

![Silhouette plots for k = 3, 4 and 5 on the four-blob data. k = 4 gives the highest mean and the most even bands.](figures/unsupmetrics-silplot.svg)

Each band is one cluster's points, sorted by score. In the figure, at $k = 3$ one band is very wide (two real groups were merged). At $k = 5$ a real group was split, giving thin bands with low scores. $k = 4$ scores best (0.72) and agrees with the elbow.

> **Trap:** these scores favour round, evenly sized, well-separated clusters, the shape K-means makes. A high score means "tight and separated", not "correct" or "useful".

## 3. Why anomaly detection is hard to evaluate

- **Anomalies are rare and usually unlabelled.** You often don't know how many there are or what they look like.
- **Models give a score, not a yes/no.** Where you draw the line (top 1%? top 20 cases?) is your decision. In sklearn, `contamination` is that guess: the share of points you expect to be anomalies.
- **Unusual isn't always interesting.** A flagged point might be a data error, not a real event.

![Isolation Forest scores on 1,000 synthetic points. Normal points form the bulk, the 12 true anomalies sit in the right tail, and the dashed line marks the top 2%.](figures/unsupmetrics-anomscores.svg)

In the figure, the true anomalies (wine) sit in the right tail. In real life you wouldn't know which those are. The 2% cut catches all 12, plus some normal points from the edge of the bulk.

So in practice you check that the flags are **stable** (do the same points get flagged with different random seeds or settings?) and that they make **sense** (for the [GB demand project](#/p/elec): do they line up with bank holidays, cold snaps or big TV events?). If someone has confirmed a few real anomalies, you can also count how many land in your top flags.

## 4. Code

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# the seven worked-example points and their clusters
X = np.array([[1, 1], [2, 1], [1, 2], [6, 5], [7, 5], [6, 6], [4, 3]], float)
labels = np.array([0, 0, 0, 1, 1, 1, 1])
print(silhouette_samples(X, labels).round(3))   # one s per point; P7 is -0.044
print(silhouette_score(X, labels))              # the average: 0.640

# elbow + silhouette for several k on your own (scaled) data X_big
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_big)
    print(k, km.inertia_, silhouette_score(X_big, km.labels_))   # inertia_ = WCSS
```

> **Trap:** distances depend on the units of each feature. Standardise first ([Scaling & encoding](#/t/scaling)), or one large-unit feature will dominate every distance.

## Quick reference

| Term | Meaning | Better |
|---|---|---|
| Centroid | The centre (average) of a cluster | |
| Inertia / WCSS | Total squared distance from each point to its centroid | Lower, but always falls with k |
| Elbow | The $k$ after which WCSS stops falling steeply | |
| $a$ | Average distance to the other points in your own cluster | Small |
| $b$ | Average distance to the nearest other cluster | Big |
| Silhouette | $(b - a)/\max(a, b)$, averaged over points; −1 to +1 | Higher |
| Contamination | Your guess of the share of anomalies; sets the cut-off | (an assumption) |
| Anomaly checks | Stability across reruns, sense against known events | |
