# What unsupervised learning means

Unsupervised learning finds structure in data that has no labels: groups, directions of variation, or unusual points, without anyone saying what the right answer is.

## Definition

- **Label**: the answer column you want to predict (spam / not spam, a house price).
- **Supervised learning**: the data has inputs $x$ **and** labels $y$. The model learns to predict $y$ and is scored against the true labels. See [What supervised means](#/t/supintro).
- **Unsupervised learning**: the data has inputs $x$ **only**. There is no $y$. The algorithm looks for **structure**: which points sit close together, which directions the data spreads along, which points don't fit.

> **Say it like this:** "Supervised learning is told the answers and learns to reproduce them. Unsupervised learning is given only the data and has to describe it."

![Left: a supervised dataset with given labels and a learned boundary. Right: the same points with no labels, where the algorithm can only look for structure, such as unusual points.](figures/unsupintro-supvsunsup.svg)

In the figure, the colours on the left were given to us, so every prediction can be checked. On the right nothing is given; the circled points are "unusual" only because they sit far from the crowd.

## Supervised vs unsupervised at a glance

| | Supervised | Unsupervised |
|---|---|---|
| Training data | $x$ and $y$ | $x$ only |
| Goal | Predict $y$ for new $x$ | Describe structure in $x$ |
| Typical tasks | Regression, classification | Clustering, dimensionality reduction, anomaly detection |
| Right answer? | Yes, the label | No single right answer |
| Example from my work | Accident severity ([US Accidents](#/p/acc)) | Unusual demand half-hours ([GB Grid Anomalies](#/p/elec)) |

## The three main families

### Clustering: grouping similar points

**Clustering** splits the data into groups (**clusters**) so that points in the same cluster are close to each other and far from other clusters.

- **[K-means](#/t/kmeans)**: you choose $k$, the number of clusters. It places $k$ centres and keeps moving each one to the middle of its nearest points. Fast and simple; likes round clusters.
- **[Hierarchical clustering](#/t/hierarchical)**: builds a tree of merges (a **dendrogram**). You cut the tree where the groups look sensible, so you don't fix $k$ in advance.
- **[DBSCAN](#/t/dbscan)**: clusters are crowded regions separated by empty gaps. Finds clusters of any shape and labels lonely points as **noise**.

![The same 120 points twice. Left: as the algorithm receives them, unlabelled. Right: the three groups K-means finds.](figures/unsupintro-clusters.svg)

In the figure, the colours on the right are the algorithm's own invention. "Cluster 1" means nothing until a person looks at what its points have in common.

**Use case:** customer segmentation, i.e. group shoppers by spending pattern and give each group its own offers.

### Dimensionality reduction: fewer columns, most of the information

**Dimensionality reduction** replaces many features with a few new ones that keep most of the information. Useful for plotting (2 dimensions, not 50), speed, and removing noise.

- **[PCA](#/t/pca)** (principal component analysis) finds new axes, called **principal components**, along the directions where the data spreads most.

![PCA on two correlated features. The first principal component points along the long axis of the cloud and keeps 95% of the spread.](figures/unsupintro-pca.svg)

In the figure, the two features move together, so one arrow (PC1) describes almost all the spread. Keeping PC1 alone loses only 5%.

**Use case:** gene-expression data, with thousands of genes per sample. A PCA plot is often the first quality check, showing batch effects or odd samples.

### Anomaly detection: finding what doesn't fit

**Anomaly detection** finds points that are rare or unlike the rest. The model learns what "normal" looks like, gives each point an **anomaly score** (higher = more unusual) and flags the highest. Common methods: DBSCAN noise points, Isolation Forest, and big errors from a model of normal behaviour.

**Use case:** fraud, machine faults, and unusual demand on an electricity grid (below).

## Why evaluation is harder without labels

- **No right answer to check against.** Grouping customers by age or by spending could both be reasonable.
- **Algorithms always return something.** K-means gives you $k$ clusters even on random noise.
- **Scores measure shape, not usefulness.** A score can say clusters are tight and well separated, not that they mean anything for the business.

So unsupervised work is judged with its own scores (inertia, silhouette), checks that results are stable, and common sense about the domain. See [Clustering & anomaly metrics](#/t/unsupmetrics).

> **Trap:** accuracy, precision and recall need true labels. They don't apply to a clustering.

## My use case: anomalies in GB electricity demand

My [GB Grid Anomalies](#/p/elec) project (in progress) is an unsupervised problem:

- **Data:** NESO half-hourly national demand for Great Britain, 2018–2023, in MW: 105,168 half-hours.
- **Question:** which half-hours have demand that is unusually high or low **for that time of day, day of week and season**?
- **Why unsupervised:** nobody has labelled half-hours as "anomalous". The model must learn normal demand and flag what departs from it.
- **EDA so far:** a strong yearly cycle (high in winter, low in summer) and a slow downward drift. The model must allow for the cycle, or every winter looks like an anomaly. Clock-change days have 46 or 50 half-hours instead of 48.
- **Evaluation plan:** check flags against known events (holidays, extreme weather, big broadcasts) and check they are stable.

## Quick reference

| Term | Meaning |
|---|---|
| Unsupervised learning | Learning from $x$ only, no labels $y$ |
| Clustering | Grouping similar points (K-means, hierarchical, DBSCAN) |
| Dimensionality reduction | Fewer features that keep most of the information (PCA) |
| Principal component | A new axis along a direction of big spread |
| Anomaly detection | Scoring how unusual each point is and flagging the extremes |
| Anomaly score | A number per point: higher = more unusual |
| Evaluation | Internal scores, stability and domain checks, not accuracy |
