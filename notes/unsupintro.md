# What unsupervised learning means

Unsupervised learning finds structure in data that has no labels: groups, directions of variation, or unusual points, without anyone saying what the right answer is.

## Definition

- **Label**: the answer column you want to predict (spam / not spam, a house price, a severity class).
- **Supervised learning**: the training data has inputs $x$ **and** labels $y$. The model learns a mapping $x \to y$ and is scored by comparing its predictions with the true labels. See [What supervised means](#/t/supintro).
- **Unsupervised learning**: the training data has inputs $x$ **only**. There is no $y$. The algorithm looks for **structure** in $x$: which points sit close together, which directions the data varies along, which points don't fit the pattern.

> **Say it like this:** "Supervised learning is told the answers and learns to reproduce them. Unsupervised learning is given only the data and has to describe it."

![Left: a supervised dataset with given labels and a learned boundary. Right: the same points with no labels, where the algorithm can only look for structure, such as unusual points.](figures/unsupintro-supvsunsup.svg)

In the figure, the left panel's colours were given to us, so we can check every prediction. In the right panel, nothing is given. The circled points are "unusual" only because they sit far from the crowd.

![The same 120 points twice. Left: as the algorithm receives them, unlabelled. Right: the three groups K-means finds.](figures/unsupintro-clusters.svg)

In the figure, the colours on the right are the algorithm's own invention. "Cluster 1" means nothing until a person looks at what its points have in common.

## Supervised vs unsupervised at a glance

| | Supervised | Unsupervised |
|---|---|---|
| Training data | $x$ and $y$ | $x$ only |
| Goal | Predict $y$ for new $x$ | Describe structure in $x$ |
| Typical tasks | Regression, classification | Clustering, dimensionality reduction, anomaly detection |
| Is there a right answer? | Yes, the label | No single right answer |
| How it's scored | Compare predictions with labels (MAE, F1, PR-AUC...) | Internal measures of structure, stability, and human judgement |
| Example from my work | Predicting accident severity ([US Accidents](#/p/acc)) | Finding unusual demand half-hours ([GB Grid Anomalies](#/p/elec)) |

## The three main families

Each family gets its own chapter. Those chapters are coming soon.

### Clustering: grouping similar points

**Clustering** splits the data into groups (**clusters**), so that points in the same cluster are similar to each other and different from points in other clusters. "Similar" usually means a short distance in feature space.

- **[K-means](#/t/kmeans)**: you choose $k$, the number of clusters. It places $k$ centres and repeatedly moves each one to the mean of the points closest to it. Fast and simple, but it assumes roughly round clusters of similar size.
- **[Hierarchical clustering](#/t/hierarchical)**: builds a tree of merges (a **dendrogram**), from every point alone up to one big group. You cut the tree at the height that gives sensible groups, so you don't need to fix $k$ in advance.
- **[DBSCAN](#/t/dbscan)**: clusters are dense regions separated by sparse gaps. It finds clusters of any shape and labels isolated points as **noise**, which also makes it useful for spotting anomalies.

**Use case:** customer segmentation. Group shoppers by spending pattern, then give each segment its own offers.

### Dimensionality reduction: fewer columns, most of the information

**Dimensionality reduction** replaces many features with a few new ones that keep most of the information. This helps with visualising (you can plot 2 dimensions, not 50), speeding up models, and removing noise and redundancy.

- **[PCA](#/t/pca)** (principal component analysis) finds new axes, the **principal components**, along the directions of greatest spread (variance). The first component keeps the most variance, the second the most of what's left, and so on. Each component is uncorrelated with the others.

![PCA on two correlated features. The first principal component points along the long axis of the cloud and keeps 95% of the spread.](figures/unsupintro-pca.svg)

In the figure, the two features move together, so one arrow (PC1) describes almost all of the variation. You could keep PC1 alone and lose only 5% of the variance.

**Use case:** gene-expression data in bioinformatics, where thousands of genes are measured per sample. A PCA plot of the first two components is often the first quality check, showing batch effects or sample outliers.

### Anomaly detection: finding what doesn't fit

**Anomaly detection** (outlier detection) finds points that are rare or unlike the rest. The model learns what "normal" looks like, gives each point an **anomaly score**, and flags the highest scores. Common methods: distance or density based (DBSCAN noise, Local Outlier Factor), Isolation Forest (anomalies are easier to isolate with random splits), and residuals from a model of normal behaviour.

**Use case:** fraud detection, machine faults, network intrusions, and unusual demand on an electricity grid (below).

## Why evaluation is harder without labels

With labels, you can count right and wrong answers. Without them:

- **There's no ground truth.** Two clusterings of the same data can both be reasonable, for example grouping customers by age or by spending.
- **Algorithms always return an answer.** K-means will give you $k$ clusters even for random noise.
- **Scores measure shape, not usefulness.** A score can tell you that clusters are compact and well separated. It can't tell you that they mean anything for the business question.
- **Choices like $k$, the distance measure and scaling change the result**, and there is no label to settle which is right.

So unsupervised work is judged with **different metrics** from supervised work: internal scores (inertia, silhouette...), stability checks, a few labelled examples when they exist, and domain sense. These are covered in [Unsupervised metrics](#/t/unsupmetrics).

> **Trap:** Accuracy, precision and recall need true labels. They don't apply to a clustering unless you have external labels to compare against, and even then you need label-free versions like the Adjusted Rand Index (clusters have no fixed names).

## My use case: anomalies in GB electricity demand

My [GB Grid Anomalies](#/p/elec) project (in progress) is an unsupervised problem:

- **Data:** NESO half-hourly national demand (ND) for Great Britain, 2018–2023, in MW. 105,168 half-hours.
- **Question:** which half-hours (and days) have demand that is unusually high or low **for that time of day, day of week and season**?
- **Why unsupervised:** nobody has labelled half-hours as "anomalous". The model has to learn what normal demand looks like and flag what departs from it.
- **What the EDA showed so far:** demand has a strong yearly cycle (high in winter, low in summer) and drifts down slowly over the six years. The model must allow for that cycle, or every winter will look like an anomaly. Clock-change days have 46 or 50 half-hours rather than 48, so timestamps need care.
- **Evaluation plan:** with no labels, results will be checked against known events (holidays, extreme weather, major broadcasts), using score distributions and stability checks, as described in [Unsupervised metrics](#/t/unsupmetrics).

## Quick reference

| Term | Meaning |
|---|---|
| Unsupervised learning | Learning from $x$ only, with no labels $y$ |
| Clustering | Grouping similar points (K-means, hierarchical, DBSCAN) |
| Cluster | A group of points the algorithm puts together |
| Dimensionality reduction | Fewer features that keep most of the information (PCA) |
| Principal component | A new axis along a direction of maximum remaining variance |
| Anomaly detection | Scoring how unusual each point is and flagging the extremes |
| Anomaly score | A number per point: higher = more unusual |
| Evaluation | Internal scores, stability and domain checks rather than accuracy |
