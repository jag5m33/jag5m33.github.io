# Isolation forest

An isolation forest finds anomalies by cutting the data at random: odd points get separated from everything else in very few cuts, normal points need many.

## The idea

> **What you need to know**
> - anomalies are **"few and different"**, so they are **easy to isolate**
> - build many **random trees**: pick a random feature, pick a random cut between its min and max, repeat until each point is alone
> - **path length** = how many cuts it took to get a point on its own
> - **short path = odd**, long path = normal; average over many trees → the **anomaly score**
> - **unsupervised**: no labels needed
>
> **Skip:** the scoring formula that turns path length into a score; `score_samples()` does it

Think of the game "guess who". A normal person in the middle of the crowd needs lots of questions before you've narrowed it down to just them. Someone very different (the only person over 2 m tall) is found with one question.

## Link to my tree notes

It splits data like a [decision tree](#/t/trees), but with two big differences:

| | Decision tree | Isolation tree |
|---|---|---|
| Labels? | yes (supervised) | **no** (unsupervised) |
| How the split is chosen | best **information gain** (lowest entropy) | **random** feature, **random** cut |
| When it stops | node pure, max depth, or gain < threshold | point is alone (or a depth limit) |
| What we read off | the leaf's class | **how deep** the point ended up |

The "forest" part is the same idea as a [random forest](#/t/randomforest): one random tree is noisy, so build many (each on a random sample of rows, like [bagging](#/t/bagging)) and **average**.

## Worked example: one demand spike

Let's say we have 10 half-hours of demand **residuals** in MW: what's left after removing the normal daily, weekly and yearly pattern (see my project below). One of them is a spike.

```
-310   -180   -90   -40   0   30   70   120   210   2400
```

**Step 1: one random tree, by hand.** The range is −310 to 2,400. A random cut anywhere in the big empty gap between 210 and 2,400 puts the spike on its own straight away.

![Top: one random cut at 1,000 isolates the spike. Bottom: in the same tree, the value 0 needs 6 cuts before it is alone.](figures/isoforest-cuts.svg)

In the figure, the gold dashed lines are the random cuts, numbered in order. The spike (wine) is alone after cut 1. The value 0 sits in the crowd, so each cut only trims a bit off: it takes 6.

- The gap from 210 to 2,400 is 2,190 MW of a 2,710 MW range, so the **first** random cut isolates the spike **81%** of the time (2,190 ÷ 2,710).
- In my 100 random trees it happened in **74** of them (random, so close to 81% but not equal).

**Step 2: average over 100 trees.** I coded the random cuts myself in Python and repeated them 100 times:

![Average number of cuts to isolate each value over 100 random trees. The spike needs 1.3; values in the middle need about 5.5.](figures/isoforest-paths.svg)

In the figure, the bars are lowest at the edges and lowest of all for the spike. Values at the edges (−310, 210) are also a bit easier to isolate than the middle: being at the edge of a crowd is slightly "odd".

**Step 3: sklearn agrees.** With the code below, `predict` flags only 2,400, and `score_samples` gives the spike −0.80 and the value 0 −0.36. **Lower = more anomalous**, so the spike is the most anomalous point. The same is true with 5 different random seeds.

> **Say it like this:** "An isolation forest doesn't learn what normal looks like. It just asks how easy each point is to cut away from the rest. The easier, the more anomalous."

## What the score looks like in 2D

![206 points: a cloud of normal points plus 6 odd ones. Shading is the isolation forest score, darkest wine at the edges. Points outside the gold line are flagged.](figures/isoforest-2d.svg)

In the figure, the score is highest (teal) in the dense middle and drops towards the edges (wine). The gold line is the threshold set by `contamination=0.03`: points outside it are flagged. It catches all 6 odd points, plus 1 normal point from the edge of the cloud.

## In sklearn

```python
import numpy as np
from sklearn.ensemble import IsolationForest

vals = [-310, -180, -90, -40, 0, 30, 70, 120, 210, 2400]
X = np.array(vals).reshape(-1, 1)      # one column, 10 rows

model = IsolationForest(n_estimators=100,   # number of random trees
                        contamination=0.1,  # expect ~10% anomalies
                        random_state=0)     # same cuts every run
model.fit(X)                         # build the trees (no y needed)

print(model.predict(X))              # 1 = normal, -1 = anomaly
print(model.score_samples(X))        # lower = more anomalous
print(model.decision_function(X))    # below 0 = flagged
```

> **Trap:** with the default `contamination="auto"` on this tiny example, −310 and 210 are flagged too, as well as the spike. The threshold is a choice, not something the model knows. With `contamination=0.1` (1 in 10) only the spike is flagged.

| Parameter | What it does | Default |
|---|---|---|
| `n_estimators` | how many random trees; more = steadier scores | 100 |
| `max_samples` | rows used to build each tree (a random sample) | min(256, n rows) |
| `contamination` | the share of points you **expect** to be anomalies; it only sets the cut-off for `predict` | `"auto"` |
| `max_features` | features each tree can choose from | 1.0 (all) |
| `random_state` | fixes the randomness so results repeat | None |

## In my GB electricity project

In [GB Grid Anomalies](#/p/elec) I'm stripping the normal daily, weekly and yearly pattern out of half-hourly demand with MSTL (the cycles found in [Autocorrelation](#/t/autocorr)), then looking for anomalies in what's left: the **residuals**.

**Why residuals, not raw demand?** Raw demand swings by thousands of MW every day. On raw data, every evening peak and every cold winter would look "odd". The residual asks the real question: *is this unusual for this time of day, weekday and season?*

```python
from sklearn.ensemble import IsolationForest

# df: one row per half-hour
# "resid" = demand minus the MSTL daily/weekly/yearly pattern
model = IsolationForest(n_estimators=200, contamination=0.005,
                        random_state=0)
df["flag"] = model.fit_predict(df[["resid"]])     # -1 = anomaly
df["score"] = model.score_samples(df[["resid"]])  # lower = odder

top = df.sort_values("score").head(20)            # 20 oddest half-hours
print(top[["settlement_date", "resid", "score"]]) # what happened then?
```

The 0.5% contamination is a starting guess, not a fact; see below.

## How do I know it worked?

There are no labels, so there's nothing to count right or wrong ([Clustering & anomaly metrics](#/t/unsupmetrics)). Instead:

- **Check against known events:** do the flags line up with storms, cold snaps, bank holidays, big matches or TV events?
- **Look at them:** plot the day around each flag. Is it a real event or a data error?
- **Stability:** rerun with different `random_state` values. Are the same half-hours flagged?
- If a few anomalies are confirmed, use **precision@k**: of the top k flags, how many are real?

## Assumptions & caveats

- **Contamination is a guess.** It decides how many points get flagged; the model can't tell you the right value.
- **Remove patterns first.** It treats every row as separate and knows nothing about time. Seasonal and daily patterns must be removed first (hence MSTL).
- **Clustered anomalies hide each other (masking).** If many odd points sit together, they form their own little crowd and take more cuts to isolate.
- **Many irrelevant features hurt.** With lots of features, most random cuts are on useless ones, so the odd points take longer to stand out.
- **Scores are relative, not probabilities.** −0.80 means "odder than −0.36", not "80% likely to be an anomaly".
- **No "why".** It flags a point but doesn't explain what makes it odd; you investigate that yourself.

## Use cases

- **Fraud:** unusual card transactions.
- **Sensor faults:** a machine reading that jumps away from normal.
- **Grid demand spikes:** my electricity project.
- **Network intrusion:** odd traffic patterns.
- A quick first pass on any table to find rows worth checking (data errors too).

## Advantages / Disadvantages

| Advantages | Disadvantages |
|---|---|
| No labels needed | Threshold (contamination) is a guess |
| Fast: trees are random, nothing to optimise; each built on a small sample | Struggles when anomalies cluster together (masking) |
| Works on many features; no scaling needed | Weaker with many irrelevant features |
| Few settings; defaults work well | Scores aren't probabilities |
| Simple idea, easy to explain | Ignores time order; patterns must be removed first |
| | Doesn't say **why** a point is odd |

## Interview one-liners

- **How does it work?** Random trees with random cuts; anomalies are isolated in few cuts, so a short average path length means anomalous.
- **How is it different from a decision tree?** No labels and no information gain: the feature and the cut are both random. We read off depth, not a class.
- **What does `contamination` do?** Sets the share of points `predict` labels −1. It doesn't change the scores.
- **Why run it on residuals?** So it finds demand that's odd *for that time*, not just every daily peak.
- **How do you evaluate it without labels?** Match flags to known events, inspect them, check stability across seeds; precision@k if a few are confirmed.

## Quick reference

| Term | Meaning |
|---|---|
| Isolation | cutting at random until a point is alone |
| Path length | number of cuts needed to isolate a point |
| Anomaly score | average path length over many trees, scaled; short path = anomalous |
| `score_samples` | lower = more anomalous |
| `predict` | 1 = normal, −1 = anomaly |
| `contamination` | expected share of anomalies; sets the cut-off |
| `max_samples` | rows per tree (default min(256, n)) |
| Masking | a cluster of anomalies hides itself |
