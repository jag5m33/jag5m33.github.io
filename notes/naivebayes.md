# Naive Bayes

Naive Bayes classifies by asking which class would most likely have produced the features it sees, using Bayes' theorem and the simplifying ("naive") assumption that the features are independent once you know the class.

## Before you start

**Terms used below**

- **Probability** $P(A)$: how likely event $A$ is, from 0 to 1.
- **Conditional probability** $P(A \mid B)$: how likely $A$ is **given that** $B$ has happened. Read the bar "$\mid$" as "given".
- **Joint probability** $P(A, B)$: how likely $A$ and $B$ both happen.
- **Class** $y$: the label to predict (spam / not spam; one of 100 clause types).
- **Features** $x_1, \dots, x_n$: what we observe (words, counts, measurements).

Text examples use [Bag of Words](#/t/bow) counts and [TF-IDF](#/t/tfidf) scores; the log trick uses [Logs & exponents](#/t/logs).

## Bayes' theorem

### Where it comes from

Start from the definition of conditional probability. The chance of $A$ and $B$ together can be written two ways:

$$
P(A, B) = P(A \mid B)\,P(B) = P(B \mid A)\,P(A)
$$

Both are the same joint probability, so set them equal and divide by $P(B)$:

$$
P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}
$$

That's all Bayes' theorem is: a way to **flip** a conditional probability. You often know $P(\text{symptom} \mid \text{disease})$, but you want $P(\text{disease} \mid \text{symptom})$.

### Each term in plain words

For classification, $A$ is the class $y$ and $B$ is the data $x$:

$$
\underbrace{P(y \mid x)}_{\text{posterior}} = \frac{\overbrace{P(x \mid y)}^{\text{likelihood}}\;\overbrace{P(y)}^{\text{prior}}}{\underbrace{P(x)}_{\text{evidence}}}
$$

| Term | Name | Plain meaning |
|---|---|---|
| $P(y)$ | **Prior** | How common the class is before looking at this example |
| $P(x \mid y)$ | **Likelihood** | If the example were of class $y$, how likely would these features be? |
| $P(x)$ | **Evidence** | How likely these features are overall, across all classes. It's the same for every class, so it just rescales |
| $P(y \mid x)$ | **Posterior** | How likely the class is **after** seeing the features. This is what we want |

The evidence is the sum of "likelihood × prior" over all classes: $P(x) = \sum_k P(x \mid y_k)\,P(y_k)$. That's why posteriors always add up to 1.

### Worked example: a medical test

A disease affects 1% of people. A test catches 90% of people who have it (**sensitivity**) and wrongly flags 5% of healthy people (**false positive rate**). You test positive. How likely is it that you have the disease?

| Step | Quantity | Calculation | Value |
|---|---|---|---|
| 1 | Prior $P(D)$ | given | 0.01 |
| 2 | Likelihood $P(+ \mid D)$ | given | 0.90 |
| 3 | $P(+ \mid \text{healthy})$ | given | 0.05 |
| 4 | Evidence $P(+)$ | $0.90 \times 0.01 + 0.05 \times 0.99$ | 0.0585 |
| 5 | Posterior $P(D \mid +)$ | $0.009 / 0.0585$ | **0.154** |

![The same calculation as counts. Of 10,000 people, 100 are ill and 90 of them test positive; of 9,900 healthy people, 495 test positive. Only 90 of the 585 positives are really ill.](figures/naivebayes-medical.svg)

In the figure, follow the two wine "test +" boxes: 90 true positives and 495 false positives. The healthy group is so much bigger that its 5% error rate swamps the 90 real cases. A positive test raises your chance from 1% to 15.4%, but it's still far more likely you're healthy.

> **Say it like this:** "The posterior is the prior updated by the evidence. A rare class needs very strong evidence before it becomes the most likely answer."

## From Bayes to a classifier

To classify, compute the posterior for every class and pick the biggest. The evidence $P(x)$ is the same for every class, so for **choosing** the class you can drop it:

$$
\hat{y} = \arg\max_{y}\; P(y)\,P(x_1, x_2, \dots, x_n \mid y)
$$

($\arg\max_y$ = "the $y$ that gives the largest value".)

### The problem

$P(x_1, \dots, x_n \mid y)$ is the chance of seeing this **exact combination** of features in class $y$. With 213,105 possible n-grams in a clause, almost every combination is unique, so you can't estimate it by counting.

### The "naive" assumption

Assume the features are **independent of each other once you know the class** (conditional independence). Independent probabilities multiply, so the joint likelihood becomes a product of one-feature likelihoods:

$$
P(x_1, \dots, x_n \mid y) = P(x_1 \mid y)\,P(x_2 \mid y)\cdots P(x_n \mid y) = \prod_{j=1}^{n} P(x_j \mid y)
$$

Each $P(x_j \mid y)$ is easy to estimate: count how often feature $j$ shows up in class $y$. So the classifier is:

$$
\hat{y} = \arg\max_{y}\; P(y)\prod_{j=1}^{n} P(x_j \mid y)
$$

### Why it's usually false, and why it still works

It's almost never true. In legal text, "governing" and "law" appear together far more often than chance; knowing one tells you a lot about the other. So the products, and the probabilities they give, are distorted.

But to **classify**, the probabilities don't have to be right. They only have to put the correct class **on top**. The distortion often affects every class in a similar way, so the ranking survives even when the numbers are off. That's why Naive Bayes classifies text well while producing poor probability estimates (see the caveats).

## The four sklearn variants

They differ only in **how $P(x_j \mid y)$ is modelled**, which depends on what kind of feature $x_j$ is.

| Variant | Feature type | $P(x_j \mid y)$ modelled as | Typical use |
|---|---|---|---|
| `GaussianNB` | Continuous numbers | A normal (bell) curve per feature per class, with that class's mean and variance | Measurements: lab values, sensor readings |
| `MultinomialNB` | Counts (or TF-IDF weights) | Share of the class's total word count that goes to word $j$ | Text classification with BoW / TF-IDF |
| `BernoulliNB` | Binary: present / absent | Fraction of the class's documents that contain word $j$ | Short texts; also uses **absence** of words as evidence |
| `ComplementNB` | Counts (or TF-IDF weights) | Like Multinomial, but estimated from the **complement** of each class (all other classes) | Text with **imbalanced** classes |

For `GaussianNB`, the likelihood of a value $x_j$ is the normal density:

$$
P(x_j \mid y) = \frac{1}{\sqrt{2\pi\sigma_{y}^2}}\exp\left(-\frac{(x_j - \mu_{y})^2}{2\sigma_{y}^2}\right)
$$

where $\mu_y$ and $\sigma_y^2$ are the mean and variance of feature $j$ among training examples of class $y$.

## Worked text example by hand (MultinomialNB)

### The corpus

Four short, already lemmatised clauses with stop words removed (in the style of my [Bag of Words](#/t/bow) sentences), two classes:

| Doc | Text | Class |
|---|---|---|
| D1 | pay fee invoice | Payment |
| D2 | pay fee notice | Payment |
| D3 | terminate notice | Termination |
| D4 | terminate agreement notice | Termination |

Vocabulary ($V = 6$ words): agreement, fee, invoice, notice, pay, terminate.

**New clause to classify:** "terminate fee notice".

### Step 1: priors

$$
P(\text{Payment}) = \frac{2}{4} = 0.5 \qquad P(\text{Termination}) = \frac{2}{4} = 0.5
$$

### Step 2: word counts per class

| Word | Count in Payment | Count in Termination |
|---|---|---|
| agreement | 0 | 1 |
| fee | 2 | 0 |
| invoice | 1 | 0 |
| notice | 1 | 2 |
| pay | 2 | 0 |
| terminate | 0 | 2 |
| **Total words** | **6** | **5** |

### Step 3: the zero-frequency problem

Without smoothing, $P(\text{word} \mid \text{class}) = \frac{\text{count}}{\text{total}}$. Then:

- $P(\text{terminate} \mid \text{Payment}) = 0/6 = 0$, so the whole Payment product is 0.
- $P(\text{fee} \mid \text{Termination}) = 0/5 = 0$, so the whole Termination product is 0.

Both classes score exactly 0, and the model can't decide. One unseen word wipes out all the other evidence, because anything times 0 is 0. This is the **zero-frequency problem**.

### Step 4: Laplace smoothing ($\alpha = 1$)

Add $\alpha$ to every count, and $\alpha V$ to every total so the probabilities still sum to 1:

$$
P(w \mid y) = \frac{\text{count}(w, y) + \alpha}{\text{total words in } y + \alpha V}
$$

- $\alpha$ (alpha) is a **pseudo-count**: we pretend every word was seen $\alpha$ extra times in every class. $\alpha = 1$ is **Laplace** smoothing; $0 < \alpha < 1$ is called **Lidstone** smoothing.
- Larger $\alpha$ pulls all word probabilities towards equal (more smoothing, less trust in the counts). Smaller $\alpha$ trusts the counts more. It's a hyperparameter to tune on validation data.

Payment denominator: $6 + 1 \times 6 = 12$. Termination denominator: $5 + 6 = 11$.

| Word | $P(w \mid \text{Payment})$ | $P(w \mid \text{Termination})$ |
|---|---|---|
| agreement | (0+1)/12 = 0.0833 | (1+1)/11 = 0.1818 |
| fee | (2+1)/12 = 0.2500 | (0+1)/11 = 0.0909 |
| invoice | (1+1)/12 = 0.1667 | (0+1)/11 = 0.0909 |
| notice | (1+1)/12 = 0.1667 | (2+1)/11 = 0.2727 |
| pay | (2+1)/12 = 0.2500 | (0+1)/11 = 0.0909 |
| terminate | (0+1)/12 = 0.0833 | (2+1)/11 = 0.2727 |

![Smoothed word probabilities for each class. The three words of the new clause are in bold. "terminate" and "notice" favour Termination; "fee" favours Payment.](figures/naivebayes-likelihoods.svg)

In the figure, no bar is zero any more: smoothing gave unseen words a small probability. Compare the bars for the three bold words: two of them lean towards Termination.

### Step 5: score each class for "terminate fee notice"

Multiply the prior by the likelihood of each word in the clause:

| Class | Prior × terminate × fee × notice | Product |
|---|---|---|
| Payment | $0.5 \times \frac{1}{12} \times \frac{3}{12} \times \frac{2}{12}$ | $\frac{1}{576} = 0.001736$ |
| Termination | $0.5 \times \frac{3}{11} \times \frac{1}{11} \times \frac{3}{11}$ | $\frac{9}{2662} = 0.003381$ |

### Step 6: normalise to get posteriors

The evidence is the sum: $0.001736 + 0.003381 = 0.005117$.

$$
P(\text{Payment} \mid x) = \frac{0.001736}{0.005117} = 0.339 \qquad P(\text{Termination} \mid x) = \frac{0.003381}{0.005117} = 0.661
$$

**Prediction: Termination.** "terminate" is 3.27 times as likely under Termination and "notice" 1.64 times; "fee" is 2.75 times as likely under Payment. Multiplying the ratios, $3.27 \times 1.64 / 2.75 = 1.95$: Termination's score is about twice Payment's ($0.003381 / 0.001736 = 1.95$).

### Step 7: the same with logs

Real documents have hundreds of words. Multiplying hundreds of small probabilities produces numbers too small for the computer to store. They **underflow** to exactly 0.0:

![Left: the product of n probabilities of 0.001 falls by a factor of 1,000 per word and becomes exactly 0.0 at n = 108. Right: the sum of their logs just keeps going down in a straight line.](figures/naivebayes-underflow.svg)

In the figure, the left curve stops at 108 words, where a 64-bit float can't go any smaller. The right line is still a normal number at 200 words (−1381.6).

The fix: take logs. The log of a product is the sum of the logs, and log keeps the order (bigger in, bigger out), so the winning class doesn't change:

$$
\log\Big(P(y)\prod_j P(x_j \mid y)\Big) = \log P(y) + \sum_j \log P(x_j \mid y)
$$

| | $\ln$ prior | $\ln$ P(terminate) | $\ln$ P(fee) | $\ln$ P(notice) | **Sum** |
|---|---|---|---|---|---|
| Payment | −0.6931 | −2.4849 | −1.3863 | −1.7918 | **−6.3561** |
| Termination | −0.6931 | −1.2993 | −2.3979 | −1.2993 | **−5.6896** |

−5.6896 is larger (less negative), so **Termination** wins again. To turn log-scores back into probabilities, use [softmax](#/t/softmax): $P(\text{Termination} \mid x) = \frac{e^{-5.6896}}{e^{-6.3561} + e^{-5.6896}} = 0.661$, the same as step 6.

When a word appears $c$ times, its log-probability is added $c$ times: the general score is $\log P(y) + \sum_j x_j \log P(w_j \mid y)$, where $x_j$ is the word's count (or its TF-IDF weight).

### Check with sklearn

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB

docs = ["pay fee invoice", "pay fee notice",
        "terminate notice", "terminate agreement notice"]
y = ["Payment", "Payment", "Termination", "Termination"]

cv = CountVectorizer()               # Bag of Words counts
X = cv.fit_transform(docs)           # 4 docs x 6 words
x_new = cv.transform(["terminate fee notice"])

mnb = MultinomialNB(alpha=1.0)       # Laplace smoothing
mnb.fit(X, y)
print(cv.get_feature_names_out())
print(np.exp(mnb.feature_log_prob_).round(4))       # P(word | class)
print(mnb.predict_joint_log_proba(x_new).round(4))  # log prior + sum of logs
print(mnb.predict_proba(x_new).round(4), mnb.predict(x_new))

cnb = ComplementNB(alpha=1.0)        # same data, complement version
cnb.fit(X, y)
print(cnb.feature_log_prob_.round(4))  # weights = -log P(word | other class)
print((x_new @ cnb.feature_log_prob_.T).round(4), cnb.predict(x_new))
```

```text
['agreement' 'fee' 'invoice' 'notice' 'pay' 'terminate']
[[0.0833 0.25   0.1667 0.1667 0.25   0.0833]
 [0.1818 0.0909 0.0909 0.2727 0.0909 0.2727]]
[[-6.3561 -5.6896]]
[[0.3393 0.6607]] ['Termination']
[[1.7047 2.3979 2.3979 1.2993 2.3979 1.2993]
 [2.4849 1.3863 1.7918 1.7918 1.3863 2.4849]]
[[4.9965 5.663 ]] ['Termination']
```

Every hand number matches: the likelihood table, the log sums (−6.3561, −5.6896) and the posteriors (0.339, 0.661).

## ComplementNB

### How it works

MultinomialNB learns each class's word probabilities **from that class's own documents**. ComplementNB learns them **from all the other classes' documents** (the class's **complement**), then asks: "which class's complement does this document fit **worst**?"

1. For class $y$, count words in every training document **not** in $y$. Smooth with $\alpha$ as before.
2. Weight for word $w$: $-\log P(w \mid \text{not } y)$. A word that's rare outside $y$ gets a big weight.
3. Score: $\text{score}(y) = \sum_j x_j \times \text{weight}_{y,j}$. Pick the class with the **highest** score.

**By hand on the toy corpus.** With only two classes, "not Payment" is Termination, so Payment's weights are minus the logs of Termination's smoothed probabilities:

- Payment: $-(\ln\tfrac{3}{11} + \ln\tfrac{1}{11} + \ln\tfrac{3}{11}) = 1.2993 + 2.3979 + 1.2993 = 4.9965$
- Termination: $-(\ln\tfrac{1}{12} + \ln\tfrac{3}{12} + \ln\tfrac{2}{12}) = 2.4849 + 1.3863 + 1.7918 = 5.6630$

Termination's complement (the Payment documents) explains "terminate fee notice" worst, so **Termination** wins, as sklearn printed above. With two classes and equal priors, ComplementNB and MultinomialNB always agree. They part ways when priors differ and, above all, with three or more classes, especially imbalanced ones.

### Why the complement helps with imbalance

A small class has very few words to learn from, so its MultinomialNB word estimates are noisy and dominated by the smoothing. Its complement contains almost the whole training set, so its estimates are stable. Every class's weights are then built from similar, large amounts of data, which removes much of MultinomialNB's pull towards the big classes. Rennie et al. (2003), who proposed it, also found it more accurate than MultinomialNB on standard text tasks.

![For LEDGAR's largest class (Governing Laws, 3,167 training clauses) and smallest (Books, 23), the number of clauses each method learns from. The complement of Books still has 59,977 clauses.](figures/naivebayes-complement.svg)

In the figure, the gold bars are what MultinomialNB has to go on; the teal bars are what ComplementNB uses. For Books, that is 23 clauses against nearly 60,000. (Counts from my EDA notebook, taken on the 60,000 training rows before de-duplication.)

### The two settings in sklearn

- `alpha` (default 1.0): Laplace/Lidstone smoothing on the complement counts, exactly as above.
- `norm` (default `False`): when `True`, each class's weight vector is divided by the sum of its weights, so they're all on the same scale. This is the "weight normalisation" step from the paper. It stops classes whose complements produce larger weights from winning just for that reason. sklearn leaves it off by default, matching the Mahout and Weka implementations.

> **Trap:** ComplementNB **ignores the class priors** in prediction (sklearn: `class_prior` is "not used", `fit_prior` only matters if training data has a single class). Its `predict_proba` is a softmax of the scores, so treat it as a ranking, not a real probability.

## In my LEDGAR project

In [LEDGAR clause classification](#/p/legal) I've just added a ComplementNB model next to the softmax logistic regression. It uses the same saved TF-IDF features (59,933 × 213,105, one- to three-word n-grams), so the comparison is fair ([src/complement_nb.py](https://github.com/jag5m33/py_legal_data/blob/main/src/complement_nb.py)):

```python
class ComplementNaiveBayes:
    def __init__(self, x_train, x_val, x_test, y_train, y_val, y_test, alpha=1.0, norm=False):
        ...
        self.model = ComplementNB(alpha=alpha, norm=norm)

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def _score(self, x, y, name):
        preds = self.model.predict(x)
        # error = 1 - MACRO F1
        f1 = f1_score(y, preds, average = 'macro')
        ...
```

What the settings mean here:

- **`alpha=1.0`**: Laplace smoothing on the complement counts. Because each complement is nearly the whole corpus, zero counts are rarer than in MultinomialNB, but they still happen for n-grams found only in one class's clauses. Alpha is worth tuning on validation (e.g. 0.01, 0.1, 0.3, 1): TF-IDF values are small fractions, so a pseudo-count of 1 is large next to them and may over-smooth.
- **`norm=False`**: sklearn's default, no weight normalisation. Trying `norm=True` is a cheap experiment.
- **Priors**: the comment in my class mentions `class_prior` and `fit_prior`. For ComplementNB they don't change predictions (see the trap above); they do matter for MultinomialNB.
- **Inputs**: Naive Bayes needs **non-negative** features. TF-IDF scores are 0 to 1, so they work. My TF-IDF settings (`sublinear_tf=True`, IDF weighting, L2 row normalisation) are the same three transforms Rennie et al. recommend feeding into ComplementNB.
- **Scoring**: the class reports accuracy, macro F1, micro F1 and error = 1 − macro F1. With 100 imbalanced classes, macro F1 is the main score, because it counts every class equally ([Classification metrics](#/t/clsmetrics)).

**Results: coming soon.** The model is wired into `model_nb.ipynb` but hasn't been run and evaluated yet. When it is, the validation macro F1 goes in the table below next to logistic regression's (0.816).

### Why ComplementNB suits LEDGAR

- **100 classes, very imbalanced.** The biggest class (Governing Laws, 3,167 clauses) is about 138 times the smallest (Books, 23). ComplementNB was built for exactly this.
- **Sparse, high-dimensional text.** Each clause has ~101 non-zero n-grams out of 213,105. Naive Bayes only touches the non-zero cells, so it trains in one fast pass.
- **A strong, cheap baseline** to check whether logistic regression's extra cost is buying real accuracy.

### Naive Bayes vs my logistic regression

| | Naive Bayes (ComplementNB) | Softmax logistic regression |
|---|---|---|
| Type | **Generative**: models how each class produces words, $P(x \mid y)P(y)$, then flips with Bayes | **Discriminative**: models $P(y \mid x)$ directly, the boundary between classes |
| Training | One pass of counting, then logs. Seconds | Iterative optimisation of 100 × 213,105 weights. Minutes |
| Correlated n-grams | Assumes independence: "governing law" and "law" are counted as separate evidence | Weights adjust jointly, so overlapping n-grams share credit |
| Tuning | Mainly `alpha` (and `norm`) | `C`, `class_weight`, `max_iter` |
| Probabilities | Poorly calibrated: too extreme (and for ComplementNB, only a ranking) | Usually better calibrated |
| Little data | Often does well quickly | Needs more data to beat NB |
| Validation macro F1 | Coming soon | 0.816 |

A classic result (Ng and Jordan, 2001) is that generative Naive Bayes gets close to its best performance with far fewer training examples, while discriminative logistic regression usually reaches a lower error once there's plenty of data. With 60,000 clauses, I'd expect logistic regression to win on accuracy and Naive Bayes to win on speed, but the results will decide.

## Assumptions & caveats

- **Conditional independence.** Assumed, and almost never true. The classification usually survives; the probabilities don't.
- **Correlated features are double-counted.** If two features carry the same information, NB counts it twice. In the figure below, one feature's information is copied up to 5 times. A copy adds nothing new, yet `GaussianNB` treats each one as fresh evidence. Text has the same issue: n-grams overlap ("governing", "governing law", "governing law state").

![P(class 1) for the same point as a feature is duplicated: 0.686 with one copy, rising to 0.980 with five, although no new information was added.](figures/naivebayes-overconfident.svg)

In the figure, the dashed line is the honest answer (0.686). Each extra copy pushes the probability further towards 1.

- **Poor calibration.** Partly because of that double-counting, NB probabilities are often far too close to 0 or 1. Use them for ranking, or calibrate them (`CalibratedClassifierCV`) if you need real probabilities.
- **Zero frequency.** Without smoothing, one unseen word zeroes a class. Always use $\alpha > 0$.
- **GaussianNB assumes each feature is normal within each class.** Skewed or multi-peaked features fit badly; a log transform can help.
- **Non-negative inputs** for Multinomial/Complement NB (counts or TF-IDF, not standardised values).
- **Priors follow the training data.** MultinomialNB's priors come from training class shares. If real-world shares differ, set `class_prior` yourself.
- **No interactions.** NB can't learn "A and B together mean something that A or B alone doesn't" (an XOR-style pattern).

## Use cases

- **Text classification**: spam filtering, sentiment, topic and document-type classification (like LEDGAR clauses).
- A **fast baseline** for any text problem before logistic regression or transformers.
- **Very high-dimensional, sparse data**, where it trains in seconds and needs little memory.
- **Small datasets**, where its strong assumption acts like regularisation.
- **Streaming / incremental learning**: `partial_fit` updates the counts as new data arrives.

## Advantages / Disadvantages

| Advantages | Disadvantages |
|---|---|
| Extremely fast to train and predict (counting) | Independence assumption is unrealistic |
| Works with little data; few hyperparameters | Probabilities badly calibrated (too extreme) |
| Handles huge sparse feature spaces well | Correlated features get double-counted |
| Simple to explain: priors and word likelihoods | Can't model feature interactions |
| Supports `partial_fit` for streaming | Usually less accurate than tuned logistic regression on large data |
| ComplementNB copes well with class imbalance | GaussianNB assumes normality per class |

## Limitations

- **A ceiling on accuracy.** Because it can't learn how features combine, it usually trails discriminative models once data is plentiful.
- **Not for probability-sensitive decisions** without calibration (risk scores, expected costs).
- **Feature engineering matters.** Results depend heavily on the text representation (BoW vs TF-IDF, n-grams, stop words).
- **ComplementNB's probabilities aren't probabilities.** They're softmaxed scores, so they rank classes but shouldn't be read as confidence.

## Quick reference

| Term | Meaning / formula |
|---|---|
| Bayes' theorem | $P(y \mid x) = P(x \mid y)P(y) / P(x)$ |
| Prior / likelihood / evidence / posterior | $P(y)$ / $P(x \mid y)$ / $P(x)$ / $P(y \mid x)$ |
| Medical test example | prior 0.01, sensitivity 0.90, FPR 0.05 → posterior 0.154 |
| Naive assumption | Features independent given the class: $P(x \mid y) = \prod_j P(x_j \mid y)$ |
| Decision rule | $\arg\max_y \; \log P(y) + \sum_j x_j \log P(w_j \mid y)$ |
| Laplace smoothing | $(\text{count} + \alpha) / (\text{total} + \alpha V)$, $\alpha = 1$ |
| Why logs | Products of many small numbers underflow to 0 (0.001 to the 108th power is 0.0) |
| GaussianNB | Continuous features, normal per class |
| MultinomialNB | Word counts / TF-IDF |
| BernoulliNB | Word present / absent |
| ComplementNB | Weights from each class's complement; for imbalanced text; ignores priors |
| Toy example | "terminate fee notice": Payment 0.339, Termination 0.661 |
| vs logistic regression | Generative vs discriminative; faster, less calibrated, assumes independence |
