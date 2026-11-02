# Stats vocabulary & phrasing

Every term in one place: what it means, and how to say it **correctly** in a sentence. In stats the wording *is* the understanding, so the wrong sentence is shown too.

**On this page**

- Data and samples: population, sample, parameter, estimate
- Describing data: mean, median, mode, spread, skew, outliers, z-score
- Distributions: normal, sampling distribution, CLT, standard error
- Confidence intervals: confidence level, margin of error, the interval
- Hypothesis tests: null, α, p-value, significant, the two errors
- Size and power: effect size, Cohen's d, power, p-hacking
- Relationships: correlation, r², adjusted R², causation
- Phrasing rules that apply everywhere

**How to read each entry:** the line under the term says what it means. ✓ is a correct sentence you can reuse. ✗ is the common wrong version, with why.

## Data and samples

### Population
The whole group you want to know about. Usually too big to measure.

✓ "The population is all adults living in London."

### Sample
The subset you actually measured.

✓ "We took a random sample of 200 residents."

✗ "Our sample shows that all Londoners…" A sample *estimates* the population; say "suggests" and give the uncertainty.

### Parameter
The true value in the population. Fixed, but unknown.

✓ "The parameter we want is the true proportion of all residents who recycle."

✗ "The parameter is 60%." 60% came from your sample, so it's an estimate, not the parameter.

### Point estimate (statistic)
The value you calculate from your sample: your single best guess at the parameter.

✓ "Our point estimate of the recycling rate is 60% (120 of 200)."

## Describing data

### Mean
The sum divided by the count. Pulled towards outliers and long tails.

✓ "The mean delivery time is 4.2 days."

### Median
The middle value when sorted. Half the values are below it, half above.

✓ "The median salary is £32,000: half of employees earn less and half earn more."

✗ "The average salary is £32,000" when you mean the median. "Average" usually means the mean, so name which one.

### Mode
The most common value. Mostly useful for categories.

✓ "The most common payment method (the mode) is card."

### Range
Largest minus smallest. Decided entirely by the two extremes.

✓ "Ages range from 18 to 64, a range of 46 years."

### Variance
The average squared distance from the mean. In squared units, so hard to read on its own.

✓ "Variance is in squared units, so we report the standard deviation instead."

### Standard deviation (SD)
The typical distance of a single value from the mean, in the data's own units.

✓ "Delivery times have a mean of 4 days and a standard deviation of 1.5 days."

✗ "The standard deviation of our estimate is 1.5." The SD describes individual values. The wobble of an estimate is the **standard error**.

### Skew
Lopsidedness. Named after the side of the **long tail**, not the side where the data bunch up.

✓ "Income is right-skewed: a long tail of high earners pulls the mean above the median."

✗ "Income is left-skewed because most people are on the left." The bunching is on the left, so the tail (and the name) is on the right.

### Quartiles and IQR
Quartiles split sorted data into four equal parts. The IQR (Q3 − Q1) is the spread of the middle half.

✓ "The IQR is 12 to 20, so the middle half of orders are between £12 and £20."

### Outlier
A value far from the rest, e.g. beyond 1.5 × IQR from the quartiles.

✓ "Using the 1.5 × IQR rule, three values are flagged as potential outliers; we checked them before deciding what to do."

✗ "These are outliers, so they're errors and we removed them." Outlier means unusual, not wrong. Check first.

### z-score
How many standard deviations a value is from the mean.

✓ "A z-score of 2 means the value is 2 standard deviations above the mean."

✗ "A z-score of 2 means it's twice the mean."

## Distributions

### Distribution
The pattern of how often each value turns up.

✓ "Delivery times are right-skewed, with most orders arriving in 2 to 5 days."

### Normal distribution
The symmetric bell curve. About 68% of values within 1 SD of the mean, 95% within 2, 99.7% within 3.

✓ "Heights are roughly normally distributed, so about 95% of people are within two standard deviations of the mean."

✗ "The data are normal" with no check. Say "roughly normal" and say how you checked (histogram, Q-Q plot).

### Sampling distribution
The spread of an estimate (like the mean) across all the samples you could have taken.

✓ "The sampling distribution of the mean shows how much the mean would vary from sample to sample."

### Central Limit Theorem (CLT)
For a big enough sample, the **sample mean** is roughly normal, whatever the shape of the data.

✓ "Because of the CLT, the sample mean is approximately normal even though the delivery times themselves are skewed."

✗ "With a big sample, the data become normal." The data keep their shape. It's the distribution of the *mean* that becomes normal.

### Standard error (SE)
How much your estimate would wobble if you took a fresh sample. SD ÷ √n, so it shrinks as the sample grows.

✓ "The standard error of the mean is 2 points, so the estimate would typically move by about 2 points between samples."

✗ "The standard error shows how spread out the data are." That's the SD.

## Confidence intervals

### Confidence level
How often the **method** captures the true value, if you repeated it many times. Chosen before you start: 90%, 95% or 99%.

✓ "We used a 95% confidence level."

### Margin of error
How far the interval reaches either side of the estimate: critical value × SE.

✓ "The estimate is 60% with a margin of error of ±8.9 percentage points."

### Confidence interval (CI)
Estimate ± margin of error: a range of plausible values for the parameter.

✓ "We are 99% confident that the true proportion of residents who recycle is between 51.1% and 68.9%."

✓ (about the method) "If we repeated this sampling many times, about 99% of the intervals built this way would contain the true proportion."

✗ "There's a 99% chance the true proportion is between 51.1% and 68.9%." The true value is fixed. It's the interval that changes from sample to sample.

✗ "99% of residents fall between 51.1% and 68.9%." The interval is about the parameter, not about individuals.

### Critical value
The cutoff score (z* or t*) at the edge of the middle band. 1.96 for 95%.

✓ "For a 95% interval, the critical value is 1.96."

## Hypothesis tests

### Null hypothesis (H₀)
"Nothing is going on": no effect, no difference. Assumed true until the data say otherwise.

✓ "The null hypothesis is that the new page layout has no effect on the conversion rate."

### Alternative hypothesis (H₁)
The effect you're testing for.

✓ "The alternative is that the conversion rate differs between the two layouts."

### Significance level (α)
The false-alarm risk you accept, set **before** testing. Usually 0.05.

✓ "We set α = 0.05 before collecting the data."

✗ "We used α = 0.10 because the result was 0.07." Choosing α after seeing the result defeats the point.

### Test statistic
How far your result is from what H₀ predicts, in standard errors.

✓ "The test statistic is z = 2.83: the result is 2.83 standard errors from what the null predicts."

### p-value
How easily a result at least this extreme would turn up by chance alone, **if the null were true**.

✓ "If there were really no difference, a result at least this extreme would happen only 0.47% of the time (p = 0.0047)."

✗ "There's a 0.47% chance the null hypothesis is true." The p-value assumes the null is true, so it can't measure that.

✗ "There's a 0.47% chance the result is due to chance." Same mistake in different words.

✗ "p = 0.0001, so the effect is huge." p is about evidence, not size.

### Statistically significant
The p-value is at or below α. It means "unlikely to be chance", not "important".

✓ "The difference is statistically significant at the 5% level (p = 0.003)."

✗ "The result is significant, so it matters." A tiny, useless difference can be significant in a big sample. Report the effect size too.

### Reject / fail to reject the null
The only two verdicts a test can give.

✓ "p = 0.003 is below 0.05, so we reject the null. There's significant evidence that conversion differs between the layouts."

✓ "p = 0.21 is above 0.05, so we fail to reject the null. There isn't enough evidence of a difference."

✗ "We accept the null." / "We proved there's no difference." Absence of evidence isn't evidence of absence.

### Type I error (false positive)
Rejecting the null when it's actually true. Its chance is α.

✓ "A Type I error here would mean flagging a clean athlete as doping."

### Type II error (false negative)
Failing to reject the null when there really is an effect. Its chance is β.

✓ "A Type II error here would mean missing an athlete who is doping."

## Size and power

### Effect size
How big the effect is, in a number that doesn't grow with sample size.

✓ "The new layout raised conversion by 2 percentage points, from 10% to 12%."

### Cohen's d
The gap between two means divided by the spread of individuals: how many person-to-person spreads wide the effect is.

✓ "d = 0.5: the groups differ by half a standard deviation, a medium effect."

### Power
The chance a test catches a real effect of a **stated size**. Power = 1 − β. Aim for 80%.

✓ "With 64 people per group, the study has 80% power to detect a medium effect (d = 0.5) at α = 0.05."

✗ "The study has 80% power." Power always needs "to detect an effect of what size?"

### Power analysis
Working out the sample size you need **before** collecting data.

✓ "A power analysis showed we need 64 people per group to have an 80% chance of detecting d = 0.5."

### p-hacking
Running many tests (or peeking repeatedly) and reporting only the ones that came out significant.

✓ "We tested 20 features at α = 0.05, so about one would look significant by chance alone; we corrected for multiple testing."

## Relationships

### Correlation (r)
How strongly two things move together in a straight line, from −1 to +1.

✓ "Height and weight have a strong positive correlation (r = 0.7)."

✗ "Height and weight are 70% correlated." r isn't a percentage.

### Spearman correlation
Pearson's r calculated on the ranks. Catches any steady up-or-down relationship, even a curved one, and resists outliers.

✓ "We used Spearman because the relationship is steady but curved, and there are outliers."

✗ "Spearman is for non-linear relationships." It's for monotonic ones (always rising or always falling).

### r² (R²)
The share of the variation in the outcome that the other variable explains.

✓ "Height explains about 49% of the variation in weight (r² = 0.49)."

✗ "49% of weight is caused by height." Explained isn't caused.

✗ "They're 49% related."

### Adjusted R²
R² with a penalty for every predictor, so useless features don't make the model look better.

✓ "Adding the extra features raised R² but not adjusted R², so they weren't adding real information."

### Correlation vs causation
Two things moving together doesn't mean one causes the other. A third factor may drive both.

✓ "Ice cream sales and drownings are correlated, probably because both rise in hot weather."

✗ "Ice cream sales cause drownings."

## Phrasing rules that apply everywhere

- **Put it in context.** Say what the number is about: "the true proportion of residents who recycle", not "p".
- **Give the uncertainty.** An estimate without a CI, SE or n is half an answer.
- **Evidence, not proof.** Stats gives "evidence for", never "proves".
- **Percentage points vs percent.** Going from 10% to 12% is **2 percentage points**, or a **20% increase**. Not "a 2% increase".
- **Significant is not important.** Pair every p-value with an effect size.
- **Name your average.** Say "mean" or "median", not just "average".
- **Associated with, not causes**, unless it was a controlled experiment.
