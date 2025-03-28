**Chapter 15: Hypothesis Testing**

Building upon descriptive statistics and probability distributions, this chapter introduces the formal framework of **statistical hypothesis testing**, a cornerstone of scientific inference used to make decisions about populations or physical models based on sample data. When analyzing astronomical data, we often want to go beyond simply describing it; we want to answer specific questions like: "Is the mean brightness of this star sample significantly different from that one?", "Does the distribution of galaxy sizes match the predictions of my cosmological model?", or "Is this apparent correlation between two variables statistically significant, or could it have arisen by chance?". Hypothesis testing provides a structured procedure for addressing such questions quantitatively, evaluating the strength of evidence provided by the data against a specific claim or "null hypothesis." We will cover the fundamental logic of hypothesis testing, including the concepts of null and alternative hypotheses, test statistics, p-values, significance levels, and Type I/II errors. We will then explore several common statistical tests frequently used in astrophysics – including t-tests for comparing means, Chi-squared tests for comparing counts or assessing goodness-of-fit, and the Kolmogorov-Smirnov (K-S) test for comparing distributions – demonstrating their implementation using `scipy.stats`. Finally, we will briefly discuss important caveats and potential pitfalls associated with hypothesis testing, such as the problems of multiple testing and publication bias.

**15.1 The Framework: Null and Alternative Hypotheses**

Statistical hypothesis testing provides a formal procedure for making decisions based on data in the presence of uncertainty. It starts by formulating two competing statements about the population or process being studied: the **null hypothesis** (H₀) and the **alternative hypothesis** (H₁ or H<0xE2><0x82><0x90>). These hypotheses must be mutually exclusive and exhaustive, meaning one and only one of them must be true.

The **null hypothesis (H₀)** typically represents a default assumption, a statement of "no effect," "no difference," or that any observed pattern is simply due to random chance. It's the hypothesis we aim to challenge or find evidence against. Examples in astrophysics include: "The mean magnitude of stars in cluster A is the same as in cluster B" (H₀: μ<0xE2><0x82><0x90> = μ<0xE1><0xB5><0xA7>), "The distribution of galaxy ellipticities is uniform" (H₀: Distribution = Uniform), or "The observed number of supernovae matches the rate predicted by model X" (H₀: Observed Counts = Expected Counts). H₀ is usually formulated in a way that allows us to calculate the expected distribution of results if it were true.

The **alternative hypothesis (H₁ or H<0xE2><0x82><0x90>)** represents the claim or effect we are interested in detecting. It contradicts the null hypothesis. It could state that there *is* a difference, an effect, or that the observed pattern is *not* due to chance alone. Examples corresponding to the H₀ above would be: "The mean magnitudes are different" (H₁: μ<0xE2><0x82><0x90> ≠ μ<0xE1><0xB5><0xA7>), "The distribution of ellipticities is not uniform", or "The observed supernova count differs significantly from model X's prediction". Alternative hypotheses can be **two-sided** (e.g., μ<0xE2><0x82><0x90> ≠ μ<0xE1><0xB5><0xA7>, allowing for a difference in either direction) or **one-sided** (e.g., μ<0xE2><0x82><0x90> > μ<0xE1><0xB5><0xA7>, specifying the direction of the expected difference). The choice depends on the specific research question.

The core logic of hypothesis testing is somewhat counterintuitive: we start by *assuming the null hypothesis (H₀) is true*. We then examine our observed sample data and calculate how likely it is to obtain data as extreme as, or more extreme than, what we observed, *if H₀ were actually true*. If this probability (the p-value, discussed next) is very small, it casts doubt on our initial assumption (H₀). We conclude that the data provides significant evidence *against* H₀ and in favor of the alternative hypothesis H₁. If the probability is not small, we conclude that we do not have sufficient evidence to reject H₀.

Crucially, hypothesis testing does *not* prove the alternative hypothesis is true, nor does it prove the null hypothesis is true. Failing to reject H₀ simply means the data are consistent with the null hypothesis (within the bounds of random chance); it doesn't mean H₀ is correct. Rejecting H₀ only means the data are unlikely to have occurred if H₀ were true, suggesting H₁ is a better explanation. The strength of evidence against H₀ is quantified by the p-value.

When making a decision based on the test, there are two possible types of errors:
*   **Type I Error (False Positive):** Rejecting the null hypothesis (H₀) when it is actually true. The probability of making a Type I error is denoted by α (alpha) and is typically set in advance by the researcher (e.g., α = 0.05).
*   **Type II Error (False Negative):** Failing to reject the null hypothesis (H₀) when it is actually false (i.e., the alternative hypothesis H₁ is true). The probability of making a Type II error is denoted by β (beta). The **power** of a test is defined as 1 - β, which is the probability of correctly rejecting H₀ when it is false.

There is an inherent trade-off between α and β. Making the criterion for rejecting H₀ stricter (decreasing α to reduce false positives) generally increases the probability of failing to detect a real effect when it exists (increasing β, decreasing power). Conversely, making the test more sensitive to detecting effects (increasing power, decreasing β) often increases the risk of false positives (increasing α). The choice of α reflects the researcher's tolerance for making a Type I error versus a Type II error, often based on the consequences of each type of error in the specific scientific context. A common, though arbitrary, choice for the significance level is α = 0.05.

The process involves calculating a **test statistic** from the sample data. This is a numerical value whose distribution *under the assumption that H₀ is true* is known (or can be approximated). Common test statistics include the t-statistic (for comparing means), the Chi-squared (χ²) statistic (for comparing counts or frequencies), and the Kolmogorov-Smirnov (K-S) statistic (for comparing distributions). We then compare the calculated value of the test statistic from our sample to its known distribution under H₀ to determine the probability of observing such an extreme value by chance.

Formulating clear, testable null and alternative hypotheses is the critical first step in any hypothesis testing procedure. The null hypothesis should be specific enough to allow calculation of expected outcomes or the distribution of a test statistic under its assumption. The alternative hypothesis should reflect the scientific question being investigated. This framework provides a structured way to evaluate evidence and make objective decisions, acknowledging the inherent uncertainty in sample data.

While widely used, the frequentist hypothesis testing framework based on p-values has limitations and is subject to misinterpretation (e.g., equating a non-significant p-value with evidence *for* the null hypothesis). Bayesian approaches (Chapters 17, 18) offer an alternative framework for evaluating evidence and comparing models, often providing more intuitive interpretations in terms of posterior probabilities of hypotheses. However, understanding the logic and common tests within the classical hypothesis testing framework remains essential for interpreting much of the existing scientific literature and for applying standard statistical procedures.

**15.2 Test Statistics, P-values, and Significance Levels (alpha)**

Once the null (H₀) and alternative (H₁) hypotheses are defined, the core of the hypothesis testing procedure involves quantifying how compatible the observed sample data is with the null hypothesis. This is achieved by calculating a **test statistic** and its associated **p-value**.

A **test statistic** is a single numerical value calculated from the sample data that summarizes the deviation between the observed data and what would be expected if the null hypothesis were true. The specific formula for the test statistic depends on the type of data and the hypothesis being tested. For example, when comparing a sample mean (x̄) to a hypothesized population mean (μ₀), the t-statistic `t = (x̄ - μ₀) / (s / sqrt(n))` measures how many standard errors the sample mean is away from the hypothesized value. When comparing observed counts in categories to expected counts, the Chi-squared statistic `χ² = Σ [(Observed - Expected)² / Expected]` measures the overall discrepancy. The key property of a test statistic is that its probability distribution *assuming H₀ is true* must be known or well-approximated (e.g., t-distribution, Chi-squared distribution, F-distribution, standard normal distribution).

After calculating the test statistic from our sample data, we compare it to its known distribution under H₀. This comparison allows us to determine the probability of obtaining a test statistic value as extreme as, or more extreme than, the one observed, purely by random chance, *if H₀ were actually true*. This probability is the **p-value**.

More formally, the **p-value** is the probability, calculated under the assumption that H₀ is true, of observing a test statistic value at least as extreme (in the direction(s) indicated by the alternative hypothesis H₁) as the value actually computed from the sample.
*   For a **two-sided** alternative hypothesis (H₁: parameter ≠ value), "extreme" means values far from the expected H₀ value in *either* direction (usually the tails of the distribution).
*   For a **one-sided** alternative hypothesis (e.g., H₁: parameter > value), "extreme" means values far from the expected H₀ value in *that specific direction* (e.g., the upper tail of the distribution).

A **small p-value** (typically less than a predetermined threshold α) indicates that the observed data (or data even more extreme) would be very unlikely to occur if the null hypothesis were true. This suggests that the data provides strong evidence *against* the null hypothesis, leading us to **reject H₀** in favor of H₁. A **large p-value** (greater than or equal to α) indicates that the observed data (or more extreme data) is reasonably plausible under the assumption that H₀ is true. This means we **fail to reject H₀**; the data are consistent with the null hypothesis, and we lack sufficient evidence to support the alternative hypothesis.

The threshold used to make this decision is the **significance level**, denoted by **α (alpha)**. This value represents the probability of making a Type I error (rejecting H₀ when it is true) that the researcher is willing to tolerate. It is chosen *before* conducting the test. A conventional, but arbitrary, choice for α in many scientific fields is 0.05 (or 5%). This means we are willing to accept a 5% chance of incorrectly rejecting a true null hypothesis (a false positive). Other common choices are α = 0.01 or α = 0.10, depending on the context and the relative costs of Type I versus Type II errors.

The decision rule is simple:
*   If **p-value ≤ α**: Reject H₀. The result is considered "statistically significant" at the level α.
*   If **p-value > α**: Fail to reject H₀. The result is considered "not statistically significant" at the level α.

It is crucial to interpret the p-value correctly. A p-value is *not* the probability that the null hypothesis is true, nor is it the probability that the alternative hypothesis is true. It is specifically the probability of the observed data (or more extreme data) occurring *given that the null hypothesis is true*. A common misinterpretation is treating a non-significant result (p > α) as evidence *for* the null hypothesis; it simply means the current data does not provide strong enough evidence to reject it. Similarly, statistical significance (p ≤ α) does not automatically imply practical or scientific significance; a very small effect might be statistically significant with a large enough sample size but irrelevant in practice.

```python
# --- Code Example: Conceptual Interpretation of p-value ---
import numpy as np
from scipy import stats

print("Conceptual interpretation of p-value and significance level:")

# Imagine we performed a test (e.g., t-test) and obtained these results:
test_statistic_value = 2.5 
p_value_calculated = 0.018 # Example calculated p-value

# Define our chosen significance level (alpha) *before* the test
alpha = 0.05
print(f"\nChosen Significance Level (alpha): {alpha}")
print(f"Calculated p-value from test: {p_value_calculated}")

# Make the decision based on p-value vs alpha
print("\nDecision:")
if p_value_calculated <= alpha:
    print(f"  Since p-value ({p_value_calculated:.3f}) <= alpha ({alpha}), we REJECT the null hypothesis (H0).")
    print(f"  The result is statistically significant at the {alpha*100:.0f}% level.")
else:
    print(f"  Since p-value ({p_value_calculated:.3f}) > alpha ({alpha}), we FAIL TO REJECT the null hypothesis (H0).")
    print(f"  The result is NOT statistically significant at the {alpha*100:.0f}% level.")

# --- Interpretation Caveats ---
print("\nImportant Interpretation Notes:")
print(f"- A p-value of {p_value_calculated:.3f} means: 'IF H0 were true, the probability of observing a test statistic")
print(f"  at least as extreme as {test_statistic_value} is {p_value_calculated:.3f} (or {p_value_calculated*100:.1f}%).'")
print("- It does NOT mean P(H0 is true) = 0.018 or P(H1 is true) = 1 - 0.018.")
print("- Failing to reject H0 does NOT prove H0 is true, only that the evidence against it is weak.")
print("-" * 20)

# Explanation: This code block simulates the final step of hypothesis testing.
# 1. It assumes a test was performed, yielding a specific p-value (0.018).
# 2. It defines the pre-chosen significance level `alpha` (0.05).
# 3. It implements the decision rule: compares the p-value to alpha.
# 4. Based on the comparison, it prints a statement indicating whether the null 
#    hypothesis is rejected or not, and whether the result is statistically significant 
#    at the chosen alpha level.
# 5. Crucially, it includes print statements clarifying the correct interpretation 
#    of the p-value and warning against common misinterpretations.
```

Reporting the exact p-value is generally more informative than simply stating whether the result was "significant" or "not significant" based on an arbitrary α threshold. The p-value itself quantifies the strength of the evidence against H₀. A p-value of 0.049 and a p-value of 0.0001 both lead to rejecting H₀ at α=0.05, but the latter represents much stronger evidence against the null hypothesis.

In summary, the test statistic measures the discrepancy between the observed data and the null hypothesis, while the p-value quantifies the probability of observing such a discrepancy (or greater) by chance if H₀ were true. Comparing the p-value to a pre-defined significance level α allows us to make a formal decision: reject H₀ if p ≤ α, or fail to reject H₀ if p > α. Correctly interpreting the meaning of the p-value and understanding the associated potential for Type I and Type II errors are crucial for drawing sound conclusions from hypothesis tests.

**15.3 Common Tests: t-test, Chi-squared test, Kolmogorov-Smirnov (K-S) test**

While the general framework of hypothesis testing is consistent, the specific test statistic used and its corresponding distribution under the null hypothesis depend on the type of data and the specific question being asked. This section introduces three widely used statistical tests often employed in astrophysical analyses: the t-test (for comparing means), the Chi-squared (χ²) test (for comparing observed vs. expected counts or frequencies), and the Kolmogorov-Smirnov (K-S) test (for comparing distributions). The `scipy.stats` module provides implementations for these and many other standard tests.

**1. t-tests:** These tests are used to compare means.
*   **One-sample t-test:** Tests whether the mean of a single sample (x̄) is significantly different from a known or hypothesized population mean (μ₀). The null hypothesis is H₀: μ = μ₀. The test statistic is `t = (x̄ - μ₀) / (s / sqrt(n))`, where `s` is the sample standard deviation and `n` is the sample size. Under H₀ (and assuming the data are approximately normally distributed or `n` is large enough via CLT), this statistic follows a Student's t-distribution with `n-1` degrees of freedom. `scipy.stats.ttest_1samp(sample_data, popmean=mu0)` performs this test, returning the t-statistic and the two-sided p-value.
*   **Two-sample t-test (independent samples):** Tests whether the means of two *independent* groups (x̄₁ and x̄₂) are significantly different. The null hypothesis is H₀: μ₁ = μ₂. The test statistic calculation is slightly more complex, depending on whether the variances of the two groups are assumed to be equal or unequal (Welch's t-test is often preferred as it doesn't assume equal variances). Under H₀ (and normality/large sample assumptions), the statistic follows a t-distribution with degrees of freedom calculated based on the sample sizes and variances. `scipy.stats.ttest_ind(sample1_data, sample2_data, equal_var=False)` performs Welch's t-test (recommended default), returning the t-statistic and p-value. This is common for comparing properties (like mean magnitude or metallicity) between two distinct populations (e.g., stars in different clusters, galaxies in different environments).
*   **Paired t-test:** Tests whether the mean difference between paired measurements (e.g., measurements on the same subjects/objects before and after a treatment, or using two different methods on the same objects) is significantly different from zero. `scipy.stats.ttest_rel(sample1_data, sample2_data)` performs this test.

**2. Chi-squared (χ²) tests:** These tests are generally used for categorical data or comparing observed frequencies/counts to expected frequencies.
*   **Chi-squared Goodness-of-Fit Test:** Tests whether the observed frequencies (O<0xE1><0xB5><0xA2>) in several categories match the expected frequencies (E<0xE1><0xB5><0xA2>) derived from a specific hypothesis or distribution. The null hypothesis is H₀: Observed frequencies match expected frequencies. The test statistic is `χ² = Σ [(O<0xE1><0xB5><0xA2> - E<0xE1><0xB5><0xA2>)² / E<0xE1><0xB5><0xA2>]` summed over all categories. Under H₀, this statistic approximately follows a Chi-squared distribution with `k - 1 - p` degrees of freedom, where `k` is the number of categories and `p` is the number of parameters estimated from the data to calculate the expected frequencies. `scipy.stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)` calculates the χ² statistic and p-value. This is often used to test if binned data (like a histogram) follows a predicted model distribution (Sec 15.5).
*   **Chi-squared Test for Independence (Contingency Tables):** Tests whether two categorical variables are independent. Data is presented in a contingency table (rows represent categories of variable 1, columns represent categories of variable 2, cells contain observed counts). H₀: The two variables are independent. The test calculates expected frequencies for each cell assuming independence and computes the χ² statistic comparing observed and expected counts across all cells. `scipy.stats.chi2_contingency(contingency_table)` performs this test, returning the χ² statistic, p-value, degrees of freedom, and expected frequencies table. Example: Testing if galaxy morphology type is independent of environment (e.g., field vs. cluster).

**3. Kolmogorov-Smirnov (K-S) test:** This test is used to compare distributions, specifically their **Cumulative Distribution Functions (CDFs)**. It is a non-parametric test, meaning it doesn't assume the data follows a specific distribution like Gaussian.
*   **One-sample K-S test:** Tests whether a sample of data comes from a specific, hypothesized continuous distribution (e.g., Gaussian, uniform, exponential). H₀: The sample data is drawn from the specified distribution. The test statistic (D) is the maximum absolute difference between the empirical CDF (ECDF) calculated from the sample data and the theoretical CDF of the hypothesized distribution. `scipy.stats.kstest(sample_data, cdf_function)` performs this test, where `cdf_function` can be a string name of a distribution in `scipy.stats` (e.g., `'norm'`, `'uniform'`) along with its parameters specified via `args`, or a callable function providing the theoretical CDF.
*   **Two-sample K-S test:** Tests whether two independent samples of data are drawn from the *same* underlying (but unspecified) continuous distribution. H₀: Both samples come from the same distribution. The test statistic (D) is the maximum absolute difference between the ECDFs of the two samples. `scipy.stats.ks_2samp(sample1_data, sample2_data)` performs this test. This is very useful for comparing distributions of a property (like planet radii, galaxy sizes, stellar ages) between two different populations without assuming the shape of the distribution.

```python
# --- Code Example: Applying Common Hypothesis Tests ---
# Note: Requires scipy installation.
import numpy as np
from scipy import stats

print("Applying common statistical tests using scipy.stats:")

# --- One-sample t-test Example ---
# Sample data (e.g., measured periods of a variable star)
sample_periods = np.array([5.1, 5.3, 4.9, 5.2, 5.4, 5.0, 5.15])
# Null hypothesis: True mean period is 5.0 days
mu0 = 5.0
print(f"\nOne-sample t-test: Is mean period = {mu0} days?")
t_stat_1samp, p_val_1samp = stats.ttest_1samp(sample_periods, popmean=mu0)
print(f"  t-statistic = {t_stat_1samp:.3f}, p-value = {p_val_1samp:.4f}")
alpha = 0.05
if p_val_1samp <= alpha: print("  -> Reject H0 (mean is likely different from 5.0)")
else: print("  -> Fail to reject H0 (consistent with mean = 5.0)")

# --- Two-sample t-test Example ---
# Sample data (e.g., metallicities of stars in two clusters)
cluster_A_feh = np.random.normal(loc=-0.1, scale=0.1, size=30)
cluster_B_feh = np.random.normal(loc=-0.3, scale=0.12, size=40)
print(f"\nTwo-sample t-test: Are mean metallicities different?")
# Use Welch's t-test (equal_var=False) by default, generally safer
t_stat_ind, p_val_ind = stats.ttest_ind(cluster_A_feh, cluster_B_feh, equal_var=False)
print(f"  t-statistic = {t_stat_ind:.3f}, p-value = {p_val_ind:.4f}")
if p_val_ind <= alpha: print("  -> Reject H0 (means are likely different)")
else: print("  -> Fail to reject H0 (means are consistent)")

# --- Chi-squared Goodness-of-Fit Example ---
# Observed counts in 4 categories (e.g., galaxy types)
observed_counts = np.array([50, 30, 15, 5])
# Expected counts based on a model (must sum to same total)
expected_counts = np.array([45, 35, 12, 8]) 
print(f"\nChi-squared GoF test: Do observed counts match expected?")
print(f"  Observed: {observed_counts}")
print(f"  Expected: {expected_counts}")
chi2_stat_gof, p_val_gof = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)
# Degrees of freedom = k - 1 = 4 - 1 = 3 (if expected counts fixed)
print(f"  Chi2 statistic = {chi2_stat_gof:.3f}, p-value = {p_val_gof:.4f}")
if p_val_gof <= alpha: print("  -> Reject H0 (observed counts significantly differ from expected)")
else: print("  -> Fail to reject H0 (observed counts consistent with expected)")

# --- Two-sample K-S test Example ---
# Compare distributions of radii for planets around M vs G stars (simulated)
radii_M_stars = stats.lognorm.rvs(s=0.5, loc=0, scale=1.5, size=100, random_state=1)
radii_G_stars = stats.lognorm.rvs(s=0.7, loc=0, scale=2.0, size=120, random_state=2)
print(f"\nTwo-sample K-S test: Are planet radius distributions different?")
ks_stat_2samp, p_val_2samp = stats.ks_2samp(radii_M_stars, radii_G_stars)
print(f"  K-S statistic = {ks_stat_2samp:.3f}, p-value = {p_val_2samp:.4f}")
if p_val_2samp <= alpha: print("  -> Reject H0 (distributions are likely different)")
else: print("  -> Fail to reject H0 (distributions are consistent)")

print("-" * 20)

# Explanation: This code demonstrates four common tests from `scipy.stats`:
# 1. `ttest_1samp`: Compares the mean of `sample_periods` to a hypothesized value `mu0`.
# 2. `ttest_ind`: Compares the means of two independent samples (`cluster_A_feh`, 
#    `cluster_B_feh`) using Welch's t-test (`equal_var=False`).
# 3. `chisquare`: Compares observed counts in categories to expected counts, calculating 
#    the Chi-squared statistic and p-value for goodness-of-fit.
# 4. `ks_2samp`: Compares two samples (`radii_M_stars`, `radii_G_stars`) to test if 
#    they are drawn from the same underlying continuous distribution using the K-S test.
# In each case, the test statistic and p-value are printed, and a conclusion is drawn 
# based on comparing the p-value to a significance level alpha=0.05.
```

Choosing the correct test is crucial. Use t-tests for comparing means (assuming data is roughly normal or sample sizes are large). Use Chi-squared tests for comparing counts or frequencies in categories, or for testing goodness-of-fit with binned data. Use K-S tests (or related tests like Anderson-Darling) for comparing the overall shapes of continuous distributions, especially when normality cannot be assumed. Each test has underlying assumptions (e.g., independence of samples, minimum expected counts for Chi-squared, continuity for K-S) that should be checked for validity in the specific application. `scipy.stats` provides a wide range of hypothesis tests, and consulting its documentation or statistical references is recommended to choose the most appropriate test for your specific research question and data type.

**15.4 Comparing Distributions**

A frequent task in astrophysics is comparing the distribution of a particular property between two or more different samples or populations. For example: Do galaxies in clusters have a different size distribution compared to galaxies in the field? Do metal-poor stars have a different velocity distribution than metal-rich stars? Do planets orbiting M-dwarfs have a different radius distribution than those orbiting G-dwarfs? While comparing means (using t-tests) or variances provides some information, we often want to test if the *entire shapes* of the distributions differ significantly, without necessarily assuming a specific form like Gaussian. Non-parametric tests based on the **Empirical Cumulative Distribution Function (ECDF)** are well-suited for this.

The ECDF, denoted F̂(x), for a sample {x₁, ..., x<0xE2><0x82><0x99>} is a step function that represents the fraction of data points less than or equal to a value x. It's calculated as F̂(x) = (number of x<0xE1><0xB5><0xA2> ≤ x) / n. The ECDF provides an estimate of the underlying true CDF of the population from which the sample was drawn. Comparing the ECDFs of two samples visually can give a qualitative impression of whether their distributions differ.

The **two-sample Kolmogorov-Smirnov (K-S) test**, implemented as `scipy.stats.ks_2samp(data1, data2)`, provides a quantitative way to perform this comparison for continuous distributions. As mentioned in Sec 15.3, its null hypothesis (H₀) is that both samples are drawn from the same underlying continuous distribution. The alternative hypothesis (H₁) is that they are drawn from different distributions. The test statistic, D, is the maximum absolute vertical difference between the ECDFs of the two samples: D = max | F̂₁(x) - F̂₂(x) |.

A large value of D indicates a significant difference between the shapes or locations of the two sample distributions. The `ks_2samp` function calculates D and returns a p-value. This p-value represents the probability of observing a maximum ECDF difference as large as D (or larger) if H₀ were true (i.e., if both samples really came from the same distribution). A small p-value (≤ α) leads to rejecting H₀, suggesting the distributions are significantly different. The K-S test is sensitive to differences in location (mean/median), scale (spread), and shape (skewness, kurtosis) of the distributions.

The K-S test is powerful because it makes no assumption about the underlying distribution shape (it's non-parametric). However, it is known to be more sensitive to differences near the *center* of the distributions than in the tails. It also formally requires the distributions to be continuous; applying it to discrete data can sometimes be done but requires caution in interpreting the p-value.

Another popular non-parametric test for comparing distributions is the **Anderson-Darling (A-D) test**. The two-sample A-D test, available in `scipy.stats` as `anderson_ksamp([sample1, sample2])`, also tests the null hypothesis that two independent samples are drawn from the same underlying distribution. Unlike the K-S test which uses the maximum difference, the A-D test statistic gives more weight to differences in the *tails* of the distributions. This often makes the A-D test more powerful (more likely to detect a true difference) than the K-S test, especially if the distributions differ primarily in their tails. The `anderson_ksamp` function returns the test statistic and critical values for different significance levels (α), rather than a direct p-value (though approximate p-value calculation might be possible). The decision rule involves comparing the calculated A-D statistic to the critical value for the chosen α: if the statistic exceeds the critical value, H₀ is rejected.

```python
# --- Code Example 1: Comparing Distributions with K-S and A-D Tests ---
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

print("Comparing two sample distributions using K-S and Anderson-Darling tests:")

# Generate two samples from slightly different distributions
np.random.seed(101)
sample1 = stats.norm.rvs(loc=0, scale=1, size=200) # Standard Normal
sample2 = stats.norm.rvs(loc=0.5, scale=1.2, size=250) # Normal with different mean and std dev

print(f"\nSample 1: N={len(sample1)}, Mean={np.mean(sample1):.2f}, Std={np.std(sample1):.2f}")
print(f"Sample 2: N={len(sample2)}, Mean={np.mean(sample2):.2f}, Std={np.std(sample2):.2f}")

# --- Two-sample K-S Test ---
print("\nPerforming two-sample K-S test...")
ks_statistic, ks_pvalue = stats.ks_2samp(sample1, sample2)
print(f"  K-S Statistic (D): {ks_statistic:.4f}")
print(f"  P-value: {ks_pvalue:.4e}") 
alpha = 0.05
if ks_pvalue <= alpha: print(f"  -> Reject H0 at alpha={alpha} (Distributions likely different)")
else: print(f"  -> Fail to reject H0 at alpha={alpha} (Distributions consistent)")

# --- Two-sample Anderson-Darling Test ---
print("\nPerforming two-sample Anderson-Darling test...")
try:
    # anderson_ksamp returns statistic and critical values for standard alpha levels
    ad_statistic, ad_critical_values, ad_significance_levels = stats.anderson_ksamp([sample1, sample2])
    print(f"  A-D Statistic: {ad_statistic:.4f}")
    print(f"  Critical Values: {ad_critical_values}")
    print(f"  Significance Levels (alpha): {ad_significance_levels}")
    
    # Check if statistic exceeds critical value for alpha = 0.05 (typically 5% is index 2)
    alpha_index = np.searchsorted(ad_significance_levels, alpha*100) 
    # Adjust index logic based on how significance levels are ordered/returned
    if alpha_index < len(ad_critical_values) and ad_statistic > ad_critical_values[alpha_index]:
         print(f"  -> Reject H0 at alpha={alpha} (A-D stat > critical value {ad_critical_values[alpha_index]:.2f})")
    elif alpha_index < len(ad_critical_values) :
         print(f"  -> Fail to reject H0 at alpha={alpha} (A-D stat <= critical value {ad_critical_values[alpha_index]:.2f})")
    else:
         print(f"  Could not determine critical value for alpha={alpha}") # Handle edge case

except ValueError as e: # Can happen if sample sizes are too small
    print(f"  Anderson-Darling test failed: {e}")
except Exception as e:
    print(f"  An error occurred during A-D test: {e}")

# --- Visualize ECDFs ---
print("\nGenerating ECDF comparison plot...")
fig, ax = plt.subplots(figsize=(7, 5))
# Calculate ECDFs manually for plotting
x1_sorted = np.sort(sample1)
y1_ecdf = np.arange(1, len(sample1) + 1) / len(sample1)
x2_sorted = np.sort(sample2)
y2_ecdf = np.arange(1, len(sample2) + 1) / len(sample2)

ax.plot(x1_sorted, y1_ecdf, drawstyle='steps-post', label='Sample 1 ECDF')
ax.plot(x2_sorted, y2_ecdf, drawstyle='steps-post', label='Sample 2 ECDF')

ax.set_xlabel("Value")
ax.set_ylabel("Cumulative Probability")
ax.set_title("Empirical Cumulative Distribution Functions (ECDFs)")
ax.legend()
ax.grid(True, alpha=0.5)
fig.tight_layout()
# plt.show()
print("ECDF plot generated.")
plt.close(fig)
print("-" * 20)

# Explanation: This code generates two samples from slightly different Normal distributions.
# 1. It performs the two-sample K-S test using `stats.ks_2samp`, obtaining the D 
#    statistic and p-value, and makes a decision based on alpha=0.05.
# 2. It performs the two-sample Anderson-Darling test using `stats.anderson_ksamp`. 
#    This returns the A-D statistic and an array of critical values corresponding 
#    to standard significance levels (e.g., 25%, 10%, 5%, 2.5%, 1%). The code 
#    finds the index for alpha=0.05 and compares the statistic to the critical value 
#    to make the decision.
# 3. It manually calculates and plots the ECDFs for both samples using `steps-post` 
#    style. The maximum vertical distance between these two step functions corresponds 
#    visually to the K-S D statistic. The plot helps visualize the differences 
#    the tests are quantifying.
```

Visually inspecting the ECDFs or normalized histograms/KDEs of the two samples is always recommended alongside performing the statistical tests. Plots can reveal the nature of the difference (shift in location, change in spread, difference in shape) that might be missed by looking only at the test statistic or p-value.

When comparing distributions, especially in astrophysics where sample sizes can be large, it's important to distinguish between statistical significance and practical significance. With very large samples, even tiny, physically unimportant differences between distributions can become statistically significant (yielding very small p-values). Always consider the magnitude of the difference (e.g., the D statistic for K-S, or visual inspection of ECDFs/histograms) in the context of your scientific question, rather than relying solely on the p-value threshold.

In summary, non-parametric tests like the two-sample Kolmogorov-Smirnov (K-S) test and the Anderson-Darling (A-D) test provide powerful tools for quantitatively comparing the overall distributions of two independent samples without assuming specific distributional forms. Implemented in `scipy.stats` as `ks_2samp` and `anderson_ksamp`, they test the null hypothesis that both samples originate from the same underlying distribution, complementing visual comparisons using ECDFs or histograms and providing objective measures of distributional differences crucial for comparing astrophysical populations.

**15.5 Assessing Goodness-of-Fit**

A common task related to hypothesis testing is assessing **goodness-of-fit**: determining how well a specific theoretical model or probability distribution describes an observed dataset. We want to answer the question: "Is the observed data consistent with being drawn from this proposed model distribution, or is the discrepancy between data and model statistically significant?" Several tests can be adapted for this purpose, primarily the Chi-squared and Kolmogorov-Smirnov tests.

The **Chi-squared (χ²) goodness-of-fit test** is typically used when the data is **binned**, either naturally (categorical data) or by creating a histogram from continuous data. The procedure involves:
1.  Defining `k` bins or categories.
2.  Counting the number of observed data points (O<0xE1><0xB5><0xA2>) falling into each bin `i`.
3.  Calculating the number of data points expected (E<0xE1><0xB5><0xA2>) in each bin `i` *according to the model being tested*. This might involve integrating the model's PDF over the bin range and multiplying by the total number of data points `n`.
4.  Calculating the Chi-squared statistic: `χ² = Σ [ (O<0xE1><0xB5><0xA2> - E<0xE1><0xB5><0xA2>)² / E<0xE1><0xB5><0xA2> ]` summed over all `k` bins.
5.  Comparing the calculated χ² value to the Chi-squared distribution with appropriate degrees of freedom (`df`).

The null hypothesis (H₀) is that the data *is* drawn from the model distribution. A large χ² value indicates a significant discrepancy between observed and expected counts, leading to a small p-value and rejection of H₀. The degrees of freedom calculation is crucial: `df = k - 1 - p`, where `k` is the number of bins and `p` is the number of *free parameters* of the model that were *estimated from the data itself* in order to calculate the expected frequencies E<0xE1><0xB5><0xA2>. If the model and its parameters are completely specified beforehand (no parameters estimated from the data), then p=0 and df=k-1. `scipy.stats.chisquare(f_obs=O, f_exp=E)` calculates the χ² statistic and the p-value, but requires the user to correctly determine and interpret the result based on the appropriate `df`. A common rule of thumb for the χ² approximation to be valid is that the expected count E<0xE1><0xB5><0xA2> should be reasonably large (e.g., E<0xE1><0xB5><0xA2> ≥ 5) in most bins; bins might need to be merged if expected counts are too low.

```python
# --- Code Example 1: Chi-squared Goodness-of-Fit for Binned Data ---
import numpy as np
from scipy import stats

print("Chi-squared Goodness-of-Fit Example:")

# Example: Testing if die rolls are consistent with a fair die.
# Roll a die 120 times. Expected count for each face = 120 / 6 = 20.
observed_counts = np.array([18, 22, 19, 21, 17, 23]) # Observed frequencies
expected_counts = np.array([20, 20, 20, 20, 20, 20]) # Expected for fair die

print(f"\nObserved Die Roll Counts: {observed_counts}")
print(f"Expected Die Roll Counts (Fair): {expected_counts}")

# Perform the Chi-squared goodness-of-fit test
chi2_stat, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

# Degrees of freedom = k - 1 = 6 - 1 = 5 (no parameters estimated)
df = len(observed_counts) - 1 
print(f"\nChi2 statistic = {chi2_stat:.3f}")
print(f"Degrees of freedom = {df}")
print(f"P-value = {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value <= alpha:
    print(f"-> Reject H0 at alpha={alpha}. Evidence suggests the die may be biased.")
else:
    print(f"-> Fail to reject H0 at alpha={alpha}. Observed counts are consistent with a fair die.")
print("-" * 20)

# Explanation: This code tests if observed die roll frequencies are consistent 
# with the null hypothesis of a fair die (equal expected frequencies).
# 1. It defines the observed counts and the expected counts under H0.
# 2. It uses `scipy.stats.chisquare` to calculate the Chi2 statistic and p-value.
# 3. It determines the correct degrees of freedom (k-1, since no parameters were estimated).
# 4. It interprets the p-value relative to alpha=0.05. In this case, the p-value 
#    will likely be large, indicating the observed deviations are consistent with 
#    random chance for a fair die.
```

The **one-sample Kolmogorov-Smirnov (K-S) test** provides an alternative goodness-of-fit test, particularly useful for **unbinned continuous data**. As introduced in Sec 15.3, it compares the empirical CDF (ECDF) of the sample data to the theoretical CDF of the hypothesized distribution. H₀: The data were drawn from the specified theoretical distribution. The test statistic D is the maximum absolute difference between the ECDF and the theoretical CDF.

`scipy.stats.kstest(data, cdf_function, args=())` performs the test. `data` is the 1D array of sample data. `cdf_function` can be a string name (e.g., `'norm'`) or a callable function representing the theoretical CDF. `args` is a tuple containing the *fixed parameters* of the theoretical distribution (e.g., `args=(mean, std_dev)` for `'norm'`). The function returns the D statistic and the p-value. A small p-value leads to rejecting H₀, indicating the data likely does not come from the hypothesized distribution.

A critical limitation of the standard K-S test (and its p-value calculation) is that it assumes the parameters of the theoretical distribution (`args`) are pre-specified and *not* estimated from the data itself. If you estimate parameters (like the mean and standard deviation for a Gaussian test) from the *same* data you are testing, the standard K-S p-value becomes inaccurate (usually overly conservative, making it harder to reject H₀). While modifications like the Lilliefors test exist specifically for testing normality with estimated parameters, `scipy.stats.kstest` does not automatically account for this. Therefore, the K-S test is most accurately used when comparing data to a model with *a priori* fixed parameters. If parameters are estimated, the resulting p-value should be interpreted with caution, or alternative tests (like Chi-squared on binned data after accounting for estimated parameters in `df`) might be more appropriate.

```python
# --- Code Example 2: K-S Goodness-of-Fit Test ---
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

print("Kolmogorov-Smirnov Goodness-of-Fit Example:")

# Generate sample data (e.g., from a Normal distribution)
np.random.seed(42)
sample_data = stats.norm.rvs(loc=5.0, scale=2.0, size=100)

# Test 1: Test against the CORRECT distribution (H0 is true)
print("\nTesting data against Normal(loc=5, scale=2)...")
# Provide distribution name and *correct* parameters via args
ks_stat1, p_value1 = stats.kstest(sample_data, 'norm', args=(5.0, 2.0))
print(f"  K-S statistic = {ks_stat1:.4f}, p-value = {p_value1:.4f}")
if p_value1 > 0.05: print("  -> Fail to reject H0 (Correctly identifies consistency)")
else: print("  -> Reject H0 (Incorrect for this data)")

# Test 2: Test against a DIFFERENT distribution (H0 is false)
print("\nTesting data against Uniform(min=0, max=10)...")
# Define the CDF function for Uniform[0,10]
def uniform_cdf(x): return stats.uniform.cdf(x, loc=0, scale=10)
# Can also use string 'uniform' with args=(loc, scale) if scipy supports it well
ks_stat2, p_value2 = stats.kstest(sample_data, uniform_cdf) 
# Or: ks_stat2, p_value2 = stats.kstest(sample_data, 'uniform', args=(0, 10))
print(f"  K-S statistic = {ks_stat2:.4f}, p-value = {p_value2:.4e}")
if p_value2 <= 0.05: print("  -> Reject H0 (Correctly identifies inconsistency)")
else: print("  -> Fail to reject H0 (Incorrect)")

# Test 3: Test against Normal BUT estimating parameters from data (Caution!)
sample_mean = np.mean(sample_data)
sample_std = np.std(sample_data, ddof=1)
print(f"\nTesting data against Normal(loc={sample_mean:.2f}, scale={sample_std:.2f}) (estimated pars)...")
ks_stat3, p_value3 = stats.kstest(sample_data, 'norm', args=(sample_mean, sample_std))
print(f"  K-S statistic = {ks_stat3:.4f}, p-value = {p_value3:.4f}")
print("  WARNING: P-value is unreliable because parameters were estimated from data.")

# Visualize ECDF vs Theoretical CDF for Test 1
print("\nGenerating ECDF vs Theoretical CDF plot...")
fig, ax = plt.subplots(figsize=(7, 5))
# Empirical CDF
n_data = len(sample_data)
ax.step(np.sort(sample_data), np.arange(1, n_data + 1) / n_data, 
        where='post', label='Empirical CDF (ECDF)')
# Theoretical CDF
x_plot = np.linspace(sample_data.min(), sample_data.max(), 200)
cdf_theoretical = stats.norm.cdf(x_plot, loc=5.0, scale=2.0)
ax.plot(x_plot, cdf_theoretical, 'r-', label='Theoretical Normal CDF (H0)')

ax.set_title("K-S Test: ECDF vs. Theoretical CDF")
ax.set_xlabel("Data Value")
ax.set_ylabel("Cumulative Probability")
ax.legend()
ax.grid(True, alpha=0.5)
# plt.show()
print("Plot generated.")
plt.close(fig)

print("-" * 20)

# Explanation: This code demonstrates the one-sample K-S test.
# 1. It generates data known to be from a Normal(5, 2) distribution.
# 2. Test 1: It runs `kstest` comparing the data against the *correct* theoretical 
#    Normal CDF (passing the true parameters 5.0 and 2.0 via `args`). The resulting 
#    p-value is expectedly large (H0 is true).
# 3. Test 2: It runs `kstest` comparing the data against a *different* theoretical 
#    distribution (Uniform CDF). The resulting p-value is expectedly very small (H0 is false).
# 4. Test 3: It *estimates* the mean and standard deviation from the sample data and 
#    uses *these estimates* in `kstest`. While the K-S statistic might still be 
#    calculated, a crucial warning is printed because the standard p-value calculation 
#    is invalid in this case.
# 5. It generates a plot comparing the data's ECDF (a step function) with the 
#    theoretical CDF used in Test 1, visually showing the basis for the K-S statistic 
#    (the maximum vertical distance between the two curves).
```

Other goodness-of-fit tests exist, such as the Anderson-Darling test (often more sensitive than K-S, especially in the tails, available as `scipy.stats.anderson`) and tests based on likelihood ratios (Chapter 18). The choice depends on whether data is binned or unbinned, whether model parameters are pre-specified or estimated, and the specific aspects of the fit (tails vs. center) one is most interested in testing. Assessing goodness-of-fit provides a crucial quantitative check on whether a chosen model or distribution provides an adequate description of the observed data.

**15.6 Pitfalls: Multiple Testing, P-hacking**

While hypothesis testing provides a valuable framework for statistical decision-making, it's essential to be aware of potential pitfalls and common misuses that can lead to spurious or unreliable scientific conclusions. Two significant issues are the **multiple testing problem** and **p-hacking**.

The **multiple testing problem** (also known as the "look-elsewhere effect" in physics) arises when multiple hypothesis tests are performed simultaneously on the same dataset or within the same study. If we use a standard significance level α = 0.05, we expect to get a false positive (rejecting a true H₀) about 5% of the time *for a single test*. However, if we perform many independent tests (say, 100 tests), the probability of getting *at least one* false positive by chance increases dramatically. For 100 independent tests where H₀ is true for all, the probability of at least one false positive is 1 - (0.95)¹⁰⁰ ≈ 0.994, meaning it's almost certain we'll find a "significant" result just by chance!

This is a major issue in modern astrophysics where large datasets allow for countless potential correlations or comparisons to be examined. For example, searching for periodic signals in a light curve across many frequencies, looking for correlations between dozens of galaxy properties, or scanning an image for significant sources involves performing numerous implicit or explicit hypothesis tests. If the significance threshold α is not adjusted to account for the number of tests performed, spurious "discoveries" are highly likely.

Several methods exist to correct for multiple testing, aiming to control either the **Family-Wise Error Rate (FWER)** – the probability of making *at least one* Type I error – or the **False Discovery Rate (FDR)** – the expected *proportion* of rejected null hypotheses that are actually false positives.
*   The **Bonferroni correction** is the simplest FWER control method: if performing `m` tests, use a stricter significance level α' = α / m for each individual test. This is often too conservative, reducing statistical power significantly.
*   Other FWER methods (like Holm-Bonferroni) provide more power than simple Bonferroni.
*   **FDR control methods**, like the Benjamini-Hochberg procedure, are often preferred when performing a large number of tests (e.g., in genomics or large surveys). They aim to control the proportion of false discoveries among all discoveries made, offering greater power than FWER control while still managing the overall rate of false positives. `statsmodels.sandbox.stats.multicomp.multipletests` in the `statsmodels` Python package provides implementations of various correction methods.

Recognizing when multiple testing occurs and applying appropriate corrections (or at least acknowledging the increased risk of false positives) is crucial for the credibility of results derived from large-scale data exploration.

**P-hacking** refers to a related set of questionable research practices where researchers consciously or unconsciously manipulate data analysis choices to obtain statistically significant p-values (p ≤ α). This can happen in various ways:
*   **Selective reporting:** Only reporting tests or analyses that yielded significant results, while ignoring non-significant ones.
*   **Optional stopping:** Collecting data until a significant result is found and then stopping data collection.
*   **Flexible data analysis:** Trying different statistical tests, different data transformations, different outlier removal criteria, or different model specifications until one yields p ≤ 0.05.
*   **HARKing (Hypothesizing After the Results are Known):** Formulating the hypothesis *after* exploring the data and finding an apparently significant pattern, then presenting it as if it were an *a priori* hypothesis.

These practices dramatically inflate the true Type I error rate, leading to a scientific literature potentially filled with spurious findings that are unlikely to replicate. P-hacking exploits the flexibility in data analysis choices to "fish" for significance. While often not outright fraudulent, it represents poor scientific practice driven by pressures to publish "positive" results or confirmation bias.

Combating p-hacking requires promoting **transparency** and **preregistration**. Researchers should ideally specify their primary hypotheses and analysis plan *before* analyzing the data (preregistration), distinguishing clearly between planned confirmatory analyses and exploratory analyses. Reporting *all* performed analyses and their results (regardless of statistical significance), clearly justifying analysis choices, performing sensitivity analyses (checking if results hold under different reasonable analysis choices), and focusing on effect sizes and confidence intervals alongside p-values can also mitigate the problem. Within astrophysics, the complexity of data reduction pipelines can also offer avenues for unconscious p-hacking if choices are made based on achieving a desired outcome.

Furthermore, **publication bias** – the tendency for journals to preferentially publish statistically significant "positive" results over non-significant "null" results – exacerbates the problem, creating a distorted view of the evidence in the literature. Encouraging the publication of well-conducted studies with null results is also important.

In summary, while hypothesis testing is a useful tool, it must be applied thoughtfully. The multiple testing problem requires adjusting significance thresholds or controlling the false discovery rate when many tests are performed. P-hacking represents a misuse of statistical flexibility and must be countered through transparency, preregistration, and focusing on robust scientific practice rather than solely chasing statistical significance. Awareness of these pitfalls is essential for both performing and critically evaluating statistical claims in astrophysical research.

**Application 15.A: Comparing Exoplanet Radius Distributions (K-S Test)**

**Objective:** This application demonstrates the use of the two-sample Kolmogorov-Smirnov (K-S) test (Sec 15.3, 15.4) to quantitatively compare the overall distributions of a continuous variable (exoplanet radius) between two distinct populations (planets orbiting M-dwarf stars vs. planets orbiting G-dwarf stars). It illustrates how hypothesis testing can address questions about whether different formation environments or host star types lead to different planetary outcomes.

**Astrophysical Context:** Understanding the diversity of exoplanets and how their properties depend on the characteristics of their host stars is a major goal of exoplanet science. For instance, M-dwarf stars are much smaller, cooler, and longer-lived than Sun-like G-dwarf stars. Do the planets they host typically have different sizes? Visually comparing histograms might be suggestive, but a statistical test is needed to determine if any observed difference in the radius distributions is statistically significant or could simply be due to random sampling variations.

**Data Source:** A catalog of confirmed exoplanets containing planet radius (`pl_rade` in Earth radii) and host star spectral type (or temperature/mass, allowing classification into M-dwarf and G-dwarf categories). This data can be obtained from archives like the NASA Exoplanet Archive or the Exoplanet EU catalog. We will simulate representative data if needed.

**Modules Used:** `astropy.table.Table` (or `pandas.DataFrame`) to handle the catalog data, `numpy` for array manipulation, `scipy.stats.ks_2samp` to perform the K-S test, and `matplotlib.pyplot` for visualizing the distributions (optional but recommended).

**Technique Focus:** Applying a non-parametric two-sample hypothesis test (`ks_2samp`) appropriate for comparing continuous distributions without assuming normality. Formulating the null hypothesis (H₀: Radius distributions for M-dwarf planets and G-dwarf planets are identical) and the alternative hypothesis (H₁: Distributions are different). Interpreting the returned K-S statistic (D) and, more importantly, the p-value. Making a decision based on comparing the p-value to a chosen significance level (α).

**Processing Step 1: Load and Prepare Data:** Read the exoplanet catalog into an Astropy Table or Pandas DataFrame. Filter the table to include only confirmed planets with reliable radius measurements and host star classifications. Create two separate NumPy arrays: one containing the radii (`pl_rade`) of planets orbiting M-dwarfs, and another containing radii for planets orbiting G-dwarfs. Handle any missing radius values appropriately (e.g., remove corresponding rows).

**Processing Step 2: Perform K-S Test:** Apply the two-sample K-S test using `ks_statistic, p_value = stats.ks_2samp(radii_M_stars, radii_G_stars)`. This function calculates the maximum difference (D) between the empirical CDFs of the two radius samples and computes the p-value associated with that difference under the null hypothesis.

**Processing Step 3: Interpret Results:** Print the calculated K-S statistic and the p-value. Compare the p-value to a pre-defined significance level (e.g., α = 0.05). If p ≤ α, state that the null hypothesis is rejected, concluding that there is statistically significant evidence that the radius distributions differ between the two host star types. If p > α, state that we fail to reject the null hypothesis, concluding that the observed data does not provide sufficient evidence to claim a significant difference in the distributions (at the chosen α level).

**Processing Step 4: Visualization (Recommended):** Create plots to visually compare the distributions. Options include:
    *   Overlaying normalized histograms (using density=True) for both samples on the same axes. Choose binning carefully (e.g., logarithmic bins might be appropriate for radii).
    *   Plotting the Empirical Cumulative Distribution Functions (ECDFs) for both samples on the same axes. The maximum vertical separation between the ECDF curves corresponds to the K-S statistic D.
These plots help visualize the nature of any difference detected by the test (or lack thereof).

**Output, Testing, and Extension:** The primary output is the K-S statistic, the p-value, and the statistical conclusion based on comparing p to α. The visualization plots are also key outputs. **Testing** involves checking the data filtering steps and ensuring the correct radius values are extracted for each group. Verify the interpretation of the p-value. Run the test on two random subsamples drawn from the *same* group (e.g., two halves of the G-dwarf sample) – the p-value should generally be large (> α) in this case. **Extensions:** (1) Perform the comparison using the Anderson-Darling test (`stats.anderson_ksamp`) and compare the conclusion. (2) Compare distributions for other parameters, like orbital period or planet mass. (3) Split the samples further (e.g., by discovery method – Transit vs. RV) and compare distributions within those subgroups. (4) Instead of just comparing M vs G dwarfs, compare multiple stellar type bins (e.g., M, K, G, F) using appropriate multi-sample comparison tests if available, or pairwise K-S tests with corrections for multiple testing (Sec 15.6).

```python
# --- Code Example: Application 15.A ---
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from astropy.table import Table # Assume data in Table
import io # Simulate file

print("Comparing Exoplanet Radius Distributions (M vs G dwarfs) using K-S Test:")

# --- Step 1: Simulate/Load Data ---
# Simulate plausible data for demonstration
np.random.seed(2024)
n_m_planets = 150
n_g_planets = 200
# M dwarfs often have smaller planets - e.g., lognormal centered smaller
radii_m = stats.lognorm.rvs(s=0.6, loc=0.1, scale=1.2, size=n_m_planets) 
# G dwarfs might have wider range, including giants
radii_g = stats.lognorm.rvs(s=0.8, loc=0.1, scale=2.5, size=n_g_planets) 
# Clip unrealistic values for demo
radii_m = np.clip(radii_m, 0.3, 10)
radii_g = np.clip(radii_g, 0.3, 20)

print(f"\nGenerated {n_m_planets} radii for M-dwarf planets (Mean={np.mean(radii_m):.2f})")
print(f"Generated {n_g_planets} radii for G-dwarf planets (Mean={np.mean(radii_g):.2f})")

# --- Step 2: Perform K-S Test ---
print("\nPerforming two-sample K-S test...")
ks_statistic, ks_pvalue = stats.ks_2samp(radii_m, radii_g)

# --- Step 3: Interpret Results ---
print(f"  K-S Statistic (D): {ks_statistic:.4f}")
print(f"  P-value: {ks_pvalue:.4e}") 
alpha = 0.05
print(f"\nDecision (alpha = {alpha}):")
if ks_pvalue <= alpha: 
    print(f"  Reject H0. Evidence suggests the radius distributions for planets")
    print(f"  around M dwarfs and G dwarfs are significantly different (p <= {alpha}).")
else: 
    print(f"  Fail to reject H0. The data is consistent with the radius distributions")
    print(f"  being the same for M and G dwarfs (p > {alpha}).")

# --- Step 4: Visualization (ECDFs) ---
print("\nGenerating ECDF comparison plot...")
fig, ax = plt.subplots(figsize=(8, 5))

# Calculate and plot ECDFs
x_m_sorted = np.sort(radii_m)
y_m_ecdf = np.arange(1, len(radii_m) + 1) / len(radii_m)
ax.step(x_m_sorted, y_m_ecdf, where='post', label=f'M-Dwarf Planets (N={n_m_planets})')

x_g_sorted = np.sort(radii_g)
y_g_ecdf = np.arange(1, len(radii_g) + 1) / len(radii_g)
ax.step(x_g_sorted, y_g_ecdf, where='post', label=f'G-Dwarf Planets (N={n_g_planets})')

# Optional: Mark the K-S statistic location (approximate)
# This requires finding the x value where ECDF difference is max, more complex
# max_diff_idx = np.argmax(np.abs(np.interp(x_common, x1, y1) - np.interp(x_common, x2, y2)))

ax.set_xlabel("Planet Radius (Earth Radii)")
ax.set_ylabel("Cumulative Probability (ECDF)")
ax.set_title("ECDF Comparison of Planet Radii (M vs G dwarfs)")
ax.legend(loc='lower right')
ax.grid(True, alpha=0.5)
ax.set_xscale('log') # Radii often viewed on log scale
ax.set_xlim(0.3, 30) # Adjust limits
ax.set_ylim(0, 1.05)

fig.tight_layout()
# plt.show()
print("ECDF plot generated.")
plt.close(fig)
print("-" * 20)

# Explanation: This application simulates planet radius data for M and G dwarf hosts.
# 1. It generates two arrays `radii_m` and `radii_g` from slightly different distributions.
# 2. It performs the two-sample K-S test using `stats.ks_2samp(radii_m, radii_g)`.
# 3. It prints the resulting K-S statistic and p-value, then interprets the result 
#    by comparing the p-value to alpha=0.05, stating whether the null hypothesis 
#    (that distributions are the same) is rejected or not.
# 4. It generates a plot showing the Empirical CDFs for both samples using `plt.step`. 
#    This plot visually illustrates the difference (or similarity) quantified by the K-S test. 
#    A logarithmic x-axis is used, common for planet radii.
```

**Application 15.B: Testing Model Predictions for Galaxy Cluster Counts (Chi-squared)**

**Objective:** This application demonstrates the use of the Chi-squared (χ²) goodness-of-fit test (Sec 15.3, 15.5) to evaluate whether observed counts of objects (galaxy clusters) in different bins (e.g., redshift or mass bins) are statistically consistent with the counts predicted by a specific theoretical model (e.g., a cosmological model).

**Astrophysical Context:** Galaxy clusters, the largest gravitationally bound structures in the universe, are sensitive probes of cosmology. Their abundance and spatial distribution depend on cosmological parameters like the matter density (Ω<0xE1><0xB5><0x89>), the amplitude of density fluctuations (σ₈), and dark energy properties. Cosmological models predict the expected number density of clusters as a function of mass and redshift. Comparing these predictions with observed cluster counts derived from surveys (using X-ray, Sunyaev-Zel'dovich effect, or optical detection methods) provides a powerful test of the underlying cosmological model.

**Data Source:** Two sets of numbers are required:
    1.  `observed_counts`: An array representing the number of galaxy clusters detected in several predefined bins (e.g., 5 redshift bins or 4 mass bins). These counts might come from analyzing survey data.
    2.  `expected_counts`: An array of the *same length* representing the number of clusters *predicted* by a specific theoretical model (e.g., a ΛCDM model with specific parameters) within those exact same bins. This typically involves integrating a theoretical halo mass function over the survey volume and mass/redshift limits of each bin, accounting for survey selection effects.
For this application, we will simulate plausible observed and expected counts.

**Modules Used:** `numpy` (for arrays), `scipy.stats.chisquare` (to perform the test), `matplotlib.pyplot` (optional visualization).

**Technique Focus:** Applying the Chi-squared goodness-of-fit test. Formulating the null hypothesis (H₀: The observed counts are consistent with the expected counts from the model). Calculating the χ² statistic `χ² = Σ [(O - E)² / E]` using `scipy.stats.chisquare`. Determining the correct degrees of freedom (`df = k - 1 - p`, where k is number of bins, p is number of model parameters fitted *to this data* to get the expected counts). Interpreting the resulting p-value to assess the goodness-of-fit.

**Processing Step 1: Define Observed and Expected Counts:** Create two NumPy arrays, `observed_counts` and `expected_counts`, representing the counts in `k` corresponding bins. Ensure the total counts are reasonably similar (though the test accounts for differences). *Crucially*, ensure that expected counts `E` are not too low (e.g., E ≥ 5 is a common guideline) in most bins; if necessary, bins should be merged *before* the test.

**Processing Step 2: Perform Chi-squared Test:** Use `chi2_stat, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)`. This calculates the statistic and a p-value based on the χ² distribution.

**Processing Step 3: Determine Degrees of Freedom:** This is a critical step requiring context *beyond* just the input arrays. Let `k` be the number of bins (categories) compared. Let `p` be the number of free parameters in the theoretical model that were adjusted or fitted *using the observed_counts data itself* in order to generate the `expected_counts`. If the `expected_counts` were derived entirely independently of the `observed_counts` (e.g., from a completely fixed prior model), then p=0 and `df = k - 1`. If, for example, the overall normalization of the model was adjusted to match the total observed count, then p=1 and `df = k - 1 - 1 = k - 2`. If multiple cosmological parameters were fitted to these cluster counts, `p` would be the number of those fitted parameters. Incorrectly specifying `df` leads to an incorrect interpretation of the p-value returned by `scipy.stats.chisquare` (which often assumes df=k-1 if `p` isn't accounted for).

**Processing Step 4: Interpret Results:** Compare the p-value returned by `chisquare` to the chosen significance level α (e.g., 0.05). If p ≤ α, reject H₀ and conclude that the observed counts are statistically inconsistent with the model's predictions. If p > α, fail to reject H₀ and conclude that the data are consistent with the model (the model provides an acceptable fit). It's also common to report the "reduced Chi-squared," χ²/df. Values close to 1 suggest a good fit, values much larger than 1 suggest a poor fit, and values much smaller than 1 might indicate overestimated errors or overfitting (though interpretation depends heavily on correct df and error assumptions).

**Processing Step 5: Visualization (Optional):** Plot the observed counts (perhaps with Poisson error bars, `sqrt(O)`) and the expected counts versus the bin centers or ranges. Visually inspecting where the largest discrepancies occur can provide insight into *why* a model might be a poor fit.

**Output, Testing, and Extension:** The output includes the calculated χ² statistic, the determined degrees of freedom (`df`), the p-value, and the conclusion regarding the model's goodness-of-fit. An optional plot visualizes the comparison. **Testing** involves verifying the calculation of `df` based on how the `expected_counts` were generated. Double-check that expected counts meet the minimum threshold (e.g., ≥ 5). Test with data where observed perfectly matches expected (χ² should be 0, p-value 1). Test with data having large discrepancies (χ² should be large, p-value small). **Extensions:** (1) Incorporate observational uncertainties (e.g., Poisson errors `sqrt(O)`) into the χ² calculation, potentially using a modified statistic like `Σ [(O - E)² / σ²]` where σ² includes uncertainty from both O and E (if model has uncertainty). (2) If model parameters *were* fitted, use the best-fit χ² value along with `df` to assess goodness-of-fit. (3) Compare the goodness-of-fit (e.g., using χ²/df or p-values, carefully considering df) for different theoretical models providing different `expected_counts` for the same `observed_counts`. (4) If dealing with low counts per bin, consider using alternatives like Fisher's exact test (for 2x2 tables) or Monte Carlo simulations to assess significance instead of relying on the χ² approximation.

```python
# --- Code Example: Application 15.B ---
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

print("Testing Model Predictions for Cluster Counts (Chi-squared GoF):")

# Step 1: Define Observed and Expected Counts in bins (e.g., redshift bins)
# Simulate data for demonstration
bins_edges = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1])
bin_centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])
k_bins = len(bin_centers)

# Assume these observed counts came from a survey
observed_counts = np.array([25, 45, 30, 15, 8])
# Assume these expected counts came from a specific LCDM model (fixed parameters)
expected_counts = np.array([28.5, 41.0, 28.0, 16.5, 10.0]) 

# Basic check: Ensure total counts are reasonably similar (not required by test itself)
print(f"\nObserved Counts: {observed_counts} (Total: {np.sum(observed_counts)})")
print(f"Expected Counts (Model): {expected_counts} (Total: {np.sum(expected_counts):.1f})")

# Check if expected counts are too low (e.g., < 5) - not an issue here
if np.any(expected_counts < 5):
     print("\nWarning: Some expected counts are low (< 5), Chi-squared approx might be less reliable.")

# Step 2: Perform Chi-squared Test
print("\nPerforming Chi-squared Goodness-of-Fit test...")
chi2_stat, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

# Step 3: Determine Degrees of Freedom
# Assume the model parameters were *fixed* beforehand (not fitted to these counts)
p_params_fitted = 0 
df = k_bins - 1 - p_params_fitted 
print(f"  Chi2 statistic = {chi2_stat:.3f}")
print(f"  Degrees of freedom (k-1-p) = {k_bins} - 1 - {p_params_fitted} = {df}")
print(f"  P-value = {p_value:.4f}")

# Step 4: Interpret Results
alpha = 0.05
print(f"\nDecision (alpha = {alpha}):")
if p_value <= alpha:
    print(f"  Reject H0. The observed counts are statistically inconsistent")
    print(f"  with the model predictions (p <= {alpha}).")
else:
    print(f"  Fail to reject H0. The observed counts are consistent")
    print(f"  with the model predictions (p > {alpha}).")
    
# Calculate reduced Chi-squared
reduced_chi2 = chi2_stat / df
print(f"\nReduced Chi-squared (Chi2/df) = {reduced_chi2:.3f}")
print("  (Values near 1 indicate a good fit for the correct df)")

# Step 5: Visualization (Optional)
print("\nGenerating comparison plot...")
fig, ax = plt.subplots(figsize=(7, 5))
# Calculate Poisson errors for observed counts (sqrt(N))
observed_err = np.sqrt(observed_counts) 
ax.errorbar(bin_centers, observed_counts, yerr=observed_err, fmt='o', color='black', 
            label='Observed Counts', capsize=3, markersize=5)
ax.step(bins_edges, np.concatenate(([expected_counts[0]], expected_counts)), where='post', 
        color='red', label='Expected Counts (Model)')
# Alternative: ax.bar(bin_centers, expected_counts, width=np.diff(bins_edges), alpha=0.5, color='red')

ax.set_xlabel("Redshift Bin Center (Example)")
ax.set_ylabel("Number of Clusters")
ax.set_title("Galaxy Cluster Counts: Observed vs. Model")
ax.legend()
ax.grid(True, alpha=0.4)
ax.set_ylim(bottom=0) # Counts cannot be negative

fig.tight_layout()
# plt.show()
print("Comparison plot generated.")
plt.close(fig)
print("-" * 20)

# Explanation: This application simulates observed cluster counts in redshift bins 
# and expected counts from a fixed cosmological model.
# 1. It defines the observed and expected counts arrays.
# 2. It performs the Chi-squared test using `stats.chisquare`.
# 3. It determines the degrees of freedom `df`, crucially assuming *no* model 
#    parameters were fitted to these specific observed counts (p=0).
# 4. It interprets the resulting p-value to decide if the observed counts are 
#    consistent with the model's prediction at the alpha=0.05 level. It also 
#    calculates the reduced Chi-squared.
# 5. It generates a plot comparing the observed counts (with error bars) to the 
#    expected counts from the model, visually showing the goodness-of-fit.
```

**Summary**

This chapter introduced the fundamental principles and common techniques of statistical hypothesis testing, a critical framework for making decisions about scientific claims based on sample data in the presence of uncertainty. It began by outlining the core logic: formulating a specific null hypothesis (H₀, typically representing "no effect" or consistency with a default model) and an alternative hypothesis (H₁, representing the effect or difference of interest), then calculating the probability (p-value) of observing data as extreme as or more extreme than the actual sample data *if the null hypothesis were true*. The concepts of test statistics (quantifying deviation from H₀), p-values (quantifying the evidence against H₀), significance levels (α, the threshold for rejecting H₀, typically 0.05), and the potential for Type I (false positive) and Type II (false negative) errors were explained.

Several widely used hypothesis tests were then detailed, along with their implementation in `scipy.stats`. These included t-tests (`ttest_1samp`, `ttest_ind`) for comparing sample means to hypothesized values or comparing means between two groups; Chi-squared tests (`chisquare`, `chi2_contingency`) for comparing observed counts or frequencies in categories against expected values (goodness-of-fit) or testing for independence between categorical variables; and the non-parametric Kolmogorov-Smirnov (K-S) test (`kstest`, `ks_2samp`) for comparing the overall shape of a sample distribution against a theoretical CDF or comparing the distributions of two independent samples without assuming normality. The application of Chi-squared and K-S tests specifically for assessing goodness-of-fit between data and models was also discussed, highlighting the importance of correctly determining degrees of freedom for the Chi-squared test and the caveat regarding estimated parameters for the K-S test. Finally, the chapter cautioned against common pitfalls, particularly the multiple testing problem (inflated false positives when performing many tests, requiring corrections like Bonferroni or FDR control) and p-hacking (questionable research practices aimed at achieving statistical significance), emphasizing the need for transparency, preregistration, and careful interpretation beyond simple p-value thresholds.

---

**References for Further Reading:**

1.  **Wall, J. V., & Jenkins, C. R. (2012).** *Practical Statistics for Astronomers* (2nd ed.). Cambridge University Press. [https://doi.org/10.1017/CBO9781139168491](https://doi.org/10.1017/CBO9781139168491)
    *(Provides accessible explanations of hypothesis testing concepts (p-values, significance) and common tests like t-tests, Chi-squared, and K-S tests in an astronomical context.)*

2.  **Feigelson, E. D., & Babu, G. J. (2012).** *Modern Statistical Methods for Astronomy: With R Applications*. Cambridge University Press. [https://doi.org/10.1017/CBO9781139179009](https://doi.org/10.1017/CBO9781139179009)
    *(Offers in-depth coverage of hypothesis testing principles, various parametric and non-parametric tests including those discussed, and goodness-of-fit assessment.)*

3.  **Ivezić, Ž., Connolly, A. J., VanderPlas, J. T., & Gray, A. (2014).** *Statistics, Data Mining, and Machine Learning in Astronomy*. Princeton University Press. (Relevant chapters often available online, e.g., Chapter 4 on hypothesis testing: [http://press.princeton.edu/titles/10159.html](http://press.princeton.edu/titles/10159.html))
    *(Provides a rigorous treatment of hypothesis testing within a broader statistical framework for astronomy, including discussions of p-values, significance, and common tests.)*

4.  **The SciPy Community. (n.d.).** *SciPy Reference Guide: Statistical functions (scipy.stats)*. SciPy. Retrieved January 16, 2024, from [https://docs.scipy.org/doc/scipy/reference/stats.html](https://docs.scipy.org/doc/scipy/reference/stats.html)
    *(Official documentation for `scipy.stats`, providing details on the implementation and usage of hypothesis tests like `ttest_1samp`, `ttest_ind`, `chisquare`, `ks_2samp`, `kstest`, `anderson_ksamp`, etc., relevant to Sec 15.3-15.5.)*

5.  **Benjamini, Y., & Hochberg, Y. (1995).** Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. *Journal of the Royal Statistical Society, Series B (Methodological)*, *57*(1), 289–300. [http://www.jstor.org/stable/2346101](http://www.jstor.org/stable/2346101)
    *(The seminal paper introducing the Benjamini-Hochberg procedure for controlling the False Discovery Rate (FDR), a key technique for addressing the multiple testing problem discussed in Sec 15.6.)*
