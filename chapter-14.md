**Chapter 14: Descriptive Statistics and Error Analysis**

After establishing the foundations of probability theory and common distributions in the previous chapter, we now turn to the practical task of summarizing and characterizing datasets using **descriptive statistics**. When confronted with a collection of measurements – whether stellar magnitudes, galaxy redshifts, photon arrival times, or simulation particle properties – we need quantitative tools to describe its central tendency, spread, and shape, and to understand the relationships between different measured quantities. This chapter introduces fundamental descriptive statistics, covering measures of location (mean, median, mode) and dispersion (variance, standard deviation, interquartile range). We will explore how to calculate these using Python libraries like NumPy and SciPy. Crucially, we also delve into the essential topic of **error analysis**, discussing how to propagate measurement uncertainties through calculations to estimate the uncertainty in derived quantities. Techniques for visualizing distributions, specifically histograms and Kernel Density Estimates (KDEs), are presented as vital tools for exploration. We then discuss measures of correlation and covariance to quantify relationships between variables. Finally, recognizing that real astrophysical data often contains outliers, we introduce robust statistical estimators (like median absolute deviation and sigma clipping using `astropy.stats`) that provide reliable summaries even in the presence of non-Gaussian behavior or contaminants.

**14.1 Measures of Central Tendency (Mean, Median, Mode)**

When analyzing a dataset, one of the first questions we typically ask is: "What is a typical or central value?" Measures of **central tendency** aim to provide a single number that summarizes the center or location of a distribution of data points. The three most common measures are the mean, the median, and the mode, each providing a different perspective on what constitutes the "center" and exhibiting different sensitivities to the data's characteristics. Understanding their definitions, properties, and appropriate use cases is fundamental to descriptive statistics.

The **arithmetic mean**, often simply called the "mean" or "average," is the most widely known measure of central tendency. It is calculated by summing all the values in a dataset and dividing by the total number of values. If we have a dataset {x₁, x₂, ..., x<0xE2><0x82><0x99>}, the sample mean (denoted x̄) is calculated as: x̄ = (1/n) * Σ<0xE2><0x82><0x99> x<0xE1><0xB5><0xA2>. In Python, the mean is easily calculated using `numpy.mean(data_array)`. The mean utilizes all data points in its calculation and is mathematically tractable, forming the basis for many statistical tests and models.

However, the mean has a significant drawback: it is highly sensitive to **outliers** or extreme values. A single very large or very small value in the dataset can disproportionately pull the mean towards it, potentially making the mean a poor representation of the "typical" value for the bulk of the data, especially if the distribution is skewed. For instance, in a dataset of galaxy luminosities, a single extremely bright quasar included in the sample could dramatically inflate the mean luminosity, giving a misleading impression of the typical galaxy brightness.

The **median** provides a more robust alternative measure of central tendency, less affected by outliers. The median is the middle value of a dataset when the values are sorted in ascending order. If the dataset has an odd number of points (n), the median is the value at position (n+1)/2. If the dataset has an even number of points (n), the median is typically defined as the average of the two middle values (at positions n/2 and n/2 + 1). In Python, the median is calculated using `numpy.median(data_array)`. Because the median only depends on the rank order of the values, changing even extreme outlier values will not affect it as long as they remain on the same side of the central value(s). This makes the median a preferred measure of central location for skewed distributions or datasets potentially contaminated by outliers.

For example, consider the dataset {1, 2, 3, 4, 100}. The mean is (1+2+3+4+100)/5 = 110/5 = 22, which is heavily influenced by the outlier 100 and doesn't represent the cluster of values 1-4 well. The sorted dataset is {1, 2, 3, 4, 100}. The median is the middle value (the 3rd value), which is 3. This value is much more representative of the typical magnitude of the majority of the data points.

The **mode** is the value that appears most frequently in a dataset. A dataset can have one mode (unimodal), two modes (bimodal), or multiple modes (multimodal). The mode is particularly useful for describing categorical data (e.g., the most common morphological type of galaxy in a sample) but can also be applied to numerical data, especially discrete data. For continuous data, finding the mode often involves identifying the peak(s) of the distribution's histogram or probability density function. In Python, `scipy.stats.mode(data_array)` can be used to find the mode(s) and their count(s). The mode is insensitive to outliers but can be less informative for distributions without a clear peak or with multiple peaks of similar height. It's also not always uniquely defined.

Choosing the appropriate measure of central tendency depends on the nature of the data distribution and the specific question being asked. For symmetric distributions without significant outliers (like approximately Gaussian data), the mean, median, and mode will be close together, and the mean is often preferred due to its mathematical properties. For skewed distributions or data with outliers, the median often provides a more representative measure of the typical value. The mode is most relevant when identifying the most common category or value.

```python
# --- Code Example 1: Calculating Mean, Median, Mode ---
import numpy as np
from scipy import stats

print("Calculating measures of central tendency:")

# Sample data (slightly skewed with an outlier)
data = np.array([15.1, 15.3, 15.0, 15.5, 15.2, 15.4, 14.9, 15.3, 15.1, 18.5]) 
print(f"\nSample Data: {data}")

# Calculate Mean
mean_value = np.mean(data)
print(f"\nMean: {mean_value:.3f}") 
print("  (Sensitive to the outlier 18.5)")

# Calculate Median
median_value = np.median(data)
print(f"\nMedian: {median_value:.3f}") 
# Sorted: [14.9, 15.0, 15.1, 15.1, 15.2, 15.3, 15.3, 15.4, 15.5, 18.5]
# Average of 5th (15.2) and 6th (15.3) elements = 15.25
print("  (Robust to the outlier 18.5)")

# Calculate Mode
# scipy.stats.mode returns mode value(s) and count(s)
# Use keepdims=False for simpler output if expecting single mode
# Use axis=None to compute mode of flattened array
try:
    # Note: mode behavior might change in newer SciPy, may need keepdims=True/False
    mode_result = stats.mode(data, axis=None, keepdims=False) 
    print(f"\nMode: {mode_result.mode}") 
    print(f"  Count: {mode_result.count}") 
    # Values 15.1 and 15.3 both appear twice
    # Older SciPy might only return one, newer might return array if multimodal
    if isinstance(mode_result.mode, np.ndarray) and len(mode_result.mode) > 1:
        print("  (Dataset is multimodal)")
    elif np.isscalar(mode_result.mode):
         # Check if count is low, indicating mode might not be meaningful
         if mode_result.count <= 1 and len(data) > 1 :
              print("  (Mode may not be meaningful - no value repeated significantly)")
    else: # Handle potential different return types
        print(f"  Mode value: {mode_result.mode}")


except NotImplementedError:
    print("\nMode calculation might require specific handling for multimodal data in this SciPy version.")
except Exception as e:
     print(f"\nError calculating mode: {e}")

print("-" * 20)

# Explanation: This code defines a small NumPy array `data` containing mostly values
# around 15, but with one outlier at 18.5. 
# It calculates the mean using `np.mean()`, showing it's pulled upwards by the outlier.
# It calculates the median using `np.median()`, which gives a value (15.25) closer 
# to the bulk of the data, demonstrating its robustness.
# It calculates the mode using `scipy.stats.mode()`. The output shows the mode value 
# (which might be 15.1 or 15.3, or both depending on the SciPy version and exact 
# implementation handling ties) and how many times it occurred. It includes checks 
# for multimodality or cases where the mode isn't very informative.
```

In astrophysical contexts, the choice matters. When reporting the average magnitude of stars in a cluster potentially contaminated by non-members, the median magnitude is often more reliable than the mean. When describing the typical redshift of galaxies in a cluster known to contain foreground/background interlopers, the median redshift (or a more robust estimator like the biweight location, Sec 14.6) is preferred. However, when calculating quantities based on physical conservation laws (like the center of mass, which involves averaging positions weighted by mass), the mean is the appropriate measure. Always consider the properties of your data and the goals of your analysis when choosing a measure of central tendency.

**14.2 Measures of Dispersion (Variance, Std Dev, IQR)**

While measures of central tendency describe the typical location of data, **measures of dispersion** (or spread) quantify how much the individual data points vary or scatter around that central value. Describing the spread is just as important as describing the center for characterizing a distribution. A dataset tightly clustered around its mean is very different from one spread out over a wide range, even if they share the same mean or median. Common measures of dispersion include the range, variance, standard deviation, and interquartile range (IQR).

The simplest measure is the **range**, calculated as the difference between the maximum and minimum values in the dataset (`max(data) - min(data)`). While easy to compute, the range is extremely sensitive to outliers, as it depends only on the two most extreme values. It provides little information about the spread of the bulk of the data and is generally not a robust measure.

The **variance** (σ²) provides a more statistically meaningful measure of spread based on the average squared deviation of each data point from the mean (μ or x̄). For a population, σ² = E[(X - μ)²]. For a sample dataset {x₁, ..., x<0xE2><0x82><0x99>} with sample mean x̄, the **sample variance** (s²) is calculated as: s² = [1 / (n - 1)] * Σ<0xE2><0x82><0x99> (x<0xE1><0xB5><0xA2> - x̄)². Note the use of `n - 1` in the denominator (Bessel's correction) which provides an unbiased estimator of the true population variance from the sample. In Python, `numpy.var(data_array, ddof=1)` calculates the sample variance (`ddof=1` specifies the `n-1` denominator; the default `ddof=0` calculates the population variance using `n`). The variance has units that are the square of the original data units (e.g., mag², (km/s)²), which can sometimes be awkward to interpret directly.

The **standard deviation** (σ or s) is simply the square root of the variance: s = sqrt(s²). It is arguably the most commonly used measure of dispersion because it has the same units as the original data, making it more directly interpretable as a typical deviation from the mean. For a Gaussian distribution, approximately 68% of the data falls within one standard deviation (μ ± σ) of the mean, 95% within two (μ ± 2σ), and 99.7% within three (μ ± 3σ). In Python, `numpy.std(data_array, ddof=1)` calculates the sample standard deviation (again, use `ddof=1` for the unbiased estimate from a sample).

Like the mean, both the variance and standard deviation are calculated using all data points and are therefore sensitive to outliers. Extreme values contribute heavily to the sum of squared deviations, potentially inflating the variance and standard deviation, giving an exaggerated sense of the spread of the majority of the data if outliers are present.

A more robust measure of dispersion, analogous to the median, is the **Interquartile Range (IQR)**. It is based on **quartiles**, which divide the sorted data into four equal parts. The first quartile (Q1) is the value below which 25% of the data lies (the 25th percentile). The second quartile (Q2) is the median (50th percentile). The third quartile (Q3) is the value below which 75% of the data lies (the 75th percentile). The IQR is defined as the difference between the third and first quartiles: IQR = Q3 - Q1. This range encompasses the central 50% of the data and is completely insensitive to outlier values beyond Q1 and Q3. In Python, quartiles (and other percentiles) can be calculated using `numpy.percentile(data_array, [25, 50, 75])`, and the IQR can be calculated directly using `scipy.stats.iqr(data_array)`.

The IQR is often used in conjunction with the median to provide a robust summary of the data's location and spread. It is particularly useful for visualizing distributions via box plots, where the box typically spans the IQR. While robust, the IQR only describes the spread of the central half of the data and ignores information in the tails.

```python
# --- Code Example 1: Calculating Measures of Dispersion ---
import numpy as np
from scipy import stats

print("Calculating measures of dispersion:")

# Use the same data as Sec 14.1 example (with outlier)
data = np.array([15.1, 15.3, 15.0, 15.5, 15.2, 15.4, 14.9, 15.3, 15.1, 18.5]) 
# Data without outlier for comparison
data_no_outlier = data[data < 17.0] 

print(f"\nSample Data: {data}")
print(f"Data without Outlier: {data_no_outlier}")

# --- Range ---
range_with_outlier = np.ptp(data) # Peak-to-peak = max - min
range_no_outlier = np.ptp(data_no_outlier)
print(f"\nRange (with outlier): {range_with_outlier:.2f}")
print(f"Range (no outlier): {range_no_outlier:.2f} (Much smaller)")

# --- Variance and Standard Deviation (Sample versions, ddof=1) ---
var_with_outlier = np.var(data, ddof=1)
std_with_outlier = np.std(data, ddof=1)
var_no_outlier = np.var(data_no_outlier, ddof=1)
std_no_outlier = np.std(data_no_outlier, ddof=1)

print(f"\nSample Variance (with outlier): {var_with_outlier:.3f}")
print(f"Sample Std Dev (with outlier): {std_with_outlier:.3f}")
print(f"Sample Variance (no outlier): {var_no_outlier:.3f}")
print(f"Sample Std Dev (no outlier): {std_no_outlier:.3f} (Much smaller)")
print("  (Std Dev highly affected by the outlier)")

# --- Quartiles and IQR ---
q1_with, med_with, q3_with = np.percentile(data, [25, 50, 75])
iqr_with = q3_with - q1_with 
# Alternatively: iqr_with = stats.iqr(data) # interpolation='midpoint' might differ slightly from percentile
q1_no, med_no, q3_no = np.percentile(data_no_outlier, [25, 50, 75])
iqr_no = q3_no - q1_no 
# Alternatively: iqr_no = stats.iqr(data_no_outlier)

print(f"\nIQR (with outlier): {iqr_with:.3f} (Q1={q1_with:.3f}, Q3={q3_with:.3f})")
print(f"IQR (no outlier): {iqr_no:.3f} (Q1={q1_no:.3f}, Q3={q3_no:.3f})")
print("  (IQR is much less affected by the outlier)")
print("-" * 20)

# Explanation: This code uses the same sample data with an outlier (18.5).
# It calculates the Range (`np.ptp`), Sample Variance (`np.var(ddof=1)`), and 
# Sample Standard Deviation (`np.std(ddof=1)`) for both the original data and 
# data with the outlier removed. This clearly shows how sensitive these measures 
# are to the single extreme value.
# It then calculates the 1st and 3rd quartiles (25th and 75th percentiles) using 
# `np.percentile` and computes the Interquartile Range (IQR = Q3 - Q1). It compares 
# the IQR for the data with and without the outlier, demonstrating that the IQR, 
# describing the spread of the central 50%, is much more robust to the outlier. 
# (Note: `scipy.stats.iqr` might give slightly different results due to different 
# interpolation methods for quartiles in even-sized datasets compared to np.percentile).
```

Similar to central tendency, the choice of dispersion measure depends on the data and goals. Standard deviation is widely used, mathematically convenient, and easily interpretable for Gaussian-like distributions (e.g., quoting measurement errors as ±1σ). However, for skewed data or data with outliers, the standard deviation can be misleadingly large. In such cases, the IQR provides a more robust measure of the spread of the bulk of the data. Other robust measures like the Median Absolute Deviation (MAD) will be discussed in Section 14.6. Providing both a robust measure (median, IQR/MAD) and non-robust measures (mean, std dev) can sometimes give a more complete picture of the distribution's location and spread, especially highlighting the influence of outliers or skewness. Remember that variance and standard deviation measure spread *around the mean*, while IQR measures the spread of the central 50% *independent of the mean*.

**14.3 Handling Uncertainties and Error Propagation**

Astrophysical measurements are never perfectly precise; they always carry some degree of **uncertainty** or **error**. This uncertainty arises from various sources, including random noise (e.g., photon shot noise from the source and background, detector readout noise), systematic effects (e.g., imperfect calibration, instrument artifacts, model approximations), and statistical limitations (e.g., finite sample size). Reporting a measurement without its associated uncertainty (e.g., Flux = 10.5 ± 0.2 mJy) is scientifically incomplete. Furthermore, when we use these uncertain measurements in calculations to derive new quantities, the uncertainties from the input measurements propagate through the calculation, leading to an uncertainty in the final derived result. **Error propagation** is the process of estimating the uncertainty in a derived quantity based on the uncertainties of the input variables and the mathematical relationship between them.

The most common framework for error propagation involves using calculus, specifically first-order Taylor expansions, assuming the errors are relatively small and uncorrelated. Let y be a quantity derived from one or more measured variables x₁, x₂, ... via a function y = f(x₁, x₂, ...). Let σ<0xE1><0xB5><0xA2> represent the standard deviation (uncertainty) associated with the measurement of x<0xE1><0xB5><0xA2>. If the uncertainties in x<0xE1><0xB5><0xA2> are small and **uncorrelated**, the variance in the derived quantity y (σ<0xE1><0xB5><0xA7>²) can be approximated by:

σ<0xE1><0xB5><0xA7>² ≈ Σ<0xE1><0xB5><0xA2> [ (∂f / ∂x<0xE1><0xB5><0xA2>)² * σ<0xE1><0xB5><0xA2>² ]

where (∂f / ∂x<0xE1><0xB5><0xA2>) is the partial derivative of the function f with respect to the variable x<0xE1><0xB5><0xA2>, evaluated at the measured values of x<0xE1><0xB5><0xA2>. The standard deviation (uncertainty) in y is then σ<0xE1><0xB5><0xA7> = sqrt(σ<0xE1><0xB5><0xA7>²). This formula essentially adds the contributions of each input uncertainty to the output variance in quadrature, weighted by how sensitive the function f is to changes in each input variable (the partial derivative).

From this general formula, we can derive rules for common operations:
*   **Addition/Subtraction:** If y = a*x₁ ± b*x₂, then σ<0xE1><0xB5><0xA7>² ≈ (a*σ₁)² + (b*σ₂)² (where σ₁ = σ<0xE2><0x82><0x99>₁, etc.). Note that errors add in quadrature even for subtraction.
*   **Multiplication/Division:** If y = a * x₁ * x₂, then (σ<0xE1><0xB5><0xA7>/y)² ≈ (σ₁/x₁)² + (σ₂/x₂)² (relative errors add in quadrature). If y = a * x₁ / x₂, the same rule applies: (σ<0xE1><0xB5><0xA7>/y)² ≈ (σ₁/x₁)² + (σ₂/x₂)² .
*   **Power Law:** If y = a * xⁿ, then (σ<0xE1><0xB5><0xA7>/y) ≈ |n| * (σ<0xE2><0x82><0x99>/x) (relative error is multiplied by the absolute value of the exponent).
*   **General Function of One Variable:** If y = f(x), then σ<0xE1><0xB5><0xA7> ≈ |df/dx| * σ<0xE2><0x82><0x99>. For example, if y = ln(x), then σ<0xE1><0xB5><0xA7> ≈ (1/x) * σ<0xE2><0x82><0x99> = σ<0xE2><0x82><0x99>/x (uncertainty in ln(x) is approx the relative uncertainty in x). If y = exp(x), then σ<0xE1><0xB5><0xA7> ≈ exp(x) * σ<0xE2><0x82><0x99> = y * σ<0xE2><0x82><0x99>.

These formulas allow us to estimate the uncertainty in quantities derived through simple calculations. They are based on a first-order approximation and assume the uncertainties σ<0xE1><0xB5><0xA2> are small enough that the function f is approximately linear over the range defined by the uncertainties. They also crucially assume the errors in the input variables x<0xE1><0xB5><0xA2> are **uncorrelated**.

If the errors in the input variables are **correlated**, the general formula becomes more complex, involving covariance terms:
σ<0xE1><0xB5><0xA7>² ≈ Σ<0xE1><0xB5><0xA2> [ (∂f / ∂x<0xE1><0xB5><0xA2>)² * σ<0xE1><0xB5><0xA2>² ] + 2 * Σ<0xE1><0xB5><0xA2><j [ (∂f / ∂x<0xE1><0xB5><0xA2>) * (∂f / ∂x<0xE2><0x82><0x97>) * Cov(x<0xE1><0xB5><0xA2>, x<0xE2><0x82><0x97>) ]
where Cov(x<0xE1><0xB5><0xA2>, x<0xE2><0x82><0x97>) is the covariance between x<0xE1><0xB5><0xA2> and x<0xE2><0x82><0x97>. Positive correlation increases the resulting variance, while negative correlation decreases it. Handling correlated errors requires knowledge of the covariance matrix of the input variables.

Manually applying these propagation formulas, especially for complex functions or when correlations are present, can be tedious and prone to algebraic mistakes. Several Python packages exist to automate this process. The **`uncertainties`** package is particularly popular. It allows you to define numbers with associated uncertainties (e.g., `x = ufloat(value, std_dev)`). You can then perform calculations directly on these "ufloat" objects, and the package automatically propagates the uncertainties (including correlations if specified) using the standard formulas (or exact methods for some functions).

```python
# --- Code Example 1: Manual Error Propagation ---
import numpy as np

print("Manual Error Propagation Example:")

# Measured quantities with uncorrelated uncertainties (value, error)
# Example: Flux density F and angular size theta to calculate surface brightness SB ~ F / theta^2
F = 10.0 # mJy
sigma_F = 0.5 # mJy
theta = 2.0 # arcsec
sigma_theta = 0.1 # arcsec

print(f"Inputs: F = {F} +/- {sigma_F} mJy")
print(f"        theta = {theta} +/- {sigma_theta} arcsec")

# Calculate derived quantity: SB = F / theta^2 (ignore constant factors)
SB = F / (theta**2)

# --- Propagate errors ---
# Partial derivatives:
# dSB/dF = 1 / theta^2
# dSB/dtheta = -2 * F / theta^3

# Calculate variance contributions
var_F_contrib = ( (1 / theta**2)**2 ) * (sigma_F**2)
var_theta_contrib = ( (-2 * F / theta**3)**2 ) * (sigma_theta**2)

# Total variance (assuming uncorrelated errors)
sigma_SB_sq = var_F_contrib + var_theta_contrib
# Total standard deviation
sigma_SB = np.sqrt(sigma_SB_sq)

print(f"\nDerived Surface Brightness (SB = F / theta^2): {SB:.3f}")
print(f"Calculated Uncertainty in SB (sigma_SB): {sigma_SB:.3f}")
print(f"Result: SB = {SB:.3f} +/- {sigma_SB:.3f} (units: mJy / arcsec^2)")

# --- Compare relative errors ---
rel_err_F = sigma_F / F
rel_err_theta = sigma_theta / theta
rel_err_SB = sigma_SB / SB
# Formula for y = x1 / x2^2 gives (sig_y/y)^2 = (sig_x1/x1)^2 + (-2 * sig_x2/x2)^2
rel_err_SB_formula = np.sqrt( rel_err_F**2 + (2 * rel_err_theta)**2 )

print(f"\nRelative error F: {rel_err_F:.3f}")
print(f"Relative error theta: {rel_err_theta:.3f}")
print(f"Calculated relative error SB: {rel_err_SB:.3f}")
print(f"Relative error SB via formula: {rel_err_SB_formula:.3f} (Should match)")
print("-" * 20)

# Explanation: This code manually propagates errors for the calculation SB = F / theta^2.
# 1. It defines the input values (F, theta) and their uncertainties (sigma_F, sigma_theta).
# 2. It calculates the derived value SB.
# 3. It calculates the partial derivatives of SB with respect to F and theta.
# 4. It calculates the contribution to the variance of SB from the uncertainty in F 
#    and from the uncertainty in theta using the formula (df/dx)^2 * sigma_x^2.
# 5. It adds these variances (assuming uncorrelated errors) to get the total variance 
#    in SB, and takes the square root to get the standard deviation sigma_SB.
# 6. It also calculates the relative errors and verifies the result using the direct 
#    formula for relative errors in division and powers.
```

```python
# --- Code Example 2: Using the 'uncertainties' package ---
# Note: Requires installation: pip install uncertainties
import numpy as np
# Need ufloat and umath (for math functions on ufloats)
try:
    from uncertainties import ufloat
    from uncertainties import umath 
    uncertainties_installed = True
except ImportError:
    uncertainties_installed = False
    print("NOTE: 'uncertainties' package not installed. Skipping example.")

print("\nError Propagation using the 'uncertainties' package:")

if uncertainties_installed:
    # Define inputs as ufloat objects (value, std_dev)
    F_u = ufloat(10.0, 0.5)
    theta_u = ufloat(2.0, 0.1)
    print(f"Inputs: F = {F_u}")
    print(f"        theta = {theta_u}")

    # Perform the calculation directly on ufloat objects
    # SB = F / theta^2
    SB_u = F_u / (theta_u**2) 
    
    # The result SB_u automatically contains the propagated uncertainty
    print(f"\nDerived SB (ufloat): {SB_u}")
    print(f"  Value: {SB_u.nominal_value:.3f}")
    print(f"  Uncertainty: {SB_u.std_dev:.3f}")
    
    # Example with logarithm
    # Note: Use umath for functions like log, exp, sin on ufloats
    log_F_u = umath.log10(F_u) 
    print(f"\nLog10(F): {log_F_u}")
    # Expected uncertainty: sigma_F / (F * ln(10)) 
    expected_log_err = (0.5 / (10.0 * np.log(10)))
    print(f"  Expected log10 error: {expected_log_err:.4f} vs {log_F_u.std_dev:.4f}")

else:
     print("Skipping 'uncertainties' package demonstration.")
print("-" * 20)

# Explanation: This code uses the `uncertainties` package.
# 1. It defines F and theta as `ufloat` objects, providing value and standard deviation.
# 2. It performs the calculation `SB_u = F_u / (theta_u**2)` directly using these objects. 
#    The `uncertainties` package automatically calculates the result's nominal value 
#    and propagates the standard deviation using the standard formulas.
# 3. The resulting `SB_u` object stores both the value (`.nominal_value`) and the 
#    propagated uncertainty (`.std_dev`), which match the manual calculation.
# 4. It also shows using `umath.log10` to calculate the log and its propagated error.
```

Another approach for error propagation, especially useful for complex functions where derivatives are hard to compute or when errors are large and non-Gaussian, is using **Monte Carlo simulation**. This involves generating many random samples for each input variable based on its assumed probability distribution (e.g., Gaussian with mean=value, stddev=error, using `rng.normal()`), calculating the derived quantity `y` for each set of random inputs, and then analyzing the distribution of the resulting `y` values. The standard deviation of the `y` distribution provides an estimate of the uncertainty σ<0xE1><0xB5><0xA7>, and the histogram of `y` reveals its full probability distribution, which might be non-Gaussian. This method is computationally more intensive but more general and robust than first-order propagation.

Understanding and correctly applying error propagation is fundamental to quantitative science. It allows us to determine the significance of results (e.g., is a measured difference between two quantities larger than their combined uncertainties?), compare results with theoretical predictions that also have uncertainties, and perform statistically sound model fitting (where uncertainties often act as weights). Using tools like the `uncertainties` package or Monte Carlo methods can significantly simplify and improve the reliability of error propagation in complex Python analyses.

**14.4 Visualizing Distributions (Histograms, KDEs)**

While numerical summaries like mean, median, standard deviation, and IQR provide concise descriptions of a dataset's central tendency and spread, they don't capture the full shape of the distribution. **Visualizing the distribution** is crucial for gaining intuition about the data, identifying potential multimodality (multiple peaks), skewness (asymmetry), outliers, and generally understanding how the data values are spread out. Two primary techniques for visualizing the distribution of a single continuous variable are histograms and Kernel Density Estimates (KDEs).

A **histogram** is created by dividing the range of data values into a series of adjacent intervals or **bins**, and then counting the number of data points that fall into each bin. The result is typically displayed as a bar chart, where the height (or area, if normalized) of each bar represents the frequency or density of data within that bin. Matplotlib's `ax.hist(data, bins=...)` is the standard function for creating histograms (as seen in Sec 13.5, 13.6).

The appearance and interpretation of a histogram are highly sensitive to the choice of **binning** – specifically, the number of bins or the width of the bins. Using too few bins can over-smooth the distribution, hiding important features like multiple peaks. Using too many bins, especially with limited data, can result in a very noisy histogram where bin heights fluctuate dramatically due to small number statistics, making it hard to discern the underlying shape. Choosing the "optimal" number or width of bins is non-trivial and depends on the dataset size and underlying distribution shape.

Several rules-of-thumb or algorithms exist for estimating a reasonable number of bins. Common rules include Sturges' rule (`k = 1 + log₂(n)`), Scott's rule (bin width proportional to σ / n^(1/3)), and the **Freedman-Diaconis rule** (bin width proportional to IQR / n^(1/3)). The Freedman-Diaconis rule is often preferred as it uses the robust IQR and is less sensitive to outliers than rules based on the standard deviation (σ). Python libraries provide tools for automatic bin calculation: `numpy.histogram_bin_edges(data, bins='auto')` attempts several rules, while `astropy.visualization.hist(data, bins='freedman'|'scott'|'blocks'|...)` offers direct implementation of these rules and more advanced methods like Bayesian Blocks (`bins='blocks'`) which creates bins of variable width adapted to data density. Using these automatic methods is often better than arbitrarily choosing a fixed number of bins.

```python
# --- Code Example 1: Histograms with Different Binning ---
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import hist as astro_hist

print("Comparing histogram binning methods:")

# Generate some sample data (e.g., bimodal distribution)
np.random.seed(0)
data1 = np.random.normal(loc=5, scale=1.0, size=300)
data2 = np.random.normal(loc=12, scale=1.5, size=200)
combined_data = np.concatenate((data1, data2))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Fixed number of bins (potentially poor choice)
n_bins_fixed = 10
axes[0].hist(combined_data, bins=n_bins_fixed, density=True, alpha=0.7)
axes[0].set_title(f"Fixed Bins (n={n_bins_fixed})")
axes[0].set_ylabel("Density")
axes[0].grid(True, alpha=0.5)

# Plot 2: Freedman-Diaconis rule (often robust)
# astro_hist calculates and plots directly
astro_hist(combined_data, bins='freedman', ax=axes[1], density=True, alpha=0.7)
axes[1].set_title("Freedman-Diaconis Bins")
axes[1].grid(True, alpha=0.5)

# Plot 3: Sturges' rule (simpler, common default)
astro_hist(combined_data, bins='sturges', ax=axes[2], density=True, alpha=0.7)
axes[2].set_title("Sturges' Bins")
axes[2].grid(True, alpha=0.5)

# Add common X label
for ax in axes: ax.set_xlabel("Data Value")

fig.tight_layout()
# plt.show()
print("Generated histograms with different binning strategies.")
plt.close(fig)
print("-" * 20)

# Explanation: This code generates data from two overlapping Gaussian distributions 
# to create a bimodal sample. It then creates three histograms of this data:
# 1. Using `ax.hist` with an arbitrary fixed number of bins (10).
# 2. Using `astropy.visualization.hist` with `bins='freedman'`, which applies the 
#    robust Freedman-Diaconis rule to determine the bin width.
# 3. Using `astropy.visualization.hist` with `bins='sturges'`, a simpler rule based 
#    only on the number of data points.
# Comparing the plots visually shows how the choice of binning affects the histogram's 
# appearance. The fixed binning might merge the two peaks, while Freedman might 
# resolve them better than Sturges, depending on the data. `density=True` normalizes 
# the histograms so their areas integrate to 1, allowing comparison with PDFs.
```

While histograms are intuitive, their appearance depends on bin placement and width, and they produce a jagged representation of the underlying distribution. **Kernel Density Estimation (KDE)** provides an alternative, smoother visualization. KDE works by placing a smooth kernel function (typically a Gaussian) centered on each data point and then summing these kernels to create a continuous estimate of the probability density function. The smoothness of the resulting KDE curve is controlled by a parameter called the **bandwidth**, analogous to the bin width in a histogram. Choosing an appropriate bandwidth is crucial: too small a bandwidth results in a noisy curve reflecting individual data points, while too large a bandwidth over-smooths the distribution, hiding features.

Python libraries like `scipy.stats.gaussian_kde` and `seaborn.kdeplot` provide tools for performing KDE and plotting the results. `scipy.stats.gaussian_kde` takes the data array and automatically estimates a suitable bandwidth (using rules like Scott's or Silverman's rule-of-thumb), returning an object that can be evaluated at any point `x` to get the estimated density. `seaborn.kdeplot` is a higher-level function built on top of this (or similar methods) that directly generates a smooth KDE plot, often with options for automatic bandwidth selection and easy integration with pandas DataFrames.

```python
# --- Code Example 2: Comparing Histogram and KDE ---
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns # For easy KDE plotting
import pandas as pd # Seaborn often works well with DataFrames

print("Comparing Histogram and Kernel Density Estimate (KDE):")

# Use the same bimodal data from previous example
np.random.seed(0)
data1 = np.random.normal(loc=5, scale=1.0, size=300)
data2 = np.random.normal(loc=12, scale=1.5, size=200)
combined_data = np.concatenate((data1, data2))
data_df = pd.DataFrame({'value': combined_data}) # Put in DataFrame for Seaborn

fig, ax = plt.subplots(figsize=(8, 5))

# Plot histogram (using Freedman bins again)
counts, bin_edges, _ = ax.hist(combined_data, bins='freedman', density=True, 
                               alpha=0.5, label='Histogram (Freedman)')
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# Plot KDE using Seaborn (often easiest)
sns.kdeplot(data=data_df, x='value', ax=ax, color='red', lw=2, 
            label='KDE (Seaborn default BW)')

# # Alternative: Using scipy.stats.gaussian_kde
# try:
#     kde = stats.gaussian_kde(combined_data)
#     x_kde = np.linspace(combined_data.min(), combined_data.max(), 200)
#     pdf_kde = kde(x_kde)
#     ax.plot(x_kde, pdf_kde, color='green', lw=2, linestyle='--', label='KDE (SciPy default BW)')
# except Exception as e:
#     print(f"Scipy KDE failed: {e}")

ax.set_title("Histogram vs. Kernel Density Estimate")
ax.set_xlabel("Data Value")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True, alpha=0.5)

fig.tight_layout()
# plt.show()
print("Generated plot comparing Histogram and KDE.")
plt.close(fig)
print("-" * 20)

# Explanation: This code uses the same bimodal data.
# 1. It plots a histogram using Freedman-Diaconis bins for reference.
# 2. It uses `seaborn.kdeplot` to calculate and plot a smooth Kernel Density Estimate 
#    directly on the same axes. Seaborn automatically handles bandwidth selection 
#    and plotting. (Commented code shows the alternative using `scipy.stats.gaussian_kde` 
#    which requires calculating the PDF values manually over a range and then plotting).
# Comparing the histogram and the KDE curve shows how KDE provides a smoother 
# representation of the distribution's shape, potentially making features like the 
# two modes more apparent than in the binned histogram, although it also involves 
# smoothing choices (the bandwidth).
```

KDEs provide a visually appealing, smooth representation of the distribution, free from binning artifacts. They can be particularly effective at revealing multiple modes or subtle features. However, KDEs also involve assumptions (the choice of kernel, usually Gaussian) and smoothing parameters (bandwidth selection), which can influence the resulting curve. They can also artificially smooth over sharp features or extend density into regions where no data exists, especially in the tails. Both histograms (with careful binning choices) and KDEs are valuable tools for visualizing distributions, often best used in conjunction to provide complementary views of the data.

**14.5 Correlation and Covariance**

Often in astrophysics, we measure multiple properties for each object in a sample, and we are interested in understanding how these properties relate to each other. For instance, does a galaxy's star formation rate correlate with its stellar mass? Is the period of a variable star related to its luminosity? Does the X-ray flux of an AGN correlate with its radio flux? **Correlation** and **covariance** are statistical measures used to quantify the degree and direction of a *linear* association between two random variables (or data columns).

**Covariance** measures how two variables change together. If X and Y are two random variables with means μ<0xE2><0x82><0x99> and μ<0xE1><0xB5><0xA7>, their covariance is defined as Cov(X, Y) = E[ (X - μ<0xE2><0x82><0x99>) * (Y - μ<0xE1><0xB5><0xA7>) ]. For a sample dataset with pairs {(x₁, y₁), ..., (x<0xE2><0x82><0x99>, y<0xE2><0x82><0x99>)} and sample means x̄, ȳ, the sample covariance is calculated as: Cov(x, y) = [1 / (n - 1)] * Σ<0xE1><0xB5><0xA2> (x<0xE1><0xB5><0xA2> - x̄) * (y<0xE1><0xB5><0xA2> - ȳ).
*   If variables tend to increase together (when X is above its mean, Y tends to be above its mean), the product (x<0xE1><0xB5><0xA2> - x̄)(y<0xE1><0xB5><0xA2> - ȳ) will often be positive, leading to a positive covariance.
*   If one variable tends to increase when the other decreases, the product will often be negative, leading to a negative covariance.
*   If there's no consistent linear relationship, positive and negative products tend to cancel out, leading to a covariance near zero.
The magnitude of the covariance depends on the units of the variables X and Y, making it hard to interpret directly as a measure of the *strength* of the relationship. Cov(X, X) is simply the variance of X, Var(X).

The **Pearson correlation coefficient** (often denoted *r*) provides a *normalized* measure of the *linear* correlation between two variables, removing the dependence on units. It is calculated by dividing the covariance by the product of the standard deviations of the two variables:
r = Corr(X, Y) = Cov(X, Y) / (σ<0xE2><0x82><0x99> * σ<0xE1><0xB5><0xA7>)
The value of *r* always lies between -1 and +1, inclusive.
*   r = +1 indicates a perfect positive linear relationship (all points lie on a line with positive slope).
*   r = -1 indicates a perfect negative linear relationship (all points lie on a line with negative slope).
*   r = 0 indicates no *linear* relationship between the variables.
Values between 0 and +1 indicate varying degrees of positive linear correlation (points tend to cluster around a line with positive slope), while values between 0 and -1 indicate varying degrees of negative linear correlation. The closer |r| is to 1, the stronger the linear association.

It is crucial to emphasize that the Pearson correlation coefficient *r* only measures the strength and direction of a **linear** relationship. Two variables can have a strong non-linear relationship (e.g., a parabolic relationship) but still have a correlation coefficient close to zero if the overall linear trend is flat. Therefore, **correlation does not imply causation**, and a low correlation does not necessarily mean there is no relationship at all. Visualizing the relationship with a scatter plot (Sec 6.4) is always essential alongside calculating correlation coefficients.

In Python, `numpy.cov(x, y, ddof=1)` calculates the 2x2 covariance matrix between two 1D arrays `x` and `y`. The diagonal elements are the variances (Var(x), Var(y)), and the off-diagonal elements are the covariance Cov(x, y) = Cov(y, x). `numpy.corrcoef(x, y)` calculates the 2x2 correlation matrix. The diagonal elements are always 1 (correlation of a variable with itself), and the off-diagonal elements are the Pearson correlation coefficient *r* between x and y.

```python
# --- Code Example 1: Calculating Covariance and Correlation ---
import numpy as np
import matplotlib.pyplot as plt

print("Calculating Covariance and Pearson Correlation:")

# Generate sample data with some correlation
np.random.seed(1)
n_points = 100
# Variable X
x = np.random.normal(loc=10, scale=2, size=n_points)
# Variable Y correlated with X, plus noise
y = 0.8 * x - 5.0 + np.random.normal(loc=0, scale=1.5, size=n_points) 

print(f"\nGenerated {n_points} pairs of (x, y) data.")

# --- Calculate Covariance Matrix ---
# rowvar=True (default) means rows are variables, columns observations
# rowvar=False means columns are variables (more common if x, y are 1D arrays)
# We stack x and y as columns, so use rowvar=False
# Alternatively, provide a list [x, y] and use default rowvar=True
cov_matrix = np.cov(x, y, ddof=1) 
# cov_matrix = np.cov(np.vstack((x, y)), ddof=1) # Alternative if using vstack
print("\nCovariance Matrix:")
print(cov_matrix)
print(f"  Var(x) = {cov_matrix[0, 0]:.3f} (Compare to np.var(x, ddof=1) = {np.var(x, ddof=1):.3f})")
print(f"  Var(y) = {cov_matrix[1, 1]:.3f} (Compare to np.var(y, ddof=1) = {np.var(y, ddof=1):.3f})")
print(f"  Cov(x, y) = {cov_matrix[0, 1]:.3f}")

# --- Calculate Correlation Matrix ---
corr_matrix = np.corrcoef(x, y)
print("\nCorrelation Matrix:")
print(corr_matrix)
pearson_r = corr_matrix[0, 1]
print(f"  Pearson Correlation Coefficient (r) = {pearson_r:.3f}")
print("  (Indicates a strong positive linear correlation)")

# --- Visualize with Scatter Plot ---
print("\nGenerating scatter plot...")
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(x, y, alpha=0.7)
ax.set_xlabel("Variable X")
ax.set_ylabel("Variable Y")
ax.set_title(f"Scatter Plot (r = {pearson_r:.3f})")
ax.grid(True, alpha=0.5)
fig.tight_layout()
# plt.show()
print("Scatter plot generated.")
plt.close(fig)
print("-" * 20)

# Explanation: This code simulates two correlated variables, x and y (where y 
# is linearly related to x plus some noise).
# 1. It calculates the 2x2 covariance matrix using `np.cov(x, y, ddof=1)`. The diagonal 
#    elements are the variances of x and y, and the off-diagonal element is the 
#    covariance between x and y. 
# 2. It calculates the 2x2 correlation matrix using `np.corrcoef(x, y)`. The off-diagonal 
#    element is the Pearson correlation coefficient `r`.
# 3. It prints the matrices and the extracted `r` value, noting that its value (close to 1) 
#    indicates a strong positive linear relationship, as expected from the data generation.
# 4. It generates a scatter plot of y vs x, visually confirming the linear trend 
#    quantified by the correlation coefficient.
```

When dealing with more than two variables (e.g., columns in a table), `np.cov` and `np.corrcoef` can accept a 2D array where each row (if `rowvar=True`) or each column (if `rowvar=False`) represents a variable. They return an NxN matrix where N is the number of variables, showing all pairwise covariances or correlations. This is useful for exploring relationships within multivariate datasets.

The significance of a calculated correlation coefficient should also be considered. Even random data might show a non-zero sample correlation *r* by chance, especially for small sample sizes. Statistical tests (often based on the t-distribution) can be used to calculate a p-value, representing the probability of observing a correlation as strong as or stronger than the calculated *r* if the true correlation were actually zero. `scipy.stats.pearsonr(x, y)` calculates both the Pearson *r* and the associated p-value. A small p-value (e.g., < 0.05) suggests the observed correlation is statistically significant.

Understanding correlations is vital in astrophysics for identifying physical relationships, building predictive models, and understanding the interdependence of measured parameters. Covariance matrices are also fundamental inputs for multivariate statistical techniques like Principal Component Analysis (PCA, Chapter 23) and for propagating errors when derived quantities depend on multiple, potentially correlated, input variables (as mentioned in Sec 14.3). Always remember to visualize data with scatter plots alongside calculating correlation coefficients to avoid misinterpreting purely linear measures.

**14.6 Robust Statistics (MAD, Sigma Clipping)**

As highlighted in the discussions of the mean (Sec 14.1) and standard deviation (Sec 14.2), classical statistical estimators can be highly sensitive to **outliers** – data points that lie far away from the bulk of the distribution. These outliers might arise from measurement errors (e.g., cosmic ray hits, detector artifacts), genuine but rare astrophysical phenomena (e.g., a flaring star in a quiescent sample), or contamination of a sample (e.g., foreground stars in a galaxy survey). Because methods like the mean and standard deviation involve sums or sums of squares over *all* data points, extreme values can dominate the calculation and yield results that poorly represent the characteristics of the majority of the data. In many astrophysical situations, we need **robust statistical estimators** that are less influenced by such outliers.

The **median** (Sec 14.1) is a prime example of a robust estimator of central tendency. Robust estimators of dispersion are also crucial. The Interquartile Range (IQR, Sec 14.2) provides one robust measure of spread, but another widely used and often more convenient measure is the **Median Absolute Deviation (MAD)**. The MAD is defined as the median of the absolute deviations of the data points from the median of the data itself: MAD = median( |x<0xE1><0xB5><0xA2> - median(x)| ). Like the median, the MAD is resistant to outliers; extreme values have little impact on its calculation.

While the MAD itself is a robust measure of spread, it's often convenient to have a robust estimator of the standard deviation (σ), especially for comparison with σ or for use in methods assuming Gaussian-like behavior. For data drawn from an underlying Gaussian distribution, there is a known relationship between the MAD and σ: σ ≈ 1.4826 * MAD. Therefore, the quantity **MAD_std = 1.4826 * MAD** serves as a robust estimator of the standard deviation. If the data is indeed Gaussian, MAD_std will be very close to the standard deviation calculated directly. If the data contains outliers, MAD_std will provide a much more stable estimate of the spread of the underlying "good" data distribution than the standard deviation, which would be inflated by the outliers. `astropy.stats.median_absolute_deviation(data, axis=None)` computes the MAD, and multiplying by ~1.4826 gives the robust standard deviation estimate.

Another common technique for dealing with outliers, particularly when assuming the bulk of the data follows an approximately Gaussian distribution, is **sigma clipping**. This is an iterative procedure:
1.  Calculate a measure of central tendency (mean or median) and dispersion (standard deviation or MAD_std) for the current dataset.
2.  Identify data points that lie more than a specified number of standard deviations (e.g., 3σ) away from the center.
3.  Remove (or "clip") these outlier points.
4.  Repeat steps 1-3 using the remaining points until no more points are clipped or a maximum number of iterations is reached.
The final central tendency and dispersion calculated from the clipped dataset provide robust estimates, assuming the clipping process correctly removed only the outliers and not part of the underlying distribution's tails.

The `astropy.stats` module provides a powerful function `sigma_clip(data, sigma=3.0, maxiters=5, cenfunc=np.ma.median, stdfunc=stats.mad_std, ...)`. This function performs iterative sigma clipping. Key arguments include:
*   `data`: The input data array (can be a NumPy array or MaskedArray).
*   `sigma`: The number of standard deviations (or equivalent, if using `stdfunc=mad_std`) to use for clipping threshold.
*   `maxiters`: The maximum number of iterations.
*   `cenfunc`: The function used to calculate the center (default is robust median `np.ma.median`). Can be set to `np.ma.mean`.
*   `stdfunc`: The function used to calculate the spread for the sigma threshold (default is robust `stats.mad_std` from `astropy.stats`). Can be set to `np.ma.std`.
The function returns a `MaskedArray` where the clipped (outlier) values are masked (`mask=True`). You can then perform calculations on this masked array (e.g., `np.ma.mean(clipped_data)`) to get robust estimates based only on the unmasked data.

```python
# --- Code Example 1: Calculating MAD and MAD_std ---
import numpy as np
from astropy.stats import median_absolute_deviation, mad_std
from scipy import stats # For comparison IQR

print("Calculating robust dispersion measures (MAD):")

# Use the same data with outlier from previous sections
data = np.array([15.1, 15.3, 15.0, 15.5, 15.2, 15.4, 14.9, 15.3, 15.1, 18.5]) 
data_no_outlier = data[data < 17.0] 

print(f"\nSample Data: {data}")

# Calculate standard deviation (sensitive to outlier)
std_dev = np.std(data, ddof=1)
print(f"\nStandard Deviation (ddof=1): {std_dev:.3f}")

# Calculate MAD using astropy.stats
# axis=None computes for flattened array
mad_value = median_absolute_deviation(data, axis=None) 
print(f"\nMedian Absolute Deviation (MAD): {mad_value:.3f}")

# Calculate robust standard deviation estimate from MAD
robust_std_dev = mad_std(data, axis=None) # Equivalent to 1.4826 * mad_value
print(f"Robust Std Dev estimate (MAD_std): {robust_std_dev:.3f}")
print("  (Much smaller than std dev, less affected by outlier)")

# Compare with IQR / 1.349 (another robust std estimate for Gaussian)
iqr_value = stats.iqr(data) # Get IQR
robust_std_iqr = iqr_value / 1.34896 # Factor for Gaussian
print(f"\nIQR: {iqr_value:.3f}")
print(f"Robust Std Dev estimate (IQR/1.349): {robust_std_iqr:.3f}") 
print("-" * 20)

# Explanation: This code calculates the standard deviation (sensitive to the outlier 18.5) 
# using np.std. It then calculates the Median Absolute Deviation (MAD) using 
# `astropy.stats.median_absolute_deviation`, which is robust. It also calculates 
# the robust estimate of the standard deviation using `astropy.stats.mad_std` 
# (which is just MAD * 1.4826). Comparing `std_dev` and `robust_std_dev` clearly 
# shows that the MAD-based estimate is much smaller and less influenced by the outlier. 
# It also calculates an alternative robust standard deviation estimate based on the IQR 
# for comparison.
```

```python
# --- Code Example 2: Using Sigma Clipping ---
import numpy as np
from astropy.stats import sigma_clip, mad_std
from scipy import stats # For regular std dev comparison
import numpy.ma as ma # For masked array operations

print("Applying iterative sigma clipping:")

# Use the same data with outlier
data = np.array([15.1, 15.3, 15.0, 15.5, 15.2, 15.4, 14.9, 15.3, 15.1, 18.5]) 
print(f"\nOriginal Data: {data}")
print(f"Original Mean: {np.mean(data):.3f}, Original StdDev: {np.std(data, ddof=1):.3f}")

# Apply sigma_clip using robust median and MAD_std
# Clip values more than 3 standard deviations (estimated robustly) away
print("\nApplying sigma_clip (sigma=3, cenfunc=median, stdfunc=mad_std)...")
clipped_data = sigma_clip(
    data, 
    sigma=3.0, 
    maxiters=5, 
    cenfunc=np.ma.median, # Use median for center
    stdfunc=mad_std        # Use MAD_std for spread
    # masked=True is default, returns MaskedArray
)

print(f"\nClipped Data (MaskedArray): {clipped_data}")
print(f"Mask: {clipped_data.mask}") # Should be True only for the outlier 18.5
print(f"Number of clipped points: {np.sum(clipped_data.mask)}")

# Calculate statistics on the unclipped data
mean_clipped = ma.mean(clipped_data) # Use np.ma.mean for masked arrays
median_clipped = ma.median(clipped_data)
std_clipped = ma.std(clipped_data, ddof=1)

print(f"\nStatistics of CLIPPED data:")
print(f"  Mean (clipped): {mean_clipped:.3f}")
print(f"  Median (clipped): {median_clipped:.3f}")
print(f"  Std Dev (clipped, ddof=1): {std_clipped:.3f}")
print("  (Mean and Std Dev are now much closer to values without the outlier)")
print("-" * 20)

# Explanation: This code applies `astropy.stats.sigma_clip` to the data with an outlier.
# It uses robust estimators (`cenfunc=np.ma.median`, `stdfunc=mad_std`) and a 3-sigma 
# threshold. The function returns a NumPy MaskedArray `clipped_data` where the outlier 
# (18.5) is masked out (`mask=True`). 
# It prints the masked array and confirms the outlier was masked. 
# Finally, it calculates the mean, median, and standard deviation of the *unmasked* 
# data using functions from `numpy.ma` (masked array module). These clipped statistics 
# are much closer to the values expected for the bulk of the data, demonstrating the 
# effectiveness of sigma clipping in removing outlier influence when calculating 
# standard descriptive statistics.
```

Choosing between using robust estimators directly (median, MAD_std, IQR) or applying sigma clipping depends on the situation and assumptions. If you suspect non-Gaussian outliers and want a quick, robust summary, median and MAD_std are excellent choices. If you believe the underlying distribution is primarily Gaussian but contaminated by a few distinct outliers, iterative sigma clipping can be very effective at isolating the "good" data before calculating potentially less robust statistics like the mean or standard deviation on the cleaned sample. Other robust estimators, like the biweight location and scale, offer further alternatives often used in astronomical literature, available in `astropy.stats` as `biweight_location` and `biweight_scale`.

In conclusion, robust statistics provide essential tools for summarizing astrophysical data that may deviate from ideal Gaussian assumptions or contain outliers. The median offers a robust alternative to the mean for central tendency. The MAD (and MAD_std) or IQR provide robust measures of spread, less sensitive to extreme values than the standard deviation. Sigma clipping offers an iterative method to identify and remove outliers before applying standard statistical calculations. Employing these robust techniques, particularly those readily available in `astropy.stats`, leads to more reliable characterizations of data distributions commonly encountered in astrophysics.

**Application 14.A: Robust Analysis of Star Cluster Photometry**

**Objective:** This application demonstrates the practical use of robust statistics (Sec 14.6) and distribution visualization (Sec 14.4) for analyzing potentially contaminated photometric data of a star cluster. We will calculate both standard (mean, std dev) and robust (median, MAD_std) statistics for stellar magnitudes or colors, apply sigma clipping to identify outliers (likely non-member field stars), and visualize the results using histograms and a color-magnitude diagram (CMD). Reinforces Sec 14.1, 14.2, 14.4, 14.6.

**Astrophysical Context:** Studying star clusters (open or globular) provides insights into stellar evolution and the formation history of galaxies. Photometric measurements (magnitudes in different filters) for stars in the cluster's vicinity are plotted on a Color-Magnitude Diagram (CMD), which reveals characteristic sequences (main sequence, giant branch, etc.). However, observations often include foreground and background field stars not physically associated with the cluster. These field stars act as outliers or contaminants in the CMD and can significantly bias standard statistical analyses (like calculating the mean cluster color or magnitude spread) if not properly handled.

**Data Source:** A photometric catalog (e.g., FITS table or CSV file, `cluster_phot.csv`) containing magnitudes for stars in a region covering a star cluster. Key columns would be an identifier, RA, Dec, a magnitude (e.g., `G_mag`), and a color index (e.g., `BP_RP_color`), possibly derived from Gaia data or ground-based observations. The sample is assumed to contain both true cluster members and contaminating field stars.

**Modules Used:** `astropy.table.Table` (to read/handle catalog), `numpy` (for calculations), `astropy.stats` (for `sigma_clip`, `mad_std`), `matplotlib.pyplot` (for plotting CMD and histograms).

**Technique Focus:** Applying both standard (`np.mean`, `np.std`) and robust (`np.median`, `astropy.stats.mad_std`) statistics to a potentially contaminated dataset. Using `astropy.stats.sigma_clip` with robust settings (`cenfunc=median`, `stdfunc=mad_std`) to identify and mask likely outliers based on their deviation from the main distribution (e.g., in color or magnitude space, or both). Visualizing the impact of outliers and sigma clipping using histograms (`plt.hist`) and a scatter plot (CMD, Sec 6.4) where outliers are marked differently.

**Processing Step 1: Load Data:** Read the photometric catalog into an Astropy Table `phot_table = Table.read('cluster_phot.csv', ...)`. Identify the relevant magnitude and color columns (e.g., 'G_mag', 'BP_RP').

**Processing Step 2: Calculate Initial Statistics:** Calculate the mean, standard deviation, median, and MAD_std for a key column (e.g., 'BP_RP' color) using the full, unclipped dataset. Print these values, anticipating that the mean and standard deviation might be affected by outliers.

**Processing Step 3: Apply Sigma Clipping:** Use `sigma_clip()` on the data. Since field stars might deviate in both color and magnitude, applying clipping iteratively or on multiple dimensions might be considered (though simple clipping on color or magnitude is often a starting point). Let's clip based on color: `clipped_color = sigma_clip(phot_table['BP_RP'], sigma=3, cenfunc=np.ma.median, stdfunc=mad_std)`. This returns a `MaskedArray` where likely outliers (field stars far from the main cluster sequence color) are masked.

**Processing Step 4: Calculate Clipped Statistics:** Calculate the mean, standard deviation, median, and MAD_std again, but this time using the *masked* array returned by `sigma_clip` (e.g., `mean_clipped = ma.mean(clipped_color)`). Compare these clipped statistics to the initial statistics calculated in Step 2, highlighting the robustness of median/MAD_std and the change in mean/std after clipping.

**Processing Step 5: Visualization:** Create two plots:
    *   A histogram comparison: Plot normalized histograms of the 'BP_RP' color for the full dataset and the sigma-clipped dataset (using `phot_table['BP_RP'][~clipped_color.mask]`) on the same axes to visualize the removal of outliers.
    *   A Color-Magnitude Diagram: Create a scatter plot of 'G_mag' vs 'BP_RP'. Color the points based on the `clipped_color.mask` (e.g., blue for kept points, red for clipped points) to visually identify the outliers removed by the clipping process relative to the cluster sequence. Remember to invert the magnitude axis.

**Output, Testing, and Extension:** The output includes the printed comparison of standard vs. robust statistics before and after clipping, and the two plots (histogram comparison and CMD with outliers marked). **Testing** involves checking if the clipped statistics are significantly different from the unclipped mean/std but potentially close to the unclipped median/MAD_std. Visually inspect the CMD to confirm that the points marked as clipped by `sigma_clip` indeed appear to be field stars away from the main cluster sequence. Adjust the `sigma` parameter in `sigma_clip` (e.g., 2.5 or 4) and observe its effect on the number of clipped points and the resulting statistics. **Extensions:** (1) Apply `sigma_clip` simultaneously to both magnitude and color (e.g., by clipping on residuals from a fitted sequence, or using multi-dimensional clipping techniques if available). (2) Use more sophisticated clustering algorithms (like DBSCAN, Chapter 23) on positional and kinematic data (if available) to identify members before calculating photometric statistics. (3) Calculate robust statistics using biweight estimators (`astropy.stats.biweight_location`, `biweight_scale`) and compare them to median/MAD_std and clipped mean/std.

```python
# --- Code Example: Application 14.A ---
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, MaskedColumn
from astropy.stats import sigma_clip, mad_std
import numpy.ma as ma
import os
import io # To simulate file

print("Robust analysis of simulated cluster photometry:")

# --- Setup: Simulate Cluster + Field Photometry ---
np.random.seed(42)
n_cluster = 300
n_field = 100
# Cluster main sequence + giant branch (simplified)
g_mag_cl = np.random.normal(18, 2, n_cluster)
# Color depends on magnitude for main sequence, constant for giants (crude model)
color_cl = np.piecewise(g_mag_cl, 
                        [g_mag_cl < 17, g_mag_cl >= 17], # Conditions
                        [lambda x: 0.1 + 0.3*(x-14) + np.random.normal(0, 0.05, x.shape), # MS color func
                         lambda x: 1.2 + np.random.normal(0, 0.1, x.shape)]) # GB color func
# Field stars
g_mag_fld = np.random.uniform(14, 24, n_field)
color_fld = np.random.uniform(-0.5, 2.5, n_field)
# Combine
g_all = np.concatenate((g_mag_cl, g_mag_fld))
color_all = np.concatenate((color_cl, color_fld))
# Keep track of true membership (for plotting/verification only)
true_member = np.concatenate((np.ones(n_cluster), np.zeros(n_field))).astype(bool)

# Create Astropy Table
phot_table = Table({
    'G_mag': g_all,
    'BP_RP': color_all,
    'IsMember_Truth': true_member # For visualization verification
})
print(f"\nCreated simulated table with {len(phot_table)} stars (cluster + field).")
# --- End Setup ---

try:
    color_col = phot_table['BP_RP'] # Column to analyze

    # Step 2: Calculate Initial Statistics
    mean_orig = np.mean(color_col)
    std_orig = np.std(color_col, ddof=1)
    median_orig = np.median(color_col)
    mad_std_orig = mad_std(color_col) # Robust std dev estimate
    
    print("\nInitial Statistics for BP_RP color (All Stars):")
    print(f"  Mean: {mean_orig:.3f}")
    print(f"  Std Dev: {std_orig:.3f}")
    print(f"  Median: {median_orig:.3f}")
    print(f"  MAD_std: {mad_std_orig:.3f}")

    # Step 3: Apply Sigma Clipping on Color
    print("\nApplying 3-sigma clipping based on median/MAD_std...")
    # Returns a MaskedArray where outliers are masked (mask=True)
    clipped_color = sigma_clip(color_col, sigma=3.0, maxiters=5, 
                               cenfunc=np.median, stdfunc=mad_std)
    n_clipped = np.sum(clipped_color.mask)
    print(f"  Number of points clipped: {n_clipped} ({n_clipped/len(color_col)*100:.1f}%)")

    # Step 4: Calculate Clipped Statistics
    mean_clip = ma.mean(clipped_color)
    std_clip = ma.std(clipped_color, ddof=1)
    median_clip = ma.median(clipped_color) # Should be similar to original median
    mad_std_clip = mad_std(clipped_color.compressed()) # mad_std needs unmasked array
    
    print("\nStatistics for BP_RP color (After Sigma Clipping):")
    print(f"  Mean (clipped): {mean_clip:.3f}")
    print(f"  Std Dev (clipped): {std_clip:.3f}")
    print(f"  Median (clipped): {median_clip:.3f}")
    print(f"  MAD_std (clipped): {mad_std_clip:.3f}")
    print("  (Mean/Std should be less affected by field stars now)")

    # Step 5: Visualization
    print("\nGenerating plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # --- Histogram Comparison ---
    # Use automatic binning robust to outliers ('freedman')
    ax1.hist(color_col, bins='freedman', density=True, alpha=0.5, 
             label='All Stars')
    # Plot histogram of only the unmasked (clipped) data
    ax1.hist(clipped_color.compressed(), bins='freedman', density=True, alpha=0.7, 
             histtype='step', linewidth=1.5, label='Sigma-Clipped Stars')
    ax1.set_xlabel("BP - RP Color (mag)")
    ax1.set_ylabel("Normalized Density")
    ax1.set_title("Color Distribution Before/After Clipping")
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    # --- Color-Magnitude Diagram ---
    # Plot clipped points (mask is True) in red
    ax2.scatter(phot_table['BP_RP'][clipped_color.mask], phot_table['G_mag'][clipped_color.mask], 
                s=10, alpha=0.5, c='red', label='Clipped (Outliers)')
    # Plot kept points (mask is False) in blue
    ax2.scatter(phot_table['BP_RP'][~clipped_color.mask], phot_table['G_mag'][~clipped_color.mask], 
                s=5, alpha=0.7, c='blue', label='Kept (Cluster?)')
    ax2.set_xlabel("BP - RP Color (mag)")
    ax2.set_ylabel("G Magnitude (mag)")
    ax2.set_title("CMD showing Sigma-Clipped Stars")
    ax2.invert_yaxis()
    ax2.legend()
    ax2.grid(True, alpha=0.4)
    
    fig.tight_layout()
    # plt.show()
    print("Plots generated.")
    plt.close(fig)

except Exception as e:
    print(f"An error occurred: {e}")

print("-" * 20)
```

**Application 14.B: Estimating Uncertainty in Asteroid Rotation Amplitude**

**Objective:** This application provides a practical example of basic error propagation (Sec 14.3), estimating the uncertainty in a quantity derived from measurements that themselves have uncertainties. Specifically, we will calculate the amplitude of an asteroid's light curve (difference between maximum and minimum brightness) and propagate the uncertainties from the individual flux measurements to the derived amplitude.

**Astrophysical Context:** Asteroids often exhibit periodic changes in brightness as they rotate, due to irregular shapes or variations in surface reflectivity (albedo). Measuring the amplitude and period of this variation from a time-series photometry (a light curve) provides information about the asteroid's shape elongation and rotation state. Accurately quantifying the uncertainty in the measured amplitude is important for comparing with shape models or studying populations.

**Data Source:** A light curve dataset for an asteroid, typically a text file or FITS table (`asteroid_lc.txt`) containing columns for Time, Flux (or Magnitude), and Flux_Error (or Magnitude_Error). We will simulate this data.

**Modules Used:** `numpy` (for basic calculations and potentially loading data), `matplotlib.pyplot` (for visualization), possibly `astropy.table.Table` (if reading a table), and optionally the `uncertainties` package for comparison.

**Technique Focus:** Identifying maximum and minimum flux values and their corresponding errors from a noisy light curve. Applying the error propagation formula for subtraction (Rule: If Y = X₁ - X₂, then σ<0xE1><0xB5><0xA7>² ≈ σ₁² + σ₂²) to find the uncertainty in the amplitude A = Flux_max - Flux_min. Visualizing the light curve with error bars. Optionally, comparing the manual propagation with the result from the `uncertainties` package.

**Processing Step 1: Load Data:** Read the time, flux, and flux_error data into NumPy arrays. If loading from text, `np.loadtxt` might be suitable if purely numerical. Let the arrays be `time`, `flux`, `flux_err`.

**Processing Step 2: Find Max/Min Flux and Errors:** Find the index corresponding to the maximum flux value: `idx_max = np.argmax(flux)`. Get the maximum flux `flux_max = flux[idx_max]` and its corresponding error `err_flux_max = flux_err[idx_max]`. Similarly, find the index for the minimum flux: `idx_min = np.argmin(flux)`. Get `flux_min = flux[idx_min]` and `err_flux_min = flux_err[idx_min]`. (Note: This simple max/min assumes the peak/trough are well-sampled; fitting a model might be more robust in reality).

**Processing Step 3: Calculate Amplitude:** The amplitude is simply `amplitude = flux_max - flux_min`.

**Processing Step 4: Propagate Error Manually:** Apply the error propagation rule for subtraction (errors add in quadrature): `variance_amplitude = err_flux_max**2 + err_flux_min**2`. The uncertainty (standard deviation) in the amplitude is `sigma_amplitude = np.sqrt(variance_amplitude)`. Print the amplitude and its calculated uncertainty.

**Processing Step 5: Visualization and Optional Comparison:** Create a plot of the light curve using `plt.errorbar(time, flux, yerr=flux_err, fmt='.', capsize=3, alpha=0.5)` to show the data points and their uncertainties. Optionally, perform the same calculation using the `uncertainties` package: `fmax_u = ufloat(flux_max, err_flux_max)`, `fmin_u = ufloat(flux_min, err_flux_min)`, `amp_u = fmax_u - fmin_u`. Compare `amp_u.nominal_value` and `amp_u.std_dev` with the manually calculated `amplitude` and `sigma_amplitude`.

**Output, Testing, and Extension:** The output includes the calculated light curve amplitude and its propagated uncertainty (e.g., Amplitude = 0.25 ± 0.03 units). The plot shows the light curve data with error bars. **Testing** involves verifying the amplitude calculation is correct. Check if the propagated error `sigma_amplitude` is larger than the individual errors `err_flux_max` and `err_flux_min` (as expected when adding errors in quadrature). Compare the manual result with the `uncertainties` package result; they should match for this simple case. **Extensions:** (1) Fit a sinusoidal or polynomial model to the light curve maxima and minima instead of just taking the single max/min points to get more robust estimates of the peak/trough levels and their uncertainties, then propagate those uncertainties. (2) Convert the flux amplitude to a magnitude amplitude using the formula ΔMag ≈ 1.0857 * (ΔFlux / Flux_avg) and propagate the uncertainties through this non-linear conversion (requiring the formula σ<0xE1><0xB5><0xA7> ≈ |df/dx| * σ<0xE2><0x82><0x99>). (3) Perform a Monte Carlo error propagation: generate many simulated light curves by adding Gaussian noise (scaled by `flux_err`) to the observed `flux` values, find the amplitude for each simulated curve, and calculate the standard deviation of the resulting amplitude distribution. Compare this MC result to the analytical propagation.

```python
# --- Code Example: Application 14.B ---
import numpy as np
import matplotlib.pyplot as plt
# Optional: for comparison
try:
    from uncertainties import ufloat
    uncertainties_installed = True
except ImportError:
    uncertainties_installed = False

print("Estimating Uncertainty in Asteroid Light Curve Amplitude:")

# Step 1: Simulate Light Curve Data (Time, Flux, Flux_Error)
np.random.seed(99)
n_points = 100
time = np.linspace(0, 8, n_points) # 8 hours observation
# Simulate flux variation (e.g., double-peaked sinusoid) + noise
period = 4.0 # hours
amplitude_true = 0.15 # True semi-amplitude
mean_flux = 1.0
flux = mean_flux + amplitude_true * np.sin(2 * np.pi * time / period * 2 + 0.5) 
# Add noise based on flux level (e.g., sqrt(flux) photon noise scaling)
flux_err = 0.02 * np.sqrt(flux / mean_flux) # Example error model
flux_noisy = flux + np.random.normal(0, flux_err, n_points)

print(f"\nGenerated {n_points} light curve data points.")

# Step 2: Find Max/Min Flux and Corresponding Errors
idx_max = np.argmax(flux_noisy)
flux_max = flux_noisy[idx_max]
err_flux_max = flux_err[idx_max]

idx_min = np.argmin(flux_noisy)
flux_min = flux_noisy[idx_min]
err_flux_min = flux_err[idx_min]

print("\nIdentifying Max/Min Flux:")
print(f"  Max Flux = {flux_max:.4f} +/- {err_flux_max:.4f} (at time {time[idx_max]:.2f})")
print(f"  Min Flux = {flux_min:.4f} +/- {err_flux_min:.4f} (at time {time[idx_min]:.2f})")

# Step 3: Calculate Amplitude (Peak-to-Peak)
amplitude = flux_max - flux_min
print(f"\nCalculated Amplitude (Max - Min): {amplitude:.4f}")

# Step 4: Propagate Error Manually
# Variance = err_max^2 + err_min^2 (errors add in quadrature for subtraction)
variance_amplitude = err_flux_max**2 + err_flux_min**2
sigma_amplitude = np.sqrt(variance_amplitude)
print(f"Propagated Uncertainty (sigma_Amplitude): {sigma_amplitude:.4f}")
print(f"\nResult: Amplitude = {amplitude:.4f} +/- {sigma_amplitude:.4f}")

# Step 5: Visualization
print("\nGenerating light curve plot with error bars...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.errorbar(time, flux_noisy, yerr=flux_err, fmt='.', color='black', 
            ecolor='gray', elinewidth=1, capsize=2, alpha=0.7, label='Observed Data')
# Highlight max and min points
ax.plot(time[idx_max], flux_max, 'ro', ms=8, label=f'Max ({flux_max:.3f})')
ax.plot(time[idx_min], flux_min, 'bo', ms=8, label=f'Min ({flux_min:.3f})')

ax.set_xlabel("Time (hours)")
ax.set_ylabel("Relative Flux")
ax.set_title("Simulated Asteroid Light Curve")
ax.legend()
ax.grid(True, alpha=0.4)
fig.tight_layout()
# plt.show()
print("Plot generated.")
plt.close(fig)

# Optional: Comparison with 'uncertainties' package
if uncertainties_installed:
    print("\nComparison with 'uncertainties' package:")
    fmax_u = ufloat(flux_max, err_flux_max)
    fmin_u = ufloat(flux_min, err_flux_min)
    amp_u = fmax_u - fmin_u
    print(f"  Amplitude (uncertainties pkg): {amp_u:.4f}")
    # Check if results match
    print(f"  Match nominal value? {np.isclose(amp_u.nominal_value, amplitude)}")
    print(f"  Match std dev? {np.isclose(amp_u.std_dev, sigma_amplitude)}")
else:
    print("\n'uncertainties' package not installed, skipping comparison.")

print("-" * 20)
```

**Summary**

This chapter focused on fundamental descriptive statistics and error analysis techniques essential for summarizing and interpreting astrophysical datasets. It introduced measures of central tendency – the mean (average, sensitive to outliers), the median (middle value, robust to outliers), and the mode (most frequent value) – discussing their calculation using NumPy and SciPy and their appropriate use cases depending on data distribution characteristics. Complementing these, measures of dispersion were covered, including the range (highly sensitive), the variance and standard deviation (measuring spread around the mean, also sensitive to outliers), and the robust Interquartile Range (IQR, measuring the spread of the central 50% of data). The crucial topic of error propagation was addressed, explaining the standard first-order approximation method for estimating the uncertainty in a derived quantity based on the uncertainties of input variables, covering rules for common operations and highlighting the importance of assuming uncorrelated errors for the basic formulas. The `uncertainties` package was introduced as a tool for automating these calculations.

Recognizing that simple summary statistics don't capture the full picture, the chapter explored methods for visualizing distributions, primarily histograms (`matplotlib.pyplot.hist`) and Kernel Density Estimates (KDEs, via `scipy.stats.gaussian_kde` or `seaborn.kdeplot`). The importance of appropriate binning for histograms (using rules like Freedman-Diaconis via `astropy.visualization.hist`) and bandwidth selection for KDEs was emphasized. Methods for quantifying linear relationships between two variables were presented, defining covariance and the Pearson correlation coefficient (*r*), explaining their calculation (`numpy.cov`, `numpy.corrcoef`) and interpretation, while cautioning against inferring causation and stressing the need for visual inspection via scatter plots. Finally, acknowledging the prevalence of outliers in astronomical data, the chapter introduced robust statistical estimators: the Median Absolute Deviation (MAD) and the related robust standard deviation estimate (`astropy.stats.mad_std`), and the iterative outlier rejection technique of sigma clipping (`astropy.stats.sigma_clip`), demonstrating how these methods provide more reliable data summaries in the presence of contamination or non-Gaussian behavior.

---

**References for Further Reading:**

1.  **Wall, J. V., & Jenkins, C. R. (2012).** *Practical Statistics for Astronomers* (2nd ed.). Cambridge University Press. [https://doi.org/10.1017/CBO9781139168491](https://doi.org/10.1017/CBO9781139168491)
    *(Provides concise explanations of descriptive statistics, error propagation, correlation, and robust methods tailored for astronomers.)*

2.  **Feigelson, E. D., & Babu, G. J. (2012).** *Modern Statistical Methods for Astronomy: With R Applications*. Cambridge University Press. [https://doi.org/10.1017/CBO9781139179009](https://doi.org/10.1017/CBO9781139179009)
    *(Offers more in-depth coverage of descriptive statistics, distribution visualization (histograms, KDEs), correlation, and robust estimators in an astronomical context.)*

3.  **Ivezić, Ž., Connolly, A. J., VanderPlas, J. T., & Gray, A. (2014).** *Statistics, Data Mining, and Machine Learning in Astronomy*. Princeton University Press. (Relevant chapters often available online, e.g., Chapter 3 for descriptive stats/distributions: [http://press.princeton.edu/titles/10159.html](http://press.princeton.edu/titles/10159.html))
    *(Covers descriptive statistics, error analysis, correlation, visualization, and robust methods comprehensively within a broader data science context for astronomy.)*

4.  **Astropy Project Contributors. (n.d.).** *Astropy Documentation: Astronomical Statistics (astropy.stats)*. Astropy Documentation. Retrieved January 16, 2024, from [https://docs.astropy.org/en/stable/stats/](https://docs.astropy.org/en/stable/stats/)
    *(Official documentation for `astropy.stats`, providing details and examples for robust statistics like `mad_std`, `sigma_clip`, biweight estimators, and advanced histogram binning algorithms, relevant to Sec 14.4 and 14.6.)*

5.  **Lebigot, E. O. (n.d.).** *Uncertainties Python package*. uncertainties. Retrieved January 16, 2024, from [https://pythonhosted.org/uncertainties/](https://pythonhosted.org/uncertainties/)
    *(Documentation for the `uncertainties` package, which automates error propagation calculations as demonstrated conceptually in Sec 14.3.)*
