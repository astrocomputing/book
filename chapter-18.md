**Chapter 18: Model Fitting and Model Selection**

Building on the parameter estimation techniques introduced in the previous chapters (Maximum Likelihood and Bayesian MCMC), this chapter addresses the practical workflow of fitting mathematical or physical models to astrophysical data and, critically, evaluating how well different models describe the data or choosing between competing models. We begin by discussing ways to define models programmatically in Python, ranging from simple functions to the more structured framework provided by `astropy.modeling`. We revisit common fitting techniques like **least-squares fitting** (using `scipy.optimize.curve_fit`) and **Chi-squared minimization** (using `scipy.optimize.minimize`), clarifying their connection to Maximum Likelihood Estimation under Gaussian errors. The core focus then shifts to **model selection**: how do we objectively decide if adding complexity to a model (e.g., more parameters) provides a significantly better fit, or choose between fundamentally different physical models? We explore common frequentist approaches, including the **Likelihood Ratio Test (LRT)** for nested models, and information criteria like the **Akaike Information Criterion (AIC)** and **Bayesian Information Criterion (BIC)**, which penalize model complexity. We then introduce the Bayesian approach to model comparison based on calculating the **Bayesian evidence (marginal likelihood)** and comparing models using **Bayes Factors**, highlighting the interpretation of evidence ratios. Finally, techniques like **cross-validation** for assessing a model's predictive performance on unseen data are briefly discussed, providing a toolkit for rigorously evaluating and comparing the models used to interpret astrophysical observations and simulations.

**18.1 Defining Models in Python**

Before any model fitting can occur, the mathematical or physical model itself must be implemented in a way that allows evaluation within a Python environment. The model essentially defines the predicted value (or distribution) of the dependent variable(s) given specific values of the independent variable(s) and a set of model parameters. Python offers several ways to represent these models, ranging in complexity and structure depending on the application's needs.

The simplest approach is often to define the model as a standard **Python function**. This function typically takes the independent variable(s) (e.g., `x`, or `time`, `wavelength`) as the first argument(s) and the model parameters (e.g., `p1`, `p2`, `...`) as subsequent arguments. Inside the function, the mathematical formula relating inputs and parameters is implemented, usually using NumPy for numerical calculations, and the function returns the predicted model value(s). For example, a linear model `y = mx + c` could be `def linear_model(x, m, c): return m * x + c`. A Gaussian profile `A * exp(-(x-μ)²/(2σ²))` could be `def gaussian(x, A, mu, sigma): return A * np.exp(-(x - mu)**2 / (2 * sigma**2))`. This functional approach is straightforward and often sufficient for use with general-purpose fitting routines like `scipy.optimize.minimize` or `scipy.optimize.curve_fit`.

```python
# --- Code Example 1: Defining Models as Python Functions ---
import numpy as np
import matplotlib.pyplot as plt # For plotting example

print("Defining models as simple Python functions:")

# --- Linear Model ---
def linear_model(x, slope, intercept):
    """A simple linear model: y = slope * x + intercept."""
    return slope * x + intercept

# --- Gaussian Profile Model ---
def gaussian_model(x, amplitude, mean, stddev):
    """A Gaussian profile: amplitude * exp(-(x-mean)^2 / (2*stddev^2))."""
    # Add check for valid width
    if stddev <= 0:
        # Return NaN or raise error, depending on fitting context
        # Returning array of NaNs avoids crashing some fitters
        return np.full_like(x, np.nan) 
    return amplitude * np.exp(-(x - mean)**2 / (2 * stddev**2))

# --- Power Law Model (as used in Ch 16) ---
def power_law_model(x, norm, index):
    """Power law model y = norm * x^index."""
    # Need to handle x<=0 if index is not integer
    x_safe = np.asarray(x)
    if np.any(x_safe <= 0) and not np.issubdtype(type(index), np.integer):
        # Return NaN where x is non-positive
        result = np.full_like(x_safe, np.nan, dtype=np.float64)
        mask = x_safe > 0
        result[mask] = norm * (x_safe[mask]**index)
        return result
    return norm * (x_safe**index)

# --- Example Usage and Plotting ---
x_values = np.linspace(0, 10, 50)

# Evaluate models with some parameters
y_line = linear_model(x_values, slope=2.0, intercept=1.0)
y_gauss = gaussian_model(x_values, amplitude=5.0, mean=5.0, stddev=1.0)
# Need positive x for power law if index is negative
x_power = np.linspace(1, 10, 50) 
y_power = power_law_model(x_power, norm=10.0, index=-1.5)

print("\nEvaluated model functions at example parameters.")

# Plot the models
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(x_values, y_line); ax[0].set_title("Linear Model")
ax[1].plot(x_values, y_gauss); ax[1].set_title("Gaussian Model")
ax[2].plot(x_power, y_power); ax[2].set_title("Power Law Model"); ax[2].set_yscale('log'); ax[2].set_xscale('log')
for axi in ax: axi.grid(True); axi.set_xlabel("x")
ax[0].set_ylabel("y"); ax[1].set_ylabel("y"); ax[2].set_ylabel("y")
fig.tight_layout()
# plt.show()
print("Generated plots of the example model functions.")
plt.close(fig)

print("-" * 20)

# Explanation: This code defines three common astrophysical models (linear, Gaussian, 
# power law) as simple Python functions. Each function takes the independent variable `x` 
# and the model parameters as arguments and returns the calculated model value using 
# NumPy functions. Basic checks for invalid parameter values (like negative stddev 
# or non-positive x for power law) are included. The code then demonstrates evaluating 
# these functions with example parameters and plotting the resulting model curves 
# using Matplotlib, illustrating their basic usage. These function definitions 
# are suitable inputs for many fitting routines.
```

While simple functions work well, they lack inherent structure for managing parameter metadata (like names, default values, bounds, units, fixed status) or easily combining multiple model components. For more complex modeling scenarios, the **`astropy.modeling`** framework provides a powerful object-oriented approach. It offers a library of pre-defined common 1D and 2D models (e.g., `Gaussian1D`, `Lorentz1D`, `PowerLaw1D`, `Const1D`, `Linear1D`, `Polynomial1D`, `Gaussian2D`, `Moffat2D`, `Disk2D`, etc.) and allows users to easily create custom models by defining their mathematical form.

In `astropy.modeling`, a model is represented by a model class instance (e.g., `g = models.Gaussian1D(amplitude=1.0, mean=0.0, stddev=1.0)`). This object holds the parameter values as attributes (`g.amplitude`, `g.mean`, `g.stddev`), which are themselves special `Parameter` objects that can store bounds, fixed status, and potentially units. Evaluating the model is done by calling the instance: `y_model = g(x_values)`.

The key advantage of `astropy.modeling` is its ability to **combine models** using standard arithmetic operators (`+`, `-`, `*`, `/`, `**`) or functional composition (`|`). For example, to create a model of a Gaussian line on a linear background, you can simply write: `combined_model = models.Gaussian1D(...) + models.Linear1D(...)`. Astropy automatically handles the combined parameter set and evaluation. This makes building complex, multi-component models significantly easier and more structured than writing monolithic functions.

```python
# --- Code Example 2: Using astropy.modeling ---
# Note: Requires astropy installation
from astropy.modeling import models
import numpy as np
import matplotlib.pyplot as plt

print("Defining and combining models using astropy.modeling:")

# --- Instantiate Pre-defined Models ---
# Gaussian line (parameters: amplitude, mean, stddev)
# Provide initial guesses or default values
gauss_comp = models.Gaussian1D(amplitude=10.0, mean=6563.0, stddev=5.0, 
                               name="Halpha") # Optional name

# Linear continuum (parameters: slope, intercept)
continuum_comp = models.Linear1D(slope=0.01, intercept=5.0, 
                                 name="Continuum")

print("\nInstantiated individual models:")
print(gauss_comp)
print(continuum_comp)

# --- Combine Models using '+' ---
# The parameters from both components are automatically combined
spec_model = gauss_comp + continuum_comp
print("\nCombined Model (Gaussian + Linear):")
print(spec_model)
print(f"  Parameter names in combined model: {spec_model.param_names}")
# Note names are prefixed by component name or index: amplitude_0, mean_0, stddev_0, slope_1, intercept_1

# --- Evaluate the Combined Model ---
wavelengths = np.linspace(6500, 6620, 121)
# Evaluate by calling the model instance
flux_combined = spec_model(wavelengths) 
# Access and modify parameters via attributes
spec_model.amplitude_0 = 12.0 # Change amplitude of the Gaussian component
spec_model.intercept_1 = 4.5 # Change continuum intercept
flux_combined_modified = spec_model(wavelengths)
print("\nEvaluated combined model at example wavelengths.")

# --- Plotting ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(wavelengths, flux_combined, label='Initial Combined Model')
ax.plot(wavelengths, flux_combined_modified, '--', label='Modified Combined Model')
# Plot individual components (using their instances)
ax.plot(wavelengths, gauss_comp(wavelengths), ':', label=f'{gauss_comp.name} Component (Mod.)')
ax.plot(wavelengths, continuum_comp(wavelengths), ':', label=f'{continuum_comp.name} Component (Mod.)')

ax.set_xlabel("Wavelength (arb. units)")
ax.set_ylabel("Flux (arb. units)")
ax.set_title("Using astropy.modeling to Combine Models")
ax.legend()
ax.grid(True, alpha=0.4)
# plt.show()
print("Generated plot showing combined model and components.")
plt.close(fig)

print("-" * 20)

# Explanation: This code uses `astropy.modeling`.
# 1. It instantiates pre-defined models `Gaussian1D` and `Linear1D` from `astropy.models`, 
#    providing initial parameter values. Models can optionally be named.
# 2. It combines these two models simply using the `+` operator to create a compound 
#    model `spec_model`. The parameter names in the combined model are automatically 
#    managed (e.g., `amplitude_0`, `slope_1`).
# 3. It evaluates the combined model by calling the instance `spec_model(wavelengths)`.
# 4. It shows how individual parameters of the combined model can be accessed and 
#    modified using attribute access (e.g., `spec_model.amplitude_0 = 12.0`).
# 5. It plots the evaluated combined model before and after modification, and also 
#    plots the individual components separately, illustrating the ease of building 
#    and evaluating multi-component models.
```

`astropy.modeling` also includes a framework for **fitting** these model objects to data, often using wrappers around `scipy.optimize` or specialized algorithms (like Levenberg-Marquardt). Fitters within `astropy.modeling` understand the model structure, allowing parameters to be fixed (`param.fixed = True`) or tied to other parameters via mathematical expressions during the fit. This provides a higher-level interface compared to directly using `scipy.optimize.minimize` with simple Python functions, particularly advantageous for complex or multi-component models. We might revisit `astropy.modeling.fitting` when discussing specific applications.

Creating **custom models** within the `astropy.modeling` framework is also possible by subclassing `models.FittableModel` and defining the `evaluate` method (which performs the calculation) and specifying the parameters. This allows integrating bespoke physical models into the structured environment provided by Astropy, enabling easier combination with standard models and use with Astropy's fitting routines.

Choosing how to define your model – as a simple function or using the `astropy.modeling` framework – depends on the complexity of the model and the fitting procedure. Simple functions are often sufficient for basic MLE using `scipy.optimize`. For multi-component models, models with complex parameter constraints, or when wanting to leverage Astropy's integrated fitting routines, the `astropy.modeling` framework offers significant advantages in structure, convenience, and power.

**18.2 Least-Squares Fitting (`scipy.optimize.curve_fit`)**

One of the most common and intuitive methods for fitting a model to data is **least-squares fitting**. The core idea is to find the model parameters that minimize the sum of the squared differences (residuals) between the observed data points and the values predicted by the model. This approach is widely used due to its simplicity and its direct connection to Maximum Likelihood Estimation (MLE) when the data uncertainties are Gaussian and uniform. The `scipy.optimize` module provides a convenient function, `curve_fit`, specifically designed for performing least-squares fits to non-linear models defined by Python functions.

Let `y = f(x, θ)` be the model function, where `x` is the independent variable, `y` is the dependent variable, and `θ` represents the set of model parameters we want to estimate. Given a dataset `(xᵢ, yᵢ)`, the standard (unweighted) least-squares method seeks to find the parameters `θ̂` that minimize the **Sum of Squared Residuals (SSR)** or **Residual Sum of Squares (RSS)**:

SSR(θ) = Σ<0xE1><0xB5><0xA2> [ y<0xE1><0xB5><0xA2> - f(x<0xE1><0xB5><0xA2>, θ) ]²

If the uncertainties (standard deviations σ<0xE1><0xB5><0xA2>) on the data points `yᵢ` are known and assumed to be Gaussian, then minimizing the SSR is equivalent to maximizing the Gaussian log-likelihood (ignoring constant terms) *only if all σ<0xE1><0xB5><0xA2> are equal*. A statistically better approach in this case is **weighted least squares**, which corresponds directly to MLE for independent Gaussian errors. This involves minimizing the **Chi-squared (χ²)** statistic:

χ²(θ) = Σ<0xE1><0xB5><0xA2> [ (y<0xE1><0xB5><0xA2> - f(x<0xE1><0xB5><0xA2>, θ))² / σ<0xE1><0xB5><0xA2>² ]

The `scipy.optimize.curve_fit` function provides an interface to perform non-linear least-squares fitting, including weighted fits if uncertainties are provided. Its basic usage is:
`popt, pcov = curve_fit(model_func, x_data, y_data, p0=initial_guess, sigma=y_error, absolute_sigma=True, ...)`

*   `model_func`: A Python function defining the model, e.g., `f(x, param1, param2, ...)`. The first argument must be the independent variable `x`, followed by the model parameters.
*   `x_data`, `y_data`: The arrays containing the observed independent and dependent variable values.
*   `p0` (optional): An array or list providing initial guesses for the model parameters. Crucial for non-linear models to guide the fit.
*   `sigma` (optional): An array of the same size as `y_data` containing the 1-sigma uncertainties (σ<0xE1><0xB5><0xA2>) on the `y_data` values. If provided, `curve_fit` performs a weighted least-squares fit (minimizing Chi-squared).
*   `absolute_sigma=True`: This important flag tells `curve_fit` that the provided `sigma` values represent absolute 1-sigma uncertainties. If `True`, the returned covariance matrix `pcov` directly estimates the covariance of the fitted parameters based on these errors. If `False` (or `sigma` not provided), `curve_fit` assumes `sigma` represents only *relative* weights, performs an unweighted or relatively weighted fit, and then *rescales* `pcov` based on the residual variance, which is often not statistically appropriate if the absolute scale of errors is meaningful. **Always use `absolute_sigma=True` if your `sigma` input represents actual error estimates.**
*   Other arguments allow specifying bounds (`bounds`), different minimization methods (`method`), etc.

`curve_fit` returns two values:
*   `popt`: A NumPy array containing the optimal parameter values found by the least-squares fit (these are the MLEs if `sigma` was provided and errors are Gaussian).
*   `pcov`: The estimated **covariance matrix** of the `popt` parameters. The diagonal elements `pcov[i, i]` are the estimated variances (σ<0xE1><0xB5><0x82>²) of the parameters, and their square roots `np.sqrt(np.diag(pcov))` give the estimated 1-sigma standard errors on the parameters. The off-diagonal elements `pcov[i, j]` give the estimated covariances between parameter `i` and parameter `j`. This covariance matrix is derived from the curvature (Hessian) of the objective function (SSR or χ²) at the minimum, analogous to the Hessian-based uncertainties in MLE (Sec 16.4).

```python
# --- Code Example 1: Using scipy.optimize.curve_fit ---
# Note: Requires scipy installation.
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

print("Fitting a model using scipy.optimize.curve_fit:")

# --- Define Model Function (e.g., Gaussian from Sec 18.1) ---
def gaussian_model(x, amplitude, mean, stddev):
    """A Gaussian profile: amplitude * exp(-(x-mean)^2 / (2*stddev^2))."""
    # Basic check for width
    if stddev <= 0: return np.inf
    return amplitude * np.exp(-(x - mean)**2 / (2 * stddev**2))

# --- Simulate Data ---
np.random.seed(0)
x_data = np.linspace(0, 10, 50)
true_params = [5.0, 5.0, 1.0] # True Amp, Mean, Stddev
y_true = gaussian_model(x_data, *true_params)
y_error = 0.3 # Assume constant error for simplicity
y_observed = y_true + np.random.normal(0, y_error, size=x_data.shape)
print("\nSimulated Gaussian data generated.")

# --- Perform the Fit ---
# Initial guesses for parameters [Amp, Mean, Stddev]
initial_guess = [4.0, 5.1, 1.1] 
print(f"Initial Guess: {initial_guess}")

print("Running curve_fit...")
try:
    # Provide the model function, x data, y data, initial guess, and errors
    popt, pcov = curve_fit(
        gaussian_model, 
        x_data, 
        y_observed, 
        p0=initial_guess, 
        sigma=np.full_like(y_observed, y_error), # Pass error array
        absolute_sigma=True # Crucial: Treat sigma as absolute errors
        # bounds=([0, -np.inf, 1e-6], [np.inf, np.inf, np.inf]) # Optional bounds
    )
    
    # --- Extract Results ---
    fit_amp, fit_mean, fit_stddev = popt
    print("\nFit successful!")
    print(f"  Fitted Amplitude: {fit_amp:.3f} (True: {true_params[0]})")
    print(f"  Fitted Mean: {fit_mean:.3f} (True: {true_params[1]})")
    print(f"  Fitted Stddev: {fit_stddev:.3f} (True: {true_params[2]})")
    
    # Calculate standard errors from covariance matrix
    perr = np.sqrt(np.diag(pcov))
    print("\nEstimated Parameter Standard Errors:")
    print(f"  sigma(Amplitude): {perr[0]:.3f}")
    print(f"  sigma(Mean): {perr[1]:.3f}")
    print(f"  sigma(Stddev): {perr[2]:.3f}")
    print("\nEstimated Covariance Matrix:\n", pcov)

    # --- Plotting ---
    print("\nGenerating plot...")
    plt.figure(figsize=(8, 5))
    plt.errorbar(x_data, y_observed, yerr=y_error, fmt='o', label='Data', capsize=3)
    x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
    plt.plot(x_smooth, gaussian_model(x_smooth, *popt), 'r-', label='curve_fit Result')
    # plt.plot(x_smooth, gaussian_model(x_smooth, *true_params), 'g--', label='True Model')
    plt.xlabel("X"); plt.ylabel("Y"); plt.legend(); plt.title("curve_fit Example")
    plt.grid(True, alpha=0.5)
    # plt.show()
    print("Plot generated.")
    plt.close()
    
except RuntimeError as e:
     print(f"\nFit failed: {e}. Check initial guess or model.")
except Exception as e:
     print(f"\nAn error occurred: {e}")

print("-" * 20)

# Explanation: This code fits a Gaussian model to simulated data using `curve_fit`.
# 1. It defines the `gaussian_model` function as required by `curve_fit` (x first, then params).
# 2. It simulates data `x_data`, `y_observed` with known `y_error`.
# 3. It provides an `initial_guess` `p0` for the parameters [Amp, Mean, Stddev].
# 4. It calls `curve_fit`, passing the model function, data, guess, and crucially, 
#    the `sigma` array containing errors and `absolute_sigma=True`. This ensures 
#    a weighted fit (minimizing Chi-squared) and correctly scaled parameter errors.
# 5. It unpacks the optimal parameters (`popt`) and the covariance matrix (`pcov`).
# 6. It calculates the standard errors by taking the square root of the diagonal 
#    elements of `pcov`.
# 7. It prints the fitted parameters and their errors, and optionally plots the data 
#    and the best-fit curve.
```

`curve_fit` is very convenient for relatively simple non-linear model fitting when data uncertainties are available and approximately Gaussian. It directly returns both the best-fit parameters and their estimated covariance matrix based on the least-squares minimization. It wraps the underlying `scipy.optimize.least_squares` or `leastsq` functions which use variations of the Levenberg-Marquardt algorithm, generally efficient for least-squares problems.

However, `curve_fit` has limitations. It specifically minimizes the sum of squares (weighted or unweighted). If your data's error distribution is significantly non-Gaussian (e.g., Poisson counts, data with outliers), minimizing Chi-squared (which `curve_fit` does when `sigma` is given) is no longer equivalent to maximizing the true likelihood. In such cases, directly defining the correct negative log-likelihood function and using the more general `scipy.optimize.minimize` (as in Chapter 16) is statistically more appropriate.

Furthermore, while `curve_fit` returns the covariance matrix `pcov`, interpreting the full uncertainty landscape, especially correlations between parameters, might still benefit from visualizing the likelihood surface or using MCMC methods (Chapter 17). Providing good initial guesses (`p0`) is often essential for `curve_fit` to converge correctly for non-linear models, as the underlying algorithms can easily get stuck in local minima.

Despite these points, `curve_fit` remains a valuable and widely used tool for quick and convenient fitting of functions to data, especially when Gaussian errors dominate and a reasonable initial guess is available. Its direct return of parameter estimates and their covariance matrix makes it very user-friendly for many common fitting tasks.

**18.3 Chi-squared Minimization for Binned Data**

While `curve_fit` (Sec 18.2) is excellent for fitting models to individual data points `(xᵢ, yᵢ, σᵢ)`, another common scenario involves fitting a model to **binned data**, typically represented as a **histogram**. For example, we might have the number of galaxies observed in different redshift bins, the number of photons detected in different energy channels of a spectrum, or the number of stars counted in different magnitude ranges. In these cases, we often want to compare the observed counts per bin (O<0xE1><0xB5><0xA2>) to the counts predicted by a model (E<0xE1><0xB5><0xA2>(θ)) which depends on some parameters θ, and find the parameters that minimize the discrepancy.

Assuming the counts in each bin are large enough (typically O<0xE1><0xB5><0xA2> > 5-10) to be approximately Gaussian-distributed, or if the underlying process is Poisson and the expected counts are large, the standard approach is **Chi-squared (χ²) minimization**. We aim to find the parameters θ that minimize the χ² statistic, defined as:

χ²(θ) = Σ<0xE1><0xB5><0xA2> [ (O<0xE1><0xB5><0xA2> - E<0xE1><0xB5><0xA2>(θ))² / σ<0xE1><0xB5><0xA2>² ]

Here, O<0xE1><0xB5><0xA2> is the observed count in bin `i`, E<0xE1><0xB5><0xA2>(θ) is the expected count in bin `i` predicted by the model with parameters θ, and σ<0xE1><0xB5><0xA2> is the uncertainty associated with the observed count O<0xE1><0xB5><0xA2>.

The uncertainty σ<0xE1><0xB5><0xA2> is crucial. If the observed counts O<0xE1><0xB5><0xA2> arise from a counting process (like photon counts or object counts), they are often assumed to follow Poisson statistics, where the variance is equal to the mean. If the *true* mean count in bin `i` is μ<0xE1><0xB5><0xA2> ≈ E<0xE1><0xB5><0xA2>, then the variance is also approximately E<0xE1><0xB5><0xA2>. In this case, σ<0xE1><0xB5><0xA2>² ≈ E<0xE1><0xB5><0xA2>(θ), and the statistic becomes Pearson's Chi-squared:
χ²<0xE1><0xB5><0x96>(θ) = Σ<0xE1><0xB5><0xA2> [ (O<0xE1><0xB5><0xA2> - E<0xE1><0xB5><0xA2>(θ))² / E<0xE1><0xB5><0xA2>(θ) ]

Alternatively, if the observed counts O<0xE1><0xB5><0xA2> themselves are large enough (e.g., > 10-20), the Poisson variance can be approximated by the observed count itself: σ<0xE1><0xB5><0xA2>² ≈ O<0xE1><0xB5><0xA2>. In this case, the statistic becomes Neyman's Chi-squared (often used in particle physics):
χ²<0xE1><0xB5><0x8A>(θ) = Σ<0xE1><0xB5><0xA2> [ (O<0xE1><0xB5><0xA2> - E<0xE1><0xB5><0xA2>(θ))² / O<0xE1><0xB5><0xA2> ]
(Neyman's χ² requires all O<0xE1><0xB5><0xA2> > 0). If external uncertainties σ<0xE1><0xB5><0xA2> are provided (e.g., derived from propagating errors during binning), then the general χ² formula using those σ<0xE1><0xB5><0xA2> should be used. For Poisson data, especially with low counts, minimizing the **Cash statistic** (C = 2 Σ [E<0xE1><0xB5><0xA2> - O<0xE1><0xB5><0xA2> * ln(E<0xE1><0xB5><0xA2>)]) derived directly from the Poisson likelihood is statistically preferred over Chi-squared, as it doesn't rely on Gaussian approximations.

Assuming we use one of the χ² formulations (most commonly Pearson's χ² or the general form with known σ<0xE1><0xB5><0xA2>), finding the best-fit parameters θ̂ involves minimizing χ²(θ). Since the expected counts E<0xE1><0xB5><0xA2>(θ) often depend non-linearly on the parameters θ, this requires numerical optimization, typically using `scipy.optimize.minimize`. The workflow is very similar to the MLE example in Sec 16.6:
1.  Define a Python function `model_expected_counts(bin_info, params)` that calculates the expected counts E<0xE1><0xB5><0xA2>(θ) in each bin given the bin definitions (`bin_info`, e.g., bin edges) and the parameters `params`. This often involves integrating a model PDF over the bin ranges and multiplying by the expected total number of counts (which might itself be a parameter).
2.  Define the objective function `chi_squared_binned(params, bin_info, observed_counts, errors)` that calls `model_expected_counts`, calculates the appropriate χ² sum (e.g., `np.sum(((observed_counts - expected_counts)**2) / errors**2)`), and returns it. `errors` would typically be `np.sqrt(observed_counts)` or `np.sqrt(expected_counts)` or externally provided errors.
3.  Provide initial guesses and potentially bounds for the parameters.
4.  Call `minimize(chi_squared_binned, initial_guess, args=(...), method=...)`.
5.  Extract the best-fit parameters `θ̂` from `result.x` and estimate uncertainties from the Hessian (Cov ≈ 2 * H<0xE1><0xB5><0xA2><0xE2><0x82><0x99>ᵢ²⁻¹).

```python
# --- Code Example: Chi-squared Minimization for Histogram Fit ---
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt

print("Fitting a model to binned data (histogram) using Chi-squared minimization:")

# Step 1: Simulate Observed Binned Data (Histogram)
np.random.seed(0)
# Generate data from a Gaussian distribution
true_mean = 5.0; true_std = 1.5
underlying_data = np.random.normal(true_mean, true_std, size=500)
# Create histogram (observed counts O_i)
n_bins = 10
observed_counts, bin_edges = np.histogram(underlying_data, bins=n_bins, range=(0, 10))
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
bin_width = bin_edges[1] - bin_edges[0]
print(f"\nObserved Counts in {n_bins} bins: {observed_counts}")

# --- Define Model and Chi-squared ---
# Model: Gaussian PDF. Parameters theta = [amplitude, mean, stddev]
# Note: Amplitude here is normalization N*width, not peak height
def expected_counts_gaussian(bin_edges_local, total_N, mu, sigma):
    """Calculate expected counts in bins for a Gaussian model."""
    # Use stats.norm.cdf to get probability in each bin
    bin_probabilities = stats.norm.cdf(bin_edges_local[1:], loc=mu, scale=sigma) - \
                        stats.norm.cdf(bin_edges_local[:-1], loc=mu, scale=sigma)
    return total_N * bin_probabilities

def chi_squared_hist(params, bin_edges_local, obs_counts):
    """Chi-squared function for fitting Gaussian to histogram."""
    # Amplitude parameter 'A' is related to total count N
    # Let's fit Mean (mu) and Std Dev (sigma), fixing total N
    mu, sigma = params[0], params[1]
    total_N = np.sum(obs_counts) # Fix total normalization to observed count
    
    if sigma <= 0: return np.inf
        
    # Calculate expected counts E_i
    expected_counts = expected_counts_gaussian(bin_edges_local, total_N, mu, sigma)
    
    # Calculate errors sigma_i for weighting (use sqrt(O_i) for simplicity if counts > ~5)
    # Or use sqrt(E_i) (Pearson). Handle O_i = 0 if using sqrt(O_i).
    # Use sqrt(max(1, O_i)) to avoid division by zero if O_i=0
    errors_sq = np.maximum(1, obs_counts) # Variance approx = Observed counts (Neyman-like)
    # errors_sq = expected_counts # Pearson Chi2 (use if E_i >= 5 generally)
    # errors_sq = provided_errors**2 # If external errors given
    
    # Calculate Chi-squared, handling bins where expected might be zero
    valid_bins = expected_counts > 1e-9 # Avoid division by zero in case model predicts zero
    if not np.all(errors_sq[valid_bins] > 0): return np.inf # Avoid division by zero error
        
    chisq = np.sum( ((obs_counts[valid_bins] - expected_counts[valid_bins])**2) / errors_sq[valid_bins] )
    
    return chisq if np.isfinite(chisq) else np.inf

# --- Perform Optimization ---
# Initial guesses [mu, sigma]
initial_guess = [np.mean(underlying_data), np.std(underlying_data)] 
print(f"\nInitial Guess [mu, sigma]: {np.round(initial_guess, 2)}")
bounds = [(None, None), (1e-6, None)] # sigma > 0

print("Running minimize...")
result = minimize(
    chi_squared_hist,
    initial_guess,
    args=(bin_edges, observed_counts),
    method='L-BFGS-B',
    bounds=bounds
)

# --- Results and Uncertainties ---
mle_params = None
if result.success:
    mle_params = result.x
    min_chisq = result.fun
    k_bins_fit = len(observed_counts) 
    # Parameters fitted = mu, sigma. Total N was fixed. So p=2.
    p_params_fitted = 2 
    dof = k_bins_fit - p_params_fitted 
    print("\nOptimization successful!")
    print(f"  MLE for mu: {mle_params[0]:.3f} (True: {true_mean})")
    print(f"  MLE for sigma: {mle_params[1]:.3f} (True: {true_std})")
    print(f"  Minimum Chi-squared: {min_chisq:.2f} (dof={dof})")
    # P-value from chi2 distribution: 1 - stats.chi2.cdf(min_chisq, df=dof)
    p_value_fit = 1.0 - stats.chi2.cdf(min_chisq, df=dof)
    print(f"  Goodness-of-fit p-value: {p_value_fit:.3f}")

    # Error estimation (conceptual, needs robust Hessian)
    print("  (Error estimation requires robust Hessian calculation)")

else:
    print("\nOptimization failed!")

# --- Plotting ---
if mle_params is not None:
    print("\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(8, 5))
    # Plot histogram data
    ax.bar(bin_centers, observed_counts, width=bin_width, alpha=0.6, label='Observed Counts')
    ax.errorbar(bin_centers, observed_counts, yerr=np.sqrt(np.maximum(1, observed_counts)), 
                fmt='none', ecolor='black', capsize=3)
    # Plot best-fit model expected counts
    total_N_obs = np.sum(observed_counts)
    expected_fit = expected_counts_gaussian(bin_edges, total_N_obs, mle_params[0], mle_params[1])
    ax.plot(bin_centers, expected_fit, 'ro-', label='Best-Fit Model')
    
    ax.set_xlabel("Data Value Bins")
    ax.set_ylabel("Number of Counts")
    ax.set_title(f"Chi-squared Fit to Binned Data (χ²/dof = {min_chisq/dof:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    # plt.show()
    print("Plot generated.")
    plt.close(fig)

print("-" * 20)

# Explanation: This code fits a Gaussian model to binned (histogram) data.
# 1. It simulates data from a known Gaussian and creates observed counts `O_i` using `np.histogram`.
# 2. It defines `expected_counts_gaussian` which calculates expected counts `E_i` in each 
#    bin for given `mu`, `sigma` by integrating the Gaussian PDF (using CDF differences). 
#    It fixes the total normalization `N` to the sum of observed counts.
# 3. It defines `chi_squared_hist` which calculates the Chi-squared statistic, using 
#    `sqrt(O_i)` as the uncertainty `sigma_i` in the denominator (approximating Poisson error).
# 4. It uses `minimize` to find the `mu` and `sigma` that minimize this Chi-squared value.
# 5. It extracts the best-fit parameters, calculates degrees of freedom (`dof = bins - params_fitted`), 
#    and calculates the goodness-of-fit p-value using `scipy.stats.chi2.cdf`.
# 6. It plots the observed histogram counts (with error bars) and the expected counts 
#    from the best-fit model for visual comparison.
```

The value of the minimized Chi-squared statistic (χ²<0xE1><0xB5><0x9B><0xE1><0xB5><0xA2><0xE1><0xB5><0x8A>) itself provides a measure of the **goodness-of-fit**. Comparing χ²<0xE1><0xB5><0x9B><0xE1><0xB5><0xA2><0xE1><0xB5><0x8A> to the Chi-squared distribution with the correct degrees of freedom (`df = k - 1 - p`) yields a p-value. This p-value addresses the hypothesis "H₀: The data *is* drawn from the fitted model." A very small p-value (e.g., < 0.01 or < 0.05) indicates a poor fit – the model is unlikely to have generated the observed counts. A large p-value (e.g., > 0.1) suggests the data is consistent with the model. The **reduced Chi-squared** (χ²<0xE1><0xB5><0x9B><0xE1><0xB5><0xA2><0xE1><0xB5><0x8A>/df) is often reported; values close to 1 generally indicate a good fit, while values much larger than 1 indicate a poor fit or underestimated errors, and values much smaller than 1 might suggest overestimated errors or overfitting.

Chi-squared minimization is a powerful technique for fitting models to binned data when counts are sufficiently large for Gaussian approximations to hold. It directly connects to MLE and provides both best-fit parameters (via minimization) and a quantitative measure of goodness-of-fit (via the minimized χ² value and associated p-value). For low-count data, likelihood functions based directly on the Poisson distribution (e.g., Cash statistic) should be preferred.

**18.4 Model Comparison (Frequentist): Likelihood Ratio Test, AIC, BIC**

Often in astrophysics, we have multiple competing models that could potentially explain our data, or we might consider adding complexity (more parameters) to a simpler model. For example, is an emission line better described by a single Gaussian or two overlapping Gaussians? Is a galaxy's rotation curve adequately explained by luminous matter alone, or is adding a dark matter halo component statistically justified? We need objective methods to **compare** these different models and decide which one provides the best or most appropriate description of the data, balancing goodness-of-fit with model complexity. Frequentist statistics offers several tools for this, including the Likelihood Ratio Test (LRT), the Akaike Information Criterion (AIC), and the Bayesian Information Criterion (BIC).

The **Likelihood Ratio Test (LRT)** is specifically designed for comparing two **nested models**. A model M₁ is nested within a more complex model M₂ if M₁ is a special case of M₂ that can be obtained by fixing one or more parameters of M₂ to specific values (often zero). For example, a single Gaussian model is nested within a two-Gaussian model (by setting the amplitude of the second Gaussian to zero). A linear model `y = c` is nested within `y = mx + c` (by setting `m=0`).

The LRT compares the maximum likelihood values achieved by the two nested models. Let L₁ be the maximum likelihood achieved by the simpler model M₁ (with k₁ parameters), and L₂ be the maximum likelihood for the more complex model M₂ (with k₂ > k₁ parameters). Since M₂ has more freedom, L₂ will always be greater than or equal to L₁. The question is whether the *increase* in likelihood is large enough to justify the added complexity of M₂. The LRT statistic is defined as:
D = -2 * ln(L₁ / L₂) = -2 * (ln L₁ - ln L₂)
Wilks' Theorem states that, under the null hypothesis that the simpler model M₁ is correct, this statistic D approximately follows a **Chi-squared (χ²) distribution with `k₂ - k₁` degrees of freedom** (the number of extra parameters in M₂).

We can then calculate a p-value from this χ² distribution. A small p-value (e.g., ≤ 0.05) indicates that the observed increase in likelihood with the more complex model M₂ is unlikely to have occurred by chance if M₁ were true. This leads us to reject M₁ in favor of the more complex model M₂. A large p-value suggests the added complexity of M₂ did not provide a statistically significant improvement in fit compared to M₁, so we might prefer the simpler model M₁ based on parsimony (Occam's Razor). The LRT is a powerful tool but is strictly applicable only when comparing nested models.

For comparing **non-nested models**, or more generally accounting for model complexity when comparing any models (nested or not), **Information Criteria** like AIC and BIC are widely used. These criteria aim to select the model that best predicts future data or represents the most parsimonious description, balancing goodness-of-fit (how well the model fits the current data) with model complexity (number of parameters).

The **Akaike Information Criterion (AIC)** is defined as:
AIC = -2 * ln L(θ̂) + 2 * k
where ln L(θ̂) is the maximum log-likelihood achieved by the model with `k` free parameters. The model with the **lower** AIC value is preferred. AIC estimates the information lost when a given model is used to represent the process that generates the data; lower AIC indicates less information loss and better expected predictive performance. The `2k` term acts as a penalty for model complexity – adding more parameters increases `k` and thus increases AIC, unless the increase in ln L is substantial enough to overcome the penalty.

The **Bayesian Information Criterion (BIC)** (also known as the Schwarz Criterion) is similar but imposes a stronger penalty for model complexity, particularly for larger datasets:
BIC = -2 * ln L(θ̂) + k * ln(n)
where `n` is the number of data points used in the fit. Again, the model with the **lower** BIC value is preferred. The penalty term `k * ln(n)` increases more rapidly with the number of parameters `k` and the sample size `n` compared to AIC's `2k` penalty. BIC is derived from a Bayesian argument approximating the model evidence (Sec 18.5) under certain assumptions and tends to favor simpler models more strongly than AIC, especially for large `n`.

To use AIC or BIC for model comparison:
1.  Fit each candidate model (M₁, M₂, ...) to the *same* dataset using Maximum Likelihood Estimation (finding θ̂ and ln L(θ̂) for each).
2.  Calculate AIC and BIC for each model using the corresponding max log-likelihood, number of parameters (`k`), and number of data points (`n`).
3.  Compare the AIC (or BIC) values. The model with the minimum value is considered the best according to that criterion.
Differences in AIC or BIC values between models (ΔAIC = AIC<0xE1><0xB5><0xA2> - AIC<0xE1><0xB5><0x9B><0xE1><0xB5><0xA2><0xE1><0xB5><0x8A>, ΔBIC = BIC<0xE1><0xB5><0xA2> - BIC<0xE1><0xB5><0x9B><0xE1><0xB5><0xA2><0xE1><0xB5><0x8A>) are often used to gauge the strength of evidence in favor of the model with the lower value. Rules of thumb exist (e.g., Δ > 2 suggests positive evidence, Δ > 6 strong evidence, Δ > 10 very strong evidence), but these should be interpreted cautiously.

```python
# --- Code Example: Calculating AIC and BIC ---
# Assume we have fitted two models (e.g., Model 1: single Gaussian, Model 2: double Gaussian)
# to the *same* dataset of size 'n', and obtained the maximum log-likelihood values 
# and number of parameters for each.

import numpy as np

print("Calculating AIC and BIC for model comparison:")

# --- Hypothetical Fit Results ---
# Model 1 (Simpler, e.g., single Gaussian + const)
logL_model1 = -55.2 # Maximum log-likelihood achieved
k_model1 = 4      # Number of free parameters (e.g., cont, amp, cen, wid)

# Model 2 (More Complex, e.g., double Gaussian + const)
logL_model2 = -51.8 # Higher log-likelihood (better fit to data)
k_model2 = 7      # Number of free parameters (cont, amp1,cen1,wid1, amp2,cen2,wid2)

# Number of data points used in the fit
n_data = 100

print(f"\nModel 1: logL = {logL_model1:.2f}, k = {k_model1}")
print(f"Model 2: logL = {logL_model2:.2f}, k = {k_model2}")
print(f"Number of data points (n) = {n_data}")

# --- Calculate AIC ---
# AIC = -2 * logL + 2 * k
aic_model1 = -2 * logL_model1 + 2 * k_model1
aic_model2 = -2 * logL_model2 + 2 * k_model2

print("\nAIC Calculation:")
print(f"  AIC Model 1 = {aic_model1:.2f}")
print(f"  AIC Model 2 = {aic_model2:.2f}")
delta_aic = aic_model2 - aic_model1
print(f"  ΔAIC (AIC2 - AIC1) = {delta_aic:.2f}")

# --- Calculate BIC ---
# BIC = -2 * logL + k * log(n)
bic_model1 = -2 * logL_model1 + k_model1 * np.log(n_data)
bic_model2 = -2 * logL_model2 + k_model2 * np.log(n_data)

print("\nBIC Calculation:")
print(f"  BIC Model 1 = {bic_model1:.2f}")
print(f"  BIC Model 2 = {bic_model2:.2f}")
delta_bic = bic_model2 - bic_model1
print(f"  ΔBIC (BIC2 - BIC1) = {delta_bic:.2f}")

# --- Interpretation ---
print("\nInterpretation:")
print(f"  Model with lower AIC is preferred by AIC: Model {np.argmin([aic_model1, aic_model2]) + 1}")
print(f"  Model with lower BIC is preferred by BIC: Model {np.argmin([bic_model1, bic_model2]) + 1}")

# Assess strength based on delta (using common rules of thumb)
if delta_aic < -2: print("  AIC provides positive evidence for Model 2 over Model 1.")
elif delta_aic > 2: print("  AIC provides positive evidence for Model 1 over Model 2.")
else: print("  AIC difference is small; models are comparable.")

if delta_bic < -6: print("  BIC provides strong evidence for Model 2 over Model 1.")
elif delta_bic > 6: print("  BIC provides strong evidence for Model 1 over Model 2.")
elif delta_bic < -2: print("  BIC provides positive evidence for Model 2 over Model 1.")
elif delta_bic > 2: print("  BIC provides positive evidence for Model 1 over Model 2.")
else: print("  BIC difference is small; models are comparable.")
# Note: In this example, Model 2 has higher likelihood, but also more parameters. 
# Depending on the exact values and n, BIC might penalize the extra parameters 
# more heavily than AIC and could potentially prefer Model 1 despite lower likelihood.

print("-" * 20)

# Explanation: This code simulates the final step of model comparison using AIC/BIC.
# 1. It assumes two models have been fitted to the same data (size `n_data`), yielding 
#    maximum log-likelihoods (`logL_model1`, `logL_model2`) and having `k_model1` 
#    and `k_model2` free parameters, respectively.
# 2. It calculates AIC = -2*logL + 2*k for both models.
# 3. It calculates BIC = -2*logL + k*ln(n) for both models.
# 4. It compares the AIC values and BIC values, identifying the model with the lower 
#    value as preferred by each criterion.
# 5. It calculates the differences (ΔAIC, ΔBIC) and provides a basic interpretation 
#    of the strength of evidence based on common rules of thumb (e.g., |Δ| > 2 or > 6). 
# This demonstrates how to apply these criteria once the maximum likelihood fits 
# for competing models have been obtained.
```

AIC and BIC provide practical tools for model selection based on balancing fit quality and complexity. They are relatively easy to calculate once the maximum likelihood fit is obtained. However, they are based on asymptotic approximations and specific assumptions (AIC assumes the model is a good approximation to reality; BIC assumes one of the models *is* the true model). They provide relative comparisons – indicating which model is *better* among the set considered, not whether any model is absolutely "correct" or a good fit in an absolute sense (goodness-of-fit tests like Chi-squared are needed for that). Different criteria (AIC vs. BIC) can sometimes favor different models, reflecting their different underlying assumptions and complexity penalties. Bayesian model comparison using Bayes Factors (Sec 18.5) offers another powerful alternative rooted in Bayesian probability theory.

**18.5 Bayesian Model Comparison: Bayes Factors and the Evidence**

The Bayesian framework offers a conceptually different and often powerful approach to **model comparison**, complementing the frequentist methods (LRT, AIC, BIC) discussed previously. Instead of relying on penalized likelihoods or asymptotic approximations, Bayesian model comparison directly uses probability theory, via Bayes' Theorem, to calculate the **relative probability** of competing models given the observed data. The key quantities involved are the **Bayesian evidence** (or marginal likelihood) for each model and the **Bayes Factor**.

Recall Bayes' Theorem applied to model comparison. Let M₁, M₂ be two competing models. We are interested in the posterior probability ratio P(M₂|D) / P(M₁|D), which tells us how much more probable Model 2 is compared to Model 1 *after* observing the data D. Using Bayes' Theorem for each model (P(Mᵢ|D) ∝ P(D|Mᵢ) * P(Mᵢ)), the ratio becomes:

P(M₂|D) / P(M₁|D) = [ P(D|M₂) / P(D|M₁) ] * [ P(M₂) / P(M₁) ]

The term [ P(M₂) / P(M₁) ] is the **prior odds ratio**, representing our relative belief in the two models *before* seeing the data. Often, if we have no strong prior preference, we assume equal prior probabilities P(M₁) = P(M₂), making the prior odds ratio equal to 1.

The crucial term is **K = P(D|M₂) / P(D|M₁)**, known as the **Bayes Factor** in favor of Model 2 over Model 1. It represents the ratio of the **evidence** (or **marginal likelihood**) for each model. The evidence P(D|Mᵢ) for a model Mᵢ is the probability of observing the data D integrated over *all possible parameter values* θ<0xE1><0xB5><0xA2> allowed by that model, weighted by the parameters' prior distribution P(θ<0xE1><0xB5><0xA2>|Mᵢ):

P(D|Mᵢ) = Zᵢ = ∫ P(D|θ<0xE1><0xB5><0xA2>, Mᵢ) * P(θ<0xE1><0xB5><0xA2>|Mᵢ) dθ<0xE1><0xB5><0xA2>

Here, P(D|θ<0xE1><0xB5><0xA2>, Mᵢ) is the likelihood function L(θ<0xE1><0xB5><0xA2>|D, Mᵢ) for model Mᵢ. The evidence Zᵢ thus represents the *average* likelihood of the model across its entire parameter space, weighted by the prior. It automatically incorporates a penalty for model complexity (Occam's Razor): models with more parameters, or very wide priors, spread their predictive power over a larger volume, often resulting in a lower average likelihood (evidence) compared to simpler models that provide a similarly good fit to the data within a more constrained parameter space, unless the extra complexity is strongly required by the data.

The Bayes Factor K = Z₂ / Z₁ directly quantifies the extent to which the data D favors Model 2 over Model 1. If K > 1, the data support M₂ more than M₁. If K < 1, the data support M₁ more. The magnitude of K indicates the strength of evidence. Common interpretations (e.g., the Jeffreys scale) provide qualitative guidelines:
*   1 < K < 3: Weak evidence for M₂
*   3 < K < 10 or 12: Moderate or Substantial evidence for M₂
*   10 or 12 < K < 100 or 150: Strong evidence for M₂
*   K > 100 or 150: Decisive evidence for M₂
(Similar categories apply for K < 1 by considering 1/K in favor of M₁). Note that these thresholds are just conventions. It's often more informative to report the value of K (or ln K) itself.

Calculating the evidence Z = ∫ L(θ|D)P(θ) dθ is the main computational challenge in Bayesian model comparison. This multi-dimensional integral over the entire parameter space is generally intractable analytically and computationally demanding. Standard MCMC methods like Metropolis-Hastings or `emcee` primarily sample the *posterior* P(θ|D) ∝ L(θ|D)P(θ) and do *not* directly calculate the normalization constant Z.

However, certain advanced computational techniques are specifically designed to estimate the evidence:
*   **Nested Sampling** algorithms (like those implemented in `dynesty`, Sec 17.4) are particularly well-suited for evidence calculation. They work by exploring the likelihood landscape within nested prior volume contours and directly estimate the integral Z as part of the process, along with producing posterior samples. The `.results.logz` attribute from a `dynesty` run provides an estimate of the natural logarithm of the evidence (ln Z), and `.results.logzerr` provides its numerical uncertainty.
*   Other methods include Thermodynamic Integration (stepping between prior and posterior), Annealed Importance Sampling, or methods based on posterior sample densities (like the harmonic mean estimator, though often unreliable). Nested sampling is currently one of the most popular and robust methods for evidence calculation in astrophysics.

To compare two models M₁ and M₂ using Bayes Factors derived from nested sampling:
1.  Run a nested sampling algorithm (e.g., `dynesty`) separately for each model (M₁ and M₂), fitting them to the *same* dataset D. This requires defining the log-likelihood function and the prior transform function for *each* model.
2.  Extract the estimated log-evidence for each model from the results: ln Z₁ and ln Z₂.
3.  Calculate the natural logarithm of the Bayes Factor: ln K = ln(Z₂ / Z₁) = ln Z₂ - ln Z₁.
4.  Interpret ln K. Positive values favor M₂, negative values favor M₁. Common thresholds for |ln K| are: 1-3 (substantial), 3-5 (strong), >5 (decisive).

```python
# --- Code Example: Conceptual Bayesian Model Comparison using Dynesty Evidence ---
# Note: Requires dynesty installation. Assumes dynesty runs were performed for two models.
import numpy as np
# import dynesty # Would be needed to run the sampler

print("Conceptual Bayesian Model Comparison using Bayes Factors from Evidence:")

# --- Assume Nested Sampling results are available for two models ---
# (These would come from running sampler1.run_nested() and sampler2.run_nested())

# Model 1 Results (e.g., single Gaussian fit)
logz1 = -60.5  # Example estimated log evidence for Model 1
logz1_err = 0.1 # Estimated numerical uncertainty on logz1

# Model 2 Results (e.g., double Gaussian fit)
logz2 = -58.2  # Example estimated log evidence for Model 2
logz2_err = 0.12 # Estimated numerical uncertainty on logz2

print(f"\nModel 1 Log Evidence (ln Z1) = {logz1:.2f} +/- {logz1_err:.2f}")
print(f"Model 2 Log Evidence (ln Z2) = {logz2:.2f} +/- {logz2_err:.2f}")

# --- Calculate Log Bayes Factor ---
# ln K = ln(Z2 / Z1) = ln Z2 - ln Z1
ln_bayes_factor = logz2 - logz1
# Propagate uncertainty (variances add)
ln_bayes_factor_err = np.sqrt(logz1_err**2 + logz2_err**2)

print(f"\nLog Bayes Factor (ln K = ln Z2 - ln Z1) = {ln_bayes_factor:.2f} +/- {ln_bayes_factor_err:.2f}")

# --- Interpretation (using common ln K thresholds) ---
print("\nInterpretation:")
if ln_bayes_factor > 5.0:
    strength = "Decisive"
elif ln_bayes_factor > 2.5: # Or 3
    strength = "Strong"
elif ln_bayes_factor > 1.0:
    strength = "Substantial"
elif ln_bayes_factor > 0.0:
     strength = "Weak"
elif ln_bayes_factor > -1.0:
     strength = "Weak (Favors M1)"
elif ln_bayes_factor > -2.5: # Or -3
     strength = "Substantial (Favors M1)"
elif ln_bayes_factor > -5.0:
     strength = "Strong (Favors M1)"
else:
    strength = "Decisive (Favors M1)"

if abs(ln_bayes_factor) < 1.0:
    print(f"  Evidence difference is weak (|ln K| < 1). Models are comparable.")
else:
    if ln_bayes_factor > 0:
         print(f"  {strength} evidence in favor of Model 2 over Model 1.")
    else:
         print(f"  {strength} evidence in favor of Model 1 over Model 2.")
         
# Optionally calculate odds ratio if priors were specified
# prior_odds = P(M2) / P(M1) # e.g., 1.0 if equal priors
# posterior_odds = np.exp(ln_bayes_factor) * prior_odds
# print(f"\nPosterior Odds (assuming prior odds = 1): {posterior_odds:.2f} : 1 in favor of Model 2")

print("-" * 20)

# Explanation: This code demonstrates comparing two models using pre-calculated 
# log evidence values (ln Z1, ln Z2) obtained from nested sampling runs (e.g., using dynesty).
# 1. It assumes `logz1`, `logz1_err`, `logz2`, `logz2_err` have been obtained.
# 2. It calculates the natural log of the Bayes Factor `ln K = ln Z2 - ln Z1`.
# 3. It propagates the numerical uncertainties on log Z to get the uncertainty on ln K.
# 4. It provides an interpretation of `ln K` based on common qualitative thresholds 
#    (e.g., |ln K| > 1, 2.5, 5 corresponding to substantial, strong, decisive evidence).
#    A positive ln K favors Model 2, negative favors Model 1.
# 5. It conceptually shows how to calculate posterior odds if prior odds are known.
# This illustrates the final step of Bayesian model comparison once evidence values 
# are available from a suitable computational method like nested sampling.
```

Bayesian model comparison via Bayes Factors provides a self-consistent probabilistic framework for comparing models, naturally incorporating Occam's Razor by penalizing overly complex models that spread their predictions too thinly over parameter space (resulting in lower evidence Z). It directly answers the question: "How much more probable is one model compared to another, given the data and our prior assumptions?" Unlike AIC/BIC which rely on asymptotic approximations and specific penalty terms, Bayes Factors derive directly from probability theory, although they are sensitive to the choice of prior distributions P(θ|Mᵢ), especially parameter ranges. The computational cost of estimating the evidence accurately remains the primary practical challenge, but methods like nested sampling have made it increasingly feasible for many astrophysical problems.

**18.6 Cross-Validation Techniques**

While likelihood-based methods (LRT, AIC, BIC) and Bayesian evidence provide valuable tools for comparing models based on how well they fit the *current* data (potentially with complexity penalties), another important aspect of model evaluation is assessing its **predictive performance** on *new, unseen* data. A model that fits the existing data extremely well (e.g., a high-order polynomial perfectly interpolating noisy points) might perform poorly when predicting new data points because it has "overfit" the noise in the training set. **Cross-validation (CV)** techniques provide a way to estimate a model's predictive performance by repeatedly splitting the available data into training and testing subsets.

The most common form is **k-fold cross-validation**. The procedure is:
1.  Randomly divide the original dataset D into `k` roughly equal-sized, non-overlapping subsets (or "folds"). Common choices for `k` are 5 or 10.
2.  Repeat `k` times:
    *   Select one fold `i` to be the **test set**.
    *   Use the remaining `k-1` folds as the **training set**.
    *   Fit the model(s) of interest using only the training set.
    *   Evaluate the performance of the fitted model(s) on the held-out test set (fold `i`) using a suitable metric (e.g., Mean Squared Error for regression, accuracy or AUC for classification). Store this performance score.
3.  Average the `k` performance scores obtained across the `k` iterations. This average score provides an estimate of how well the model is expected to perform on unseen data drawn from the same underlying distribution.

This process ensures that every data point is used for testing exactly once, while the model is always trained on a substantial fraction (`(k-1)/k`) of the data. By averaging performance across multiple different test sets, k-fold CV provides a more robust estimate of generalization performance compared to a single train/test split (which can be sensitive to how the split was made).

Cross-validation is primarily used for two purposes:
1.  **Model Selection:** Fit several different competing models (e.g., linear regression vs. random forest, or models with different complexity levels) using the k-fold CV procedure. Choose the model that yields the best average performance score on the held-out test sets across the folds. This selects the model likely to generalize best to new data.
2.  **Hyperparameter Tuning:** Many models (especially in machine learning, Part IV) have "hyperparameters" that control their complexity but are not fitted directly from the data (e.g., the degree of a polynomial, the regularization strength in Ridge regression, the number of trees in a Random Forest). Cross-validation can be used within a grid search or randomized search to find the hyperparameter values that yield the best average predictive performance on the test folds, helping to avoid overfitting.

Python's `scikit-learn` library (primarily used for machine learning, Part IV) provides excellent tools for cross-validation, readily applicable even for simpler statistical models. The `sklearn.model_selection.cross_val_score(estimator, X, y, cv=k, scoring=...)` function automates the k-fold CV process.
*   `estimator`: A scikit-learn compatible model object (with `.fit()` and `.score()` or `.predict()` methods). You might need to wrap simple `scipy` fits or custom models into a compatible structure.
*   `X`, `y`: The feature matrix and target variable data.
*   `cv=k`: Specifies the number of folds.
*   `scoring`: Defines the performance metric to use (e.g., `'neg_mean_squared_error'`, `'r2'`, `'accuracy'`, `'roc_auc'`).
The function returns an array of the scores obtained for each of the `k` folds. You typically analyze the mean and standard deviation of these scores. `sklearn.model_selection.GridSearchCV` combines cross-validation with hyperparameter tuning.

```python
# --- Code Example: Conceptual k-fold Cross-Validation with Scikit-learn ---
# Note: Requires scikit-learn installation: pip install scikit-learn
# Uses a simple linear regression model for illustration.

import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error # For scoring

print("Conceptual k-fold Cross-Validation using scikit-learn:")

# --- Simulate Data ---
np.random.seed(1)
n_points = 100
X = np.random.rand(n_points, 1) * 10 # Single feature
true_slope = 1.5
true_intercept = 3.0
y = true_intercept + true_slope * X.flatten() + np.random.normal(0, 1.5, n_points) # Target variable
X = X # Keep X as 2D array for scikit-learn
print(f"\nGenerated {n_points} data points for linear regression.")

# --- Define Model (Estimator) ---
model = LinearRegression()
print(f"Model: {model}")

# --- Perform k-fold Cross-Validation ---
k = 5 # Number of folds
print(f"\nPerforming {k}-fold cross-validation...")
# Define the cross-validation splitter
cv_splitter = KFold(n_splits=k, shuffle=True, random_state=42) 

# Use cross_val_score to get scores for each fold
# Scoring='neg_mean_squared_error' because score functions aim to be maximized
# Lower MSE is better, so higher negative MSE is better.
try:
    mse_scores = cross_val_score(
        model, 
        X, 
        y, 
        cv=cv_splitter, 
        scoring='neg_mean_squared_error' 
    )
    
    # Scores are negative MSE, convert back to positive MSE
    rmse_scores = np.sqrt(-mse_scores) 
    
    print(f"\nCross-validation RMSE scores for each fold: {np.round(rmse_scores, 3)}")
    print(f"  Mean RMSE across folds: {np.mean(rmse_scores):.3f}")
    print(f"  Std Dev of RMSE across folds: {np.std(rmse_scores):.3f}")
    print("  (This mean RMSE estimates the model's expected prediction error on unseen data)")

except ImportError:
    print("\nError: scikit-learn is required ('pip install scikit-learn').")
except Exception as e:
    print(f"\nAn error occurred during cross-validation: {e}")

print("-" * 20)

# Explanation: This code demonstrates k-fold cross-validation conceptually for a simple 
# linear regression model using scikit-learn.
# 1. It simulates feature data `X` and target data `y` with a linear relationship plus noise.
# 2. It defines the model using `sklearn.linear_model.LinearRegression()`.
# 3. It specifies `k=5` folds and creates a `KFold` splitter object (which handles 
#    dividing the data into train/test sets for each fold). `shuffle=True` randomizes 
#    the data order before splitting.
# 4. `cross_val_score` automatically performs the k-fold process: for each fold, it 
#    trains the `model` on k-1 folds and evaluates it on the held-out fold using the 
#    specified `scoring` metric ('neg_mean_squared_error').
# 5. It returns an array `mse_scores` containing the score for each fold. The code 
#    converts these to Root Mean Squared Error (RMSE) and prints the scores for each 
#    fold, along with their mean and standard deviation. The mean RMSE provides an 
#    estimate of the model's typical prediction error on new data.
```

Cross-validation provides a valuable perspective on model performance focused on prediction accuracy rather than just goodness-of-fit to the training data. It is particularly crucial in machine learning contexts where overfitting is a major concern. While computationally more intensive than calculating AIC/BIC (as it requires fitting the model `k` times), k-fold CV gives a more direct estimate of generalization error. However, it doesn't inherently penalize model complexity in the same way as AIC/BIC or Bayesian evidence, although models that overfit will typically show poor performance on the held-out test folds. It's a complementary tool in the model assessment toolkit, particularly useful for selecting between different model types or tuning hyperparameters based on expected predictive power.

**Application 18.A: Comparing Pulsar Spin-Down Models using AIC/BIC**

**Objective:** This application demonstrates frequentist model comparison techniques (Sec 18.4) – specifically the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) – to choose between competing models describing pulsar timing data. We will fit a simple spin-down model and a more complex model (e.g., including braking index or timing noise component) to simulated pulsar timing residuals using Maximum Likelihood Estimation (or equivalent Chi-squared minimization) and then use AIC/BIC to determine which model provides a more appropriate balance between goodness-of-fit and model complexity.

**Astrophysical Context:** Pulsars are rotating neutron stars whose rotation gradually slows down due to the emission of electromagnetic radiation or particle winds. This "spin-down" can be precisely measured by timing the arrival of pulses over long periods. The simplest model assumes spin-down occurs due to magnetic dipole radiation, leading to a specific relationship between the pulse frequency (ν) and its derivatives (ν̇, ν̈), often parameterized by a "braking index" n (where ν̇ ∝ -νⁿ, with n=3 for pure dipole). Real pulsars often deviate from this simple model due to effects like magnetic field evolution, glitches (sudden spin-ups), or "timing noise" (stochastic variations in rotation). Choosing the appropriate model to describe a pulsar's spin evolution is crucial for understanding its physical properties and long-term behavior.

**Data Source:** Simulated time-of-arrival (TOA) residuals for a pulsar observed over several years. Residuals represent the difference between observed TOAs and those predicted by a simple constant-frequency model. We will simulate data consistent with a quadratic spin-down model (ν and ν̇ terms) plus potentially a cubic term (ν̈ related to braking index) or added noise. Data consists of arrays: `time` (e.g., MJD) and `residuals` (in phase or seconds). We also need uncertainties `errors` on the residuals, assumed Gaussian.

**Modules Used:** `numpy` (for data simulation and calculations), `scipy.optimize.minimize` (for fitting via Chi-squared minimization, equivalent to MLE for Gaussian errors), `matplotlib.pyplot` (for visualization).

**Technique Focus:** Defining multiple (nested) models for the timing residuals. Fitting each model to the data using Chi-squared minimization (Sec 18.3) to find the best-fit parameters and the minimum Chi-squared value (χ²_min). Calculating the maximum log-likelihood from the minimum Chi-squared (ln L = -0.5 * χ²_min + constant; the constant cancels in differences). Calculating AIC = χ²_min + 2k and BIC = χ²_min + k*ln(n) for each model, where `k` is the number of free parameters and `n` is the number of data points. Comparing AIC/BIC values to select the preferred model (lower value is better).

**Processing Step 1: Simulate/Load Data:** Generate or load `time`, `residuals`, `errors` arrays. Simulate residuals consistent with, e.g., `res(t) = c + f*t + 0.5*fd*t² [+ 1/6*fdd*t³] + noise`.

**Processing Step 2: Define Models:**
    *   Model 1 (Quadratic): `model1(t, params=[c, f, fd]) = c + f*t + 0.5*fd*t**2`. Has k₁=3 parameters.
    *   Model 2 (Cubic/Braking Index): `model2(t, params=[c, f, fd, fdd]) = c + f*t + 0.5*fd*t**2 + (1./6.)*fdd*t**3`. Has k₂=4 parameters. (Model 1 is nested within Model 2).
    *   Define the Chi-squared objective function `chi_squared(params, t, res, err, model_func)` that takes parameters, data, and the appropriate model function.

**Processing Step 3: Fit Models:**
    *   Find initial guesses for parameters for both models.
    *   Use `minimize(chi_squared, guess1, args=(..., model1), ...)` to find best-fit `params1` and minimum `chisq1`.
    *   Use `minimize(chi_squared, guess2, args=(..., model2), ...)` to find best-fit `params2` and minimum `chisq2`. Ensure fits converge successfully.

**Processing Step 4: Calculate AIC/BIC:**
    *   Get number of data points `n = len(time)`.
    *   AIC₁ = chisq1 + 2 * k₁
    *   AIC₂ = chisq2 + 2 * k₂
    *   BIC₁ = chisq1 + k₁ * np.log(n)
    *   BIC₂ = chisq2 + k₂ * np.log(n)
    *   Calculate ΔAIC = AIC₂ - AIC₁ and ΔBIC = BIC₂ - BIC₁.

**Processing Step 5: Compare and Conclude:** Print AIC and BIC values. The model with the lower value is preferred by that criterion. Interpret the ΔAIC and ΔBIC values using rules of thumb (e.g., Δ > 6 indicates strong preference for the model with the lower value). State the conclusion about whether the data significantly supports the inclusion of the cubic term (fdd). Optionally, plot the residuals and both model fits.

**Output, Testing, and Extension:** Output includes best-fit parameters for both models, the minimum Chi-squared for each, the AIC/BIC values, and the conclusion based on comparing them. An optional plot shows the fits. **Testing:** Verify fits converged. Check if AIC/BIC calculations use correct `k` and `n`. Simulate data where the cubic term is truly zero and verify AIC/BIC correctly prefer the simpler model. Simulate data where the cubic term is significant and verify AIC/BIC prefer the more complex model. **Extensions:** (1) Perform a Likelihood Ratio Test (LRT, Sec 18.4) since the models are nested: calculate D = chisq1 - chisq2 and compare to χ² distribution with df = k₂ - k₁ = 1 to get a p-value for the significance of the cubic term. (2) Fit models including sinusoidal terms to search for binary companions. (3) Fit models incorporating timing noise using Gaussian Processes or specialized pulsar timing software like TEMPO/PINT which use generalized least squares. (4) Use Bayesian model comparison with Bayes Factors (Sec 18.5) if evidence values can be calculated (e.g., using nested sampling).

```python
# --- Code Example: Application 18.A ---
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

print("Comparing Pulsar Spin-Down Models using AIC/BIC:")

# Step 1: Simulate Pulsar Timing Residuals
np.random.seed(100)
n_points = 150
time = np.linspace(0, 10, n_points) # Years
# True parameters (c=offset, f=freq_offset, fd=fdot, fdd=fddot)
# Small fdd case first, where quadratic might be preferred
params_true = [0.1, 0.05, -1e-3, 5e-6] 
# params_true = [0.1, 0.05, -1e-3, 5e-5] # Larger fdd case
res_true = params_true[0] + params_true[1]*time + 0.5*params_true[2]*time**2 + (1./6.)*params_true[3]*time**3
res_err = 0.02 # Assume constant error (microseconds or phase)
res_obs = res_true + np.random.normal(0, res_err, size=n_points)
print(f"\nGenerated {n_points} simulated timing residuals.")

# Step 2: Define Models and Chi-squared
def model_quad(t, params): # k=3
    c, f, fd = params
    return c + f*t + 0.5*fd*t**2

def model_cubic(t, params): # k=4
    c, f, fd, fdd = params
    return c + f*t + 0.5*fd*t**2 + (1./6.)*fdd*t**3

def chi_squared_psr(params, t, res, err, model_func):
    res_model = model_func(t, params)
    chisq = np.sum(((res - res_model) / err)**2)
    return chisq if np.isfinite(chisq) else np.inf

# Step 3: Fit Models
print("\nFitting models...")
# Initial guesses (can be tricky, maybe fit polynomial first)
guess1 = [0, 0, 0] # For quadratic
guess2 = [0, 0, 0, 0] # For cubic

# Fit Model 1 (Quadratic)
result1 = minimize(chi_squared_psr, guess1, args=(time, res_obs, res_err, model_quad), method='BFGS')
params1 = result1.x
chisq1 = result1.fun
k1 = len(params1)
success1 = result1.success

# Fit Model 2 (Cubic)
result2 = minimize(chi_squared_psr, guess2, args=(time, res_obs, res_err, model_cubic), method='BFGS')
params2 = result2.x
chisq2 = result2.fun
k2 = len(params2)
success2 = result2.success

if not (success1 and success2):
    print("Warning: One or both fits did not converge successfully!")
else:
    print("Both models fitted successfully.")
    print(f"Model 1 (Quadratic): k={k1}, Chi2={chisq1:.2f}")
    print(f"Model 2 (Cubic):     k={k2}, Chi2={chisq2:.2f}")

    # Step 4: Calculate AIC/BIC
    n_data = len(time)
    aic1 = chisq1 + 2 * k1
    aic2 = chisq2 + 2 * k2
    bic1 = chisq1 + k1 * np.log(n_data)
    bic2 = chisq2 + k2 * np.log(n_data)
    
    print("\nModel Comparison Criteria:")
    print(f"  AIC1 = {aic1:.2f}")
    print(f"  AIC2 = {aic2:.2f}")
    print(f"  BIC1 = {bic1:.2f}")
    print(f"  BIC2 = {bic2:.2f}")
    
    delta_aic = aic2 - aic1
    delta_bic = bic2 - bic1
    print(f"  Delta AIC (AIC2 - AIC1) = {delta_aic:.2f}")
    print(f"  Delta BIC (BIC2 - BIC1) = {delta_bic:.2f}")

    # Step 5: Compare and Conclude
    print("\nConclusion:")
    preferred_aic = "Model 2 (Cubic)" if delta_aic < -2 else ("Model 1 (Quad)" if delta_aic > 2 else "Comparable")
    preferred_bic = "Model 2 (Cubic)" if delta_bic < -2 else ("Model 1 (Quad)" if delta_bic > 2 else "Comparable") 
    # Stricter BIC threshold? e.g., delta_bic < -6
    
    print(f"  AIC prefers: {preferred_aic}")
    print(f"  BIC prefers: {preferred_bic}")
    if preferred_aic != preferred_bic:
         print("  Note: AIC and BIC give different preferences.")
         
    # --- Optional Plot ---
    print("\nGenerating plot of residuals and fits...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(time, res_obs, yerr=res_err, fmt='.', label='Data')
    time_smooth = np.linspace(time.min(), time.max(), 200)
    ax.plot(time_smooth, model_quad(time_smooth, params1), label=f'Quad Fit (χ²={chisq1:.1f})')
    ax.plot(time_smooth, model_cubic(time_smooth, params2), label=f'Cubic Fit (χ²={chisq2:.1f})')
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Timing Residuals")
    ax.set_title("Pulsar Spin-Down Model Fits")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    # plt.show()
    print("Plot generated.")
    plt.close(fig)

print("-" * 20)
```

**Application 18.B: Comparing Galaxy Rotation Curve Models (Bayesian Evidence)**

**(Paragraph 1)** **Objective:** This application demonstrates Bayesian model comparison (Sec 18.5) using Bayes Factors derived from evidence calculations (typically via nested sampling) to compare two different physical models describing a galaxy's rotation curve: one based only on luminous matter (stars and gas) and one including a dark matter halo component. Reinforces Sec 17.4 (use of `dynesty`), 18.5.

**Astrophysical Context:** Galaxy rotation curves – plots of circular rotation velocity versus distance from the galactic center – provide fundamental tests of our understanding of gravity and mass distribution in galaxies. Observations (from HI 21cm line, CO lines, or stellar kinematics) often show that rotation velocities remain flat or even rise at large radii, far beyond the visible extent of stars and gas. This discrepancy is primary evidence for the existence of **dark matter** halos surrounding galaxies. Bayesian model comparison allows us to quantitatively assess the evidence favoring a model *with* a dark matter component over one *without* it, given the observed rotation curve data.

**Data Source:** Observed rotation curve data for a galaxy, consisting of arrays: `radius` (distance from center, e.g., in kpc), `velocity` (observed rotation velocity, e.g., in km/s), and `velocity_error` (uncertainty on velocity). Additionally, information derived from photometry about the expected contribution of luminous matter (stars/gas) to the rotation curve, V<0xE1><0xB5><0x97><0xE1><0xB5><0x98><0xE1><0xB5><0x8B><0xE1><0xB5><0x8A><0xE1><0xB5><0x92><0xE1><0xB5><0x98><0xE1><0xB5><0x98><0xE1><0xB5><0x98>(r), is needed. We will simulate this data.

**Modules Used:** `dynesty` (for nested sampling and evidence estimation), `numpy`, `matplotlib.pyplot`, `astropy.units`, `astropy.constants`. Potentially `astropy.modeling` if used to define components.

**Technique Focus:** Setting up two competing physical models for the rotation curve V(r). Model 1: V²(r) = V<0xE1><0xB5><0x97><0xE1><0xB5><0x98><0xE1><0xB5><0x8A><0xE1><0xB5><0x92><0xE1><0xB5><0x98><0xE1><0xB5><0x98><0xE1><0xB5><0x98>²(r) * (M/L), where M/L (mass-to-light ratio) is the only free parameter scaling the fixed luminous contribution. Model 2: V²(r) = V<0xE1><0xB5><0x97><0xE1><0xB5><0x98><0xE1><0xB5><0x8A><0xE1><0xB5><0x92><0xE1><0xB5><0x98><0xE1><0xB5><0x98><0xE1><0xB5><0x98>²(r) * (M/L) + V<0xE1><0xB5><0x87><0xE1><0xB5><0x86><0xE1><0xB5><0x8B><0xE1><0xB5><0x92>²(r, params<0xE1><0xB5><0x87><0xE1><0xB5><0x86><0xE1><0xB5><0x8B><0xE1><0xB5><0x92>), where V<0xE1><0xB5><0x87><0xE1><0xB5><0x86><0xE1><0xB5><0x8B><0xE1><0xB5><0x92>² is the contribution from a dark matter halo model (e.g., NFW or pseudo-isothermal) with its own parameters (params<0xE1><0xB5><0x87><0xE1><0xB5><0x86><0xE1><0xB5><0x8B><0xE1><0xB5><0x92>). Defining appropriate priors, log-likelihood functions, and prior transform functions for both models. Running `dynesty` separately for each model to obtain the log-evidence estimates (ln Z₁ and ln Z₂). Calculating the log Bayes Factor ln K = ln Z₂ - ln Z₁. Interpreting ln K to determine the strength of evidence favoring the dark matter model.

**Processing Step 1: Simulate/Load Data:** Generate or load `radius`, `velocity`, `velocity_error` arrays. Also need the *fixed* luminous contribution profile V<0xE1><0xB5><0x97><0xE1><0xB5><0x98><0xE1><0xB5><0x8A><0xE1><0xB5><0x92><0xE1><0xB5><0x98><0xE1><0xB5><0x98><0xE1><0xB5><0x98>(r). Simulate data where a dark matter halo is clearly needed to match the outer rotation curve.

**Processing Step 2: Define Models, Priors, Likelihoods, Transforms:**
    *   Model 1 (Luminous only): Parameter θ₁ = [M/L]. Model: V² = (M/L) * V<0xE1><0xB5><0x97><0xE1><0xB5><0x98><0xE1><0xB5><0x8A><0xE1><0xB5><0x92><0xE1><0xB5><0x98><0xE1><0xB5><0x98><0xE1><0xB5><0x98>(r)²`. Define `log_likelihood1`, `prior_transform1` (e.g., uniform M/L).
    *   Model 2 (Luminous + DM Halo): Parameters θ₂ = [M/L, HaloParam1, HaloParam2, ...]. Model: V² = (M/L) * V<0xE1><0xB5><0x97><0xE1><0xB5><0x98><0xE1><0xB5><0x8A><0xE1><0xB5><0x92><0xE1><0xB5><0x98><0xE1><0xB5><0x98><0xE1><0xB5><0x98>(r)² + V<0xE1><0xB5><0x87><0xE1><0xB5><0x86><0xE1><0xB5><0x8B><0xE1><0xB5><0x92>(r, HaloParams)²`. Define `log_likelihood2`, `prior_transform2` (priors for M/L and halo parameters).
    *   The log-likelihood function in both cases uses Gaussian assumption: ln L ∝ -0.5 * Σ [(V<0xE1><0xB5><0x92><0xE1><0xB5><0x87>ₛᵢ - V<0xE1><0xB5><0x9B><0xE1><0xB5><0x92><0xE1><0xB5><0x87><0xE1><0xB5><0x86>(rᵢ, θ))² / V<0xE1><0xB5><0x98><0xE1><0xB5><0xA3><0xE1><0xB5><0xA3>ᵢ²].

**Processing Step 3: Run Nested Sampling:**
    *   Instantiate `dynesty.NestedSampler` (or `DynamicNestedSampler`) for Model 1 using `log_likelihood1`, `prior_transform1`, `ndim1`. Run `sampler1.run_nested(...)`. Get results `res1 = sampler1.results`. Extract log evidence `logz1 = res1.logz[-1]`.
    *   Instantiate sampler for Model 2 using `log_likelihood2`, `prior_transform2`, `ndim2`. Run `sampler2.run_nested(...)`. Get results `res2 = sampler2.results`. Extract log evidence `logz2 = res2.logz[-1]`.

**Processing Step 4: Calculate and Interpret Bayes Factor:** Calculate log Bayes Factor `ln_K = logz2 - logz1`. Estimate uncertainty if `logzerr` is available. Interpret the value of `ln_K` using Jeffreys scale or similar criteria to assess the strength of evidence favoring the model with dark matter (Model 2) over the luminous-only model (Model 1).

**Processing Step 5: Visualization:** Plot the rotation curve data points with error bars. Overplot the best-fit curves (e.g., using median posterior parameters from `res1.samples_equal()` and `res2.samples_equal()`) for both Model 1 and Model 2. Visually compare how well each model describes the data, particularly at large radii.

**Output, Testing, and Extension:** Output includes the log evidence values (ln Z₁ and ln Z₂) for both models, the calculated log Bayes Factor (ln K), and its interpretation regarding the evidence for dark matter. The plot shows the data and both model fits. **Testing:** Check `dynesty` convergence diagnostics (summary plots). Ensure priors are reasonable and cover plausible parameter ranges. Verify the fits visually. Check if ln K strongly favors the DM model if the simulated data clearly requires it. **Extensions:** (1) Compare different dark matter halo profiles (e.g., NFW vs. pseudo-isothermal) using Bayes Factors. (2) Include the stellar mass-to-light ratio (M/L) as a free parameter common to both models. (3) Use real rotation curve data from a galaxy. (4) Compare the Bayesian evidence results with conclusions drawn from AIC/BIC calculated using the maximum likelihood values (which `dynesty` might also estimate).

```python
# --- Code Example: Application 18.B ---
# Note: Requires dynesty, astropy. Needs careful model implementation. Conceptual.
import numpy as np
import matplotlib.pyplot as plt
import dynesty
from astropy import units as u
from astropy import constants as const
from scipy import stats # For prior transform examples

print("Bayesian Comparison of Galaxy Rotation Curve Models (Conceptual):")

# Step 1: Simulate/Load Data
# Assume we have: radius (kpc), vel_obs (km/s), vel_err (km/s)
# And Vlum(r) profile (km/s) from luminous matter (fixed shape, scaled by M/L)
np.random.seed(1)
radius = np.array([1, 2, 3, 5, 8, 12, 18, 25, 35]) * u.kpc
# Simulate Vlum contribution (e.g., from disk model) - rises then falls
Vlum_profile = (150 * (radius.value / (radius.value + 2)) * np.exp(-radius.value/20.0)) * (u.km/u.s)
# Simulate DM halo contribution (e.g., pseudo-isothermal V ~ const at large R)
true_halo_V = 180.0 * u.km / u.s # Flat part
Vhalo_profile = true_halo_V * (radius.value / np.sqrt(radius.value**2 + 5**2)) # Rises then flattens
# True parameters
true_ML = 1.5 # Mass-to-light scaling for Vlum
# Total true velocity squared = (M/L)*Vlum^2 + Vhalo^2
vel_true_sq = true_ML * Vlum_profile**2 + Vhalo_profile**2
vel_true = np.sqrt(vel_true_sq)
# Add noise
vel_err = 10.0 * u.km / u.s
vel_obs = vel_true + np.random.normal(0, vel_err.value, size=radius.shape) * vel_err.unit

print("\nSimulated Rotation Curve Data Generated.")
print(f"Radii (kpc): {radius.value}")
print(f"Obs Velocity (km/s): {vel_obs.value.round(1)}")

# Step 2: Define Models, Priors, Likelihoods, Transforms

# --- Model 1: Luminous Only ---
ndim1 = 1 # Parameter: M/L ratio
def loglike1(theta): # theta = [ML]
    ML = theta[0]
    if ML <= 0: return -np.inf
    model_vel_sq = ML * Vlum_profile**2
    model_vel = np.sqrt(model_vel_sq)
    chisq = np.sum(((vel_obs - model_vel) / vel_err)**2)
    return -0.5 * chisq # Gaussian log-likelihood (ignoring constant)

def prior_transform1(u_vec): # u_vec is length 1
    # Uniform prior for M/L between 0.1 and 5.0
    ML = 0.1 + u_vec[0] * (5.0 - 0.1)
    return np.array([ML])

# --- Model 2: Luminous + DM Halo (Pseudo-Isothermal example) ---
# Vhalo^2 = Vinf^2 * (r^2 / (r^2 + rc^2))
# Parameters: theta = [ML, Vinf, rc]
ndim2 = 3 
def Vhalo_sq_iso(r, Vinf, rc):
    if Vinf <=0 or rc <=0: return -np.inf # Invalid physical value
    r_val = r.to(u.kpc).value
    return Vinf**2 * (r_val**2 / (r_val**2 + rc**2)) # Units (km/s)^2

def loglike2(theta): # theta = [ML, Vinf, rc]
    ML, Vinf, rc = theta
    if ML <= 0 or Vinf <= 0 or rc <= 0: return -np.inf
    model_vel_sq = ML * Vlum_profile**2 + Vhalo_sq_iso(radius, Vinf, rc)*(u.km/u.s)**2
    # Avoid sqrt of negative if model becomes unphysical
    if np.any(model_vel_sq < 0): return -np.inf
    model_vel = np.sqrt(model_vel_sq)
    chisq = np.sum(((vel_obs - model_vel) / vel_err)**2)
    return -0.5 * chisq

def prior_transform2(u_vec): # u_vec is length 3
    theta = np.zeros_like(u_vec)
    # Uniform M/L prior [0.1, 5.0]
    theta[0] = 0.1 + u_vec[0] * (5.0 - 0.1) 
    # Uniform Vinf prior [10, 300] km/s
    theta[1] = 10.0 + u_vec[1] * (300.0 - 10.0)
    # Uniform rc prior [0.1, 20] kpc
    theta[2] = 0.1 + u_vec[2] * (20.0 - 0.1)
    # Can use stats.norm.ppf(u_vec[i], loc, scale) for Gaussian priors
    return theta

# Step 3: Run Nested Sampling (Conceptual Execution)
print("\nRunning Nested Sampling (Conceptual)...")
logz1, logz2 = None, None
results1, results2 = None, None
try:
    # --- Run Model 1 ---
    # sampler1 = dynesty.NestedSampler(loglike1, prior_transform1, ndim1, nlive=200)
    # sampler1.run_nested(dlogz=0.1)
    # results1 = sampler1.results
    # logz1 = results1.logz[-1]
    # Simulate result for Model 1
    logz1 = -35.2 # Example value (likely poor fit)
    print("  (Simulated Dynesty run for Model 1 completed)")
    print(f"  Model 1 Log Evidence (ln Z1) ~ {logz1:.2f}")

    # --- Run Model 2 ---
    # sampler2 = dynesty.NestedSampler(loglike2, prior_transform2, ndim2, nlive=500)
    # sampler2.run_nested(dlogz=0.1)
    # results2 = sampler2.results
    # logz2 = results2.logz[-1]
    # Simulate result for Model 2
    logz2 = -12.5 # Example value (much better fit expected)
    print("  (Simulated Dynesty run for Model 2 completed)")
    print(f"  Model 2 Log Evidence (ln Z2) ~ {logz2:.2f}")

except ImportError:
     print("Error: dynesty package is required.")
     logz1, logz2 = None, None # Ensure they are None if dynesty not installed
except Exception as e:
     print(f"Error during nested sampling simulation: {e}")
     logz1, logz2 = None, None 

# Step 4: Calculate and Interpret Bayes Factor
if logz1 is not None and logz2 is not None:
    ln_K = logz2 - logz1 # Bayes Factor K = Z2/Z1 in favor of Model 2
    print(f"\nLog Bayes Factor (ln K = ln Z2 - ln Z1) = {ln_K:.2f}")
    # Interpretation
    if ln_K > 5.0: print("  -> Decisive evidence for Model 2 (Luminous + DM Halo).")
    elif ln_K > 2.5: print("  -> Strong evidence for Model 2.")
    # ... etc ...
    else: print("  -> Evidence inconclusive or favors Model 1.")
else:
    print("\nCould not calculate Bayes Factor due to errors.")

# Step 5: Visualization (Conceptual - needs results objects for best fit)
print("\n(Visualization would plot data vs best-fit curves for both models)")

print("-" * 20)
```

**Summary**

This chapter focused on the crucial final steps of statistical modeling: fitting models to data and comparing the suitability of different competing models. It began by exploring different ways to define models programmatically in Python, from simple functions suitable for basic fitting routines to the more structured, object-oriented approach offered by `astropy.modeling`, which facilitates building complex, multi-component models. Common fitting techniques based on minimizing the difference between data and model were revisited, specifically standard non-linear least-squares fitting using `scipy.optimize.curve_fit` (which directly provides parameter errors via the covariance matrix assuming Gaussian data uncertainties with `absolute_sigma=True`) and the more general Chi-squared minimization approach using `scipy.optimize.minimize`, highlighting their connection to Maximum Likelihood Estimation under Gaussian assumptions and their application to both point data and binned histogram data.

The core of the chapter then addressed the critical task of model selection – objectively choosing between models of differing complexity or physical basis. Frequentist approaches were presented, including the Likelihood Ratio Test (LRT) specifically for comparing nested models based on the significance of the improvement in maximum likelihood, and the widely used Information Criteria – AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion). Both AIC and BIC balance goodness-of-fit (maximum likelihood) against model complexity (number of parameters `k`, with BIC also considering sample size `n`), providing metrics where lower values indicate preferred models, but employing different penalty terms reflecting different underlying assumptions (AIC focuses on predictive accuracy, BIC approximates Bayesian evidence and tends to favor simpler models more strongly). The Bayesian approach to model comparison was then detailed, centered on calculating the Bayesian evidence (or marginal likelihood) Z = P(D|M) for each model, which represents the average likelihood over the prior parameter space and naturally penalizes model complexity. The ratio of evidences between two models yields the Bayes Factor K = Z₂/Z₁, which directly quantifies the relative probability of the models given the data, with established scales (like Jeffreys') for interpreting the strength of evidence. The computational challenge of calculating Z and the utility of nested sampling algorithms (like `dynesty`) for estimating it were emphasized. Finally, the concept of cross-validation (particularly k-fold CV) using tools like `sklearn.model_selection.cross_val_score` was introduced as a method focused on assessing a model's predictive performance on unseen data, providing a complementary perspective to methods based solely on fit quality to the training data.

---

**References for Further Reading:**

1.  **Ivezić, Ž., Connolly, A. J., VanderPlas, J. T., & Gray, A. (2014).** *Statistics, Data Mining, and Machine Learning in Astronomy*. Princeton University Press. (Relevant chapters often available online, e.g., Chapters 4 & 5 cover MLE, Bayesian inference, model selection: [http://press.princeton.edu/titles/10159.html](http://press.princeton.edu/titles/10159.html))
    *(Provides extensive coverage of model fitting, MLE, Bayesian evidence, Bayes Factors, AIC, BIC, and cross-validation in an astronomical context.)*

2.  **Feigelson, E. D., & Babu, G. J. (2012).** *Modern Statistical Methods for Astronomy: With R Applications*. Cambridge University Press. [https://doi.org/10.1017/CBO9781139179009](https://doi.org/10.1017/CBO9781139179009)
    *(Covers least squares, Chi-squared minimization, MLE, Likelihood Ratio Tests, AIC, BIC, and introduces Bayesian model comparison concepts.)*

3.  **Burnham, K. P., & Anderson, D. R. (2002).** *Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach* (2nd ed.). Springer-Verlag.
    *(A classic, comprehensive text focusing on information-theoretic approaches to model selection, particularly AIC, providing deep background for Sec 18.4.)*

4.  **Kass, R. E., & Raftery, A. E. (1995).** Bayes Factors. *Journal of the American Statistical Association*, *90*(430), 773–795. [https://doi.org/10.1080/01621459.1995.10476572](https://doi.org/10.1080/01621459.1995.10476572)
    *(A foundational review paper on Bayes Factors, explaining their calculation, interpretation (including the Jeffreys scale), and application for Bayesian model comparison, relevant to Sec 18.5.)*

5.  **Astropy Project Contributors. (n.d.).** *Astropy Documentation: Modeling (astropy.modeling)*. Astropy Documentation. Retrieved January 16, 2024, from [https://docs.astropy.org/en/stable/modeling/](https://docs.astropy.org/en/stable/modeling/)
    *(Documentation for the `astropy.modeling` framework introduced in Sec 18.1, including defining models, combining them, and using its associated fitting routines (which often wrap `scipy.optimize` but provide model-aware features).)*
