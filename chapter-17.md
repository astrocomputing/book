**Chapter 17: Parameter Estimation: Bayesian Methods**

While Maximum Likelihood Estimation (Chapter 16) provides a powerful method for finding the single best-fit parameter values, it often gives an incomplete picture of the uncertainties, especially for complex models or limited data. This chapter introduces the **Bayesian approach** to parameter estimation, a fundamentally different framework that treats parameters not as fixed unknown constants, but as random variables characterized by probability distributions that represent our state of knowledge. We begin by revisiting Bayes' Theorem, reinterpreting its components – prior, likelihood, posterior, and evidence – in the context of parameter inference. The core output of Bayesian analysis, the **posterior probability distribution**, which encapsulates all information about the parameters given the data and prior beliefs, will be highlighted. Since calculating the posterior distribution analytically is often intractable, we will focus on computational techniques, particularly **Markov Chain Monte Carlo (MCMC)** methods, which provide algorithms for drawing samples from the posterior distribution. We will introduce the concepts behind common MCMC algorithms like Metropolis-Hastings and explore practical implementation using popular Python libraries such as `emcee` (ensemble sampler) and `dynesty` (nested sampling). Methods for analyzing the output of MCMC simulations, including assessing convergence, visualizing posterior distributions using corner plots, and deriving parameter estimates and credible intervals, will be demonstrated, showcasing the rich information provided by the Bayesian framework for characterizing model parameters and their uncertainties.

**17.1 Bayesian Inference: Priors, Likelihood, Posterior, Evidence**

The Bayesian approach to statistical inference offers a conceptually different, yet powerful, framework compared to the frequentist methods discussed earlier (like hypothesis testing and Maximum Likelihood Estimation). Instead of viewing model parameters (θ) as fixed, unknown constants and calculating the probability of observing data given those parameters, Bayesian inference treats the parameters themselves as random variables whose uncertainty can be described by probability distributions. The core of Bayesian inference lies in updating our knowledge about these parameters – expressed as a **prior probability distribution** P(θ) – using observed data `D` via the **likelihood function** P(D|θ), resulting in an updated **posterior probability distribution** P(θ|D). This process is governed by **Bayes' Theorem**:

P(θ|D) = [ P(D|θ) * P(θ) ] / P(D)

Let's re-examine the terms in the context of parameter estimation:
*   **P(θ) - Prior Probability Distribution:** This distribution represents our state of knowledge or belief about the parameters θ *before* considering the specific dataset D. It can incorporate previous experimental results, physical constraints, theoretical expectations, or simply express a state of relative ignorance (uninformative priors). The choice of prior is a defining characteristic of Bayesian analysis and can influence the resulting posterior, especially when data is not very informative. Common choices include uniform priors over a plausible range, Gaussian priors centered on expected values, or priors derived from physical principles.
*   **P(D|θ) - Likelihood Function:** This is mathematically the same function used in MLE (Sec 16.1). It represents the probability (or probability density) of observing the data D *given* a specific set of parameter values θ. It quantifies how well the model, with parameters θ, predicts the data. The likelihood function contains all the information brought by the current dataset D.
*   **P(θ|D) - Posterior Probability Distribution:** This is the central result of a Bayesian analysis. It represents our updated state of knowledge or belief about the parameters θ *after* having observed the data D and incorporated our prior beliefs. The posterior distribution encapsulates all information about the parameters – including their most probable values, uncertainties, and correlations – consistent with both the prior information and the observed data. It is a full probability distribution, not just a single point estimate or standard error.
*   **P(D) - Evidence (or Marginal Likelihood):** This term represents the probability of observing the data D integrated (or summed) over all possible values of the parameters θ, weighted by the prior: P(D) = ∫ P(D|θ) * P(θ) dθ. In the context of parameter estimation for a *single* model, P(D) acts primarily as a normalization constant, ensuring that the posterior distribution P(θ|D) integrates (or sums) to 1 over all possible θ. Its value ensures that P(θ|D) is a proper probability distribution. However, the evidence P(D) becomes critically important when comparing different *models* (Bayesian Model Selection, Chapter 18), where it quantifies how well the model *as a whole* fits the data, averaged over its parameter space.

Since the evidence P(D) is constant for a given dataset D and model when considering only parameter estimation, Bayes' Theorem is often written in proportional form:

**P(θ|D) ∝ P(D|θ) * P(θ)**
**Posterior ∝ Likelihood × Prior**

This highlights the intuitive nature of Bayesian updating: our final belief (posterior) is proportional to how well the parameters explain the data (likelihood) weighted by our initial belief (prior). Data that is strongly predicted by certain parameter values (high likelihood) will increase the posterior probability for those values, while prior beliefs down-weight parameters considered implausible beforehand.

The power of the Bayesian approach lies in providing the full posterior probability distribution P(θ|D). This distribution contains much richer information than a single point estimate (like the MLE) and its standard error. From the posterior, we can derive various summary statistics:
*   **Point estimates:** The **mode** of the posterior (Maximum A Posteriori or MAP estimate, which coincides with the MLE if the prior is uniform), the **mean** of the posterior, or the **median** of the posterior can all serve as representative "best-fit" values.
*   **Uncertainty quantification:** **Credible intervals** (or credible regions for multiple parameters) represent ranges within which the true parameter value lies with a certain probability (e.g., a 95% credible interval contains the true value with 95% probability according to our posterior belief). These are calculated directly from the posterior distribution (e.g., by finding the interval containing 95% of the probability mass, often the central 95%). This interpretation is often considered more intuitive than the frequentist confidence interval.
*   **Correlations:** The shape of the multi-dimensional posterior distribution reveals correlations between parameters – how uncertainty in one parameter relates to uncertainty in another.

The main challenge in Bayesian inference is often computational. While the formula P(θ|D) ∝ P(D|θ) * P(θ) defines the *shape* of the posterior, calculating the normalization constant P(D) (the evidence) requires integrating the likelihood times the prior over the entire parameter space, which is often high-dimensional and analytically intractable. Furthermore, even without the normalization, exploring and summarizing the potentially complex, high-dimensional shape of the posterior distribution P(θ|D) to find means, medians, credible intervals, and correlations often requires sophisticated numerical methods.

This is where **Markov Chain Monte Carlo (MCMC)** methods, discussed in the next sections, become indispensable. MCMC provides algorithms that allow us to draw a large number of samples {θ₁, θ₂, ..., θ<0xE1><0xB5><0x8A>} directly from the posterior distribution P(θ|D), *without needing to calculate the normalization constant P(D)*. By analyzing the distribution of these samples (e.g., using histograms, calculating means/medians/percentiles), we can effectively map out and summarize the properties of the posterior distribution, providing estimates for parameters, their uncertainties (credible intervals), and correlations.

In summary, Bayesian inference offers a powerful framework for parameter estimation based on updating prior beliefs with observed data via the likelihood function, governed by Bayes' Theorem. Its primary output, the posterior probability distribution P(θ|D), provides a complete characterization of our knowledge about the model parameters, including uncertainties and correlations. While analytical solutions are rare, computational techniques like MCMC enable us to explore and summarize the posterior distribution effectively, making Bayesian methods increasingly practical and popular in astrophysics.

**17.2 Markov Chain Monte Carlo (MCMC)**

The core computational challenge in Bayesian inference is often exploring and characterizing the posterior probability distribution P(θ|D) ∝ L(θ|D)P(θ). Except for very simple models, this distribution can be high-dimensional (if there are many parameters θ) and have a complex shape (multiple peaks, strong correlations, non-Gaussian tails). Calculating properties like the mean, median, or credible intervals requires integrating over this potentially complicated distribution, which is usually analytically intractable and computationally difficult using standard numerical integration techniques (like grid-based methods) in high dimensions (due to the "curse of dimensionality"). **Markov Chain Monte Carlo (MCMC)** methods provide a powerful class of algorithms specifically designed to tackle this problem by generating samples from a target probability distribution (in our case, the posterior) without needing to calculate its normalization constant (the evidence P(D)).

The fundamental idea behind MCMC is to construct a **Markov chain** – a sequence of random samples {θ₀, θ₁, θ₂, ...} – whose values progressively explore the parameter space θ. A Markov chain has the property that the next state θ<0xE1><0xB5><0xA2>₊₁ depends only on the current state θ<0xE1><0xB5><0xA2> and not on the sequence of states that preceded it (the "Markov property"). MCMC algorithms are designed such that, under certain conditions, the **stationary distribution** of this Markov chain is precisely the target posterior distribution P(θ|D). This means that after an initial "burn-in" period (where the chain converges from its starting point to the high-probability regions of the posterior), the subsequent samples {θ<0xE1><0xB5><0x8D>, θ<0xE1><0xB5><0x8D>₊₁, ...} drawn from the chain effectively represent random draws from the desired posterior distribution P(θ|D).

By generating a large number of samples {θ<0xE1><0xB5><0xA2>} from the chain *after* it has converged, we obtain an empirical representation of the posterior distribution. Regions of the parameter space where the posterior probability P(θ|D) is high will be visited more frequently by the chain, resulting in a higher density of samples in those regions. Conversely, low-probability regions will be sampled less often. This collection of samples can then be used to approximate properties of the posterior distribution using standard Monte Carlo techniques.

For example, the mean of a parameter θ<0xE2><0x82><0x97> can be estimated by calculating the average value of θ<0xE2><0x82><0x97> across the MCMC samples. The median can be estimated by finding the 50th percentile of the sampled θ<0xE2><0x82><0x97> values. A 95% credible interval for θ<0xE2><0x82><0x97> can be estimated by finding the 2.5th and 97.5th percentiles of the sampled θ<0xE2><0x82><0x97> values. Correlations between parameters θ<0xE1><0xB5><0xA2> and θ<0xE2><0x82><0x97> can be visualized by creating a 2D histogram or scatter plot of the sampled pairs (θ<0xE1><0xB5><0xA2>, θ<0xE2><0x82><0x97>) – this is the basis of "corner plots" (Sec 17.5).

The key advantage of MCMC is that constructing the chain often only requires the ability to evaluate the *ratio* of posterior probabilities at two different points, P(θ'|D) / P(θ|D). Since the normalization constant P(D) cancels out in this ratio, we only need to be able to calculate the product of the likelihood and the prior, L(θ|D)P(θ), which is usually feasible even when the evidence P(D) is unknown. Specific MCMC algorithms (like Metropolis-Hastings, discussed next) provide rules for proposing moves from the current state θ<0xE1><0xB5><0xA2> to a new state θ' and accepting or rejecting that move based on this probability ratio, ensuring the chain converges to the correct target distribution.

For MCMC methods to work correctly, certain theoretical conditions must be met. The Markov chain needs to be **ergodic**, meaning it must be possible to eventually reach any state (region of parameter space with non-zero probability) from any other state (irreducibility), and the chain should not get stuck in periodic cycles (aperiodicity). If these conditions hold, the distribution of samples generated by the chain is guaranteed to converge to the unique stationary distribution, which we design to be our target posterior P(θ|D).

A critical practical aspect of MCMC is **convergence**. The initial samples generated by the chain (the "burn-in" phase) typically depend on the starting point θ₀ and do not represent the true posterior distribution. We must discard these burn-in samples before analyzing the chain. Determining when the chain has converged and the burn-in phase has ended can be challenging and often involves visual inspection of trace plots (parameter values vs. iteration number) and quantitative convergence diagnostics (like the Gelman-Rubin statistic, Sec 17.5). Running multiple independent chains starting from different, widely dispersed points in parameter space is highly recommended to assess convergence robustly.

Another important consideration is **autocorrelation**. Because each step in a Markov chain depends on the previous step, successive samples {θ<0xE1><0xB5><0xA2>, θ<0xE1><0xB5><0xA2>₊₁} are generally not independent draws from the posterior. They tend to be correlated, meaning nearby samples are similar. The **autocorrelation time** (τ) measures roughly how many steps are needed for the chain to "forget" its previous state and produce a nearly independent sample. To obtain an effective number of independent samples (N_eff ≈ Total Samples / τ) sufficient for accurately estimating posterior properties, the chain often needs to be run for many autocorrelation times beyond the burn-in period. Efficient MCMC algorithms aim to minimize τ. Thinning the chain (keeping only every k-th sample, where k ≈ τ) is sometimes done to reduce storage, but it doesn't increase the amount of information and using the full chain (after burn-in) is often preferred for analysis if computationally feasible.

In summary, MCMC methods provide a powerful computational engine for Bayesian inference. By constructing a Markov chain whose stationary distribution matches the target posterior P(θ|D), MCMC algorithms allow us to generate samples from this distribution, even when it's high-dimensional, complex, and lacks a known normalization constant. Analyzing these samples enables us to estimate parameters, quantify uncertainties via credible intervals, and explore correlations, providing a comprehensive picture of the parameter constraints offered by the data and prior information. Practical implementation requires careful consideration of convergence, burn-in, and autocorrelation.

**17.3 Introduction to MCMC Algorithms (e.g., Metropolis-Hastings)**

While the general concept of MCMC involves creating a Markov chain that samples from the posterior, several specific algorithms exist to construct such chains. The foundational algorithm, upon which many others are built, is the **Metropolis-Hastings (M-H) algorithm**. Understanding the basic steps of M-H provides insight into how MCMC samplers explore the parameter space and converge to the target distribution P(θ|D).

The Metropolis-Hastings algorithm works iteratively as follows:
1.  **Initialization:** Start the chain at an initial parameter vector θ₀.
2.  **Iteration (t=0, 1, 2, ...):** Given the current state θ<0xE1><0xB5><0x8D>, generate a candidate next state θ' using a **proposal distribution** q(θ' | θ<0xE1><0xB5><0x8D>). This distribution suggests a potential move from the current position θ<0xE1><0xB5><0x8D> to a new position θ'. A common choice is a symmetric proposal distribution, like a Gaussian centered on the current state: θ' ~ Normal(θ<0xE1><0xB5><0x8D>, Σ<0xE1><0xB5><0x96>), where Σ<0xE1><0xB5><0x96> is a proposal covariance matrix (often tuned).
3.  **Calculate Acceptance Ratio (α):** Compute the ratio of the target posterior probability density at the proposed state θ' to the density at the current state θ<0xE1><0xB5><0x8D>. Since P(θ|D) ∝ L(θ|D)P(θ), this ratio simplifies to:
    α = [ P(θ'|D) / P(θ<0xE1><0xB5><0x8D>|D) ] * [ q(θ<0xE1><0xB5><0x8D> | θ') / q(θ' | θ<0xE1><0xB5><0x8D>) ]
    The second term involving the proposal distribution ratio is needed for non-symmetric proposals (Hastings modification). If the proposal distribution `q` is symmetric (i.e., q(θ'|θ) = q(θ|θ')), like the Gaussian example, this term becomes 1, simplifying the acceptance ratio to just the ratio of posterior densities (or Likelihood × Prior):
    α = P(θ'|D) / P(θ<0xE1><0xB5><0x8D>|D) = [ L(θ'|D)P(θ') ] / [ L(θ<0xE1><0xB5><0x8D>|D)P(θ<0xE1><0xB5><0x8D>) ]
    Note that the normalization constant P(D) cancels out, which is a key advantage.
4.  **Accept or Reject:** Generate a random number `u` uniformly from [0, 1].
    *   If `u ≤ min(1, α)`, **accept** the proposed state: set θ<0xE1><0xB5><0x8D>₊₁ = θ'.
    *   If `u > min(1, α)`, **reject** the proposed state: set θ<0xE1><0xB5><0x8D>₊₁ = θ<0xE1><0xB5><0x8D> (i.e., the chain stays at the current position for this step).
5.  **Repeat:** Go back to Step 2 with the new current state θ<0xE1><0xB5><0x8D>₊₁.

The crucial step is the acceptance probability `min(1, α)`. If the proposed state θ' has a higher posterior probability than the current state θ<0xE1><0xB5><0x8D> (i.e., α > 1), the move is always accepted. If the proposed state has a lower posterior probability (α < 1), the move is accepted only *probabilistically* with probability α. This allows the chain to occasionally move "downhill" in probability, enabling it to escape local probability peaks and explore the entire distribution. It can be shown that this acceptance rule ensures the chain satisfies the detailed balance condition, guaranteeing that its stationary distribution is the target posterior P(θ|D).

The efficiency of the M-H algorithm heavily depends on the choice of the **proposal distribution** q(θ' | θ<0xE1><0xB5><0x8D>). If the proposal steps (controlled by Σ<0xE1><0xB5><0x96> in the Gaussian case) are too small, the chain explores the parameter space very slowly (high autocorrelation time τ), and the acceptance rate might be very high (most moves accepted, but tiny steps). If the proposal steps are too large, the chain frequently proposes moves to regions of much lower probability, leading to a very low acceptance rate (most moves rejected), and the chain explores inefficiently. A common rule of thumb is to tune the proposal distribution (e.g., the covariance matrix Σ<0xE1><0xB5><0x96>) so that the acceptance rate is roughly between 20% and 50% for reasonably high-dimensional problems. This often requires an initial "tuning" phase or adaptive MCMC algorithms that adjust the proposal distribution during the run.

Another MCMC algorithm is **Gibbs sampling**. It is applicable when the *conditional* probability distribution of each parameter θ<0xE1><0xB5><0xA2>, given all the *other* parameters (θ<0xE2><0x82><0x97> for j≠i) and the data D, is known and easy to sample from. The Gibbs sampler iteratively samples each parameter (or block of parameters) from its full conditional distribution, holding the other parameters fixed at their current values. That is, at step t+1:
*   Sample θ₁⁽<0xE1><0xB5><0x8D>⁺¹⁾ from P(θ₁ | θ₂⁽<0xE1><0xB5><0x8D>⁾, θ₃⁽<0xE1><0xB5><0x8D>⁾, ..., D)
*   Sample θ₂⁽<0xE1><0xB5><0x8D>⁺¹⁾ from P(θ₂ | θ₁⁽<0xE1><0xB5><0x8D>⁺¹⁾, θ₃⁽<0xE1><0xB5><0x8D>⁾, ..., D)
*   ... and so on for all parameters.
Gibbs sampling can be very efficient if the conditional distributions are simple, as all proposed moves are effectively accepted. However, it requires knowledge of these conditional distributions, which is often not available for complex models. It can also suffer from slow convergence if parameters are highly correlated.

More advanced MCMC algorithms have been developed to improve efficiency, particularly for high-dimensional or complex posteriors. **Ensemble samplers**, like the Affine-Invariant sampler implemented in `emcee` (Sec 17.4), use an ensemble ("walkers") of chains that explore the space simultaneously and propose moves based on the positions of other walkers in the ensemble. This can automatically adapt to the correlations and scales of the posterior distribution, often requiring less tuning than standard M-H. **Hamiltonian Monte Carlo (HMC)** and its variants (like NUTS - No-U-Turn Sampler, used in Stan and PyMC) introduce auxiliary momentum variables and use Hamiltonian dynamics to propose more distant, high-probability moves, potentially exploring the parameter space much more efficiently (lower autocorrelation time) for smooth, high-dimensional distributions, although each step is computationally more expensive as it requires gradient calculations. **Nested sampling** algorithms (like `dynesty`, Sec 17.4) approach the problem differently, aiming to directly estimate the Bayesian evidence P(D) while also producing posterior samples, often by exploring contours of constant likelihood within the prior volume.

Regardless of the specific algorithm, the core principles remain: construct a Markov chain designed to sample from the posterior P(θ|D) ∝ L(θ|D)P(θ), run the chain long enough to converge and sample the distribution adequately (many autocorrelation times after burn-in), and then use the collected samples to estimate the properties of the posterior distribution (means, medians, credible intervals, correlations). Python libraries implementing these algorithms provide the practical tools for applying these powerful Bayesian computation techniques.

**17.4 Using Python MCMC Libraries (`emcee`, `dynesty`)**

While understanding the theory behind MCMC algorithms is important, practical implementation in astrophysics typically relies on well-developed and widely used Python libraries that provide robust and efficient samplers. Two of the most popular choices are **`emcee`** (implementing an affine-invariant ensemble sampler) and **`dynesty`** (implementing nested sampling algorithms). These libraries offer user-friendly interfaces for defining the target posterior distribution and running the sampling process.

**`emcee`** (Foreman-Mackey et al. 2013) is particularly popular due to its ease of use and its affine-invariant ensemble sampler ("stretch move"). This algorithm uses multiple "walkers" (chains) that explore the parameter space simultaneously. The proposal for moving a specific walker is generated based on the positions of *other* walkers in the ensemble. This property makes the algorithm relatively insensitive to linear correlations between parameters and avoids the need for manual tuning of proposal distributions, which is a major advantage over basic Metropolis-Hastings.

Using `emcee` typically involves these steps:
1.  **Define the Log-Posterior Function:** Create a Python function that takes a parameter vector `theta` (representing a point in the N-dimensional parameter space) and returns the natural logarithm of the posterior probability density, ln P(θ|D). Since P(θ|D) ∝ L(θ|D)P(θ), this function usually calculates ln P(θ) + ln L(θ|D). It must handle cases where parameters are outside the prior range (e.g., return `-np.inf`). Additional arguments (like the observed data) can be passed via the `args` parameter during sampler initialization.
2.  **Initialize Walkers:** Choose the number of walkers (`nwalkers`, must be greater than 2*ndim, often significantly more) and the number of parameters (`ndim`). Provide an initial starting position for each walker, usually as a NumPy array of shape `(nwalkers, ndim)`. A common strategy is to start walkers in a small ball or Gaussian distribution centered around a preliminary estimate (e.g., from MLE or a guess).
3.  **Instantiate the Sampler:** Create an `emcee.EnsembleSampler` object: `sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn, args=[data])`.
4.  **Run Burn-in (Optional but Recommended):** Run the sampler for an initial number of steps to allow the walkers to move from their starting positions towards the high-probability region of the posterior and forget their initial state. `state = sampler.run_mcmc(initial_state, n_burn_steps)`. Discard these steps. Reset the sampler using `sampler.reset()`.
5.  **Run Production Sampling:** Run the sampler for a larger number of steps (`n_steps`) starting from the final state of the burn-in (`state`). `sampler.run_mcmc(state, n_steps, progress=True)`.
6.  **Retrieve Samples:** Access the chain of samples using `sampler.get_chain(discard=0, thin=1, flat=False)`. This returns an array of shape `(n_steps, nwalkers, ndim)`. Setting `flat=True` directly returns a flattened array of shape `(n_steps * nwalkers, ndim)`, which is often convenient for analysis after verifying convergence. `discard` and `thin` can be used here instead of separate burn-in/thinning steps if preferred.

```python
# --- Code Example 1: Conceptual emcee Usage ---
# Note: Requires installation: pip install emcee corner
import numpy as np
import emcee
import corner # For plotting results (Sec 17.5)
import matplotlib.pyplot as plt

print("Conceptual workflow using emcee:")

# Step 1: Define Log-Posterior Function (Example: fitting a line y=mx+c)
# Assume data arrays x_data, y_data, y_err exist
# Parameters theta = [m, c]
def log_likelihood_line(theta, x, y, yerr):
    m, c = theta
    model = m * x + c
    sigma2 = yerr**2
    # Gaussian log-likelihood: -0.5 * sum[ (data-model)^2/err^2 + log(2*pi*err^2) ]
    # Ignoring constant term log(2*pi*err^2) which doesn't affect sampling shape
    return -0.5 * np.sum((y - model)**2 / sigma2)

def log_prior_line(theta):
    m, c = theta
    # Example: Uniform priors within reasonable ranges
    if -10.0 < m < 10.0 and -100.0 < c < 100.0:
        return 0.0 # Log(1) = 0 for constant prior density
    return -np.inf # Log(0) = -infinity outside prior range

def log_posterior_line(theta, x, y, yerr):
    lp = log_prior_line(theta)
    if not np.isfinite(lp): # Check if parameters are outside prior
        return -np.inf
    # Return log-prior + log-likelihood
    return lp + log_likelihood_line(theta, x, y, yerr)

# --- Simulate data for example ---
np.random.seed(42)
x_data = np.sort(10 * rng.random(50))
y_err = 0.5 + 0.5 * rng.random(50)
y_data = 2.0 * x_data + 5.0 + rng.normal(0, y_err) # True m=2, c=5
# ---------------------------------

# Step 2: Initialize Walkers
ndim = 2 # Number of parameters (m, c)
nwalkers = 32 # Number of walkers
# Initial guess near plausible values (e.g., from least squares or guess)
initial_m_guess, initial_c_guess = 2.1, 4.9 
# Start walkers in a small ball around the guess
initial_pos = initial_guess + 1e-3 * rng.normal(size=(nwalkers, ndim))

# Step 3: Instantiate Sampler
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_posterior_line, args=(x_data, y_data, y_err)
)

# Step 4 & 5: Run MCMC (including burn-in implicitly via discard later)
n_steps = 2000 # Total steps per walker
print(f"\nRunning emcee with {nwalkers} walkers for {n_steps} steps...")
sampler.run_mcmc(initial_pos, n_steps, progress=True)
print("MCMC run finished.")

# Step 6: Retrieve Samples (discarding initial steps as burn-in)
burn_in = 500 # Number of steps to discard
thin = 10     # Optional thinning factor (keep every 10th step)
# flat=True combines samples from all walkers into one long array
samples_flat = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
print(f"\nRetrieved flattened samples shape: {samples_flat.shape}") 
# Shape should be approx ((n_steps-burn_in)/thin * nwalkers, ndim)

# --- Analysis (Sec 17.5) ---
# Example: Calculate median and 16/84 percentile (approx 1-sigma interval)
m_mcmc, c_mcmc = np.percentile(samples_flat, [16, 50, 84], axis=0)
print(f"\nMCMC Results (median +/- approx 1-sigma):")
print(f"  m = {m_mcmc[1]:.3f} +{m_mcmc[2]-m_mcmc[1]:.3f} / -{m_mcmc[1]-m_mcmc[0]:.3f}")
print(f"  c = {c_mcmc[1]:.3f} +{c_mcmc[2]-c_mcmc[1]:.3f} / -{c_mcmc[1]-c_mcmc[0]:.3f}")

# Example: Generate corner plot
# fig_corner = corner.corner(samples_flat, labels=["m", "c"], truths=[2.0, 5.0])
# fig_corner.show()
print("\n(Corner plot generation would visualize posterior distributions - see Sec 17.5)")

print("-" * 20)

# Explanation: This code demonstrates using `emcee` to fit a line (y=mx+c).
# 1. It defines Python functions for the log-likelihood, log-prior (uniform here), 
#    and the combined log-posterior. The log-posterior function takes parameters `theta` 
#    and any extra data (`x, y, yerr`) needed.
# 2. It initializes the number of dimensions (`ndim`) and walkers (`nwalkers`), and sets 
#    up initial positions `initial_pos` for the walkers near a guess.
# 3. It creates the `emcee.EnsembleSampler` instance, passing the log-posterior function 
#    and the data arguments.
# 4. It runs the MCMC for `n_steps` using `sampler.run_mcmc()`.
# 5. It retrieves the samples using `sampler.get_chain()`, applying `discard` for burn-in 
#    and optional `thin`ning, with `flat=True` to get a 2D array (N_samples x N_dim).
# 6. It demonstrates basic analysis: calculating the 16th, 50th (median), and 84th 
#    percentiles from the flattened samples to estimate parameter values and uncertainties.
# 7. It conceptually mentions generating a corner plot (Sec 17.5) for visualization.
```

**`dynesty`** (Speagle 2020) offers a different approach based on **nested sampling**. Nested sampling is designed primarily to calculate the Bayesian **evidence** P(D) (crucial for model comparison, Chapter 18), but it also produces **posterior samples** as a by-product. It works by exploring the likelihood function within nested contours of constant likelihood, constrained by the prior volume. It uses "live points" (samples) that are iteratively replaced by new points drawn from the prior subject to the constraint that their likelihood must exceed that of the point being replaced. Various methods exist for drawing these new points efficiently (e.g., random walks, slice sampling, clustering).

Using `dynesty` typically involves:
1.  **Define Log-Likelihood and Prior Transform Functions:** You need *two* functions:
    *   `loglike(theta)`: Calculates the log-likelihood ln L(θ|D) given parameters `theta`.
    *   `prior_transform(u)`: Takes a vector `u` of values uniformly distributed in [0, 1] (one value per parameter) and transforms it into a parameter vector `theta` drawn from the desired **prior distribution** P(θ). This requires knowing the inverse CDF (or equivalent transformation) for your chosen priors. For example, for a uniform prior on θ₁ between `a` and `b`, the transform is `theta[0] = a + u[0] * (b - a)`. For a Gaussian prior, it involves the inverse error function.
2.  **Instantiate the Sampler:** Create a `dynesty.NestedSampler` (or `DynamicNestedSampler`) object, providing the log-likelihood function, the prior transform function, the number of dimensions (`ndim`), and often the number of initial "live points" (`nlive`).
3.  **Run Sampling:** Call the sampler's `run_nested()` method, potentially specifying stopping criteria (e.g., based on estimated evidence uncertainty `dlogz`). This runs the nested sampling algorithm.
4.  **Retrieve Results:** Access the results stored in the sampler object's `.results` attribute. This is a dictionary-like object containing the estimated log-evidence (`logz`, `logzerr`), posterior samples (`samples`), associated weights (`logwt`), and other diagnostics. Equal-weighted posterior samples can often be obtained using `res.samples_equal()`.

```python
# --- Code Example 2: Conceptual dynesty Usage ---
# Note: Requires installation: pip install dynesty corner
import numpy as np
import dynesty
import corner # For plotting results (Sec 17.5)
import matplotlib.pyplot as plt
from scipy import stats # For prior transform (e.g., ppf)

print("Conceptual workflow using dynesty (Nested Sampling):")

# Step 1: Define Log-Likelihood and Prior Transform
# Use same line fitting example as emcee
# Assume data x_data, y_data, y_err exist
def log_likelihood_line_dynesty(theta): # Takes only theta
    # Access data (assumed global or passed via class/closure)
    # This is a simplification; real code needs better data handling
    m, c = theta
    model = m * x_data + c 
    sigma2 = y_err**2
    logL = -0.5 * np.sum((y_data - model)**2 / sigma2)
    return logL if np.isfinite(logL) else -1e100 # Return very small number if not finite

def prior_transform_line_dynesty(u): # Takes unit cube samples u
    # u is array of length ndim, values in [0, 1]
    theta = np.zeros_like(u)
    # Example: Uniform prior for m between -10 and 10
    m_min, m_max = -10.0, 10.0
    theta[0] = m_min + u[0] * (m_max - m_min) 
    # Example: Uniform prior for c between -100 and 100
    c_min, c_max = -100.0, 100.0
    theta[1] = c_min + u[1] * (c_max - c_min)
    # Can use stats.dist.ppf(u[i], ...) for other priors (Normal, LogUniform)
    return theta

# --- Use same simulated data ---
np.random.seed(42)
x_data = np.sort(10 * rng.random(50))
y_err = 0.5 + 0.5 * rng.random(50)
y_data = 2.0 * x_data + 5.0 + rng.normal(0, y_err) 
# -----------------------------

# Step 2: Instantiate Sampler
ndim = 2 # m, c
nlive = 100 # Number of live points (adjust based on problem)
# Use DynamicNestedSampler for potentially better efficiency
sampler_dynesty = dynesty.DynamicNestedSampler(
    log_likelihood_line_dynesty, 
    prior_transform_line_dynesty, 
    ndim=ndim, 
    # nlive=nlive # For static sampler
    # Other options: bound='multi', sample='auto', etc.
)
print(f"\nInstantiated dynesty sampler (ndim={ndim}).")

# Step 3: Run Sampling
print("Running nested sampling (may take time)...")
# run_nested takes criteria like dlogz (evidence uncertainty threshold)
# Or maxiter, maxcall etc.
# Use wt_kwargs to get equal-weighted samples later easily
sampler_dynesty.run_nested(dlogz=0.1, wt_kwargs={'pfrac': 1.0}) 
print("Nested sampling finished.")

# Step 4: Retrieve Results
results_dynesty = sampler_dynesty.results
print("\nDynesty Results Summary:")
print(results_dynesty.summary())

# Get log evidence
logz = results_dynesty.logz[-1] # Final log evidence estimate
logz_err = results_dynesty.logzerr[-1]
print(f"\nLog Evidence (log Z) = {logz:.3f} +/- {logz_err:.3f}")

# Get posterior samples (can use resample_equal for weighted samples)
# Or get equally weighted samples directly if requested via wt_kwargs in run_nested
# weights = np.exp(results_dynesty.logwt - results_dynesty.logz[-1])
# samples_posterior = dynesty.utils.resample_equal(results_dynesty.samples, weights)
samples_posterior = results_dynesty.samples_equal() # If wt_kwargs used

print(f"\nRetrieved {len(samples_posterior)} posterior samples.")

# --- Analysis (Sec 17.5) ---
# Example: Calculate median and 16/84 percentile
m_mcmc_d, c_mcmc_d = np.percentile(samples_posterior, [16, 50, 84], axis=0)
print(f"\nPosterior Results (median +/- approx 1-sigma):")
print(f"  m = {m_mcmc_d[1]:.3f} +{m_mcmc_d[2]-m_mcmc_d[1]:.3f} / -{m_mcmc_d[1]-m_mcmc_d[0]:.3f}")
print(f"  c = {c_mcmc_d[1]:.3f} +{c_mcmc_d[2]-c_mcmc_d[1]:.3f} / -{c_mcmc_d[1]-c_mcmc_d[0]:.3f}")

# Example: Generate corner plot
# fig_corner_d = corner.corner(samples_posterior, labels=["m", "c"], truths=[2.0, 5.0])
# fig_corner_d.show()
print("\n(Corner plot generation would visualize posterior distributions)")

print("-" * 20)

# Explanation: This code demonstrates using `dynesty` for the same line-fitting problem.
# 1. It defines *two* functions: `log_likelihood_line_dynesty` (calculating ln L) and 
#    `prior_transform_line_dynesty` (mapping unit cube samples `u` to parameter values 
#    drawn from the priors - uniform priors used here). Note the slightly different 
#    function signatures compared to emcee. Data is accessed as global here (simplification).
# 2. It instantiates `dynesty.DynamicNestedSampler`, providing these two functions and `ndim`.
# 3. It runs the sampler using `run_nested()`, specifying a stopping criterion based 
#    on the desired precision of the log evidence (`dlogz`). It also requests equally 
#    weighted samples using `wt_kwargs`.
# 4. It retrieves the results dictionary (`sampler_dynesty.results`). It prints a 
#    summary, extracts the log evidence estimate (`logz`), and gets equally weighted 
#    posterior samples using `results.samples_equal()`.
# 5. It performs similar analysis on the posterior samples (percentiles) as done for emcee. 
# The key differences are the need for a prior transform function and the direct calculation 
# of Bayesian evidence as a primary output alongside the posterior samples.
```

`dynesty` can be very effective, especially for problems where the posterior might be multi-modal or have complex degeneracies, or when accurate evidence calculation is paramount. Defining the `prior_transform` function correctly can sometimes be challenging for complex priors. The number of live points (`nlive`) influences the thoroughness of the exploration and the accuracy of the evidence estimate.

Both `emcee` and `dynesty` are powerful tools for Bayesian computation via sampling. `emcee` is often simpler to start with due to its reliance only on the log-posterior function and its robustness to correlations via the ensemble method. `dynesty` offers a different approach focused on evidence calculation via nested sampling, also yielding posterior samples. The choice between them might depend on the specific problem, the complexity of the posterior, whether evidence calculation is needed, and user preference. Other powerful libraries like `PyMC` and `Stan` (often accessed via Python interfaces like `cmdstanpy`) implement HMC/NUTS algorithms, which can be highly efficient for certain classes of problems (especially high-dimensional, smooth posteriors) but typically require gradient information for the log-posterior.

**17.5 Analyzing MCMC Output: Convergence, Corner Plots**

Running an MCMC sampler like `emcee` or obtaining posterior samples from `dynesty` is only the first step; critically analyzing the output is essential to ensure the results are reliable and to extract meaningful scientific conclusions. Key aspects include assessing **convergence** (has the sampler adequately explored the target posterior?), checking **autocorrelation** (how independent are the samples?), and **visualizing** the resulting posterior distribution to understand parameter estimates, uncertainties, and correlations.

**Assessing Convergence:** MCMC chains start from initial positions and need some number of steps (the "burn-in") to reach the stationary distribution (the posterior). It is crucial to discard these burn-in samples before analysis. Determining when burn-in is complete and the chain has converged is not always trivial.
*   **Visual Inspection (Trace Plots):** Plotting the value of each parameter versus the step number for multiple independent chains (or walkers in `emcee`) is essential. Initially, the chains might wander significantly depending on their starting points. Converged chains should appear to be randomly sampling around a stable value (like a "fuzzy caterpillar"), and different chains should overlap and explore the same region of parameter space. If chains get stuck in different regions or show long-term drifts, convergence has likely not been reached.
*   **Gelman-Rubin Diagnostic (R̂):** This quantitative diagnostic compares the variance *within* individual chains to the variance *between* multiple independent chains. The R̂ statistic (often called potential scale reduction factor) should approach 1 for all parameters when the chains have converged and mixed well. Values significantly larger than 1 (e.g., > 1.1 or even > 1.01 depending on desired precision) indicate poor convergence. Calculating R̂ requires running multiple independent chains. `emcee`'s documentation and packages like `ArviZ` provide tools for calculating this.
*   **Autocorrelation Analysis:** Analyzing the autocorrelation function (ACF) of the parameter chains helps determine the autocorrelation time τ (Sec 17.2). Plotting the ACF shows how quickly the correlation between samples drops as the step separation increases. τ is roughly the number of steps needed for the ACF to drop significantly (e.g., below 1/e). The total number of steps run (after burn-in) should ideally be many times larger than τ (e.g., > 50τ) to ensure a sufficient number of effectively independent samples. `emcee` provides `sampler.get_autocorr_time()` to estimate τ.

**Visualizing Posteriors (Corner Plots):** Once convergence is deemed satisfactory and burn-in samples are discarded, the remaining samples represent the posterior distribution P(θ|D). Visualizing this potentially high-dimensional distribution is crucial. The most common and effective tool for this is the **corner plot**, often generated using the `corner` Python package (`pip install corner`). A corner plot displays all 1D and 2D marginal posterior distributions for all parameters in a matrix format.
*   **Diagonal Panels:** Show the 1D marginalized posterior distribution for each individual parameter (θ<0xE1><0xB5><0xA2>), typically as a histogram or smoothed KDE. These plots reveal the best estimate (peak/median) and uncertainty (width/credible interval) for each parameter individually, integrating over the uncertainty in all other parameters.
*   **Off-Diagonal Panels:** Show the 2D marginalized posterior distribution for each pair of parameters (θ<0xE1><0xB5><0xA2>, θ<0xE2><0x82><0x97>), often as contour plots or 2D histograms. These plots are essential for revealing **correlations** or **degeneracies** between parameters. Elongated or tilted contours indicate that the parameters are correlated (e.g., increasing one requires decreasing another to maintain a good fit). Complex shapes can indicate non-linear relationships or multi-modal distributions.

The `corner.corner(samples, labels=param_names, truths=true_values, ...)` function takes the flattened array of posterior samples (shape N_samples x N_dim), optional labels for the axes, and optional true values (if known, e.g., from simulations) to overlay, and generates the full corner plot automatically. It provides numerous customization options for appearance, contours, and summary statistics displayed on the 1D histograms. Examining the corner plot provides a comprehensive visual summary of the Bayesian parameter estimation results.

```python
# --- Code Example 1: Analyzing emcee Output (Trace Plots, Autocorrelation, Corner Plot) ---
# Continues from the emcee example in Sec 17.4
# Assume 'sampler' is the EnsembleSampler object after running run_mcmc
# Assume 'ndim', 'nwalkers', 'n_steps' are defined

# --- Requires sampler object from Sec 17.4 example ---
# If running separately, re-run the emcee simulation first.
# Make sure `samples_flat` is also available if needed, 
# or recalculate `burn_in`, `thin`.

print("Analyzing emcee MCMC output:")

if 'sampler' in locals() and sampler is not None:
    # --- 1. Visual Inspection: Trace Plots ---
    print("\nGenerating Trace Plots (Example for first 100 steps)...")
    # Get the full chain first (ndim, nwalkers, nsteps)
    chain = sampler.get_chain() # Shape (n_steps, nwalkers, ndim)
    
    fig_trace, axes_trace = plt.subplots(ndim, 1, sharex=True, figsize=(8, ndim * 2))
    if ndim == 1: axes_trace = [axes_trace] # Make iterable if only one param
    labels_trace = ["m", "c"] 
    
    for i in range(ndim):
        ax = axes_trace[i]
        # Plot traces for all walkers
        ax.plot(chain[:, :, i], alpha=0.3) 
        ax.set_ylabel(labels_trace[i])
        # Optionally mark burn-in region
        if 'burn_in' in locals(): ax.axvline(burn_in, color='red', linestyle='--', label='Burn-in')
        ax.legend(loc='upper right')
    axes_trace[-1].set_xlabel("Step Number")
    fig_trace.suptitle("Trace Plots (All Walkers)")
    fig_trace.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    # plt.show()
    print("Trace plots generated (check for convergence).")
    plt.close(fig_trace)

    # --- 2. Autocorrelation Time ---
    print("\nEstimating Autocorrelation Time...")
    try:
        # Requires chain to be long enough
        autocorr_time = sampler.get_autocorr_time(tol=0) # Use tol=0 for default
        print(f"  Estimated autocorrelation time (steps): {autocorr_time}")
        # Check if run length is sufficient
        min_tau = np.min(autocorr_time) if np.all(np.isfinite(autocorr_time)) else np.inf
        n_eff_approx = (n_steps - burn_in) / min_tau if np.isfinite(min_tau) else 0
        print(f"  Approx effective samples per walker after burn-in: ~{n_eff_approx:.0f}")
        if n_eff_approx < 50: # Rule of thumb
             print("  WARNING: Chain might be too short relative to autocorrelation time.")
             
    except emcee.autocorr.AutocorrError as e_acorr:
        print(f"  Could not estimate autocorrelation time: {e_acorr}")
        print("  (Chain might be too short or not converged)")
    except Exception as e:
        print(f"  Error estimating autocorrelation time: {e}")

    # --- 3. Corner Plot (using samples after burn-in/thinning) ---
    # Ensure samples_flat exists from Sec 17.4 example
    if 'samples_flat' in locals() and samples_flat is not None:
        print("\nGenerating Corner Plot...")
        try:
            fig_corner = corner.corner(
                samples_flat, 
                labels=labels_trace, 
                truths=[2.0, 5.0], # True values for m and c
                quantiles=[0.16, 0.5, 0.84], # Show median and +/- 1 sigma
                show_titles=True, title_kwargs={"fontsize": 12}
            )
            # plt.show()
            print("Corner plot generated.")
            plt.close(fig_corner)
        except Exception as e_corner:
            print(f"  Error generating corner plot: {e_corner}")
    else:
        print("\nSkipping corner plot (samples_flat not available).")

else:
    print("\nMCMC sampler object ('sampler') not found. Run MCMC first.")

print("-" * 20)

# Explanation: This code analyzes the output from the `emcee` run in Sec 17.4.
# 1. Trace Plots: It retrieves the full chain using `sampler.get_chain()` and plots 
#    the parameter values versus step number for all walkers for each dimension (`m` and `c`). 
#    These plots visually help assess if walkers have converged to a stable region and 
#    if they are mixing well. The burn-in point is marked.
# 2. Autocorrelation Time: It attempts to estimate the autocorrelation time τ using 
#    `sampler.get_autocorr_time()`. It prints τ and calculates an approximate number 
#    of effective independent samples, warning if the chain seems too short.
# 3. Corner Plot: Assuming `samples_flat` (the flattened chain after burn-in/thinning) 
#    is available, it uses `corner.corner()` to generate the standard visualization. 
#    `labels` provides axis labels, `truths` overlays the known true values, `quantiles` 
#    marks the 16th, 50th, and 84th percentiles on the 1D histograms, and `show_titles` 
#    displays these quantiles numerically. The plot shows 1D histograms (marginalized 
#    posteriors) on the diagonal and 2D contour plots (joint posteriors) on the off-diagonal, 
#    revealing estimates, uncertainties, and correlations.
```

**Deriving Point Estimates and Credible Intervals:** From the posterior samples (e.g., `samples_flat` from `emcee` or `samples_posterior` from `dynesty`), summary statistics are easily calculated using NumPy functions:
*   **Mean:** `np.mean(samples_flat, axis=0)` gives the mean for each parameter.
*   **Median:** `np.median(samples_flat, axis=0)` or `np.percentile(samples_flat, 50, axis=0)`. The median is often preferred as a robust point estimate, especially if the 1D posterior is skewed.
*   **MAP (Mode):** Can sometimes be estimated from the peak of the 1D histograms, but can be noisy. Not always the best summary statistic.
*   **Credible Intervals:** Calculated using percentiles. For a symmetric 68.3% ("1-sigma") interval, use the 16th and 84th percentiles: `lower, median, upper = np.percentile(samples_flat, [16, 50, 84], axis=0)`. The interval is [`lower`, `upper`], often reported as `median + (upper-median) / - (median-lower)`. For a 95% interval, use `[2.5, 97.5]` percentiles.

Specialized packages like **`ArviZ`** (`pip install arviz`) provide more sophisticated tools for MCMC diagnostics (convergence tests like R̂, effective sample size N_eff), posterior visualization (including trace plots, corner plots, posterior predictive checks), and result summarization, often integrating well with various MCMC library outputs.

In conclusion, analyzing MCMC output is as important as running the sampler itself. Assessing convergence using trace plots and diagnostics like R̂ (or autocorrelation time) is essential to ensure the samples reliably represent the posterior. Visualizing the results using corner plots provides invaluable insight into parameter estimates, uncertainties, and correlations. Finally, calculating point estimates (like the median) and credible intervals (using percentiles) from the posterior samples provides the quantitative summary of the Bayesian parameter estimation results.

**17.6 Credible Intervals vs. Confidence Intervals**

A key difference between Bayesian and frequentist inference lies in the interpretation of the intervals used to express parameter uncertainty. Bayesian analysis yields **credible intervals** (or credible regions), while frequentist analysis (like MLE with Hessian errors or profile likelihood) yields **confidence intervals** (or confidence regions). While often numerically similar, especially for large datasets where posteriors approach Gaussian shapes, their formal definitions and interpretations are distinct and reflect the different philosophical underpinnings of the two approaches.

A Bayesian **(1 - α) credible interval** for a parameter θ is an interval [L, U] calculated from the posterior distribution P(θ|D) such that the probability that the true value of θ lies within this interval is (1 - α):
P(L ≤ θ ≤ U | D) = 1 - α
For example, a 95% credible interval means there is a 95% probability (according to our posterior belief) that the true parameter value falls within that range. This interpretation is often considered more intuitive and closer to the colloquial meaning of "uncertainty range" than the frequentist confidence interval. Credible intervals are typically calculated directly from the posterior samples generated by MCMC, for instance, by finding the range between the 2.5th and 97.5th percentiles for a central 95% interval (the "equal-tailed interval") or by finding the narrowest possible interval containing 95% of the posterior probability mass (the "highest posterior density interval" or HPDI).

A frequentist **(1 - α) confidence interval** for a parameter θ has a more complex interpretation related to the procedure used to construct it. It is an interval [L(D), U(D)], calculated from the observed data D using a specific procedure (e.g., based on MLE and standard errors, or profile likelihood), such that if we were to repeat the experiment many times and calculate the interval using the same procedure for each resulting dataset, then (1 - α) fraction of *those constructed intervals* would contain the true, fixed parameter value θ₀.

Critically, for any *single* calculated confidence interval [L, U] from a specific dataset, the frequentist framework does *not* allow us to say "there is a (1 - α) probability that the true value θ₀ lies within [L, U]". The true value θ₀ is considered fixed (though unknown), and the specific interval [L, U] either contains θ₀ or it doesn't. The probability statement applies to the long-run performance of the *procedure* used to generate intervals across hypothetical repeated experiments, not to the specific interval calculated from our actual data. This interpretation is formally correct within the frequentist paradigm but often feels less direct or intuitive than the Bayesian credible interval's interpretation.

For large datasets where the posterior distribution is well-approximated by a Gaussian (due to asymptotic normality of MLE and likelihood dominating the prior), the Bayesian credible interval (e.g., median ± 1.96 * posterior_std_dev for 95%) and the frequentist confidence interval (e.g., MLE ± 1.96 * standard_error_from_Hessian for 95%) often yield very similar numerical ranges. This occurs because the posterior shape mimics the likelihood shape, and the standard error approximates the posterior standard deviation.

However, differences arise, particularly for:
*   **Small sample sizes:** The posterior distribution might be significantly non-Gaussian and/or heavily influenced by the prior, leading to credible intervals that differ from confidence intervals based on asymptotic approximations.
*   **Parameters near boundaries:** If a parameter has a physical boundary (e.g., a mass must be positive), the posterior distribution will naturally respect this boundary. Credible intervals derived from the posterior will also respect it. Frequentist confidence intervals based on Gaussian approximations (like MLE ± error) might incorrectly extend beyond the physical boundary, requiring ad-hoc truncation or more complex interval construction methods. Profile likelihood intervals are generally better behaved near boundaries.
*   **Strong priors:** If informative prior information is used in the Bayesian analysis, the resulting posterior and credible intervals will incorporate this information and may differ significantly from frequentist intervals derived solely from the likelihood.

In practice, within the astrophysics literature, the distinction is sometimes blurred, and intervals derived from MCMC are often referred to generally as "confidence intervals" or "error bars." However, understanding the formal difference in interpretation is important. Bayesian credible intervals make direct probability statements about the parameter value given the observed data and prior, while frequentist confidence intervals make statements about the long-run coverage properties of the interval construction procedure. The Bayesian interpretation is arguably more aligned with the intuitive scientific goal of expressing our current state of knowledge and uncertainty about the parameters after analyzing the data.

**Application 17.A: Bayesian Fit of an Exoplanet Transit Light Curve**

**Objective:** This application demonstrates a full Bayesian parameter estimation workflow using MCMC (specifically `emcee`, Sec 17.4) to fit a physical model (a transit model) to observational data (a light curve) and derive constraints on the exoplanet's parameters, including robust uncertainties via posterior sampling. Reinforces Sec 17.1-17.5.

**Astrophysical Context:** Detecting and characterizing exoplanets via the transit method involves modeling the precise shape of the light curve dip caused when a planet passes in front of its star. The shape and timing of this dip depend on parameters like the planet-to-star radius ratio (Rp/Rs), the orbital period (P), the time of mid-transit (t0), the orbital inclination (i), the orbital semi-major axis relative to the stellar radius (a/Rs), and potentially limb darkening parameters describing the star's brightness profile. Bayesian inference using MCMC is the standard method for fitting these multi-parameter models to light curve data, as it naturally handles parameter correlations and provides full posterior probability distributions for robust uncertainty assessment.

**Data Source:** A segment of time-series photometry (light curve) data (`time`, `flux`, `flux_error`) covering at least one transit event for a confirmed or candidate exoplanet. This could be from Kepler, K2, TESS (downloaded via MAST, Chapter 10), or ground-based observations. We will simulate data for a typical transit.

**Modules Used:** `emcee` (for MCMC sampling), `numpy` (for calculations), `corner` (for visualizing posteriors), `matplotlib.pyplot` (for plotting data and model). Optionally, a specialized transit modeling package like `batman` (`pip install batman-package`) can be used to generate the light curve model efficiently, or we can use a simplified analytic approximation (e.g., Mandel & Agol 2002, or even simpler models for demonstration).

**Technique Focus:** Implementing a Bayesian analysis using MCMC. This involves: (1) Defining the transit model function `transit_model(time, params)`. (2) Defining the log-prior function `log_prior(params)` encoding prior knowledge or constraints on parameters (e.g., Rp/Rs > 0, P > 0, uniform or Gaussian priors based on previous knowledge). (3) Defining the log-likelihood function `log_likelihood(params, time, flux, flux_err)`, typically assuming Gaussian errors: ln L ∝ -0.5 * Σ [(flux - model)² / flux_err²]. (4) Defining the log-posterior function `log_posterior = log_prior + log_likelihood`. (5) Initializing `emcee` walkers near a plausible starting point (perhaps found via initial least-squares fit). (6) Running the MCMC sampler (`sampler.run_mcmc()`). (7) Assessing convergence (trace plots, autocorrelation time). (8) Discarding burn-in and analyzing the posterior samples (e.g., using `corner.corner()`, calculating percentiles for credible intervals).

**Processing Step 1: Load Data and Define Model:** Load `time`, `flux`, `flux_err`. Normalize flux if needed. Define the `transit_model` function (using `batman` or approximation) taking time and parameter vector (e.g., `params = [t0, P, Rp/Rs, a/Rs, inclination]`) as input.

**Processing Step 2: Define Priors and Posterior:** Define `log_prior(params)` returning 0 if parameters are within allowed ranges (e.g., 0 < inclination < 90 deg, Rp/Rs > 0) and -np.inf otherwise. Define `log_likelihood(params, ...)` calculating model and Chi-squared based term. Define `log_posterior(params, ...)` returning `log_prior + log_likelihood`.

**Processing Step 3: Initialize and Run MCMC:** Set `ndim`, `nwalkers`. Choose `initial_pos` (e.g., small ball around guesses for t0, P, Rp/Rs, a/Rs, i). Instantiate `emcee.EnsembleSampler`. Run burn-in phase `sampler.run_mcmc(..., n_burn_steps)`. Reset sampler `sampler.reset()`. Run production phase `sampler.run_mcmc(..., n_steps)`.

**Processing Step 4: Analyze Output:** Get the flattened chain after discarding burn-in: `samples = sampler.get_chain(discard=burn_in, flat=True)`. Generate a corner plot: `corner.corner(samples, labels=param_names, truths=...)`. Calculate median and 16th/84th percentiles for each parameter using `np.percentile(samples, [16, 50, 84], axis=0)`. Report these as parameter estimates and 1-sigma credible intervals.

**Processing Step 5: Visualization:** Plot the phase-folded light curve data (calculated using the median fitted period and t0). Overplot the best-fit transit model (using median parameter values). Optionally, overplot a random subset of models drawn from the posterior samples to visualize the model uncertainty.

**Output, Testing, and Extension:** Output includes the corner plot showing posterior distributions and correlations, printed parameter estimates with credible intervals (e.g., Rp/Rs = 0.105 +0.002/-0.003), and the plot of the data with model fits. **Testing:** Check trace plots and autocorrelation times for convergence. Ensure credible intervals are reasonable and parameters don't hit prior boundaries unexpectedly. Verify the model fit visually matches the data. **Extensions:** (1) Include limb darkening parameters in the fit. (2) Fit for systematic trends in the light curve simultaneously with the transit model (e.g., adding a linear or polynomial baseline). (3) Use `dynesty` instead of `emcee` and calculate the Bayesian evidence to compare different models (e.g., circular vs. eccentric orbit). (4) Analyze the MCMC output using `ArviZ` for more detailed diagnostics and plotting.

```python
# --- Code Example: Application 17.A ---
# Note: Requires emcee, corner, batman-package (pip install ...)
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
try:
    import batman
    batman_installed = True
except ImportError:
    batman_installed = False
    print("WARNING: batman package not found. Using simplified transit model.")

print("Bayesian Transit Fit using emcee:")

# Step 1: Simulate Transit Data 
np.random.seed(1984)
# True parameters
params_true = {'t0': 1.0, 'per': 5.0, 'rp': 0.08, 'a': 15.0, 'inc': 89.0, 
               'ecc': 0.0, 'w': 90.0, 'u1': 0.4, 'u2': 0.2, 'limb_dark': 'quadratic'}
time_obs = np.linspace(0, 15, 500) # Observe over ~3 periods
flux_err_obs = 0.0005 # Assume constant error

if batman_installed:
    # Use batman for realistic model
    params_bat = batman.TransitParams()
    for name, val in params_true.items(): setattr(params_bat, name, val)
    m = batman.TransitModel(params_bat, time_obs)
    flux_true = m.light_curve(params_bat)
else:
    # Simplified box model if batman not available
    phase = ((time_obs - params_true['t0'] + params_true['per']/2.) / params_true['per']) % 1.0 - 0.5
    duration_phase = 0.02 # Approximate duration in phase units
    in_transit = np.abs(phase) < (duration_phase / 2.0)
    flux_true = np.ones_like(time_obs)
    flux_true[in_transit] -= params_true['rp']**2 # Depth ~ (Rp/Rs)^2

flux_obs = flux_true + np.random.normal(0, flux_err_obs, time_obs.shape)
print("\nSimulated transit data generated.")

# Step 2: Define Priors, Likelihood, Posterior
# Parameters to fit (simplified: t0, per, rp, a, inc) - fixing limb darkening, ecc=0
# Use indices: 0=t0, 1=per, 2=rp, 3=a, 4=inc
def log_likelihood_transit(theta, t, f, ferr):
    t0, per, rp, a, inc = theta
    # Update batman params (or simplified model params)
    if batman_installed:
        params_model = batman.TransitParams()
        params_model.t0 = t0; params_model.per = per; params_model.rp = rp
        params_model.a = a; params_model.inc = inc; 
        params_model.ecc = 0.0; params_model.w = 90.0 # Fixed
        params_model.u = [params_true['u1'], params_true['u2']]; params_model.limb_dark = 'quadratic' # Fixed
        model_obj = batman.TransitModel(params_model, t)
        flux_model = model_obj.light_curve(params_model)
    else: # Simplified model calculation
        phase = ((t - t0 + per/2.) / per) % 1.0 - 0.5
        duration_phase = 0.02 # Needs better calc based on a, inc
        in_transit = np.abs(phase) < (duration_phase / 2.0)
        flux_model = np.ones_like(t)
        flux_model[in_transit] -= rp**2 
        
    sigma2 = ferr**2
    return -0.5 * np.sum((f - flux_model)**2 / sigma2) # Ignoring constant term

def log_prior_transit(theta):
    t0, per, rp, a, inc = theta
    # Example: Uniform priors over plausible ranges
    if (params_true['t0'] - 0.5 < t0 < params_true['t0'] + 0.5 and 
        params_true['per'] - 1.0 < per < params_true['per'] + 1.0 and
        0.0 < rp < 0.5 and # Rp/Rs must be positive and < 0.5 say
        5.0 < a < 25.0 and # a/Rs range
        80.0 < inc < 90.0): # Inclination range for transit
        return 0.0 # log(1)
    return -np.inf # Outside prior range

def log_posterior_transit(theta, t, f, ferr):
    lp = log_prior_transit(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_transit(theta, t, f, ferr)

# Step 3: Initialize and Run MCMC
ndim = 5
nwalkers = 50
# Initial guess near true values + small random offset
initial_guess = np.array([params_true['t0'], params_true['per'], params_true['rp'], 
                          params_true['a'], params_true['inc']])
initial_pos = initial_guess + 1e-4 * initial_guess * np.random.randn(nwalkers, ndim)
# Ensure initial positions are within priors (important!)
for i in range(nwalkers): 
     while not np.isfinite(log_prior_transit(initial_pos[i])):
         initial_pos[i] = initial_guess + 1e-3 * initial_guess * np.random.randn(ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_transit, 
                                args=(time_obs, flux_obs, flux_err_obs))

# Burn-in
n_burn = 500
print(f"\nRunning burn-in ({n_burn} steps)...")
state = sampler.run_mcmc(initial_pos, n_burn, progress=True)
sampler.reset()
print("Burn-in finished.")

# Production run
n_steps = 3000
print(f"Running production ({n_steps} steps)...")
sampler.run_mcmc(state, n_steps, progress=True)
print("Production run finished.")

# Step 4: Analyze Output
print("\nAnalyzing MCMC output...")
burn_discard = 0 # Already discarded via reset
thin = 15
samples = sampler.get_chain(discard=burn_discard, thin=thin, flat=True)
print(f"Sample shape after thinning: {samples.shape}")

labels = ["t0", "P", "Rp/Rs", "a/Rs", "inc"]
truths = [params_true['t0'], params_true['per'], params_true['rp'], 
          params_true['a'], params_true['inc']]

# Calculate median and percentiles
results_mcmc = []
print("\nMCMC Parameter Estimates (median, 16th, 84th percentiles):")
for i in range(ndim):
    q = np.percentile(samples[:, i], [16, 50, 84])
    results_mcmc.append(q[1]) # Store median
    print(f"  {labels[i]} = {q[1]:.5f} +{q[2]-q[1]:.5f} / -{q[1]-q[0]:.5f}")

# Generate corner plot
try:
    fig_corner = corner.corner(samples, labels=labels, truths=truths, 
                               quantiles=[0.16, 0.5, 0.84], show_titles=True)
    # fig_corner.show()
    print("Corner plot generated.")
    plt.close(fig_corner)
except Exception as e:
    print(f"Could not generate corner plot: {e}")

# Step 5: Visualization (Phase-folded plot)
print("Generating phase-folded plot with model...")
fig_lc, ax_lc = plt.subplots(figsize=(8, 5))
# Phase fold data using median fitted parameters
t0_fit, per_fit = results_mcmc[0], results_mcmc[1]
phase = ((time_obs - t0_fit + per_fit/2.) / per_fit) % 1.0 - 0.5
# Plot phased data (points)
ax_lc.scatter(phase, flux_obs, s=3, alpha=0.3, label='Data')
# Plot best-fit model (line)
time_model = np.linspace(t0_fit - 0.1*per_fit, t0_fit + 0.1*per_fit, 500) # Time around transit
phase_model = ((time_model - t0_fit + per_fit/2.) / per_fit) % 1.0 - 0.5
# Get model flux using fitted parameters
if batman_installed:
    params_fit = batman.TransitParams()
    params_fit.t0 = results_mcmc[0]; params_fit.per = results_mcmc[1]; params_fit.rp = results_mcmc[2]
    params_fit.a = results_mcmc[3]; params_fit.inc = results_mcmc[4]; 
    params_fit.ecc = 0.0; params_fit.w = 90.0
    params_fit.u = [params_true['u1'], params_true['u2']]; params_fit.limb_dark = 'quadratic'
    m_fit = batman.TransitModel(params_fit, time_model)
    flux_model_fit = m_fit.light_curve(params_fit)
else: # Simplified model
    in_transit_model = np.abs(phase_model) < (duration_phase / 2.0)
    flux_model_fit = np.ones_like(phase_model)
    flux_model_fit[in_transit_model] -= results_mcmc[2]**2
# Sort model by phase for clean line plot
sort_idx = np.argsort(phase_model)
ax_lc.plot(phase_model[sort_idx], flux_model_fit[sort_idx], 'r-', lw=2, label='Best-fit Model')

ax_lc.set_xlim(-0.05, 0.05) # Zoom on transit phase
ax_lc.set_xlabel("Orbital Phase")
ax_lc.set_ylabel("Normalized Flux")
ax_lc.set_title("Phase-Folded Transit Light Curve and MCMC Fit")
ax_lc.legend()
ax_lc.grid(True, alpha=0.4)
# plt.show()
print("Phase-folded plot generated.")
plt.close(fig_lc)

print("-" * 20)
```

**Application 17.B: Cosmological Parameter Constraints from Supernovae**

**Objective:** This application demonstrates using Bayesian inference, potentially via MCMC (`emcee`) or nested sampling (`dynesty`), to constrain cosmological parameters (like the Hubble constant H₀ and the matter density Ω<0xE1><0xB5><0x89>) by fitting a cosmological model (e.g., flat ΛCDM) to Type Ia Supernova distance modulus data. Reinforces Sec 17.1-17.5, linking statistical inference with physical models (`astropy.cosmology`).

**Astrophysical Context:** Type Ia supernovae (SNe Ia) are "standardizable candles" – their peak intrinsic luminosity can be calibrated, allowing their observed apparent brightness to be used to determine their distance. By measuring the apparent brightness (expressed as distance modulus, μ) and redshift (z) for many SNe Ia over a range of distances, astronomers can map out the expansion history of the Universe. Comparing this observed Hubble diagram (μ vs. z) to the predictions of different cosmological models (which relate μ and z based on parameters like H₀, Ω<0xE1><0xB5><0x89>, Ω<0xE2><0x82><0x8B>, w) allows constraining these fundamental cosmological parameters. Bayesian methods are standard for this type of cosmological parameter inference.

**Data Source:** A compilation of Type Ia supernova data, such as the Pantheon+ dataset or older compilations like Union2.1 or JLA. These datasets typically provide tables containing supernova identifier, redshift (`z`), apparent magnitude or distance modulus (`mu`), and the uncertainty on the distance modulus (`mu_err`), potentially including a covariance matrix for systematic uncertainties. We will simulate a small, simplified dataset.

**Modules Used:** `emcee` or `dynesty` (for sampling), `numpy`, `corner` (for plotting), `matplotlib.pyplot`, and crucially `astropy.cosmology` (to calculate theoretical distance moduli for a given cosmology).

**Technique Focus:** Setting up a Bayesian inference problem involving a physical model from `astropy.cosmology`. Defining priors P(θ) for cosmological parameters (θ = [H₀, Ω<0xE1><0xB5><0x89>], assuming flat ΛCDM where Ω<0xE2><0x82><0x8B>=1-Ω<0xE1><0xB5><0x89>). Defining the log-likelihood function ln L(θ|D) based on comparing model predictions (`astropy.cosmology.FlatLambdaCDM(...).distmod(z)`) with observed SN distance moduli (`mu_obs`), assuming Gaussian errors: ln L ∝ -0.5 * Σ [(μ<0xE1><0xB5><0x92><0xE1><0xB5><0x87>ₛ - μ<0xE1><0xB5><0x9B><0xE1><0xB5><0x92><0xE1><0xB5><0x87><0xE1><0xB5><0x86>(z<0xE1><0xB5><0xA2>, θ))² / μ<0xE1><0xB5><0x98><0xE1><0xB5><0xA3><0xE1><0xB5><0xA3>ᵢ²]. Running MCMC or nested sampling to obtain posterior samples for H₀ and Ω<0xE1><0xB5><0x89>. Visualizing results with a corner plot and deriving credible intervals.

**Processing Step 1: Load/Simulate Data:** Obtain or simulate arrays `z_sn` (redshifts), `mu_obs` (observed distance moduli), `mu_err` (errors on mu).

**Processing Step 2: Define Priors, Likelihood, Posterior:**
    *   `log_prior(params)`: Define priors for `params = [H0, Om0]`. E.g., uniform H₀ in [50, 100], uniform Ω<0xE1><0xB5><0x89> in [0, 1]. Return 0 if within bounds, -np.inf otherwise.
    *   `log_likelihood(params, z, mu, mu_err)`: Inside, unpack `H0, Om0`. Create cosmology object `cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)`. Calculate model distance moduli `mu_model = cosmo.distmod(z).value`. Compute Gaussian log-likelihood `lnL = -0.5 * np.sum(((mu - mu_model) / mu_err)**2)`. Return `lnL`.
    *   `log_posterior(params, ...)`: Return `log_prior(params) + log_likelihood(params, ...)`.

**Processing Step 3: Initialize and Run Sampler:** Choose sampler (`emcee` or `dynesty`).
    *   If `emcee`: Set `ndim=2`, `nwalkers`. Define `initial_pos` within prior bounds. Instantiate `EnsembleSampler`. Run burn-in and production phases.
    *   If `dynesty`: Define `prior_transform(u)` function mapping unit cube `u` to `[H0, Om0]` according to priors. Instantiate `NestedSampler` (or `DynamicNestedSampler`) with `log_likelihood`, `prior_transform`, `ndim`. Run `sampler.run_nested()`.

**Processing Step 4: Analyze Output:**
    *   If `emcee`: Get flattened chain `samples = sampler.get_chain(discard=burn_in, flat=True)`.
    *   If `dynesty`: Get weighted samples `results = sampler.results` and potentially resample to equal weights `samples = results.samples_equal()`. Also get log Evidence `logz = results.logz[-1]`.
    *   Generate corner plot: `corner.corner(samples, labels=['H0', 'Omega_M'], truths=[true_H0, true_Om0])`.
    *   Calculate median and credible intervals (e.g., 68% or 95%) using `np.percentile`.

**Processing Step 5: Visualization:** Plot the observed Hubble diagram (μ<0xE1><0xB5><0x92><0xE1><0xB5><0x87>ₛ vs z) with error bars. Overplot the best-fit ΛCDM model curve using the median posterior parameters. Optionally, overplot curves corresponding to random samples drawn from the posterior to visualize model uncertainty.

**Output, Testing, and Extension:** Output includes the corner plot showing the joint and marginalized posterior distributions for H₀ and Ω<0xE1><0xB5><0x89>, printed parameter estimates with credible intervals, and optionally the Hubble diagram plot with the best fit. If using `dynesty`, the estimated Bayesian evidence (log Z) is also a key output. **Testing:** Check MCMC convergence (trace plots, autocorrelation, R̂ if multiple chains). Verify posterior distributions are reasonable and contained within prior bounds. Compare derived constraints with accepted values from literature (e.g., from Planck or SH0ES, acknowledging potential tensions). **Extensions:** (1) Include systematic uncertainties using a covariance matrix in the likelihood calculation (if provided by the SN dataset). (2) Fit a different cosmological model (e.g., non-flat ΛCDM, or wCDM with a varying dark energy equation of state `w`) by modifying the `astropy.cosmology` object and adding `w` as a parameter with a prior. (3) Use `dynesty` to calculate the evidence for different models and compute Bayes Factors (Chapter 18) to perform Bayesian model comparison. (4) Combine SN data with other cosmological probes (like Baryon Acoustic Oscillations or CMB data) by adding their likelihood terms to the `log_likelihood` function (assuming independence).

```python
# --- Code Example: Application 17.B ---
# Note: Requires emcee, corner, astropy
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy import constants as const
import emcee # Using emcee for this example
import corner

print("Constraining Cosmological Parameters with Simulated SN Ia Data:")

# Step 1: Simulate SN Ia Data (Simplified Hubble Diagram)
np.random.seed(200)
n_sne = 50
# True cosmology (for simulation)
H0_true = 70.0 * (u.km / u.s / u.Mpc)
Om0_true = 0.3
cosmo_true = FlatLambdaCDM(H0=H0_true, Om0=Om0_true)

# Generate redshifts
z_sn = np.random.uniform(0.02, 0.8, n_sne)
z_sn = np.sort(z_sn)
# Calculate true distance moduli
mu_true = cosmo_true.distmod(z_sn) # Returns Quantity
# Add scatter representing intrinsic magnitude variations and measurement error
mu_err_model = 0.15 # mags - simplified constant error
mu_obs = mu_true.value + np.random.normal(0.0, mu_err_model, n_sne) 
print(f"\nGenerated {n_sne} simulated SN Ia data points.")
# print(f"Sample: z={z_sn[0]:.2f}, mu_obs={mu_obs[0]:.2f} +/- {mu_err_model}")

# Step 2: Define Priors, Likelihood, Posterior
# Parameters theta = [H0, Om0]
def log_prior_cosmo(theta):
    H0, Om0 = theta
    # Uniform priors
    if 50.0 < H0 < 100.0 and 0.0 < Om0 < 1.0:
        return 0.0
    return -np.inf

def log_likelihood_cosmo(theta, z, mu, mu_err):
    H0, Om0 = theta
    # Check for invalid physical parameters early
    if H0 <=0 or Om0 <= 0 or Om0 >= 1.0: return -np.inf
        
    try:
        # Create cosmology object within likelihood evaluation
        cosmo_model = FlatLambdaCDM(H0=H0 * (u.km / u.s / u.Mpc), Om0=Om0)
        mu_model = cosmo_model.distmod(z).value # Get value, assuming data in mags
    except (ValueError, OverflowError, RuntimeWarning): # Catch potential cosmology errors
        return -np.inf

    sigma2 = mu_err**2
    logL = -0.5 * np.sum((mu - mu_model)**2 / sigma2) # Ignoring constant
    return logL if np.isfinite(logL) else -np.inf

def log_posterior_cosmo(theta, z, mu, mu_err):
    lp = log_prior_cosmo(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_cosmo(theta, z, mu, mu_err)

# Step 3: Initialize and Run MCMC (emcee)
ndim = 2
nwalkers = 40
# Initial guess near center of prior range (or from quick fit)
initial_guess = np.array([75.0, 0.5]) 
initial_pos = initial_guess + 1e-1 * initial_guess * np.random.randn(nwalkers, ndim)
# Ensure walkers start within prior bounds
for i in range(nwalkers): 
     while not np.isfinite(log_prior_cosmo(initial_pos[i])):
         initial_pos[i] = initial_guess + 1e-1 * initial_guess * np.random.randn(ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_cosmo, 
                                args=(z_sn, mu_obs, mu_err_model))

# Burn-in
n_burn = 300
print(f"\nRunning burn-in ({n_burn} steps)...")
state = sampler.run_mcmc(initial_pos, n_burn, progress=True)
sampler.reset()
print("Burn-in finished.")

# Production run
n_steps = 1500
print(f"Running production ({n_steps} steps)...")
sampler.run_mcmc(state, n_steps, progress=True)
print("Production run finished.")

# Step 4: Analyze Output
print("\nAnalyzing MCMC output...")
burn_discard = 0 
thin = 10
samples = sampler.get_chain(discard=burn_discard, thin=thin, flat=True)
print(f"Sample shape after thinning: {samples.shape}")

labels_cosmo = ["H0", "Ωm"]
truths_cosmo = [H0_true.value, Om0_true]

# Calculate median and percentiles
results_mcmc_cosmo = []
print("\nMCMC Parameter Estimates (median, 16th, 84th percentiles):")
for i in range(ndim):
    q = np.percentile(samples[:, i], [16, 50, 84])
    results_mcmc_cosmo.append(q[1]) # Store median
    print(f"  {labels_cosmo[i]} = {q[1]:.3f} +{q[2]-q[1]:.3f} / -{q[1]-q[0]:.3f}")

# Generate corner plot
try:
    fig_corner_cosmo = corner.corner(samples, labels=labels_cosmo, truths=truths_cosmo, 
                                     quantiles=[0.16, 0.5, 0.84], show_titles=True)
    # fig_corner_cosmo.show()
    print("Corner plot generated.")
    plt.close(fig_corner_cosmo)
except Exception as e:
    print(f"Could not generate corner plot: {e}")

# Step 5: Visualization (Hubble Diagram)
print("Generating Hubble Diagram plot...")
fig_hub, ax_hub = plt.subplots(figsize=(8, 5))
ax_hub.errorbar(z_sn, mu_obs, yerr=mu_err_model, fmt='.', label='Simulated SN Data')
# Plot best-fit model
z_smooth = np.linspace(z_sn.min(), z_sn.max(), 100)
H0_fit, Om0_fit = results_mcmc_cosmo[0], results_mcmc_cosmo[1]
cosmo_fit = FlatLambdaCDM(H0=H0_fit*(u.km/u.s/u.Mpc), Om0=Om0_fit)
mu_fit = cosmo_fit.distmod(z_smooth)
ax_hub.plot(z_smooth, mu_fit, 'r-', label=f'Best Fit (H0={H0_fit:.1f}, Ωm={Om0_fit:.2f})')
# Plot true model
# mu_true_line = cosmo_true.distmod(z_smooth)
# ax_hub.plot(z_smooth, mu_true_line, 'g--', label='True Cosmology')

ax_hub.set_xlabel("Redshift (z)")
ax_hub.set_ylabel("Distance Modulus (μ)")
ax_hub.set_title("Simulated SN Ia Hubble Diagram and Best Fit")
ax_hub.legend()
ax_hub.grid(True, alpha=0.4)
fig_hub.tight_layout()
# plt.show()
print("Hubble diagram plot generated.")
plt.close(fig_hub)

print("-" * 20)
```

**Summary**

This chapter introduced the Bayesian framework for parameter estimation, presenting an alternative to the frequentist Maximum Likelihood approach covered previously. It began by recasting Bayes' Theorem in the context of inference, explaining the roles of the prior probability distribution P(θ) (representing initial beliefs about parameters θ), the likelihood function P(D|θ) (quantifying how well parameters predict the data D), the posterior probability distribution P(θ|D) (representing updated beliefs after considering data), and the evidence P(D) (the normalization constant). The key output, the posterior distribution, was highlighted as providing a complete characterization of parameter uncertainties and correlations, from which point estimates (mean, median, mode) and credible intervals can be derived. Recognizing the computational challenge of exploring and normalizing complex, high-dimensional posteriors, the chapter focused on Markov Chain Monte Carlo (MCMC) methods as the primary computational tool.

The fundamental concepts behind MCMC were explained: constructing a Markov chain whose stationary distribution matches the target posterior, allowing samples to be drawn that empirically map the posterior shape without calculating the evidence. The basic mechanics of the Metropolis-Hastings algorithm, including proposal distributions and the acceptance/rejection step based on the posterior ratio, were introduced, along with related concepts like Gibbs sampling and more advanced methods (Ensemble Samplers, HMC/NUTS, Nested Sampling). Practical implementation using popular Python libraries was demonstrated, focusing on `emcee` (using an affine-invariant ensemble sampler requiring only a log-posterior function) and `dynesty` (using nested sampling requiring log-likelihood and prior transform functions, and yielding evidence estimates alongside posterior samples). Finally, the crucial steps involved in analyzing MCMC output were detailed: assessing convergence using trace plots and diagnostics like autocorrelation time (or Gelman-Rubin R̂), visualizing the multi-dimensional posterior using corner plots generated by the `corner` package to understand estimates, uncertainties, and degeneracies, and calculating parameter summaries like medians and percentile-based credible intervals from the posterior samples. The distinct interpretation of Bayesian credible intervals versus frequentist confidence intervals was also clarified.

---

**References for Further Reading:**

1.  **Ivezić, Ž., Connolly, A. J., VanderPlas, J. T., & Gray, A. (2014).** *Statistics, Data Mining, and Machine Learning in Astronomy*. Princeton University Press. (Relevant chapters often available online, e.g., Chapter 5 on Bayesian inference: [http://press.princeton.edu/titles/10159.html](http://press.princeton.edu/titles/10159.html))
    *(Provides a comprehensive introduction to Bayesian concepts, Bayes' Theorem, priors, posteriors, MCMC methods (including Metropolis-Hastings, Gibbs, ensemble samplers), convergence diagnostics, and model selection in an astronomical context.)*

2.  **Foreman-Mackey, D., Hogg, D. W., Lang, D., & Goodman, J. (2013).** emcee: The MCMC Hammer. *Publications of the Astronomical Society of the Pacific*, *125*(925), 306–312. [https://doi.org/10.1086/670067](https://doi.org/10.1086/670067) (See also `emcee` documentation: [https://emcee.readthedocs.io/en/stable/](https://emcee.readthedocs.io/en/stable/))
    *(The paper introducing the `emcee` algorithm and package, widely used in astrophysics, discussed and demonstrated in Sec 17.4, 17.5, and Application 17.A.)*

3.  **Speagle, J. S. (2020).** DYNESTY: A dynamic nested sampling package for estimating Bayesian posteriors and evidences. *Monthly Notices of the Royal Astronomical Society*, *493*(3), 3132–3158. [https://doi.org/10.1093/mnras/staa278](https://doi.org/10.1093/mnras/staa278) (See also `dynesty` documentation: [https://dynesty.readthedocs.io/en/latest/](https://dynesty.readthedocs.io/en/latest/))
    *(The paper describing the `dynesty` nested sampling package, an alternative MCMC approach also capable of evidence estimation, discussed and demonstrated conceptually in Sec 17.4 and Application 17.B.)*

4.  **Foreman-Mackey, D. (2016).** corner.py: Scatterplot matrices in Python. *The Journal of Open Source Software*, *1*(2), 24. [https://doi.org/10.21105/joss.00024](https://doi.org/10.21105/joss.00024) (See also `corner` documentation: [https://corner.readthedocs.io/en/latest/](https://corner.readthedocs.io/en/latest/))
    *(Describes the `corner` package, the standard tool for visualizing MCMC posterior distributions as shown in Sec 17.5 and the applications.)*

5.  **Sharma, S. (2017).** Markov Chain Monte Carlo Methods for Bayesian Data Analysis in Astronomy. *Annual Review of Astronomy and Astrophysics*, *55*, 213–259. [https://doi.org/10.1146/annurev-astro-082214-122339](https://doi.org/10.1146/annurev-astro-082214-122339)
    *(A review article providing a pedagogical overview of MCMC methods and their practical application within astronomy, covering concepts from Sec 17.2-17.5.)*
