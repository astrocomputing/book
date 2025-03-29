**Chapter 36: Comparing Simulations with Observations**

While previous chapters focused on the mechanics of running simulations and analyzing their intrinsic properties, the ultimate goal of astrophysical simulation is to understand the real Universe. This chapter addresses the crucial step of **comparing simulation results with observational data**. This comparison serves multiple purposes: **validating** the simulation by testing if its underlying physical assumptions and numerical methods produce outcomes consistent with reality, **constraining** free parameters within the simulation's subgrid models by finding values that best reproduce observed trends, and using simulations as **interpretive tools** to understand the physical processes driving observed phenomena. We begin by emphasizing the fundamental importance of rigorous validation. We then delve into the techniques required to bridge the gap between theoretical simulation outputs (like density, temperature, velocity fields) and actual observables (like images, spectra, catalogs), focusing on methods for creating **mock observations**. This includes generating synthetic images (incorporating effects like stellar populations, dust, and telescope point spread functions), mock spectra or spectral cubes (using stellar population synthesis and potentially radiative transfer), and comparing derived **statistical properties** (like luminosity functions or correlation functions) between simulations and surveys. Finally, we discuss how **parameter studies**, where simulations are run with varying input parameters and compared to observations, can be used to constrain models and understand simulation sensitivities.

**36.1 The Importance of Validation**

Astrophysical simulations, despite their increasing sophistication, remain approximations of the immensely complex Universe. As discussed in Section 31.6, they involve approximations in the physical laws implemented, numerical errors inherent in the discretization and solution methods, limitations imposed by finite computational resolution, and often significant uncertainties associated with necessary **subgrid models** representing unresolved physics like star formation and feedback. Therefore, critically **validating** simulation outputs against real observational data is not just a desirable step, but an absolutely essential component of the scientific process using simulations. Without validation, simulation results remain purely theoretical exercises, disconnected from the reality they aim to explain.

Validation serves several crucial purposes. Firstly, it acts as a fundamental **test of the underlying physical model** and the simulation code itself. If a simulation based on well-established physics and implemented with careful numerical methods fails significantly to reproduce basic observed properties of the systems it models (e.g., if a galaxy formation simulation consistently produces galaxies with drastically incorrect sizes, morphologies, or stellar masses compared to observed galaxies at similar epochs), it signals either a flaw in our theoretical understanding, an error in the code implementation, or inadequate resolution or treatment of key physical processes. Identifying such discrepancies drives theoretical and numerical refinement.

Secondly, validation is essential for **calibrating and constraining subgrid models**. As discussed, many simulations rely on parameterized recipes for unresolved physics (e.g., star formation efficiency, feedback energy coupling). The free parameters in these models often lack precise theoretical determination and must be constrained empirically. This is typically done by running simulations with different subgrid parameter values and comparing the results to a set of key observational constraints (e.g., matching the observed galaxy stellar mass function, star formation rate density history, or cluster gas fractions). Finding the parameter values that allow the simulation to best reproduce these observations effectively calibrates the subgrid model, although this tuning process must be done carefully to avoid overfitting to specific datasets and to ensure the model remains predictive for other observables.

Thirdly, successful validation **builds confidence** in the simulation's ability to make reliable predictions or provide accurate interpretations in regimes where observational data might be sparse or unavailable. If a simulation accurately reproduces a wide range of observed properties, we gain more confidence in its predictions for, say, the behavior of gas flows in the circumgalactic medium (which is hard to observe directly) or the properties of galaxies at very high redshifts where observations are challenging. Validation against known benchmarks increases the credibility of using simulations as exploratory or interpretive tools.

The validation process should ideally involve comparing simulation outputs to a **wide range of observational constraints** across different scales, epochs, and physical properties. Relying on only one or two specific observables for calibration or validation can lead to models that are finely tuned to match those specific metrics but fail dramatically when confronted with other data. A robust simulation should aim to reproduce, at least statistically, the diversity and key scaling relations observed in the real Universe (e.g., mass-metallicity relation, Tully-Fisher relation, fundamental plane, clustering statistics) without excessive parameter tuning for each individual observable.

Comparing simulations and observations is rarely straightforward. Simulations produce "perfect" theoretical data (positions, velocities, temperatures, etc. for all components), while observations involve complex instrumental effects (finite resolution, noise, sensitivity limits), projection effects (we see 2D images of 3D structures), obscuration (by dust), and often measure different physical quantities than directly output by simulations (e.g., observed flux vs. intrinsic luminosity or mass). Therefore, a crucial part of the validation process involves **forward modeling** the simulation output to create **mock observations** (Sec 36.2-36.4) that mimic how the simulated system would actually appear if observed by a specific telescope or survey, including these observational effects. This allows for a more direct, "apples-to-apples" comparison.

The comparison itself can be qualitative (e.g., "Does the simulated galaxy morphology look similar to observed galaxies of the same mass?") or, preferably, **quantitative**. Quantitative comparisons often involve calculating specific **summary statistics** from both the simulation (applied to mock observations) and the real observational data and testing for statistical consistency. Examples include comparing luminosity functions, mass functions, correlation functions, size distributions, color distributions, radial profiles, etc. (Sec 36.5). Statistical tests (Chapter 15) or likelihood-based methods (Chapter 16, 17) can be used to quantify the agreement or disagreement.

It's also important to consider **uncertainties** on both sides of the comparison. Observational data have measurement errors and sample variance. Simulation results have numerical errors and potentially uncertainties arising from subgrid models or variations in initial conditions (often assessed by running multiple realizations or varying parameters). A meaningful comparison must account for these uncertainties; perfect agreement is neither expected nor required, but the simulation should reproduce observations within the combined uncertainties.

Furthermore, validation is not a one-time process. As new, more precise observational data becomes available, simulations should be re-validated against these stronger constraints. As simulation codes improve (higher resolution, better physics), their predictions need to be continually tested. This iterative cycle of simulation, comparison with observation, and refinement of models (both theoretical and subgrid) is fundamental to progress in computational astrophysics.

In conclusion, rigorous comparison with observational data – validation – is an indispensable part of the scientific method when using astrophysical simulations. It tests the underlying physics and code, constrains free parameters in subgrid models, builds confidence in simulation predictions, and drives theoretical and numerical progress. This comparison requires careful consideration of observational effects (leading to mock observations) and quantitative statistical analysis, forming the crucial link between the numerical laboratory and the real Universe.

**36.2 Creating Mock Observations**

A major hurdle in comparing simulation outputs directly with observational data is that they represent fundamentally different types of information. Simulations typically provide the full 3D physical state of the system (density, temperature, velocity, mass, position of particles/cells) in theoretical units at discrete time snapshots. Observations, on the other hand, capture projected 2D information (images) or 1D information (spectra, light curves) integrated along the line of sight, convolved with instrument responses (PSF, spectral resolution), limited by sensitivity and noise, and often measured in observer-dependent units (fluxes, magnitudes). To make a meaningful, quantitative comparison, we often need to **forward model** the simulation output to create **mock observations** or **synthetic data products** that mimic what a specific telescope or instrument would actually detect if it observed the simulated system.

The process of creating mock observations involves several steps, depending on the type of observation being simulated:
1.  **Identify Target Observable:** Determine the specific observational quantity you want to compare with (e.g., an HST F606W image, an SDSS spectrum, a WISE catalog magnitude, an ALMA CO spectral cube).
2.  **Select Relevant Simulation Data:** Identify the necessary physical fields from the simulation snapshot(s) required to generate that observable (e.g., stellar particle ages, masses, metallicities, positions for optical images; gas density, temperature, velocity, abundances for spectral lines or X-ray emission).
3.  **Model Emission/Absorption/Scattering:** Apply physical models to convert the simulation's base physical quantities into intrinsic emissivity or opacity in the relevant wavelength range. This is often the most complex step.
    *   For **stellar light** (optical/NIR images/spectra): Use **Stellar Population Synthesis (SPS)** models (like Starburst99, GALAXEV, FSPS, BPASS). Given the age, mass, and metallicity of simulation star particles, SPS models predict the intrinsic spectral energy distribution (SED) or luminosity in specific filter bands.
    *   For **gas emission lines** (e.g., Hα, [OIII], CO): Model the line emissivity based on gas density, temperature, ionization state, and chemical abundances, often requiring assumptions about ionization sources or using photoionization codes (like Cloudy) in post-processing.
    *   For **thermal dust emission** (IR/sub-mm): Model dust grain properties (composition, size distribution) and calculate emission based on dust temperature (which might need to be calculated self-consistently or assumed based on gas temperature or radiation field).
    *   For **X-ray emission** (from hot gas): Model emissivity based on gas density, temperature, and metallicity using plasma emission codes (like APEC or Cloudy).
    *   For **dust absorption/scattering** (attenuation): Model the dust distribution (often assuming a constant dust-to-gas ratio or linking to metallicity) and perform **radiative transfer** (RT) calculations to determine how much light is absorbed or scattered along the line of sight. Full 3D RT is computationally expensive, so approximations are sometimes used.
4.  **Project onto Observer's Frame:** Integrate the calculated emission (and account for absorption) along chosen lines of sight through the simulation volume to create a 2D map (for imaging) or integrate over a spatial region (for a 1D spectrum). This involves defining an observer position and orientation, handling cosmological effects (redshifting spectra, angular diameter distances), and performing the line-of-sight integration.
5.  **Include Instrumental Effects:** Convolve the projected map with the telescope's **Point Spread Function (PSF)** to simulate finite angular resolution. For spectra, convolve with the instrument's **Line Spread Function (LSF)** to simulate finite spectral resolution. Bin the data onto the detector's pixel grid or wavelength solution.
6.  **Add Noise:** Simulate observational noise based on the instrument characteristics and exposure time (e.g., background noise, Poisson noise from source counts, detector read noise).
7.  **Convert to Observable Units:** Convert the final simulated data into the units typically used for the observation being matched (e.g., counts, magnitudes, Janskys, erg/s/cm²/Å).

This full process can be extremely complex and computationally intensive, especially steps involving detailed radiative transfer or sophisticated SPS modeling. Various levels of approximation are often employed depending on the required fidelity of the mock observation.

Python libraries can assist with several of these steps. `yt` (Chapter 35) provides basic tools for projection and slicing, and can define derived fields for simple emissivity models (like X-ray bremsstrahlung). It also has limited capabilities for generating simple mock images or spectra, sometimes integrating with external tools. Libraries like `astropy.modeling` and `astropy.convolution` can be used for applying PSFs. Dedicated SPS libraries like `python-fsps` or interfaces to codes like Cloudy exist for modeling emission. Full radiative transfer post-processing often requires specialized codes like **SKIRT**, **Powderday**, **RADMC-3D**, which can sometimes be driven or analyzed using Python interfaces.

Creating realistic mock observations is crucial for robust comparison between simulations and real data. It allows testing not only whether the simulation reproduces the intrinsic physical properties correctly, but also whether these properties, when "observed" through a simulated telescope including all its effects, match the actual data obtained. This forward modeling approach accounts for selection effects and biases inherent in the observational process, leading to a more meaningful validation of the simulation. The complexity lies in accurately modeling both the intrinsic emission/absorption physics and the detailed instrument response.

**36.3 Generating Mock Images**

Generating synthetic or **mock images** from simulations is a common way to facilitate direct visual and quantitative comparison with observational imaging data. The goal is to predict how the simulated object or region would appear if observed by a specific telescope and instrument through a particular filter, including effects like resolution limits and potentially dust attenuation.

The simplest approach focuses on **stellar light** in optical or near-infrared bands. This typically involves:
1.  **Selecting Star Particles:** Identify the star particles from the simulation snapshot within the region of interest.
2.  **Assigning Luminosities:** For each star particle, determine its luminosity (or SED) in the desired filter bandpass. This requires using a **Stellar Population Synthesis (SPS)** model. Input parameters needed from the simulation for each star particle are typically its **age**, **initial mass** (or current mass), and **metallicity**. The SPS model (e.g., GALAXEV, FSPS) then predicts the corresponding luminosity L<0xE1><0xB5><0x97><0xE1><0xB5><0x92><0xE1><0xB5><0x8A><0xE1><0xB5><0x91><0xE1><0xB5><0xA3>(age, Z) based on stellar evolution tracks and spectral libraries. Python libraries like `python-fsps` provide interfaces to such models.
3.  **Projecting Luminosities:** Define an observer's viewpoint and project the 3D positions of the star particles onto a 2D image plane. Sum the luminosities of all particles falling into each pixel of the desired mock image grid. This creates an intrinsic, high-resolution luminosity map. Tools like `yt.ParticleProjectionPlot` can perform this projection, using particle luminosity as the field.
4.  **(Optional) Dust Attenuation:** If simulating observed bands affected by dust, this step is crucial but complex. Estimate the dust distribution from the simulation's gas density and metallicity. Perform **radiative transfer** calculations (often using dedicated codes like SKIRT, Powderday, RADMC-3D) to determine how much starlight from each particle is absorbed and scattered by dust along the line of sight to the observer. This modifies the projected luminosity map, dimming and reddening regions obscured by dust. This is computationally expensive.
5.  **Applying Instrumental Effects:**
    *   **PSF Convolution:** Convolve the projected (and potentially dust-attenuated) luminosity map with the **Point Spread Function (PSF)** of the target telescope/instrument (e.g., Hubble/ACS, SDSS). The PSF describes the blurring effect due to diffraction and atmospheric seeing (for ground-based). `astropy.convolution.convolve_fft` can be used for this.
    *   **Pixelation:** Resample the convolved image onto the pixel grid corresponding to the detector's pixel scale.
    *   **Noise Addition:** Add realistic observational noise (background sky noise, detector read noise, Poisson noise from the source counts) based on the simulated exposure time and instrument characteristics.
6.  **Unit Conversion:** Convert the final pixel values into observable units, such as counts per second, surface brightness (mag/arcsec²), or calibrated flux units (e.g., using simulated zeropoints).

Generating mock images for other wavelength regimes follows similar principles but requires different emission models:
*   **Mock X-ray Images:** Model X-ray emissivity (e.g., thermal bremsstrahlung, line emission) based on gas density, temperature, and metallicity using plasma codes (e.g., via `pyXSIM` which interfaces with `yt`, or AtomDB/APEC). Project emissivity along the line of sight, convolve with telescope PSF (e.g., Chandra PSF), add background and Poisson noise.
*   **Mock Radio Continuum Images:** Model synchrotron emissivity based on assumptions about magnetic field strength and cosmic ray electron distribution (often highly uncertain subgrid physics). Project emissivity, convolve with the synthesized beam (PSF) of the radio interferometer, add thermal noise.
*   **Mock HI/Molecular Line Images (Channel Maps):** Model line emissivity based on gas density, temperature, velocity, and abundance. Project emissivity within specific velocity channels along the line of sight. Convolve with spatial and spectral resolution kernels. Add noise.

```python
# --- Code Example 1: Conceptual Mock Stellar Image Generation ---
# Highly simplified: Focuses on projection and PSF convolution conceptually.
# Assumes stellar luminosities `L_band` are already assigned to particles.
# Does NOT include SPS modeling or dust radiative transfer.

import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel, convolve_fft
# Assume yt is used for projection conceptually, or use simple histogramming

print("Conceptual Mock Stellar Image Generation:")

# --- Assume we have simulation particle data ---
# N particles with 3D positions (x,y,z) and assigned luminosity L_band
np.random.seed(0)
n_parts = 5000
# Simulate particles clustered in 3D
positions = np.random.normal(loc=0, scale=1.0, size=(n_parts, 3)) 
# Assign luminosities (e.g., brighter near center)
radius_sq = np.sum(positions**2, axis=1)
luminosities = 100 * np.exp(-radius_sq / 2.0) + np.random.rand(n_parts) * 10
print(f"\nGenerated {n_parts} particle positions and luminosities.")

# --- Step 3: Project Luminosities (Simplified using 2D Histogram) ---
# Project onto X-Y plane. Define image grid parameters.
npix = 100 # Output image size
image_extent = [-3, 3] # Range in X and Y
bins = npix
# Use np.histogram2d to sum luminosities into pixels
# weight = luminosities ensures summing L, not just counting particles
image_intrinsic, xedges, yedges = np.histogram2d(
    positions[:, 0], positions[:, 1], 
    bins=bins, range=[image_extent, image_extent], 
    weights=luminosities
)
image_intrinsic = image_intrinsic.T # Transpose for imshow convention (y, x)
print("Projected luminosities onto image grid.")

# --- Step 5a: Define PSF and Convolve ---
# Simulate a simple Gaussian PSF
psf_sigma_pixels = 2.0 # PSF width in pixels
psf_kernel = Gaussian2DKernel(x_stddev=psf_sigma_pixels)
print(f"\nDefined Gaussian PSF (sigma={psf_sigma_pixels} pixels).")
print("Convolving image with PSF...")
# Use astropy.convolution (handles NaNs/boundaries if needed)
image_convolved = convolve_fft(image_intrinsic, psf_kernel, boundary='wrap') # Or 'fill', 'extend'
print("Convolution complete.")

# --- Step 5b & 6: Conceptual Noise Addition and Scaling ---
# noise = np.random.normal(0, sky_level, size=image_convolved.shape)
# image_final = image_convolved + noise 
# image_final_scaled = image_final * flux_scaling_factor 
print("\n(Conceptual: Add noise and scale to observable units)")

# --- Visualize ---
print("Generating visualization...")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# Intrinsic Image
im0 = axes[0].imshow(image_intrinsic, origin='lower', cmap='afmhot', 
                     extent=image_extent*2, interpolation='nearest')
axes[0].set_title("Intrinsic Luminosity")
fig.colorbar(im0, ax=axes[0], label="Luminosity / pixel")
# Convolved Image
im1 = axes[1].imshow(image_convolved, origin='lower', cmap='afmhot',
                     extent=image_extent*2, interpolation='nearest')
axes[1].set_title(f"Convolved with PSF (σ={psf_sigma_pixels} pix)")
fig.colorbar(im1, ax=axes[1], label="Smoothed Luminosity / pixel")
for ax in axes: ax.set_xlabel("X"); ax.set_ylabel("Y")
fig.tight_layout()
# plt.show()
print("Plots generated.")
plt.close(fig)
print("-" * 20)

# Explanation: This code conceptually demonstrates mock image generation.
# 1. It simulates 3D particle positions and assigns a luminosity `L_band` to each.
# 2. Projection: It uses `np.histogram2d` as a simple way to project and bin the 
#    particle luminosities onto a 2D grid (`image_intrinsic`), summing luminosities 
#    in each pixel using the `weights` argument. (Tools like `yt` provide more 
#    sophisticated projection).
# 3. PSF Convolution: It defines a simple `Gaussian2DKernel` representing the PSF 
#    and uses `astropy.convolution.convolve_fft` to smooth the intrinsic image, 
#    simulating the effect of telescope resolution.
# 4. Noise/Scaling: It conceptually notes the steps of adding realistic noise and 
#    converting to observable units.
# 5. Visualization: It displays the intrinsic projected image and the PSF-convolved image 
#    side-by-side, showing the smoothing effect of the PSF.
# This illustrates key steps but omits the complex SPS and radiative transfer parts.
```

Creating realistic mock images is often computationally demanding but essential for tasks like: testing source detection algorithms, validating morphological analysis techniques, predicting the appearance of simulated objects for comparison with specific observations (e.g., generating mock Hubble or JWST images from a simulation), and constraining physical parameters (like dust content or star formation history) by matching mock images to real ones. Python tools provide building blocks (SPS interfaces, convolution, basic projection), but full radiative transfer often requires dedicated external codes.

**36.4 Generating Mock Spectra and Spectral Cubes**

Complementary to mock images, generating **mock spectra** (1D flux vs. wavelength) or **mock spectral cubes** (3D data with two spatial and one spectral axis, like from IFUs or radio interferometers) from simulations is crucial for comparison with spectroscopic observations. This process typically involves modeling the intrinsic spectrum emitted by simulation elements (star particles or gas cells) and then performing line-of-sight integration and instrumental blurring.

**Mock Stellar Spectra:** To generate the integrated spectrum of a simulated galaxy or stellar system:
1.  **Select Star Particles:** Identify relevant star particles in the simulation snapshot.
2.  **Obtain Intrinsic Spectra:** For each star particle (with known age, mass, metallicity), use an **SPS model** (like FSPS, BPASS via `python-fsps` or other tools) to predict its full intrinsic **Spectral Energy Distribution (SED)**, F<0xE1><0xB5><0x8A><0xE1><0xB5><0x91><0xE1><0xB5><0xA3>(λ, age, Z).
3.  **(Optional) Dust Attenuation:** Model the dust distribution and use **radiative transfer** calculations to attenuate the spectrum of each star particle along the line of sight to the observer. This reddens the spectrum and reduces its overall flux.
4.  **Sum Spectra:** Sum the (potentially attenuated) spectra from all relevant star particles, possibly weighting by luminosity or applying aperture masks to simulate observing only a specific region (like a spectrograph slit or fiber).
5.  **Kinematic Broadening:** Account for Doppler shifts and broadening due to the line-of-sight velocities and velocity dispersion of the star particles contributing to the integrated spectrum. This involves shifting and convolving the summed spectrum with the appropriate line-of-sight velocity distribution (LOSVD).
6.  **Instrumental Effects:** Convolve the resulting spectrum with the instrumental **Line Spread Function (LSF)** to simulate finite spectral resolution, and resample onto the spectrograph's wavelength grid. Add appropriate noise.
7.  **Unit Conversion:** Convert to observed flux units (e.g., erg/s/cm²/Å).

**Mock Gas Emission/Absorption Line Spectra:**
1.  **Select Gas Cells/Particles:** Identify gas elements along the desired line(s) of sight.
2.  **Model Emissivity/Opacity:** Calculate the intrinsic emissivity (for emission lines like Hα, [OIII], CO) or opacity (for absorption lines) of the desired spectral line(s) in each gas element based on its density, temperature, velocity, ionization state, and chemical abundances. This might involve simple recipes, photoionization modeling (e.g., Cloudy), or assuming excitation conditions.
3.  **Line-of-Sight Integration (Radiative Transfer):** Solve the radiative transfer equation along the line of sight, integrating the emissivity and accounting for opacity (and potentially scattering) to calculate the emergent spectrum. For optically thin emission lines, this might simplify to summing the Doppler-shifted emissivity along the line of sight.
4.  **Instrumental Effects:** Convolve with instrumental spatial (PSF) and spectral (LSF) resolution, resample onto detector grid/wavelength solution, add noise.

**Mock Spectral Cubes (IFU/Radio):** Generating a mock data cube involves performing the spectral generation process (stellar and/or gas) for *each* spatial pixel (spaxel) in the desired field of view. This typically involves:
1.  Defining a grid of sightlines corresponding to the spaxels.
2.  For each sightline, performing the line-of-sight integration of emission/absorption (Steps 1-3 above) to generate a 1D spectrum for that spaxel.
3.  Applying instrumental effects (PSF spatial coupling between spaxels, LSF convolution spectrally).
4.  Assembling the resulting spectra into a 3D data cube (x, y, wavelength).
This is computationally intensive as it requires generating potentially thousands of spectra. `yt` has some capabilities for generating simplified spectral cubes.

```python
# --- Code Example 1: Conceptual Mock Spectrum Generation (Stellar) ---
# Highly simplified: Focuses on combining SPS results conceptually.
# Assumes SPS library `sps_model` exists and returns spectra.
# Omits dust, kinematics, LSF, noise for clarity.

import numpy as np
import matplotlib.pyplot as plt
# Assume SPS model interface exists:
# def get_sps_spectrum(age, metallicity, mass): 
#    # Returns wavelength_array, luminosity_density_lambda_array
#    # Units: e.g., Angstrom, Lsun/Angstrom
#    # Placeholder implementation:
#    wave = np.linspace(3000, 9000, 1000) # Angstrom
#    # Crude model: blackbody scaled by mass, peak depends on age/Z (very approx)
#    T = 10000 * (age / 1e9)**(-0.2) * (metallicity / 0.02)**(-0.1)
#    from astropy.modeling.physical_models import BlackBody
#    bb = BlackBody(temperature=T*u.K)
#    spec = bb(wave*u.AA) * (1*u.sr) * np.pi * (1*u.Rsun**2) # Example scaling
#    return wave, (spec.to(u.Lsun/u.AA).value * mass / 1.0) # Scale by mass

print("Conceptual Mock Integrated Stellar Spectrum Generation:")

# --- Assume simulation provides star particle properties ---
n_stars = 100
star_ages_gyr = np.random.uniform(0.1, 10.0, n_stars) # Ages in Gyr
star_metallicities = 10**(np.random.uniform(-1.0, 0.2, n_stars)) # Z relative to solar
star_masses = 10**(np.random.normal(0, 0.5, n_stars)) # Log-normal mass distribution

print(f"\nGenerated properties for {n_stars} star particles.")

# --- Step 2 & 3: Get Intrinsic Spectra and Sum ---
# Assume `get_sps_spectrum` exists (as defined conceptually above)
print("Summing intrinsic spectra from SPS model (conceptual)...")
total_spectrum = None
wavelengths = None

for i in range(n_stars):
    # Get intrinsic spectrum for this star particle
    # wave, spec = get_sps_spectrum(star_ages_gyr[i], star_metallicities[i], star_masses[i])
    # Simulate getting spectrum
    if i == 0:
         wavelengths = np.linspace(3000, 9000, 1000)
         total_spectrum = np.zeros_like(wavelengths)
    # Add crude blackbody based on age/metallicity (highly simplified!)
    temp = 10000 * (star_ages_gyr[i])**(-0.2) * (star_metallicities[i])**(-0.1)
    term = (const.h * const.c) / (wavelengths * u.AA * const.k_B * (temp*u.K))
    term = np.clip(term.decompose().value, -np.inf, 700) # Avoid overflow
    bb_spec = 1.0 / (np.exp(term) - 1.0) / wavelengths**5 # Proportional to B_lambda
    total_spectrum += bb_spec * star_masses[i] # Weighted sum (crude)

print("Summation complete.")
# Note: Real SPS involves complex libraries, dust is ignored, kinematics ignored.

# --- Step 5, 6, 7: Conceptual Instrumental Effects & Units ---
# Convolve with LSF (e.g., Gaussian kernel)
# Resample onto instrument wavelength grid
# Add noise
# Convert to observed flux units (erg/s/cm2/A) using distance
print("\n(Conceptual: Apply kinematic broadening, LSF, noise, unit conversion)")
final_flux = total_spectrum * 1e-17 # Arbitrary scaling for plotting

# --- Plot Result ---
if final_flux is not None:
    print("Generating plot of mock integrated spectrum...")
    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, final_flux)
    plt.xlabel("Wavelength (Angstrom)")
    plt.ylabel("Flux Density (Arbitrary Units)")
    plt.title("Conceptual Mock Integrated Stellar Spectrum")
    plt.grid(True, alpha=0.4)
    # plt.yscale('log')
    # plt.show()
    print("Plot generated.")
    plt.close()

print("-" * 20)

# Explanation: This code conceptually outlines generating an integrated stellar spectrum.
# 1. It simulates properties (age, metallicity, mass) for star particles.
# 2. It conceptually defines or calls a function `get_sps_spectrum` that would return 
#    the intrinsic spectrum (luminosity density vs wavelength) for a single stellar 
#    population based on its properties (using a highly simplified blackbody proxy here).
# 3. It loops through the star particles, gets the spectrum for each, and sums them up 
#    to get the `total_spectrum` of the integrated population.
# 4. It conceptually notes the crucial subsequent steps: accounting for dust attenuation 
#    (often via external RT codes), kinematic broadening (Doppler shifts/dispersion), 
#    convolution with the instrumental Line Spread Function (LSF), resampling, adding 
#    noise, and converting to observed flux units using the simulated object's distance.
# 5. It plots the resulting (highly simplified) spectrum. This illustrates the workflow 
#    but highlights the complexity involved in realistic spectral synthesis.
```

Generating mock spectra and cubes is vital for interpreting spectroscopic observations. It allows testing whether the stellar populations, gas properties, and kinematics in a simulation are consistent with observed line strengths, ratios, widths, and velocity fields. It is also essential for designing spectroscopic surveys, predicting observable signatures, and developing analysis pipelines for instruments like JWST/NIRSpec, MUSE, or ALMA. While computationally demanding, especially when including radiative transfer or detailed kinematics, creating mock spectra provides a crucial link between the physical conditions in simulations and the rich diagnostic information available from spectroscopy.

**36.5 Statistical Comparisons**

While visual comparison of mock observations (images, spectra) with real data is informative, rigorous validation of simulations often requires quantitative **statistical comparisons**. Instead of comparing individual objects, we compare the statistical *distributions* of properties derived from a large sample of objects in the simulation (or mock observation) with the corresponding distributions derived from observational surveys. Agreement (or disagreement) in these statistical measures provides a powerful test of the simulation's ability to reproduce the observed universe's population properties.

Common statistical properties used for comparison include:
*   **Luminosity Functions (LFs):** The number density of objects (e.g., galaxies, quasars) as a function of their luminosity or absolute magnitude in a specific bandpass (e.g., Φ(M<0xE1><0xB5><0xA3>), number per Mpc³ per magnitude). Simulations predict LFs by applying SPS models (Sec 36.4) to simulated galaxies and counting them in magnitude bins. Comparing simulated LFs to observed LFs (e.g., from SDSS, GAMA) tests whether the simulation correctly reproduces the abundance of objects across different brightness levels.
*   **Stellar Mass Functions (SMFs):** The number density of galaxies as a function of their stellar mass (Φ(M<0xE2><0x82><0x9B>), number per Mpc³ per dex in mass). This requires estimating stellar masses for simulated galaxies (often related to integrated star formation history) and comparing to SMFs derived observationally (often requiring careful mass estimation from photometry or spectra). The SMF tests the simulation's modeling of galaxy assembly and feedback processes that regulate stellar mass growth.
*   **Halo Mass Functions (HMFs):** For N-body simulations, the number density of dark matter halos as a function of halo mass (e.g., M₂₀₀<0xE1><0xB5><0x84>). This is a fundamental prediction of cosmological models and can be compared (though often indirectly) to observational estimates derived from cluster counts, weak lensing, or galaxy clustering.
*   **Correlation Functions / Power Spectra:** These statistics quantify the spatial **clustering** of objects (galaxies, halos). The **two-point correlation function ξ(r)** measures the excess probability (compared to random) of finding pairs of objects separated by a distance `r`. Its Fourier transform, the **power spectrum P(k)**, measures the variance of density fluctuations as a function of spatial wavenumber `k`. Comparing the clustering measured in simulations (from particle or halo positions) to that observed in large galaxy redshift surveys (like SDSS/BOSS/eBOSS/DESI) provides sensitive tests of cosmology and galaxy bias (how galaxies trace the underlying dark matter distribution). Baryon Acoustic Oscillations (BAO) appear as specific features in ξ(r) and P(k), acting as standard rulers.
*   **Distribution of Other Properties:** Comparisons can be made for distributions of galaxy colors, sizes, morphologies, star formation rates, metallicities, gas fractions, black hole masses, etc., comparing histograms or CDFs from simulations (via mock catalogs) with observational data using statistical tests like K-S or A-D (Sec 15.4).
*   **Scaling Relations:** Comparing relationships between different properties, such as the star formation main sequence (SFR vs. M<0xE2><0x82><0x9B>), the mass-metallicity relation, the Tully-Fisher relation (rotation velocity vs. luminosity/mass for spirals), or the Fundamental Plane (relating size, surface brightness, velocity dispersion for ellipticals). Simulations should reproduce the observed slopes, normalizations, and scatter of these key scaling relations.

Performing these statistical comparisons involves several steps:
1.  **Generate Mock Catalog/Data:** Create a catalog of simulated objects (halos, galaxies) with the relevant properties (positions, masses, luminosities, colors, SFRs, etc.) derived from the simulation snapshot(s), potentially by applying mock observation effects (selection cuts, magnitude limits, errors) matching the target observational survey.
2.  **Calculate Statistic:** Compute the desired statistic (e.g., LF, SMF, correlation function) from both the simulated mock catalog and the real observational data using consistent methods and binning. Python libraries like `numpy`, `scipy.stats`, `astropy.stats`, and potentially specialized cosmology libraries (`halotools`, `Corrfunc`) are used.
3.  **Estimate Uncertainties:** Determine uncertainties for both the simulated statistic (e.g., from sample variance within the simulation volume, often estimated using multiple realizations or jackknife/bootstrap resampling) and the observational statistic (including measurement errors, sample variance, systematic uncertainties).
4.  **Compare Quantitatively:** Plot the simulated statistic (often with error bars/regions representing simulation variance) against the observational data points (with their error bars). Perform statistical tests (e.g., Chi-squared goodness-of-fit if comparing binned functions with well-defined errors) to quantify the level of agreement or disagreement, taking uncertainties into account.

```python
# --- Code Example 1: Conceptual Comparison of Luminosity Functions ---
import numpy as np
import matplotlib.pyplot as plt

print("Conceptual comparison of Simulated vs Observed Luminosity Function:")

# --- Assume these data are loaded/calculated ---
# Observational Data (e.g., from a survey paper table)
obs_mag_bins = np.array([-22, -21, -20, -19, -18, -17]) # Bin centers (Abs Mag)
obs_phi = 10**np.array([-2.5, -2.8, -3.1, -3.5, -4.0, -4.5]) # Number density (Mpc^-3 mag^-1)
obs_phi_err = obs_phi * 0.2 # Example 20% errors

# Simulation Mock Catalog Results (calculated from mock galaxy catalog)
sim_mag_bins = np.array([-22, -21, -20, -19, -18, -17]) # MUST use same bins
sim_phi = 10**np.array([-2.6, -2.85, -3.05, -3.6, -4.1, -4.6])
# Simulation errors might come from volume variance (e.g., sqrt(N)/V or jackknife)
sim_phi_err = sim_phi * 0.15 

print("\nLoaded/Calculated LF data (conceptual).")

# --- Create Comparison Plot ---
print("Generating comparison plot...")
fig, ax = plt.subplots(figsize=(7, 5))

# Plot observational data
ax.errorbar(obs_mag_bins, obs_phi, yerr=obs_phi_err, fmt='o', color='black', 
            label='Observed LF', capsize=3)

# Plot simulation results
ax.errorbar(sim_mag_bins, sim_phi, yerr=sim_phi_err, fmt='s', color='red', 
            linestyle='--', alpha=0.8, label='Simulation Mock LF')

# Customize plot
ax.set_xlabel("Absolute Magnitude (M)")
ax.set_ylabel("Number Density Φ (Mpc⁻³ mag⁻¹)")
ax.set_yscale('log') # LFs are usually plotted log scale
# Magnitude axes often decrease to the right (brighter left)
# ax.invert_xaxis() 
ax.set_title("Luminosity Function: Simulation vs. Observation")
ax.legend()
ax.grid(True, alpha=0.4)
fig.tight_layout()
# plt.show()
print("Plot generated.")
plt.close(fig)

# --- Conceptual Quantitative Comparison (Chi-squared) ---
print("\nConceptual Quantitative Comparison (Chi-squared):")
# Assuming errors are Gaussian and uncorrelated, and represent 1-sigma
# Combine errors in quadrature if both have significant uncertainty? Or just use obs error?
# Let's use obs_phi_err as sigma_i for simplicity here.
try:
    chisq = np.sum( ((obs_phi - sim_phi) / obs_phi_err)**2 )
    n_bins = len(obs_mag_bins)
    # Degrees of freedom depends if simulation was tuned to this data!
    # Assume 0 free parameters fitted here, df = n_bins
    dof = n_bins 
    reduced_chisq = chisq / dof
    print(f"  Chi-squared = {chisq:.2f}")
    print(f"  Degrees of Freedom = {dof}")
    print(f"  Reduced Chi-squared = {reduced_chisq:.2f}")
    # p_value = 1.0 - stats.chi2.cdf(chisq, df=dof)
    # print(f"  Goodness-of-fit p-value = {p_value:.3f}")
    if reduced_chisq > 1.5: print("  -> Model might be poor fit or errors underestimated.")
    elif reduced_chisq < 0.5: print("  -> Model might be overfitting or errors overestimated.")
    else: print("  -> Model provides a reasonable fit (statistically consistent).")
except Exception as e_chi2:
    print(f"  Could not calculate Chi-squared: {e_chi2}")

print("-" * 20)

# Explanation: This code conceptually compares a simulated luminosity function (LF) 
# with observational data.
# 1. It assumes arrays containing magnitude bins (`obs_mag_bins`, `sim_mag_bins`), 
#    number densities (`obs_phi`, `sim_phi`), and their associated errors 
#    (`obs_phi_err`, `sim_phi_err`) have been obtained.
# 2. It creates a plot using `matplotlib.pyplot.errorbar` to display both the 
#    observed data points and the simulation results with their respective error bars 
#    on the same axes (typically log scale for density vs magnitude). This allows 
#    for visual comparison.
# 3. It conceptually performs a quantitative comparison using a simple Chi-squared 
#    calculation, summing the squared differences between observation and simulation, 
#    weighted by the observational errors squared. 
# 4. It calculates the reduced Chi-squared (Chi2/dof), assuming degrees of freedom 
#    equal to the number of bins (implying no parameters were fit to this specific data), 
#    and provides a basic interpretation of whether the fit is statistically reasonable.
# This illustrates the process of comparing statistical distributions derived from 
# simulations and observations.
```

Statistical comparisons provide the most rigorous way to validate simulations and constrain models. Discrepancies identified through these comparisons often point towards missing physics in the simulation (especially subgrid models), inaccuracies in the underlying cosmological model, or unaccounted-for systematic effects in the observational data. Iteratively improving simulations to better match a wide range of statistical observations is a core driver of progress in computational astrophysics.

**36.6 Parameter Studies and Constraining Models**

Simulations often contain **free parameters**, particularly within the **subgrid physics models** (Sec 31.6, 34.4) used to represent unresolved processes like star formation efficiency, feedback energy coupling, or AGN feedback mechanisms. The values of these parameters are often poorly constrained by first principles and can significantly impact the simulation outcomes (e.g., galaxy stellar masses, star formation rates). **Parameter studies** – running suites of simulations where these subgrid parameters (or sometimes cosmological parameters) are systematically varied – combined with comparison to observations (Sec 36.5), provide a crucial method for **constraining** these parameters and understanding the simulation's sensitivity to them.

The basic workflow for a parameter study involves:
1.  **Identify Key Parameters:** Determine which parameters (subgrid or cosmological) are expected to have the most significant impact on the observables you aim to match and are most uncertain.
2.  **Define Parameter Ranges:** Choose plausible ranges and sampling strategies (e.g., grid sampling, Latin hypercube sampling) for varying the selected parameters.
3.  **Run Simulation Suite:** Execute a series of simulations, each with a different combination of parameter values within the chosen ranges. This often requires significant computational resources (HPC clusters).
4.  **Post-process and Calculate Observables:** For each simulation run, perform the necessary analysis (Sec 33.5, 34.6) and generate mock observations (Sec 36.2-36.4) or statistical summaries (Sec 36.5) corresponding to the target observational constraints.
5.  **Compare with Observations:** Quantify the agreement between the results from each simulation run and the real observational data using appropriate statistical measures (e.g., Chi-squared, likelihood values).
6.  **Constrain Parameters:** Identify the region(s) in the parameter space where the simulation results are statistically consistent with the observations. This provides constraints on the plausible values of the free parameters.

This process essentially treats the simulation itself as a complex, computationally expensive model whose parameters we want to estimate by fitting it to observational data. The comparison statistics (like Chi-squared or likelihood) calculated in Step 5 can be used within statistical inference frameworks (like MLE or Bayesian inference) to find the "best-fit" parameters and their uncertainties.

For example, to constrain the efficiency of supernova feedback (parameter ε<0xE2><0x82><0x9B><0xE2><0x82><0x99>), one might run several galaxy formation simulations varying ε<0xE2><0x82><0x9B><0xE2><0x82><0x99> over a range. For each run, calculate the simulated galaxy stellar mass function (SMF). Compare each simulated SMF to the observed SMF using a Chi-squared statistic. The value of ε<0xE2><0x82><0x9B><0xE2><0x82><0x99> that yields the minimum Chi-squared represents the best-fit value constrained by the SMF data. Analyzing how Chi-squared varies with ε<0xE2><0x82><0x9B><0xE2><0x82><0x99> can provide an estimate of the uncertainty on this best-fit value (similar to profile likelihood, Sec 16.5).

Due to the high computational cost of running many full physics simulations, direct fitting using MCMC (Chapter 17) or extensive grid searches across many parameters is often infeasible. Several strategies are used to make parameter studies more tractable:
*   **Targeted Variation:** Focus on varying only one or two key parameters at a time, keeping others fixed, to understand individual sensitivities (though this ignores parameter correlations).
*   **Lower Resolution/Volume Runs:** Perform initial parameter exploration using smaller, lower-resolution simulations before running expensive high-resolution simulations for the most promising parameter regions.
*   **Emulation:** Build a statistical "emulator" model (often using Gaussian processes or machine learning) that learns to predict the simulation output (e.g., the resulting SMF or power spectrum) as a function of the input parameters, based on a limited number of actual simulation runs. This fast emulator can then be used within MCMC or optimization routines to explore the parameter space much more efficiently than running the full simulation repeatedly.
*   **History Matching:** An iterative approach where parameter space regions inconsistent with observations are progressively ruled out based on comparisons with simulation runs, focusing computational effort on plausible regions.

Parameter studies are essential not only for constraining unknown subgrid physics parameters but also for understanding the **systematic uncertainties** associated with simulation predictions. By varying parameters within plausible ranges, one can assess how sensitive the simulation results are to these choices, providing an estimate of the theoretical uncertainty originating from the subgrid modeling itself. This is crucial for making robust comparisons between simulations and observations.

Furthermore, parameter studies help identify **degeneracies**. Different combinations of parameters might produce similar observable outcomes, making it difficult to constrain individual parameters uniquely using only one type of observation. Combining multiple observational constraints (e.g., SMF *and* galaxy clustering *and* gas fractions) within the parameter study is often necessary to break these degeneracies and obtain tighter constraints.

Performing parameter studies requires significant computational resources, careful experimental design (how to sample parameter space), robust analysis pipelines to calculate observables from simulation outputs consistently, and appropriate statistical methods for comparing simulation results with observations and inferring parameter constraints. Python plays a crucial role in orchestrating these workflows, managing simulation inputs/outputs, running analysis scripts (often using `yt`), and performing the final statistical comparisons.

**(Code examples for running full simulation parameter studies are highly complex and system-dependent, involving HPC job submission and management beyond simple Python scripts. Conceptual illustration is difficult here.)**

**Application 36.A: Generating a Mock SDSS Image from a Galaxy Simulation**

**(Paragraph 1)** **Objective:** This application demonstrates the process of creating a simple mock observation (Sec 36.2, 36.3), specifically generating a synthetic optical image resembling one from the Sloan Digital Sky Survey (SDSS) based on the output of a galaxy formation simulation. It involves assigning luminosities to star particles, projecting them, convolving with a PSF, and adding noise. Reinforces Sec 36.2, 36.3, utilizing concepts from simulation analysis (e.g., `yt` or `h5py`) and image processing (`astropy.convolution`).

**(Paragraph 2)** **Astrophysical Context:** Galaxy formation simulations predict the 3D distribution, masses, ages, and metallicities of stellar populations within simulated galaxies. To compare these predictions directly with observations from imaging surveys like SDSS, astronomers need to generate mock images that show how the simulated galaxy would actually appear through the telescope and filter, including observational effects like blurring (PSF) and noise. Comparing the morphology, colors (by generating images in multiple bands), and surface brightness profiles of mock images with real SDSS data provides a powerful validation test for the simulation's ability to reproduce realistic galaxy appearances.

**(Paragraph 3)** **Data Source:** A snapshot file (`galaxy_sim.hdf5`) from a hydrodynamical galaxy formation simulation (e.g., IllustrisTNG, EAGLE, FIRE) containing star particle data: 3D positions (`Coordinates`), `Masses` (or initial masses), `StellarFormationTime` (or age), and `GFM_Metallicity` (or similar metallicity field). We also need the simulation time/redshift and box size/cosmological parameters if converting to apparent magnitudes later.

**(Paragraph 4)** **Modules Used:** `yt` (or `h5py` for reading), `numpy`, `astropy.units`, `astropy.constants`, `astropy.convolution` (for PSF), `matplotlib.pyplot`. A **Stellar Population Synthesis (SPS)** library like `fsps` (`python-fsps`) or pre-computed SPS tables are conceptually needed to assign realistic luminosities (simulated here).

**(Paragraph 5)** **Technique Focus:** Mock image generation workflow. (1) Loading star particle data (position, mass, age, metallicity) from simulation snapshot. (2) Assigning luminosity in a target band (e.g., SDSS r-band) to each star particle, conceptually using an SPS model based on its age and metallicity (simulated with a simple placeholder here). (3) Projecting these luminosities onto a 2D pixel grid aligned with a chosen line of sight using binning (`np.histogram2d` or `yt.ParticleProjectionPlot`). (4) Creating a Point Spread Function (PSF) model (e.g., a simple Gaussian representing atmospheric seeing for SDSS). (5) Convolving the intrinsic projected image with the PSF using `astropy.convolution.convolve_fft`. (6) Adding realistic background noise and potentially Poisson noise. (7) Displaying the final mock image. (Note: Dust attenuation is omitted for simplicity).

**(Paragraph 6)** **Processing Step 1: Load Data:** Use `yt.load()` or `h5py` to load the simulation snapshot. Extract arrays for star particle positions (`pos_stars`), masses (`mass_stars`), ages (`age_stars`, calculated from formation time and snapshot time), and metallicities (`Z_stars`).

**(Paragraph 7)** **Processing Step 2: Assign Luminosities (Conceptual SPS):** For each star particle `i`, calculate its luminosity `L_r[i]` in the SDSS r-band based on `age_stars[i]`, `Z_stars[i]`, and `mass_stars[i]`. **This is the most complex step in reality, requiring an SPS model.** For this example, we'll use a highly simplified placeholder: `L_r = mass_stars * (age_stars / 1e9)**(-1.0) * 10**(-0.4 * Z_stars)`. *This is NOT physically accurate but serves to provide varying luminosities for the demonstration.*

**(Paragraph 8)** **Processing Step 3: Project Luminosities:** Define the image grid (number of pixels `npix`, field of view `fov`). Choose a projection axis (e.g., z-axis). Use `np.histogram2d(pos_stars[:, 0], pos_stars[:, 1], bins=npix, range=[[-fov/2, fov/2], [-fov/2, fov/2]], weights=L_r)` to create the 2D intrinsic luminosity map `image_intrinsic`.

**(Paragraph 9)** **Processing Step 4: Convolve with PSF:** Define a PSF, e.g., a Gaussian representing SDSS seeing: `psf_sigma_pixels = 1.5 / pixel_scale` (where pixel_scale is arcsec/pixel, e.g., 0.396 for SDSS). Create the kernel `psf = Gaussian2DKernel(x_stddev=psf_sigma_pixels)`. Convolve the intrinsic image: `image_convolved = convolve_fft(image_intrinsic, psf)`.

**(Paragraph 10)** **Processing Step 5: Add Noise and Display:** Simulate background noise (e.g., `sky_level = ...; noise = np.random.normal(0, sky_level, image_convolved.shape)`). Add noise: `image_mock = image_convolved + noise`. Display `image_mock` using `plt.imshow`, perhaps using `simple_norm` or `LogNorm` for scaling. Add appropriate labels and title.

**Output, Testing, and Extension:** Output is the generated mock SDSS r-band image as a plot or saved FITS file. **Testing:** Verify the image shows a blurred representation of the projected star particle distribution. Check if the overall brightness and size appear reasonable for a galaxy. Compare the visual morphology to real SDSS images of galaxies with similar simulated masses/properties. **Extensions:** (1) Use a real SPS library (`python-fsps`) to calculate accurate luminosities based on age/metallicity. (2) Generate images in multiple bands (g, r, i) and create a mock color composite image. (3) Implement dust attenuation using a simple screen model or by linking to gas density from the simulation (requires hydro simulation). (4) Use a more realistic PSF model (e.g., Moffat profile or PSF image from SDSS). (5) Convert the mock image flux units to SDSS nanomaggies or magnitudes for quantitative comparison. (6) Generate mock images for a large sample of simulated galaxies and compare their statistical morphological properties (e.g., concentration, asymmetry) with observed distributions.

```python
# --- Code Example: Application 36.A ---
# Note: Conceptual, requires libraries. Simulates SPS and noise addition.
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.visualization import simple_norm
# import yt # or h5py for actual data loading
# import fsps # for realistic SPS

print("Generating Mock SDSS r-band Image from Simulated Galaxy:")

# Step 1: Simulate/Load Star Particle Data
np.random.seed(42)
n_stars = 10000
# Simulate particles concentrated towards center, representing a galaxy
scale_radius = 2.0 # kpc
pos_stars = np.random.normal(loc=0.0, scale=scale_radius, size=(n_stars, 3)) # 3D positions (kpc)
# Simulate masses (e.g., from IMF)
mass_stars = 10**(np.random.normal(loc=np.log10(0.5), scale=0.5, size=n_stars)) * 1e9 # Msun
# Simulate ages (e.g., uniform spread + older bulge)
age_stars = np.random.uniform(1, 10, n_stars) * 1e9 # years
rad_xy = np.sqrt(pos_stars[:,0]**2 + pos_stars[:,1]**2)
age_stars[rad_xy < 0.5*scale_radius] = np.random.uniform(8, 12, np.sum(rad_xy < 0.5*scale_radius)) * 1e9
# Simulate metallicities (e.g., higher near center)
Z_stars = 0.02 * 10**(-0.5 * rad_xy / scale_radius + np.random.normal(0, 0.1, n_stars))
print(f"\nGenerated {n_stars} star particle properties (pos, mass, age, Z).")

# Step 2: Assign Luminosities (Highly Simplified Placeholder!)
print("Assigning r-band luminosities (simplified model)...")
# L ~ Mass * Age^(-gamma) * Metal_Factor (VERY CRUDE)
# This step REQUIRES a real SPS model for accurate results.
L_r = mass_stars * (age_stars / 1e9)**(-0.8) * (Z_stars / 0.02)**(-0.2)
# Add arbitrary scaling factor for visualization
L_r = L_r / 1e10 
print("Luminosities assigned.")

# Step 3: Project Luminosities onto Image Grid
print("Projecting luminosities onto 2D grid...")
npix = 128 # Pixels per side
fov = 15.0 # Field of view in kpc
image_range = [[-fov/2, fov/2], [-fov/2, fov/2]]
pixel_size = fov / npix
# Project onto x-y plane, summing luminosities in each pixel
image_intrinsic, _, _ = np.histogram2d(
    pos_stars[:, 0], pos_stars[:, 1], 
    bins=npix, range=image_range, 
    weights=L_r
)
image_intrinsic = image_intrinsic.T # Transpose (y, x)

# Step 4: Define PSF and Convolve
# Assume SDSS seeing ~1.5 arcsec. Need pixel scale. 
# Assume distance -> pixel scale in kpc/pixel. Say 1 pix = 0.1 kpc here.
# So PSF sigma ~ 1.5 arcsec / (0.1 kpc/pix) -> need distance to convert arcsec to kpc.
# Let's just assume a PSF width in pixels for simplicity.
psf_sigma_pix = 2.0 # pixels
psf = Gaussian2DKernel(x_stddev=psf_sigma_pix)
print(f"\nConvolving with Gaussian PSF (sigma = {psf_sigma_pix} pixels)...")
image_convolved = convolve_fft(image_intrinsic, psf, boundary='wrap')

# Step 5: Add Noise and Display
print("Adding simulated background noise...")
sky_level = np.median(image_convolved[image_convolved > 0]) * 0.01 # Example: 1% of median signal
noise = np.random.normal(0, sky_level, size=image_convolved.shape)
image_mock = image_convolved + noise
print("Noise added.")

print("\nGenerating plots...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
# Use simple_norm for display scaling
norm_intrinsic = simple_norm(image_intrinsic, stretch='log', percent=99.9)
norm_convolved = simple_norm(image_convolved, stretch='log', percent=99.9)
norm_mock = simple_norm(image_mock, stretch='log', percent=99.9)

im0 = axes[0].imshow(image_intrinsic, origin='lower', cmap='magma', norm=norm_intrinsic)
axes[0].set_title("Intrinsic Projection (Log)"); fig.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(image_convolved, origin='lower', cmap='magma', norm=norm_convolved)
axes[1].set_title("PSF Convolved (Log)"); fig.colorbar(im1, ax=axes[1])
im2 = axes[2].imshow(image_mock, origin='lower', cmap='magma', norm=norm_mock)
axes[2].set_title("Convolved + Noise (Log)"); fig.colorbar(im2, ax=axes[2])

for ax in axes: ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Mock Galaxy Image Generation Steps")
fig.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()
print("Plots generated.")
plt.close(fig)

print("-" * 20)
```

**Application 36.B: Comparing Simulated vs Observed Galaxy Mass Function**

**(Paragraph 1)** **Objective:** This application demonstrates a quantitative statistical comparison (Sec 36.5) between a cosmological simulation and observational data, specifically focusing on the galaxy stellar mass function (SMF) – the number density of galaxies per unit volume per unit logarithmic mass. It involves extracting galaxy stellar masses from a simulation catalog, calculating the simulated SMF, loading an observed SMF from literature data, and plotting them together for comparison, potentially including a basic quantitative assessment like Chi-squared.

**(Paragraph 2)** **Astrophysical Context:** The galaxy stellar mass function (SMF) is a fundamental statistical description of the galaxy population, charting the relative abundance of galaxies of different stellar masses. Its shape and evolution with redshift are strongly influenced by the underlying cosmology (primarily through the dark matter halo mass function) and the complex baryonic physics governing galaxy formation, particularly star formation efficiency and feedback processes (which regulate mass growth). Comparing SMFs predicted by cosmological simulations (hydrodynamical or semi-analytic models built on N-body) with accurately measured observational SMFs provides a key test for galaxy formation models and their subgrid physics implementations. Discrepancies often highlight areas where models need refinement (e.g., overproducing massive galaxies, underproducing dwarfs).

**(Paragraph 3)** **Data Source:**
    1.  **Simulation Data:** A galaxy catalog extracted from a cosmological simulation snapshot (e.g., using halo finders like Rockstar/Consistent-Trees or SUBFIND applied to N-body/hydro simulations like Bolshoi, MultiDark, IllustrisTNG, EAGLE). The catalog must contain stellar mass (`M_stellar`, typically in M<0xE2><0x82><0x99><0xE1><0xB5><0x98><0xE1><0xB5><0x8A>/h or similar) for each simulated galaxy (or subhalo). We also need the volume of the simulation box (`box_volume`, in (Mpc/h)³ or similar).
    2.  **Observational Data:** Tabulated observational SMF data (`smf_observed.txt`) from a relevant survey and redshift (e.g., from SDSS, GAMA, CANDELS). This typically provides bin centers or edges in log₁₀(M<0xE2><0x82><0x9B>), the number density Φ (`phi_obs`, in Mpc⁻³ dex⁻¹), and associated uncertainties (`phi_err`).

**(Paragraph 4)** **Modules Used:** `numpy` (for calculations, histograms), `astropy.table.Table` or `pandas.DataFrame` (for loading catalogs/data), `matplotlib.pyplot` (for plotting). `scipy.stats.chisquare` (optional, for quantitative comparison). `astropy.units` (optional, for handling units consistently).

**(Paragraph 5)** **Technique Focus:** Statistical comparison between simulation and observation (Sec 36.5). Reading simulation catalog data (stellar masses). Calculating a histogram of log₁₀(M<0xE2><0x82><0x9B>) from the simulation data using `np.histogram`. Converting histogram counts to number density Φ by dividing by the bin width (in dex) and the simulation volume. Loading observational SMF data (bin centers, Φ, error). Plotting both the simulated SMF (with uncertainties, e.g., Poisson error `sqrt(N)/V/binwidth`) and the observed SMF (with error bars) on the same log-log plot. Optionally performing a Chi-squared test to quantify the agreement, being mindful of correlated errors and parameter fitting if applicable.

**(Paragraph 6)** **Processing Step 1: Load/Simulate Simulation Catalog:** Load or simulate a galaxy catalog containing stellar masses `M_stellar` (ensure correct units, often includes `h`). Get the simulation `box_volume` (in appropriate units, e.g., (Mpc/h)³).

**(Paragraph 7)** **Processing Step 2: Calculate Simulated SMF:**
    *   Convert stellar masses to consistent units (e.g., M<0xE2><0x82><0x99><0xE1><0xB5><0x98><0xE1><0xB5><0x8A>) and take log₁₀: `log_M_sim = np.log10(stellar_mass_Msun)`.
    *   Define mass bins (e.g., `mass_bin_edges = np.arange(8.0, 12.0, 0.2)` for log₁₀(M/M<0xE2><0x82><0x99><0xE1><0xB5><0x98><0xE1><0xB5><0x8A>)). Calculate bin centers `mass_bin_centers`. Calculate bin width `bin_width_dex = mass_bin_edges[1] - mass_bin_edges[0]`.
    *   Calculate histogram counts `N_sim` in each bin using `np.histogram(log_M_sim, bins=mass_bin_edges)`.
    *   Calculate number density: `phi_sim = N_sim / box_volume / bin_width_dex`.
    *   Estimate Poisson errors on density: `phi_err_sim = np.sqrt(N_sim) / box_volume / bin_width_dex`.

**(Paragraph 8)** **Processing Step 3: Load Observational SMF:** Read the tabulated observational data (`obs_mass_bins`, `phi_obs`, `phi_err`) from a file using `np.loadtxt`, `Table.read`, or `pd.read_csv`. Ensure mass bins and density units are consistent with the simulation data.

**(Paragraph 9)** **Processing Step 4: Plot Comparison:** Create a plot using `plt.errorbar`. Plot `phi_obs` vs `obs_mass_bins` with `yerr=phi_err`. Overplot `phi_sim` vs `sim_mass_bin_centers` with `yerr=phi_err_sim`. Use logarithmic scale for the y-axis (`plt.yscale('log')`). Add labels, title, and legend.

**(Paragraph 10)** **Processing Step 5: Quantitative Comparison (Optional):** If the observational bins match the simulation bins and errors are well-defined, calculate the Chi-squared statistic `chisq = np.sum(((phi_obs - phi_sim_interp) / combined_err)**2)`, where `phi_sim_interp` might require interpolation if bins differ slightly, and `combined_err` accounts for both observational and simulation uncertainties (simulation uncertainty might come from volume/sample variance, often harder to estimate than simple Poisson error). Calculate degrees of freedom `dof` (number of bins minus number of parameters potentially tuned in the simulation to match the SMF). Calculate reduced Chi-squared and/or p-value to assess goodness-of-fit.

**Output, Testing, and Extension:** Output is the plot comparing the simulated and observed galaxy stellar mass functions, and potentially the quantitative Chi-squared goodness-of-fit result. **Testing:** Verify units are consistent between simulation and observation. Check if binning is performed correctly. Ensure error estimation (Poisson or otherwise) is reasonable. Visually assess the agreement/disagreement between the simulation and observations on the plot. **Extensions:** (1) Perform the comparison at different redshifts if multiple simulation snapshots and observational datasets are available. (2) Compare results from simulations run with different subgrid physics parameters (e.g., varying feedback efficiency) to see how they affect the SMF and which provides a better match to observations (parameter study). (3) Calculate and compare other statistical distributions like the galaxy color distribution or size distribution. (4) Use more sophisticated statistical methods (e.g., likelihood analysis incorporating covariance between bins if available for observations) for a more rigorous comparison.

```python
# --- Code Example: Application 36.B ---
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table # Optional for reading obs data
import io # To simulate file loading

print("Comparing Simulated vs Observed Galaxy Stellar Mass Function:")

# Step 1: Simulate/Load Simulation Catalog Data
np.random.seed(10)
n_sim_gals = 50000
# Simulate log(Mass) based on a Schechter function shape (more realistic than uniform)
log_M_star = 10.5 - np.random.exponential(scale=0.8, size=n_sim_gals) 
log_M_star = np.clip(log_M_star, 8.0, 11.8) # Range 10^8 to ~10^11.8 Msun
sim_box_volume = (100.0)**3 # Example: (Mpc/h)^3 volume
print(f"\nGenerated {n_sim_gals} simulated galaxy stellar masses.")
print(f"Simulation Volume: {sim_box_volume} (Mpc/h)^3") # Assuming units

# Step 2: Calculate Simulated SMF
print("\nCalculating simulated SMF...")
mass_bin_edges = np.arange(8.0, 12.1, 0.2) # Log10(M*/Msun) bins
mass_bin_centers = 0.5 * (mass_bin_edges[:-1] + mass_bin_edges[1:])
bin_width_dex = mass_bin_edges[1] - mass_bin_edges[0]

# Histogram counts
N_sim, _ = np.histogram(log_M_star, bins=mass_bin_edges)
# Calculate number density (Phi = N / V / dM)
phi_sim = N_sim / sim_box_volume / bin_width_dex
# Estimate Poisson errors (simple sqrt(N))
phi_err_sim = np.sqrt(N_sim) / sim_box_volume / bin_width_dex
# Avoid log(0) errors - set density to small value if count is 0
phi_sim[N_sim == 0] = 1e-9 
phi_err_sim[N_sim == 0] = 0 
print("Simulated SMF calculated.")

# Step 3: Load Observational SMF (Simulated Data)
# In reality, load from file: e.g., obs_data = Table.read('obs_smf.txt', format='ascii')
obs_smf_text = """
# LogMstar_center  LogPhi   LogPhi_Err_low LogPhi_Err_high
8.1             -1.5     0.10           0.12
8.3             -1.6     0.09           0.10
8.5             -1.7     0.08           0.09
8.7             -1.8     0.07           0.08
8.9             -1.9     0.06           0.07
9.1             -2.0     0.05           0.06
9.3             -2.1     0.05           0.05
9.5             -2.2     0.05           0.05
9.7             -2.3     0.05           0.05
9.9             -2.4     0.05           0.05
10.1            -2.5     0.05           0.05
10.3            -2.6     0.06           0.06
10.5            -2.7     0.07           0.07
10.7            -2.9     0.08           0.08
10.9            -3.2     0.10           0.10
11.1            -3.6     0.12           0.12
11.3            -4.1     0.15           0.15
11.5            -4.7     0.20           0.20
11.7            -5.4     0.30           0.30
11.9            -6.2     0.40           0.40
"""
try:
    obs_data = np.loadtxt(io.StringIO(obs_smf_text))
    obs_mass_bins = obs_data[:, 0]
    # Convert LogPhi to Phi, handle errors (assuming symmetric log error for simplicity)
    phi_obs = 10**obs_data[:, 1]
    # Error propagation: sigma(phi) = phi * ln(10) * sigma(logphi)
    # Using average log error here for simplicity
    avg_log_err = (obs_data[:, 2] + obs_data[:, 3]) / 2.0
    phi_err_obs = phi_obs * np.log(10) * avg_log_err 
    print("\nLoaded and processed observational SMF data (simulated).")
except Exception as e:
     print(f"\nCould not load observational data: {e}")
     obs_mass_bins, phi_obs, phi_err_obs = None, None, None

# Step 4: Plot Comparison
if phi_obs is not None:
    print("Generating comparison plot...")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot observational data
    ax.errorbar(obs_mass_bins, phi_obs, yerr=phi_err_obs, fmt='o', color='black', 
                label='Observed SMF (Simulated Data)', capsize=3, markersize=5)

    # Plot simulation results
    ax.errorbar(mass_bin_centers, phi_sim, yerr=phi_err_sim, fmt='s', color='red', 
                linestyle='--', alpha=0.8, label='Simulation SMF')

    # Customize plot
    ax.set_xlabel("log₁₀(M<0xE2><0x82><0x9B> / M<0xE2><0x82><0x99><0xE1><0xB5><0x98><0xE1><0xB5><0x8A>)")
    ax.set_ylabel("Φ (Mpc⁻³ dex⁻¹)") # Assuming these units
    ax.set_yscale('log') 
    ax.set_ylim(bottom=1e-7, top=1e-1) # Adjust ylim based on data
    ax.set_title("Galaxy Stellar Mass Function Comparison")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    # plt.show()
    print("Plot generated.")
    plt.close(fig)
else:
    print("\nSkipping plot due to error loading observational data.")

# Step 5: Quantitative Comparison (Conceptual Chi2)
# Requires careful handling of matching bins, errors, and correlations - omitted here.
print("\nConceptual Quantitative Comparison would involve:")
print(" - Interpolating sim/obs to common bins if needed.")
print(" - Calculating Chi-squared sum weighted by combined errors.")
print(" - Determining degrees of freedom (bins - fitted params).")
print(" - Calculating p-value for goodness-of-fit.")

print("-" * 20)
```

**Chapter 36 Summary**

This chapter focused on the essential process of connecting astrophysical simulation outputs back to the observable Universe, emphasizing the importance of **validation** and **comparison with observations**. It highlighted that simulations are approximations requiring testing against reality to confirm their validity, constrain free parameters (especially in subgrid models), and build confidence in their predictive power. A crucial step in this comparison is the creation of **mock observations**, which involves forward modeling simulation data (intrinsic physical properties like stellar ages/masses/metallicities or gas density/temperature/velocity) to mimic how a specific telescope or instrument would perceive the simulated system. Techniques for generating **mock images** were discussed, typically involving Stellar Population Synthesis (SPS) models to assign luminosities to star particles, projecting these onto a 2D grid, optionally including complex **dust radiative transfer** for attenuation/emission, convolving with the instrumental Point Spread Function (PSF), adding realistic noise, and converting to observable units.

Similarly, methods for generating **mock spectra** (1D) and **spectral data cubes** (3D) were outlined. This involves using SPS models for stellar contributions, modeling gas emission/absorption lines based on physical conditions (often requiring photoionization or plasma codes), accounting for **kinematic broadening** due to line-of-sight velocities and dispersions, performing line-of-sight integration (potentially including radiative transfer), and convolving with the instrumental Line Spread Function (LSF) and spatial PSF (for cubes) before adding noise. Beyond direct visual comparison of mock data, the chapter emphasized the importance of **statistical comparisons** between simulations and large observational surveys. This involves calculating summary statistics like luminosity functions, stellar mass functions, halo mass functions, correlation functions, power spectra, or distributions of other properties from both the simulation (via mock catalogs) and observational data, and quantitatively comparing them using statistical tests or likelihood methods while accounting for uncertainties on both sides. Finally, the concept of **parameter studies** was discussed, where running suites of simulations varying uncertain input parameters (subgrid or cosmological) and comparing the results to observations allows researchers to constrain these parameters, understand model sensitivities, and break degeneracies by utilizing multiple observational constraints simultaneously, often requiring significant computational resources and potentially emulation techniques.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Turk, M. J., et al. (2011).** yt: A Multi-code Analysis Toolkit... *(See reference in Chapter 35)*. (yt includes some basic capabilities for projections and mock observations).
    *(Relevant as `yt` is often used as the starting point for generating projections needed for mock observations.)*

2.  **Conroy, C. (2013).** Modeling the Panchromatic Spectral Energy Distributions of Galaxies. *Annual Review of Astronomy and Astrophysics*, *51*, 393–455. [https://doi.org/10.1146/annurev-astro-082812-141017](https://doi.org/10.1146/annurev-astro-082812-141017)
    *(A comprehensive review of Stellar Population Synthesis (SPS) modeling, crucial for generating realistic mock stellar luminosities and spectra discussed in Sec 36.3, 36.4.)* (See also `python-fsps` library: [https://dfm.io/python-fsps/current/](https://dfm.io/python-fsps/current/))

3.  **Camps, P., & Baes, M. (2015).** SKIRT: A state-of-the-art Monte Carlo radiative transfer code for simulations of dusty astrophysical systems. *Astronomy and Computing*, *12*, 35-47. [https://doi.org/10.1016/j.ascom.2015.06.001](https://doi.org/10.1016/j.ascom.2015.06.001) (See also Powderday: Narayanan, D., et al. (2021) ApJ, 912(1), 70. [https://doi.org/10.3847/1538-4357/abed7c](https://doi.org/10.3847/1538-4357/abed7c))
    *(Examples of sophisticated radiative transfer codes (SKIRT, Powderday) often used for post-processing simulations to include dust effects in mock images and spectra, relevant to Sec 36.3, 36.4.)*

4.  **Astropy Collaboration. (n.d.).** *Astropy Documentation: Convolution and Filtering (astropy.convolution)*. Astropy Documentation. Retrieved January 16, 2024, from [https://docs.astropy.org/en/stable/convolution/](https://docs.astropy.org/en/stable/convolution/)
    *(Documentation for Astropy's convolution functions (`convolve_fft`) used for applying PSFs or LSFs in mock observations, relevant to Sec 36.3, 36.4.)*

5.  **Wechsler, R. H., & Tinker, J. L. (2018).** The Connection Between Galaxies and Their Dark Matter Halos. *Annual Review of Astronomy and Astrophysics*, *56*, 435–487. [https://doi.org/10.1146/annurev-astro-081817-051756](https://doi.org/10.1146/annurev-astro-081817-051756)
    *(A review discussing the comparison between theoretical predictions (often from simulations or models built upon them, like halo occupation) and observational statistics like stellar mass functions and galaxy clustering, relevant context for Sec 36.5.)*
