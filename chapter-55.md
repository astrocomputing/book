**Chapter 55: Computational Techniques in X-ray Astronomy**

This chapter explores the computational methods specific to **X-ray astronomy**, a field that observes the Universe's hottest and most energetic phenomena, such as accretion onto black holes and neutron stars, supernova remnants, hot gas in galaxy clusters, and stellar coronae, by detecting photons with energies typically ranging from ~0.1 to 100 keV. We discuss X-ray detection principles, focusing on **photon-counting detectors** (like CCDs or microcalorimeters) used in focusing telescopes (Chandra, XMM-Newton, eROSITA, future missions like Athena) and detectors used in coded mask or collimated instruments. Data processing for X-ray **event lists** (typically FITS files recording time, position, and energy for each detected photon) is covered, including filtering (time, energy, grade), spatial region selection, and complex **background subtraction** techniques. We detail the process of **X-ray spectral analysis**: grouping event energies into spectral channels (PHA files), associating instrument **response files** (RMF - redistribution matrix file, ARF - ancillary response file), defining astrophysical source models (e.g., thermal plasma, power law, absorption models) using frameworks like **XSPEC** or **Sherpa**, and fitting these models to the data using specialized statistics suitable for Poisson counts (like C-stat or W-stat) to constrain physical parameters (temperature, abundance, column density, spectral indices). Basic **X-ray image analysis** (creating images from event lists, source detection, simple morphology) and **timing analysis** (light curves, period searches) are also introduced. Key Python tools include `Astropy`, `regions`, interfaces to HEASoft packages like `XSPEC` (via `PyXspec`) and `Sherpa` (standalone or via CIAO), and potentially `stingray` for timing.

**55.1 X-ray Detection: Focusing Telescopes, Detectors (CCDs, Calorimeters)**

Detecting X-rays from cosmic sources presents unique challenges compared to optical astronomy, primarily because X-rays are readily absorbed by Earth's atmosphere, necessitating space-based observatories, and because their high energy makes them difficult to reflect or refract using traditional lenses or mirrors. Modern X-ray telescopes (like Chandra, XMM-Newton, NuSTAR, eROSITA) typically employ **grazing incidence optics**. These use nested shells of precisely shaped mirrors (often parabolic and hyperbolic, forming Wolter telescopes) where X-rays reflect at very shallow angles (grazing incidence), allowing them to be focused onto a detector. This technique provides true imaging capabilities, enabling high angular resolution (sub-arcsecond for Chandra). Other instruments, particularly for higher energies or wide fields of view, might use non-focusing techniques like coded masks or collimators.

At the focal plane, specialized detectors are needed to register individual X-ray photons and measure their properties. The workhorse detector for many missions has been the **X-ray Charge-Coupled Device (CCD)**. When an X-ray photon hits the silicon CCD, it generates a cloud of electron-hole pairs, with the number of electrons being roughly proportional to the photon's energy (E ≈ 3.65 eV per pair in silicon). By reading out the charge accumulated in each pixel, CCDs can measure the 2D position (x, y) of the photon interaction and estimate its energy (E), albeit with limited spectral resolution (typically ΔE/E ~ few percent at best, degrading at lower energies). CCDs are photon-counting devices, recording individual events rather than integrated flux like optical CCDs often do in brighter regimes.

For higher spectral resolution, **X-ray microcalorimeters** are emerging as powerful detectors (used on missions like Hitomi, planned for Athena). These detectors measure the tiny temperature rise caused by the absorption of a single X-ray photon in a small absorber material cooled to very low temperatures (~milliKelvin). The temperature change is precisely proportional to the photon's energy, allowing for much better energy resolution (ΔE ~ few eV) compared to CCDs, enabling detailed X-ray spectroscopy. They also record position and arrival time. Other detector types include micro-channel plates (often used for timing or UV/soft X-rays) and solid-state detectors like Cadmium Zinc Telluride (CZT) used in hard X-ray instruments (like NuSTAR).

Regardless of the specific detector type, the fundamental output of most modern X-ray telescopes used for imaging and spectroscopy is an **event list**. Each detected photon interaction is recorded as an "event" with associated properties, minimally including its arrival time, detector position (e.g., detector coordinates DETX, DETY or chip coordinates X, Y), and measured energy (often represented by pulse height amplitude PHA or PI - pulse invariant energy). Additional information might include event grades (classifying the shape of the electron cloud in CCDs, related to event quality), chip ID, or status flags.

This event-based nature dictates the subsequent data processing steps. Unlike optical images where pixel values represent integrated flux over an exposure, X-ray analysis starts with lists of discrete photon events, each carrying spatial, temporal, and spectral information. Raw event lists need to be calibrated (converting detector coordinates to sky coordinates using aspect solutions, converting PHA/PI to physical energy using gain maps) and filtered before scientific analysis like imaging, spectroscopy, or timing can be performed. Understanding the principles of X-ray detection and the event-list nature of the raw data is the first step towards appreciating the computational techniques used in X-ray data analysis.

**55.2 X-ray Data: Event Lists (FITS), Good Time Intervals (GTIs)**

The standard format for storing calibrated X-ray data, particularly from focusing telescopes like Chandra and XMM-Newton, is the **event list file**, almost universally stored as a **FITS binary table** (Sec 1.6). This table contains one row for each detected and processed X-ray event (photon). Each row acts like a database record for a single photon interaction.

Common columns found in an X-ray event list FITS file include:
*   `TIME`: The arrival time of the event, usually measured in seconds relative to a reference epoch (like Mission Elapsed Time or MET, often specified in the header). High time resolution allows for timing analysis.
*   `X`, `Y`: Event coordinates on the detector plane (detector pixels or physical coordinates). Sometimes sky coordinates (`RA`, `DEC`) calculated using the aspect solution are also included, or derived later using WCS information stored in the FITS header.
*   `PHA` (Pulse Height Amplitude) or `PI` (Pulse Invariant): Integer channels representing the measured energy deposited by the photon. PHA is often the raw pulse height, while PI is typically calibrated to be roughly proportional to energy (e.g., PI channel = Energy / energy_per_channel). The mapping between PHA/PI and physical energy is provided by the instrument response files (RMF, Sec 55.4).
*   `CCD_ID` or `DET_ID`: Identifier for the specific detector chip or segment where the event occurred (for instruments with multiple detectors).
*   `GRADE` (for CCDs): An integer classifying the event morphology (shape of the charge cloud), used to filter out potentially corrupted events (like cosmic rays or background noise) and select events for which the energy calibration is most accurate.
*   Other flags indicating event status or potential issues.

The FITS header of the event list file contains crucial metadata, including World Coordinate System (WCS) information linking detector (X, Y) or sky (RA, Dec) coordinates to pixel positions if the events are binned into an image, observation parameters (telescope, instrument, filter, pointing direction, observation ID, start/stop times, exposure duration), coordinate system information (epoch, equinox), and potentially information about calibration files used during processing.

Alongside the event list(s), X-ray data products usually include **Good Time Interval (GTI) files**. These are also typically FITS binary tables specifying time intervals during the observation when the data quality was good and the instrument was operating nominally. Intervals might be excluded due to high background radiation (e.g., during passages through the Earth's radiation belts), telemetry dropouts, detector issues, or periods of unstable pointing. Scientific analysis should generally only use events that fall within the time ranges specified by the relevant GTI file(s) to ensure data quality and correct exposure time calculation. Applying GTI filtering is a standard initial step in data reduction.

Working with X-ray data computationally thus primarily involves reading and manipulating these FITS event list tables and GTI tables. Python libraries like `astropy.io.fits` (Sec 1.6) or `astropy.table.Table` (Sec 2.3) are used to read the event data into memory (or process row-by-row for very large files). Operations then involve **filtering** the table based on criteria applied to different columns (TIME, X, Y, PHA/PI, GRADE) and the GTI information.

For example, selecting events for spectral analysis from a specific source region would involve filtering the table based on:
1.  **Spatial criteria:** Selecting rows where the (X, Y) or (RA, Dec) coordinates fall within a defined source region (e.g., a circle).
2.  **Temporal criteria:** Keeping only rows where the `TIME` falls within the intervals defined in the GTI file(s).
3.  **Energy criteria:** Selecting rows where the `PHA` or `PI` channel falls within the desired energy band for analysis.
4.  **Quality criteria:** Selecting rows with specific `GRADE` values deemed acceptable.

Similarly, creating an image involves selecting events based on time, energy, and quality criteria, and then binning the selected (X, Y) coordinates into a 2D histogram. Creating a light curve involves selecting events based on spatial, energy, and quality criteria, and then binning the `TIME` column into time bins. Understanding the structure of event lists and GTI files, and how to perform efficient filtering and binning operations (often using NumPy boolean masking and histogramming functions on Astropy Table columns), is fundamental to almost all X-ray data analysis workflows.

**55.3 Data Reduction: Filtering, Region Selection, Background Handling**

Raw X-ray event lists often contain instrumental artifacts, background events, and data taken during periods of poor instrument performance. **Data reduction** is the essential process of cleaning and filtering the event list to produce scientifically usable data products (filtered event lists, images, spectra, light curves), correctly accounting for exposure time and background. This typically involves several steps, often performed using specialized software packages provided by the instrument teams (like CIAO for Chandra, SAS for XMM-Newton, HEASoft for many missions) which often have Python interfaces or can be called from Python scripts.

**1. Time Filtering (Applying GTIs):** The first step is usually to filter the event list based on the **Good Time Intervals (GTI)** files (Sec 55.2). This removes events that occurred during periods known to be problematic (e.g., high background, bad pointing). This filtering ensures that calculated fluxes or rates are based on the correct effective exposure time. Libraries like `astropy.table` or specialized functions in instrument software allow easy filtering based on GTI start/stop times.

**2. Energy Filtering:** Depending on the scientific goal, events might be filtered based on their energy (PHA or PI channel). For example, creating an image in a specific band (e.g., soft X-rays 0.5-2 keV) involves selecting only events within that PI range before binning. For spectral analysis, a broad range might be kept initially, with specific band selections made later during fitting. Filtering might also remove channels known to have poor calibration.

**3. Grade/Quality Filtering:** For CCD detectors, event **grades** classify the pattern of charge deposition across pixels. Specific grade selections (e.g., selecting only isolated single-pixel events, or specific combinations like ASCA grades 0,2,3,4,6) are often applied to remove cosmic ray events or background interactions and select events for which the energy calibration (mapping PHA/PI to energy) is most accurate, as defined by the instrument calibration team.

**4. Spatial Region Selection:** Analysis often focuses on specific regions of the sky or detector. **Region files**, commonly created using interactive tools like `ds9` and saved in standard formats (e.g., DS9 region format, FITS regions), define spatial areas of interest (circles, ellipses, boxes, polygons). Software tools (e.g., functions in CIAO/SAS, or Python libraries like `regions` - an Astropy affiliated package) are used to filter the event list, keeping only events whose coordinates fall within the specified source region(s) or, conversely, within background region(s). Creating appropriate source and background regions is crucial for accurate analysis.

```python
# --- Code Example 1: Conceptual Filtering using Astropy Table ---
# Assumes event data is loaded into an Astropy Table 'events_table'
# with columns 'TIME', 'X', 'Y', 'PI', 'GRADE'.
# Assumes GTI data is loaded into 'gti_table' with 'START', 'STOP' columns.
# Assumes regions are defined using the 'regions' package.

import numpy as np
from astropy.table import Table, vstack 
from astropy.time import Time 
# Assume regions package is installed and imported
# from regions import CircleSkyRegion, PixCoord
# from astropy.coordinates import SkyCoord
# from astropy.wcs import WCS

print("Conceptual Filtering of X-ray Event List (Astropy Table):")

# --- Simulate Data ---
# Create a dummy event table
n_events = 10000
events_table = Table({
    'TIME': np.random.uniform(1000, 5000, n_events), # Seconds
    'X': np.random.uniform(0, 1024, n_events), # Pixels
    'Y': np.random.uniform(0, 1024, n_events),
    'PI': np.random.randint(50, 500, n_events), # Energy channels
    'GRADE': np.random.choice([0, 2, 3, 4, 6, 7], n_events) # Example grades
})
events_table['TIME'].unit = 's'
print(f"\nCreated dummy event table with {len(events_table)} events.")

# Create dummy GTI table
gti_table = Table({'START': [1200.0, 3500.0], 'STOP': [3000.0, 4800.0]})
gti_table['START'].unit = 's'; gti_table['STOP'].unit = 's'
print(f"Created dummy GTI table with {len(gti_table)} intervals.")

# --- Step 1: Time Filtering (Apply GTIs) ---
print("\nApplying GTI filter...")
gti_masks = []
for gti_interval in gti_table:
    mask = (events_table['TIME'] >= gti_interval['START']) & \
           (events_table['TIME'] < gti_interval['STOP'])
    gti_masks.append(mask)
# Combine masks: event is good if it falls in *any* GTI
good_time_mask = np.logical_or.reduce(gti_masks) 
events_filtered_time = events_table[good_time_mask]
print(f"  Events remaining after GTI filter: {len(events_filtered_time)}")

# --- Step 2: Energy Filtering ---
print("\nApplying Energy filter (e.g., PI between 100 and 400)...")
energy_mask = (events_filtered_time['PI'] >= 100) & (events_filtered_time['PI'] <= 400)
events_filtered_energy = events_filtered_time[energy_mask]
print(f"  Events remaining after energy filter: {len(events_filtered_energy)}")

# --- Step 3: Grade Filtering ---
print("\nApplying Grade filter (e.g., grades 0, 2, 3, 4, 6)...")
good_grades = [0, 2, 3, 4, 6]
grade_mask = np.isin(events_filtered_energy['GRADE'], good_grades)
events_filtered_grade = events_filtered_energy[grade_mask]
print(f"  Events remaining after grade filter: {len(events_filtered_grade)}")

# --- Step 4: Spatial Region Filtering (Conceptual) ---
# Requires WCS or pixel regions defined using the 'regions' package
print("\nApplying Spatial filter (Conceptual - requires 'regions' package):")
# Assume 'source_region' is a regions.PixelRegion object (e.g., CirclePixelRegion)
# pixel_coords = PixCoord(x=events_filtered_grade['X'], y=events_filtered_grade['Y'])
# spatial_mask = source_region.contains(pixel_coords)
# events_final = events_filtered_grade[spatial_mask]
# print(f"  Events remaining after spatial filter: {len(events_final)}")
print("  (Skipping actual spatial filtering implementation)")
events_final = events_filtered_grade # Use grade-filtered for now

print("\nFinal filtered event list ready for analysis.")
print("-" * 20)

# Explanation:
# 1. Simulates an event list as an Astropy Table and a GTI table.
# 2. GTI Filtering: Creates boolean masks for each GTI interval based on event 'TIME' 
#    and combines them using logical OR to select events within any good interval.
# 3. Energy Filtering: Applies a simple range selection on the 'PI' column.
# 4. Grade Filtering: Uses `np.isin` to select events with specific allowed 'GRADE' values.
# 5. Spatial Filtering (Conceptual): Outlines how the `regions` package would be used 
#    with pixel coordinates and a region object's `.contains()` method to select events 
#    within a specific area on the detector/sky.
# This demonstrates the typical sequential application of filters to an event list table.
```

**5. Background Handling:** Astrophysical sources are always observed against a background, which can include instrumental noise (e.g., detector read noise, thermal noise), particle background (cosmic rays interacting with the detector or spacecraft), and astrophysical background (e.g., unresolved faint sources, diffuse emission like the Galactic X-ray background or cluster ICM emission). Accurately subtracting the background contribution is crucial, especially for faint sources or spectral analysis. Common strategies include:
*   **Source-free Region Subtraction:** Extracting events (or spectra/light curves) from a nearby region on the same detector assumed to be free of sources but representative of the background affecting the source region. The background counts/spectrum/light curve are then subtracted from the source region data (often after scaling by relative area). Choosing an appropriate background region is critical and can be difficult in crowded fields or near detector edges.
*   **Modeling the Background:** For spectral analysis, theoretical or empirical models for the instrumental and astrophysical background components can be included directly in the spectral fitting process (Sec 55.5) alongside the source model, allowing the background normalization to be fitted simultaneously.
*   **Blank Sky Observations:** Using separate, long observations of "blank" sky fields taken with the same instrument configuration to characterize the background, although background levels can be time-variable.
Background handling is often one of the most challenging aspects of X-ray data reduction and analysis, and requires careful consideration of the instrument, observation specifics, and potential systematic effects. Specialized tools within packages like CIAO and SAS provide sophisticated methods for background modeling and subtraction.

These data reduction steps – applying time, energy, grade, and spatial filters, and handling background – transform the raw event list into a cleaned dataset suitable for scientific analysis like imaging, spectroscopy, or timing studies, ensuring results are based on reliable data and correct exposure calculations.

**55.4 Spectral Analysis I: PHA files, RMFs, ARFs**

X-ray spectral analysis aims to understand the physical properties of a source by modeling its emitted energy spectrum. Since X-ray detectors (like CCDs) have imperfect energy resolution and detection efficiency, directly analyzing the raw distribution of detected photon energies (PHA or PI channels) is insufficient. Instead, **forward fitting** is employed: a theoretical source spectrum model is folded through the instrument's response, and the result is compared to the observed count spectrum. This requires three key components: the observed spectrum, and two calibration files describing the instrument response: the Redistribution Matrix File (RMF) and the Ancillary Response File (ARF).

**1. PHA File (Observed Spectrum):** After filtering the event list for the source region and desired time/grade/energy selections (Sec 55.3), the selected events are binned into energy channels (PHA or PI) to create the observed count spectrum. This histogram (Counts vs. Channel) is typically stored in a standard FITS binary table format called a **PHA file**. The PHA file contains columns like:
*   `CHANNEL`: The channel number (integer).
*   `COUNTS`: The number of source events detected in that channel.
*   Often includes `STAT_ERR` (statistical error, usually sqrt(COUNTS) assuming Poisson).
*   May include grouping information (specifying how channels are grouped for fitting) and quality flags.
Crucially, the PHA file header *must* link to the appropriate RMF and ARF files via keywords like `RESPFILE` and `ANCRFILE`. A corresponding **background PHA file**, extracted from a background region and scaled appropriately, is also usually required.

**2. RMF (Redistribution Matrix File):** This file describes the **energy dispersion** of the detector – how photons of a specific *true* energy are detected and registered across different *output* PHA/PI channels. Due to inherent detector limitations (e.g., charge cloud statistics in CCDs), a monoenergetic X-ray beam will produce a distribution (often roughly Gaussian) of measured channel values. The RMF is essentially a large matrix where RMF[i, j] gives the probability that an incoming photon with true energy corresponding to input channel `j` will be detected and assigned to output channel `i`. It accounts for the detector's energy resolution (width of the redistribution function) and any non-linearities in the energy scale. RMFs are complex calibration products, usually generated by instrument teams using detailed simulations and ground/in-flight calibration data, and are specific to the detector, operating mode, and often detector location or time period. They are typically stored as FITS binary tables with specific extensions (`MATRIX`, `EBOUNDS`).

**3. ARF (Ancillary Response File):** This file describes the **effective area** of the telescope and detector combination as a function of *true* photon energy. The effective area represents the energy-dependent sensitivity, combining factors like:
*   The geometric collecting area of the telescope mirrors.
*   The energy-dependent reflectivity of the grazing incidence mirrors.
*   The transmission efficiency of any filters used.
*   The quantum efficiency (QE) of the detector itself (probability of detecting a photon of a given energy that hits it).
*   Effects of vignetting (reduction of effective area off-axis).
*   Potentially other effects like extraction region efficiency or pile-up corrections.
The ARF essentially provides the conversion factor between the incident photon flux (photons/cm²/s/keV) from the source and the expected *count rate* (counts/s/keV) that would be detected by the instrument *before* energy dispersion (RMF). ARFs are also complex calibration files, specific to the source location on the detector (due to vignetting and QE variations) and instrument configuration, generated using calibration tools provided by the instrument teams. They are typically stored as FITS binary tables with columns like `ENERG_LO`, `ENERG_HI`, `SPECRESP` (effective area in cm²).

**Forward Fitting Process:** The spectral fitting process then works conceptually as follows:
1.  Define a theoretical source model S(E) (e.g., power law, thermal plasma) giving flux in photons/cm²/s/keV as a function of true energy E.
2.  Multiply the source model by the effective area A(E) from the ARF: Expected Rate(E) = S(E) * A(E) [counts/s/keV].
3.  Fold this expected rate through the energy dispersion described by the RMF matrix: Expected Counts(i) = Σ<0xE2><0x82><0x97> RMF[i, j] * Expected Rate(E<0xE2><0x82><0x97>) * ΔE<0xE2><0x82><0x97> * ExposureTime, where `i` represents the output detector channel and `j` represents the input true energy bin. This predicts the number of counts expected in each detector channel `i`.
4.  Compare these predicted counts Expected Counts(i) with the actually observed counts Counts(i) from the PHA file (potentially after background subtraction or including a background model) using a suitable **fit statistic** (Sec 55.5).
5.  Adjust the parameters of the theoretical source model S(E) iteratively until the fit statistic is minimized, finding the best-fit model parameters and their uncertainties.

This forward folding process (`Model * ARF * RMF`) accounts for the instrument's complex response, allowing inference of the true incident source spectrum S(E) from the observed count spectrum in detector channels.

Generating the PHA file from an event list requires binning events based on their PI/PHA channel. Generating the correct ARF and RMF files requires running specialized tools provided by the instrument teams (e.g., `specextract` in CIAO, `arfgen`/`rmfgen` in XMM-SAS or HEASoft) which take into account the source position, extraction region, time period, and instrument calibration database. Python tools like `PyXspec` or `Sherpa` handle the loading of these files and perform the forward folding and fitting internally. Understanding the role of the PHA, RMF, and ARF is crucial for performing and interpreting any X-ray spectral analysis.

**(No direct code example here, as creating/using PHA/RMF/ARF primarily involves specialized external tools or libraries covered in the next section.)**

**55.5 Spectral Analysis II: Modeling and Fitting (XSPEC, Sherpa, Fit Statistics)**

Once the necessary data products are prepared – the source count spectrum (PHA file), background spectrum (optional), and instrument response files (RMF and ARF) – the core of X-ray spectral analysis involves **defining astrophysical models** and **fitting** them to the observed data. This process allows astronomers to constrain the physical parameters of the X-ray emitting source. Dedicated software packages like **XSPEC** and **Sherpa**, accessible via Python interfaces, provide the standard environments for this task.

**Defining Models:** XSPEC and Sherpa offer extensive libraries of pre-defined spectral model components representing common astrophysical emission and absorption processes:
*   **Continuum Models:** `powerlaw` (synchrotron, inverse Compton), `bbody` (blackbody), `apec` or `mekal` (optically thin thermal plasma emission), `bremss` (thermal bremsstrahlung), `cutoffpl` (power law with high-energy cutoff).
*   **Absorption Models:** `phabs` or `tbabs` (photoelectric absorption by neutral interstellar gas, parameterized by column density N<0xE1><0xB5><0x8F>), `wabs` (similar, older model).
*   **Line Models:** `gaussian`, `lorentzian` (for unresolved or resolved emission/absorption lines).
*   **Reflection Models:** For modeling X-rays reflected off accretion disks.
*   **Scattering Models:** E.g., Compton scattering.
Users combine these components multiplicatively or additively to construct a composite model representing the hypothesized physical scenario. For example, an absorbed power law is often modeled as `phabs * powerlaw`. A cooling flow cluster might be `phabs * (apec + powerlaw)`. Each component has associated **parameters** (e.g., N<0xE1><0xB5><0x8F> for `phabs`; photon index Γ and normalization for `powerlaw`; temperature kT, abundance Z, normalization for `apec`) whose values are initially guessed or set and then adjusted during the fit. Users can also define their own custom models if needed.

**Fitting Process:** The goal of fitting is to find the model parameter values that best describe the observed PHA count spectrum, given the instrument response (RMF/ARF). This involves:
1.  **Loading Data and Responses:** Load the PHA, RMF, ARF (and background PHA if used) into the fitting environment (`pyxspec.Spectrum`, `sherpa.load_pha`). Ignore bad channels or group channels for sufficient counts if needed.
2.  **Defining the Model:** Specify the model expression using the package's syntax (e.g., `pyxspec.Model("phabs*powerlaw")`, `sherpa.set_source(sherpa.xsphabs.abs1 * sherpa.xspowerlaw.pl1)`).
3.  **Choosing Fit Statistic:** Select a statistic appropriate for comparing the model prediction (folded through RMF/ARF) with the observed counts. Since X-ray data often involves low counts per channel, assuming Gaussian statistics and using standard Chi-squared (χ²) is often **incorrect**. Preferred statistics for Poisson data include:
    *   **C-Statistic (`cstat` in XSPEC, `stat=cash` or `wstat` in Sherpa):** Based on the Cash likelihood statistic, optimal for Poisson data even at low counts. Does not provide an absolute goodness-of-fit measure directly.
    *   **W-Statistic (`wstat` in XSPEC):** A modified C-statistic for background-subtracted data.
    *   **Chi-squared variants for grouped data (`chi`):** Standard χ² can be used *only* if data is grouped such that each bin contains sufficient counts (e.g., >20) for Gaussian approximation to hold. Various weighting schemes exist (e.g., `chi / data`, `chi / model`).
4.  **Performing the Fit:** Run the fitting algorithm (typically based on iterative minimization, like Levenberg-Marquardt) to find the parameter values that minimize the chosen fit statistic (`pyxspec.Fit.perform()`, `sherpa.fit()`).
5.  **Assessing Goodness-of-Fit:** Evaluate how well the best-fit model describes the data. For χ², the reduced χ² (χ²/degrees_of_freedom) should be near 1. For C-stat/W-stat, absolute goodness-of-fit is harder; often requires simulations (`pyxspec.Fit.goodness`) or examination of residuals (data - model / error). Plotting data and folded model together with residuals is essential.
6.  **Estimating Parameter Uncertainties:** Determine the statistical uncertainties (e.g., 68% or 90% confidence intervals) on the best-fit parameters. Common methods include:
    *   Using the covariance matrix calculated during the fit (approximate, assumes parabolic likelihood near minimum). (`pyxspec.Fit.error`, `sherpa.conf`).
    *   Running MCMC simulations to sample the posterior probability distribution (requires Bayesian setup with priors, Sec 16/17). (`pyxspec.Bayes`, `sherpa.get_draws`).

**Python Interfaces (`PyXspec`, `Sherpa`):**
*   **`PyXspec` (`pip install pyxspec` - requires HEASoft):** Provides a Python scripting interface to nearly all XSPEC functionalities. You interact by sending XSPEC commands as strings or using dedicated Python methods on objects representing spectra, models, parameters, and fit settings. Allows automating complex spectral analysis workflows within Python.
*   **`Sherpa` (Standalone or via CIAO):** A more object-oriented Python modeling and fitting environment developed at the Chandra X-ray Center. Models, data, statistics, and fit methods are represented as Python objects. Offers flexibility in defining custom models and statistics and integrates well with other Python libraries.

```python
# --- Code Example 1: Conceptual Spectral Fit with PyXspec ---
# Note: Requires HEASoft/XSPEC installation and pyxspec python package.
# Assumes PHA, RMF, ARF files exist (e.g., source.pha, source.rmf, source.arf)

print("Conceptual X-ray Spectral Fitting with PyXspec:")

try:
    import xspec # Import the pyxspec module
    # Suppress XSPEC chatter (optional)
    # xspec.Xset.chatter = 10 
    # xspec.Xset.logChatter = 10
    print("\nPyXspec imported.")
    
    # --- Step 1: Load Data & Responses ---
    print("Loading data (source.pha)...")
    # Need to ensure files exist or handle errors
    # s1 = xspec.Spectrum("source.pha") 
    print("  (Conceptual: Spectrum('source.pha') loaded as s1)")
    # Associate responses (if not in PHA header)
    # s1.response = "source.rmf"
    # s1.response.arf = "source.arf"
    # Load background (optional)
    # xspec.AllData.background = "background.pha"
    # Ignore bad channels / Notice energy range
    # xspec.AllData.ignore("0.0-0.3, 8.0-**") # Ignore below 0.3 keV, above 8.0 keV
    print("  (Conceptual: Responses associated, channels ignored)")
    
    # --- Step 2: Define Model ---
    print("\nDefining model (e.g., phabs * powerlaw)...")
    # Model components are added by name
    # m1 = xspec.Model("phabs * powerlaw")
    print("  (Conceptual: Model('phabs*powerlaw') created as m1)")
    # Access parameters: m1.phabs.nH, m1.powerlaw.PhoIndex, m1.powerlaw.norm
    # Set initial parameter values if needed
    # m1.phabs.nH.values = 0.1 # Example N_H in 10^22 cm^-2
    # m1.powerlaw.PhoIndex.values = 1.8
    print("  (Conceptual: Initial parameter values set)")

    # --- Step 3: Choose Fit Statistic ---
    # xspec.Fit.statMethod = "cstat" # Use C-statistic for Poisson data
    print("\nFit statistic set to cstat.")

    # --- Step 4: Perform Fit ---
    print("Performing fit...")
    # xspec.Fit.perform()
    print("  (Conceptual: Fit.perform() executed)")

    # --- Step 5: Assess Goodness-of-Fit ---
    print("\nAssessing fit results...")
    # Check fit statistic value and degrees of freedom
    # fit_stat = xspec.Fit.statistic
    # dof = xspec.Fit.dof
    # print(f"  Fit Statistic ({xspec.Fit.statMethod}): {fit_stat:.2f} for {dof} dof")
    # Plot data and model
    # xspec.Plot.device = "/xs" # Open plot window
    # xspec.Plot("data", "model", "resid") # Plot data, folded model, residuals
    print("  (Conceptual: Fit statistic checked, plot 'data model resid' generated)")
    
    # --- Step 6: Estimate Parameter Uncertainties ---
    print("\nEstimating parameter errors (e.g., 90% confidence)...")
    # xspec.Fit.error("1-3") # Calculate errors for parameters 1, 2, 3
    print("  (Conceptual: Fit.error() executed for model parameters)")
    # Display results with errors
    # xspec.Fit.show() # Shows parameters with best-fit values and errors
    print("  (Conceptual: Fit.show() displays final parameters and uncertainties)")

    # Clean up (clear data/models)
    # xspec.AllData.clear()
    # xspec.AllModels.clear()

except ImportError:
     print("\nError: pyxspec package not found or XSPEC/HEASoft not set up correctly.")
except Exception as e:
     print(f"\nAn error occurred: {e}")
     
print("-" * 20)

# Explanation: This code outlines a typical spectral fitting session using PyXspec.
# 1. Loads the spectrum (`Spectrum`), background (optional), and associates response 
#    files (RMF/ARF). Ignores energy channels outside the calibrated range.
# 2. Defines an astrophysical model (`Model`) using XSPEC's string syntax (e.g., 
#    "phabs*powerlaw"). Initial guesses for parameters might be set.
# 3. Sets the fit statistic (`Fit.statMethod`) appropriate for the data (e.g., "cstat").
# 4. Performs the fit using `Fit.perform()` which minimizes the statistic.
# 5. Assesses the fit quality by checking the final statistic value and visually 
#    inspecting plots of the data, folded model, and residuals generated using `Plot`.
# 6. Calculates parameter uncertainties using `Fit.error()`. `Fit.show()` displays 
#    the final results.
# This demonstrates the standard forward-fitting workflow used in X-ray spectral analysis.
```

X-ray spectral fitting is a powerful technique for diagnosing physical conditions in high-energy sources. Python interfaces like `PyXspec` and `Sherpa` provide flexible and scriptable environments for performing these complex analyses, allowing automation, integration with other Python tools, and sophisticated statistical modeling (including Bayesian analysis via MCMC if desired). Careful attention to data preparation (filtering, background), response files, model selection, choice of fit statistic, and uncertainty estimation is crucial for obtaining reliable scientific results.

**55.6 Image and Timing Analysis Basics**

While spectroscopy provides detailed physical diagnostics, analyzing the **spatial distribution** (imaging) and **temporal variability** (timing) of X-ray emission yields complementary insights into source morphology, structure, motion, and variability mechanisms.

**X-ray Imaging:** Although individual X-ray photons are detected, binning the spatial coordinates (X, Y or RA, Dec) of filtered events (Sec 55.3) into a 2D histogram creates an **X-ray image**. This image represents the count rate or flux distribution on the sky in a chosen energy band.
*   **Creating Images:** Use `numpy.histogram2d` or tools within `astropy` or specific mission software (CIAO `dmcopy`, SAS `evselect`+`evselect`) to bin events from the filtered event list into an image grid. Proper handling of WCS information (Sec 5.6, 5.7) is essential for creating images in sky coordinates (RA, Dec).
*   **Exposure Correction:** Raw count images need to be divided by an **exposure map**, which accounts for variations in effective exposure time across the detector due to dithering, chip gaps, bad pixels, and vignetting, to convert counts to count rate or flux. Exposure maps are generated by instrument calibration tools.
*   **Smoothing:** X-ray images are often photon-limited (low counts per pixel). Smoothing the image using filters (e.g., Gaussian (`scipy.ndimage.gaussian_filter`), or adaptive smoothing algorithms like `csmooth` (CIAO) or `asmooth`) can enhance faint, extended features, but degrades angular resolution.
*   **Source Detection:** Algorithms similar to optical source detection can be used, but adapted for Poisson statistics. Common methods include sliding cell algorithms (comparing counts in a cell to surrounding background), wavelet decomposition (`wavdetect` in CIAO), or model fitting (e.g., fitting PSFs). Libraries like `photutils` (Sec 54.5, App 53.B) can be adapted, or specialized X-ray tools used.
*   **Morphological Analysis:** Measuring sizes, shapes, orientations, surface brightness profiles of extended sources (like SNRs, galaxy clusters) or analyzing the structure of jets. The `regions` package (App 55.B concept) is crucial for defining apertures.
*   **Multi-wavelength Comparison:** Overlaying X-ray images (often represented by contours) onto optical or radio images provides crucial context about the origin of the X-ray emission. Requires accurate astrometry (WCS alignment).

**X-ray Timing Analysis:** The precise arrival time recorded for each photon event enables powerful timing analysis to study variability on timescales from milliseconds to years.
*   **Light Curves:** Binning filtered events (from a specific source region and energy band) into time bins creates a **light curve** (count rate vs. time). `astropy.timeseries.BoxyLeastSquares` or `stingray` library tools can be used for creating binned or event-based light curves. Care must be taken with background subtraction and correcting for instrumental effects (dead time, dithering affecting source position on detector).
*   **Periodicity Searches:** Searching for periodic signals (e.g., from rotating neutron stars/pulsars, orbiting binaries) in the light curve or directly in the event arrival times using techniques like:
    *   **Fourier Transform (FFT):** For searching for strictly periodic signals in binned light curves.
    *   **Lomb-Scargle Periodogram:** Handles unevenly sampled data (common due to GTIs), suitable for binned light curves or event times. (`astropy.timeseries.LombScargle`).
    *   **Epoch Folding:** Folding the arrival times modulo a trial period and searching for periods that produce a significantly non-uniform folded profile (often using χ² or Z<0xE1><0xB5><0x8A>² statistics). Specialized pulsar timing software (PRESTO, Tempo2) or libraries like `stingray` provide tools.
*   **Aperiodic Variability (Timing Noise):** Analyzing the shape of the power spectral density (PSD) derived from the light curve reveals information about characteristic timescales and the nature of stochastic variability processes (e.g., in accretion flows around black holes). Libraries like `stingray` are specifically designed for PSD analysis and modeling.
*   **Time Lags and Coherence:** Comparing variability patterns in different energy bands can reveal time lags (e.g., hard X-rays lagging soft X-rays due to propagation or Comptonization effects) or coherence, providing insights into the geometry and physics of the emission region. `stingray` includes tools for cross-spectral analysis.

```python
# --- Code Example 1: Creating X-ray Image from Events ---
# Note: Uses numpy, astropy, matplotlib. Assumes events_final table exists from App 55.A concept.
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits # Only needed if loading real events
# Assume 'events_final' is an Astropy Table or similar with 'X', 'Y' columns

print("Creating X-ray Image from Event List:")

# --- Simulate 'events_final' table ---
n_final_events = 5000
# Cluster events near center (1024x1024 detector)
center_x, center_y = 512.0, 512.0
source_radius = 50.0
source_x = np.random.normal(center_x, source_radius, n_final_events)
source_y = np.random.normal(center_y, source_radius, n_final_events)
# Add some background noise events
n_bkg = 1000
bkg_x = np.random.uniform(0, 1024, n_bkg)
bkg_y = np.random.uniform(0, 1024, n_bkg)
final_x = np.concatenate((source_x, bkg_x))
final_y = np.concatenate((source_y, bkg_y))
print(f"\nSimulated {len(final_x)} filtered event coordinates.")

# --- Bin events into an image ---
binsize = 8 # Bin events into 8x8 pixel image bins
image_shape = (1024 // binsize, 1024 // binsize) # e.g., 128x128 image
x_edges = np.arange(0, 1024 + binsize, binsize)
y_edges = np.arange(0, 1024 + binsize, binsize)

print(f"\nBinning events into {image_shape} image...")
# Use histogram2d to count events in each image pixel
xray_image, _, _ = np.histogram2d(final_y, final_x, bins=(y_edges, x_edges)) 
# Note: histogram2d expects y first for (row, col) output matching imshow
print("Binning complete.")

# --- Visualize Image ---
print("Generating image plot...")
plt.figure(figsize=(7, 6))
# Use simple square root scaling for visualization
from astropy.visualization import simple_norm
norm = simple_norm(xray_image, stretch='sqrt', percent=99.5) 
plt.imshow(xray_image, origin='lower', cmap='viridis', norm=norm, 
           extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
plt.xlabel(f"Detector X (Binned {binsize}x{binsize})")
plt.ylabel(f"Detector Y (Binned {binsize}x{binsize})")
plt.title("Binned X-ray Image from Events")
plt.colorbar(label="Counts per Bin")
# plt.show()
print("Plot generated.")
plt.close()

# --- Conceptual Source Detection ---
# smoothed_image = scipy.ndimage.gaussian_filter(xray_image, sigma=2)
# threshold = np.mean(smoothed_image) + 3 * np.std(smoothed_image)
# sources = photutils.detection.find_peaks(smoothed_image, threshold=threshold, box_size=11)
# print(f"\nFound {len(sources) if sources else 0} potential sources (conceptual).")

print("-" * 20)

# Explanation:
# 1. Simulates filtered event coordinates `final_x`, `final_y`, clustered near the center.
# 2. Defines the desired image binning (`binsize`, `image_shape`, `x_edges`, `y_edges`).
# 3. Uses `np.histogram2d` to count the number of events falling into each 2D image bin, 
#    creating the `xray_image` array.
# 4. Uses `matplotlib.imshow` to display the resulting image, applying square root scaling 
#    (`simple_norm`) to better visualize faint features and the central source concentration.
# 5. Conceptually mentions source detection steps like smoothing and thresholding.
# This shows the basic process of creating an image representation from an event list.
```

Both imaging and timing analysis extract crucial information complementary to spectroscopy. Imaging reveals the spatial extent, structure, and environment of X-ray sources, while timing probes their variability, periodicities, and the dynamics of accretion or emission processes on various timescales. Python tools like Astropy, Photutils, regions, and Stingray provide essential building blocks for performing these analyses on X-ray event data.

**55.7 Python Tools: Astropy, regions, PyXspec, Sherpa, stingray**

The analysis of X-ray data, from initial reduction to final scientific interpretation, relies heavily on specialized software, and Python has become a central language for accessing and orchestrating these tools. Several key Python libraries and packages form the core toolkit for computational X-ray astronomy.

**Astropy (`astropy`)**: As the foundational package for astronomy in Python (Appendix II), Astropy provides essential components used throughout X-ray analysis:
*   `astropy.io.fits`: For reading and writing the ubiquitous FITS files containing event lists, images, spectra (PHA), response files (RMF/ARF), and GTI tables (Sec 1.6).
*   `astropy.table.Table`: For manipulating event lists or catalog data loaded from FITS tables (Sec 2.3).
*   `astropy.wcs`: For handling World Coordinate System information in FITS headers to convert between detector/image coordinates and sky coordinates (Sec 5.7).
*   `astropy.time.Time`: For handling observation times and filtering based on GTIs (Sec 4.2).
*   `astropy.units` & `astropy.constants`: For managing physical units (keV, Angstrom, cm, s, erg) and constants consistently (Sec 3).
*   `astropy.visualization`: For image display normalization and scaling (e.g., `simple_norm`).
*   `astropy.stats`: For statistical functions, potentially including sigma clipping or robust statistics (Sec 13).

**Regions (`regions`)**: An Astropy affiliated package specifically designed for creating, manipulating, reading, and writing spatial regions commonly used in astronomical analysis.
*   **Facilities:** Defines various geometric region shapes (circles, ellipses, boxes, polygons, annuli) in pixel or celestial coordinates (`PixelRegion`, `SkyRegion`). Provides methods for checking if points are contained within a region (`.contains()`), performing geometric operations, and crucially, reading/writing region file formats like the widely used DS9 region format (`.reg`). Essential for defining source and background extraction regions for imaging and spectral analysis (Sec 55.3).
*   **Documentation:** [https://astropy-regions.readthedocs.io/en/latest/](https://astropy-regions.readthedocs.io/en/latest/)

**PyXspec (`pyxspec`)**: The official Python interface to the **XSPEC** spectral fitting package, which is part of NASA's HEASoft suite.
*   **Facilities:** Allows users to control the XSPEC backend from Python. Provides Python classes and methods for loading spectra/responses (`Spectrum`), defining models (`Model`), setting parameters (`Parameter`), performing fits (`Fit`), calculating errors/confidence intervals (`Fit.error`), plotting results (`Plot`), and running simulations (`Fakeit`). Enables scripting and automation of standard XSPEC workflows. Requires a full HEASoft/XSPEC installation.
*   **Documentation:** [https://heasarc.gsfc.nasa.gov/xanadu/xspec/python/html/](https://heasarc.gsfc.nasa.gov/xanadu/xspec/python/html/)

**Sherpa (`sherpa`)**: A powerful, Python-native modeling and fitting application developed by the Chandra X-ray Center, also widely used for data from other missions.
*   **Facilities:** Provides an object-oriented Python framework for loading data (PHA spectra, images, tables), defining complex models by combining pre-defined components (from XSPEC model library or Sherpa's own) or user-defined functions, choosing fit statistics (Cash, C-stat, Chi2 variants), selecting optimization methods (Levenberg-Marquardt, Nelder-Mead, MCMC), performing fits, and estimating parameter uncertainties (confidence intervals, MCMC posteriors). Offers great flexibility and integration with the scientific Python ecosystem. Can be installed standalone or as part of CIAO (Chandra's software package).
*   **Documentation:** [https://cxc.harvard.edu/sherpa/](https://cxc.harvard.edu/sherpa/)

**Stingray (`stingray`)**: A library focused specifically on time series analysis in astronomy, particularly relevant for X-ray timing but applicable more broadly.
*   **Facilities:** Provides classes for representing event lists (`EventList`) and light curves (`Lightcurve`). Includes tools for creating light curves with different binning, calculating power spectral densities (PSDs) using FFTs (`Powerspectrum`, `AveragedPowerspectrum`), cross-spectra (`Crossspectrum`), time lags, coherence, simulating time series based on PSD models, and performing pulsation searches (epoch folding, `Accelsearch`). Developed with X-ray timing use cases strongly in mind.
*   **Documentation:** [https://stingray.readthedocs.io/en/latest/](https://stingray.readthedocs.io/en/latest/)

**Other Relevant Tools:**
*   **Mission Specific Software (CIAO, SAS, HEASoft):** While `PyXspec` and `Sherpa` provide Python interfaces, the underlying mission-specific software packages (Chandra's CIAO, XMM-Newton's SAS, NASA's HEASoft) contain essential command-line tools (often callable from Python using `subprocess`) for initial data processing, calibration, generating response files (RMFs/ARFs), creating exposure maps, and performing specialized tasks. Familiarity with the relevant mission software is often necessary.
*   **`matplotlib`, `numpy`, `scipy`:** These core libraries are used extensively alongside the specialized packages for general calculations, plotting, and statistical analysis.

This ecosystem of Python packages provides a comprehensive environment for X-ray data analysis. `Astropy` provides the foundation, `regions` handles spatial selections, `PyXspec` and `Sherpa` are the primary tools for spectral modeling and fitting, and `stingray` offers specialized capabilities for timing analysis. Often, these libraries are used together within scripts or Jupyter notebooks to build complex analysis workflows, calling external mission-specific tools where necessary.

---
**Application 55.A: Basic X-ray Spectral Fitting with `PyXspec`**

**(Paragraph 1)** **Objective:** Demonstrate the fundamental workflow of fitting a simple astrophysical model to an X-ray spectrum using the `PyXspec` Python interface (Sec 55.7) to the standard XSPEC fitting package. This involves loading the observed spectrum (PHA) and response files (RMF, ARF), defining a model, performing the fit, and inspecting the results.

**(Paragraph 2)** **Astrophysical Context:** Fitting physical models to X-ray spectra is the primary method for diagnosing conditions in high-energy sources. For example, fitting an absorbed power-law model (`phabs*powerlaw`) to the spectrum of an Active Galactic Nucleus (AGN) can constrain the absorbing column density (N<0xE1><0xB5><0x8F>), the intrinsic spectral slope (photon index Γ), and the normalization (related to flux). This provides insights into the accretion process and the environment surrounding the central supermassive black hole.

**(Paragraph 3)** **Data Source:** Requires three FITS files:
    1.  A source spectrum file (`source.pha`), containing observed counts per channel.
    2.  A Redistribution Matrix File (`source.rmf`), describing energy dispersion.
    3.  An Ancillary Response File (`source.arf`), describing effective area.
    Optionally, a background spectrum file (`background.pha`). For this application, we assume these files exist (they can be downloaded from archives for real sources or generated using the `fakeit` command within XSPEC for simulated data).

**(Paragraph 4)** **Modules Used:** `pyxspec` (the main interface), `os` (to check file existence). Requires HEASoft/XSPEC installed and configured for `pyxspec` to import correctly.

**(Paragraph 5)** **Technique Focus:** Interacting with XSPEC via the `pyxspec` Python module. (1) Importing `xspec` and potentially setting chatter levels. (2) Loading the spectrum using `s = xspec.Spectrum("source.pha")`. (3) Associating response files (`s.response = "source.rmf"`, `s.response.arf = "source.arf"`) if not automatically linked in PHA header. (4) Ignoring bad energy channels (`xspec.AllData.ignore("...")`). (5) Defining a model using XSPEC's string syntax `m = xspec.Model("phabs*powerlaw")`. (6) Accessing and potentially setting initial parameter values (`m.phabs.nH.values = ...`, `m.powerlaw.PhoIndex.values = ...`). (7) Choosing the fit statistic (`xspec.Fit.statMethod = "cstat"`). (8) Performing the fit (`xspec.Fit.perform()`). (9) Querying fit results (`xspec.Fit.statistic`, `xspec.Fit.dof`, parameter values via `m.parameterName.values`). (10) Estimating parameter uncertainties (`xspec.Fit.error(...)`).

**(Paragraph 6)** **Processing Step 1: Setup and Load Data:** Import `xspec`. Define filenames for PHA, RMF, ARF. Check if files exist. Create `Spectrum` object: `s = xspec.Spectrum(pha_file)`. Assign responses if needed. Use `AllData.ignore` to select energy range (e.g., `"0.5-7.0"`) or ignore bad channels (e.g., `"**-0.5 7.0-**"`).

**(Paragraph 7)** **Processing Step 2: Define Model:** Create `Model` object: `m = xspec.Model("phabs*powerlaw")`. Print the model components and parameters using `m.show()`. Set reasonable initial parameter values if the defaults are poor (e.g., `m(1).values = 0.1`, `m(2).values = 1.7`).

**(Paragraph 8)** **Processing Step 3: Configure Fit:** Set statistic: `Fit.statMethod = "cstat"`. Set query mode for error calculation if desired (`Fit.query = "yes"`).

**(Paragraph 9)** **Processing Step 4: Perform Fit and Get Results:** Run the fit: `Fit.perform()`. Check the output for convergence messages. Display fit summary: `Fit.show()`. The summary shows best-fit statistic, dof, and best-fit parameter values.

**(Paragraph 10)** **Processing Step 5: Error Estimation and Plotting:** Calculate uncertainties for parameters of interest (e.g., parameters 1 and 2): `Fit.error("1,2")`. Display the results again with errors: `Fit.show()`. Use `Plot.device = "/xs"` (or `/null`) and `Plot("data resid")` to visualize the fit (requires an X-window display if not `/null`). Extract parameter values and errors programmatically if needed (e.g., `nh_val = m.phabs.nH.values[0]`, `nh_err = m.phabs.nH.error`). Remember to clear data/models (`AllData.clear()`, `AllModels.clear()`) before loading new ones if running multiple fits in a script.

**Output, Testing, and Extension:** Output includes XSPEC's console messages showing fit progress and results (statistic, parameter values, errors). Optionally, plots of data/model/residuals. **Testing:** Verify the fit converges. Check if the best-fit parameters and errors are physically reasonable for the type of source being modeled. Ensure the fit statistic indicates an acceptable (though not absolutely quantifiable with C-stat) fit. Compare results obtained using different fit statistics (e.g., `chi` on grouped data). **Extensions:** (1) Add a background file (`AllData.background = ...`). (2) Fit a more complex model (e.g., `phabs*(powerlaw + gaussian)` to model an emission line). (3) Use `Fit.goodness` command (with simulations) to estimate goodness-of-fit for C-stat. (4) Run MCMC error estimation using `Fit.chain` commands. (5) Write a script to fit the same model to spectra extracted from different regions of an extended source.

```python
# --- Code Example: Application 55.A ---
# Note: Requires HEASoft/XSPEC and pyxspec installed. 
# Requires actual FITS files (PHA, RMF, ARF) - cannot run standalone.

import os
try:
    # Attempt to import pyxspec
    import xspec 
    pyxspec_ok = True
    # Example: Set verbosity lower
    xspec.Xset.chatter = 5 
    xspec.Xset.logChatter = 5 
except ImportError:
    pyxspec_ok = False
    print("NOTE: pyxspec package not found or HEASoft/XSPEC environment not set up.")

print("Basic X-ray Spectral Fitting with PyXspec (Conceptual):")

# Define dummy filenames (replace with real files)
pha_file = "source_spectrum.pha"
rmf_file = "source_response.rmf"
arf_file = "source_effective_area.arf"
# Create dummy files just so os.path.exists doesn't fail immediately
# In reality, these need to be valid FITS files.
for fname in [pha_file, rmf_file, arf_file]:
    if not os.path.exists(fname): open(fname, 'a').close()

if pyxspec_ok:
    # Check if dummy files exist (basic check)
    if not all(os.path.exists(f) for f in [pha_file, rmf_file, arf_file]):
        print(f"Error: One or more input files not found ({pha_file}, etc.)")
        print("Please provide valid PHA, RMF, ARF files for this example.")
    else:
        print("\n--- Running PyXspec Fit (Conceptual Execution) ---")
        # Clear previous settings if any
        xspec.AllData.clear()
        xspec.AllModels.clear()
        
        try:
            # Step 1: Load Data & Responses
            print(f"Loading spectrum: {pha_file}")
            s1 = xspec.Spectrum(pha_file)
            # Assume RMF/ARF are linked in header, otherwise:
            # s1.response = rmf_file
            # s1.response.arf = arf_file
            print("Spectrum loaded.")
            
            # Ignore channels (example: ignore below 0.5 keV and above 7.0 keV)
            xspec.AllData.ignore("bad") # Ignore channels marked bad in PHA file
            xspec.AllData.ignore("**-0.5 7.0-**") 
            print("Channels ignored.")

            # Step 2: Define Model
            print("\nDefining model: phabs * powerlaw")
            m1 = xspec.Model("phabs*powerlaw")
            # Show initial parameters
            m1.show() 
            # Optional: Set initial values if defaults are bad
            # m1(1).values = 0.05 # Parameter index 1 (nH)
            # m1(2).values = 1.9 # Parameter index 2 (PhoIndex)

            # Step 3: Configure Fit
            xspec.Fit.statMethod = "cstat" 
            print(f"Fit statistic: {xspec.Fit.statMethod}")
            # xspec.Fit.query = "yes" # Prompt during error calculation

            # Step 4: Perform Fit
            print("\nPerforming fit...")
            xspec.Fit.perform() 
            print("Fit complete.")

            # Step 5 & 6: Assess Fit and Errors
            print("\n--- Fit Results ---")
            xspec.Fit.show() # Show parameters, statistic, dof
            
            print("\nCalculating 90% confidence errors for parameters 1, 2...")
            # Format: "delta_stat param_indices..." (e.g., 2.706 for 1 param 90%)
            error_command = "2.706 1-2" # 90% for 2 params? Check XSPEC manual. Assume 1, 2 for example.
            xspec.Fit.error(error_command) 
            print("Error calculation complete.")
            
            print("\n--- Final Parameters with Errors ---")
            xspec.Fit.show()
            
            # Extract values programmatically (example for first two params)
            par1_val = m1(1).values[0]
            par1_err = m1(1).error[0:2] # (delta_low, delta_high) relative to best fit
            par2_val = m1(2).values[0]
            par2_err = m1(2).error[0:2]
            print(f"\nExtracted: Param 1 = {par1_val:.3f} +{par1_err[1]:.3f} / -{par1_err[0]:.3f}")
            print(f"Extracted: Param 2 = {par2_val:.3f} +{par2_err[1]:.3f} / -{par2_err[0]:.3f}")
            
            # Plotting (conceptual - requires interactive display or file output setup)
            print("\nGenerating plots (conceptual)...")
            # xspec.Plot.device = "/png" # Or "/ps" etc.
            # xspec.Plot.setRebin(minSig=3, maxBins=40) # Example rebin for plot clarity
            # xspec.Plot("data resid") 
            # xspec.Plot.device = "/null" # Close device
            print("  (Plot commands would generate data/resid plots)")

        except Exception as e_xspec:
             print(f"\nAn error occurred during XSPEC operations: {e_xspec}")
             print("  (Ensure input files are valid and HEASoft environment is correct)")

        # Clean up dummy files if they were created
        for fname in [pha_file, rmf_file, arf_file]:
             if os.path.getsize(fname) == 0: # Simple check for dummy files
                  try: os.remove(fname) 
                  except OSError: pass
else:
    print("\nSkipping PyXspec execution.")

print("-" * 20)
```

**Application 55.B: Creating and Analyzing an X-ray Image from Event List**

**(Paragraph 1)** **Objective:** This application demonstrates the basic steps involved in creating a binned X-ray image from a FITS event list file and performing rudimentary analysis (smoothing, source detection) using core scientific Python libraries like `Astropy`, `NumPy`, and `SciPy`, complemented by `Matplotlib` for visualization. Reinforces Sec 55.2, 55.3, 55.6.

**(Paragraph 2)** **Astrophysical Context:** While X-ray detectors often record individual photon events, binning these events spatially into an image allows astronomers to visualize the morphology of extended sources (like supernova remnants, galaxy clusters, nebulae) and to detect and localize point sources (like X-ray binaries or AGN). Analyzing these images often involves smoothing to enhance faint features and applying source detection algorithms to create catalogs.

**(Paragraph 3)** **Data Source:** A FITS event list file (`events.fits`) containing columns for event coordinates, typically detector coordinates like 'X' and 'Y', or potentially calibrated sky coordinates 'RA' and 'DEC'. For this application, we'll primarily use simulated detector X/Y coordinates.

**(Paragraph 4)** **Modules Used:** `astropy.io.fits` (to read event list), `numpy` (for histogramming and array operations), `scipy.ndimage` (for optional Gaussian smoothing), `matplotlib.pyplot` (for displaying the image), `astropy.visualization` (for image scaling/stretching). `photutils` could be used for more sophisticated source detection but we use a simpler method here.

**(Paragraph 5)** **Technique Focus:** Event list handling and image creation/analysis. (1) Reading the event list from a FITS table using `fits.open()` and accessing coordinate columns. (2) Defining the desired image properties (pixel size/binning factor, image dimensions). (3) Using `numpy.histogram2d` to bin the event X/Y coordinates into a 2D image array representing counts per pixel/bin. (4) Visualizing the raw counts image using `matplotlib.pyplot.imshow`, potentially using scaling functions from `astropy.visualization` (like `simple_norm` with `stretch='sqrt'` or `'log'`) appropriate for count data. (5) Optionally applying a Gaussian filter (`scipy.ndimage.gaussian_filter`) to smooth the image. (6) Performing simple source detection by identifying pixels in the (potentially smoothed) image that exceed a significance threshold above the background (e.g., mean + N * std dev). (7) Plotting the image with detected source positions marked.

**(Paragraph 6)** **Processing Step 1: Load Event List:** Use `fits.open('events.fits')` to open the file. Access the event table HDU (usually extension 1). Read the 'X' and 'Y' coordinate columns into NumPy arrays `x_coords`, `y_coords`. Close the FITS file.

**(Paragraph 7)** **Processing Step 2: Define Image Binning:** Choose an image pixel size (or binning factor if starting from detector pixels), e.g., `binsize=8`. Determine the image dimensions based on the range of `x_coords`, `y_coords` and the `binsize`. Create bin edge arrays `x_edges` and `y_edges` using `np.arange`.

**(Paragraph 8)** **Processing Step 3: Bin Events:** Create the 2D counts image using `image_counts, _, _ = np.histogram2d(y_coords, x_coords, bins=(y_edges, x_edges))`. (Note: `histogram2d` expects `y` first for conventional image row/column order).

**(Paragraph 9)** **Processing Step 4: Visualize Raw Image:** Use `plt.imshow(image_counts, origin='lower', cmap='viridis', norm=simple_norm(...))` to display the counts image. Add colorbar and labels.

**(Paragraph 10)** **Processing Step 5: Smooth and Detect Sources (Simple):** Optionally smooth the image: `image_smoothed = gaussian_filter(image_counts, sigma=2)`. Calculate a background level and noise estimate (e.g., `bkg_mean = np.median(image_smoothed)`, `bkg_std = np.std(image_smoothed[image_smoothed < threshold])` or use `sigma_clipped_stats`). Define a detection threshold (e.g., `threshold = bkg_mean + 5 * bkg_std`). Find pixels above the threshold `bright_pixels = np.where(image_smoothed > threshold)`. Plot the image again, overlaying markers (`plt.scatter`) at the `bright_pixels` coordinates.

**Output, Testing, and Extension:** Output includes the generated FITS image (or plots) showing the binned counts, potentially a smoothed version, and markers indicating detected source candidates. **Testing:** Verify the image dimensions and total counts are consistent with the input event list and binning. Check if known sources in the field (if any) are visually apparent and detected by the simple thresholding. Assess the effect of different bin sizes or smoothing sigmas. **Extensions:** (1) Use WCS information (if present in event list header) to create the image directly in sky coordinates (RA/Dec) and display with celestial axes using `astropy.visualization.wcsaxes`. (2) Apply energy filters to the event list before binning to create images in different energy bands. (3) Implement background subtraction by estimating background from a source-free region and subtracting it from the image before source detection. (4) Use more sophisticated source detection algorithms from `photutils` (like `DAOStarFinder` or `segmentation.detect_sources`) instead of simple thresholding. (5) Use the `regions` package to define circular or other apertures around detected sources and perform simple aperture photometry (sum counts within aperture, subtract background).

```python
# --- Code Example: Application 55.B ---
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import gaussian_filter
import os

print("Creating and Analyzing X-ray Image from Simulated Event List:")

# Step 1: Simulate / Load Event List
n_events_img = 20000
detector_size = 512 # Assume a 512x512 detector
# Simulate a central point source + uniform background
center_x, center_y = detector_size / 2.0, detector_size / 2.0
n_source = int(n_events_img * 0.7)
n_bkg = n_events_img - n_source
# Source events clustered around center (simple Gaussian)
source_x = np.random.normal(center_x, 15.0, n_source)
source_y = np.random.normal(center_y, 15.0, n_source)
# Background events uniform across detector
bkg_x = np.random.uniform(0, detector_size, n_bkg)
bkg_y = np.random.uniform(0, detector_size, n_bkg)
x_coords = np.concatenate((source_x, bkg_x))
y_coords = np.concatenate((source_y, bkg_y))
# Clip coords to be within detector bounds
x_coords = np.clip(x_coords, 0, detector_size - 1e-6)
y_coords = np.clip(y_coords, 0, detector_size - 1e-6)
print(f"\nGenerated {len(x_coords)} event coordinates on a {detector_size}x{detector_size} detector.")

# Step 2: Define Image Binning
binsize = 4 # Rebin into smaller image pixels (e.g., 4x4 detector pixels -> 1 image pixel)
npix_x = detector_size // binsize
npix_y = detector_size // binsize
image_shape = (npix_y, npix_x)
x_edges = np.linspace(0, detector_size, npix_x + 1)
y_edges = np.linspace(0, detector_size, npix_y + 1)
print(f"Binning into {image_shape} image (binsize={binsize}).")

# Step 3: Bin Events
# Use histogram2d - remember y comes first for image orientation
image_counts, _, _ = np.histogram2d(y_coords, x_coords, bins=(y_edges, x_edges))
print("Events binned.")

# Step 4: Visualize Raw Image
print("\nGenerating raw counts image plot...")
fig1, ax1 = plt.subplots(figsize=(7, 6))
norm_raw = simple_norm(image_counts, stretch='sqrt', percent=99.8)
im_raw = ax1.imshow(image_counts, origin='lower', cmap='viridis', norm=norm_raw,
                    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
ax1.set_xlabel(f"Detector X (Pixel / {binsize})")
ax1.set_ylabel(f"Detector Y (Pixel / {binsize})")
ax1.set_title("Binned X-ray Counts Image (Sqrt Stretch)")
fig1.colorbar(im_raw, ax=ax1, label="Counts per Image Pixel")
# plt.show()
print("Raw image plot generated.")
plt.close(fig1)

# Step 5: Smooth and Detect Sources (Simple Thresholding)
print("\nSmoothing image and performing simple source detection...")
# Smooth image slightly to reduce noise influence
image_smoothed = gaussian_filter(image_counts, sigma=1.5)

# Estimate background and threshold
# Use sigma_clipped_stats on the *original* counts image for background estimation
mean_bkg, median_bkg, std_bkg = sigma_clipped_stats(image_counts, sigma=3.0)
# Set threshold above background using smoothed image
threshold = median_bkg + 5.0 * std_bkg # 5-sigma above median background
print(f"  Background Estimate (Median): {median_bkg:.2f} counts/pixel")
print(f"  Background Std Dev (Sigma-clipped): {std_bkg:.2f} counts/pixel")
print(f"  Detection Threshold: {threshold:.2f} counts/pixel (in smoothed image)")

# Find pixels in smoothed image above threshold
bright_pixels_y, bright_pixels_x = np.where(image_smoothed > threshold)
# Convert pixel indices back to coordinate system of the plot
bright_coords_x = x_edges[bright_pixels_x] + binsize / 2.0
bright_coords_y = y_edges[bright_pixels_y] + binsize / 2.0
print(f"  Found {len(bright_pixels_x)} pixels above threshold.")

# Visualize Smoothed Image with Detected Pixels
print("Generating smoothed image plot with detections...")
fig2, ax2 = plt.subplots(figsize=(7, 6))
norm_smooth = simple_norm(image_smoothed, stretch='sqrt', percent=99.8)
im_smooth = ax2.imshow(image_smoothed, origin='lower', cmap='viridis', norm=norm_smooth,
                       extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
ax2.scatter(bright_coords_x, bright_coords_y, s=50, facecolor='none', edgecolors='red', label='Detected Pixels')
ax2.set_xlabel(f"Detector X (Pixel / {binsize})")
ax2.set_ylabel(f"Detector Y (Pixel / {binsize})")
ax2.set_title("Smoothed Image with Pixels > 5σ above Background")
fig2.colorbar(im_smooth, ax=ax2, label="Smoothed Counts")
ax2.legend(markerscale=0.6)
# plt.show()
print("Plot generated.")
plt.close(fig2)

print("-" * 20)
```

**Chapter 55 Summary**

This chapter explored the computational techniques essential for analyzing data from **X-ray astronomy**, which probes high-energy phenomena in the Universe. It covered X-ray detection using focusing telescopes with **photon-counting detectors** (like CCDs or microcalorimeters), emphasizing that the primary data product is typically an **event list** (usually a FITS table) recording the time, position, and energy (PHA/PI channel) of each detected photon. Essential **data reduction** steps were outlined, including filtering event lists based on **Good Time Intervals (GTIs)**, energy range, event grade/quality, and spatial **region selection** (often using the `regions` library), along with the critical and often complex task of **background handling** (using source-free regions or modeling). The core process of **X-ray spectral analysis** via forward fitting was detailed, explaining the roles of the observed count spectrum (**PHA file**), the detector's energy dispersion matrix (**RMF**), and the telescope/detector effective area (**ARF**).

The chapter introduced the standard software packages **XSPEC** and **Sherpa** (accessible via Python interfaces like `PyXspec` and the `sherpa` library) used for spectral modeling and fitting. This involves defining astrophysical models (combining components like `powerlaw`, `apec`, `phabs`), choosing appropriate **fit statistics** for Poisson data (C-stat, W-stat), performing the fit to find best-fit parameters, assessing goodness-of-fit, and estimating parameter uncertainties. Basic **X-ray image analysis** techniques were also covered, including creating images by binning event lists (`numpy.histogram2d`), correcting for exposure variations, smoothing (`scipy.ndimage`), basic source detection, and morphological analysis. Finally, fundamental **X-ray timing analysis** methods were introduced, such as creating light curves, searching for periodicities (FFT, Lomb-Scargle, epoch folding using tools like `astropy.timeseries` or `stingray`), and analyzing aperiodic variability through power spectral densities (`stingray`). Two applications demonstrated basic spectral fitting with `PyXspec` (conceptual) and creating/analyzing an X-ray image from a simulated event list using `Astropy`/`NumPy`.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Arnaud, K. A. (1996).** XSPEC: The First Ten Years. In G. H. Jacoby & J. Barnes (Eds.), *Astronomical Data Analysis Software and Systems V (ADASS V)* (ASP Conference Series, Vol. 101, p. 17). Astronomical Society of the Pacific. ([Link via ADS](https://ui.adsabs.harvard.edu/abs/1996ASPC..101...17A/abstract)) (See also XSPEC Manual: [https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/](https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/))
    *(The original paper describing XSPEC. The linked manual is the definitive reference for XSPEC models, commands, and statistics, relevant to Sec 55.5 and Application 55.A.)*

2.  **Freeman, P., Doe, S., & Siemiginowska, A. (2001).** Sherpa: A Mission-Independent Data Analysis Application. In *Proceedings of SPIE*, *4477*, 76-87. [https://doi.org/10.1117/12.447109](https://doi.org/10.1117/12.447109) (See also Sherpa Documentation: [https://cxc.harvard.edu/sherpa/](https://cxc.harvard.edu/sherpa/))
    *(Introduces the Sherpa modeling and fitting environment, relevant to Sec 55.5, 55.7.)*

3.  **Cash, W. (1979).** Parameter estimation in astronomy through application of the likelihood ratio. *The Astrophysical Journal*, *228*, 939–947. [https://doi.org/10.1086/156922](https://doi.org/10.1086/156922)
    *(The seminal paper introducing the Cash statistic (C-stat), the appropriate likelihood-based statistic for fitting models to Poisson-distributed count data, crucial for X-ray spectral analysis as discussed in Sec 55.5.)*

4.  **Astropy Collaboration, et al. (2022).** The Astropy Project: Sustaining and Growing a Community-oriented Python Package for Astronomy. *The Astrophysical Journal*, *935*(2), 167. [https://doi.org/10.3847/1538-4357/ac7c74](https://doi.org/10.3847/1538-4357/ac7c74) (See also `regions` docs: [https://astropy-regions.readthedocs.io/en/latest/](https://astropy-regions.readthedocs.io/en/latest/))
    *(References the core Astropy package used for FITS I/O, WCS, etc., and the affiliated `regions` package essential for spatial analysis mentioned in Sec 55.3, 55.6, App 55.B.)*

5.  **Huppenkothen, D., et al. (including Ingram, A., & Bachetti, M.). (2019).** Stingray: A modern Python library for spectral timing. *The Astrophysical Journal*, *881*(1), 39. [https://doi.org/10.3847/1538-4357/ab258d](https://doi.org/10.3847/1538-4357/ab258d) (See also Documentation: [https://stingray.readthedocs.io/en/latest/](https://stingray.readthedocs.io/en/latest/))
    *(Introduces the Stingray library for time series analysis, particularly relevant for X-ray timing discussed in Sec 55.6, 55.7.)*
