**Chapter 53: Computational Techniques in Infrared Astronomy**

This chapter focuses on the computational methods employed in **infrared (IR) astronomy**, a wavelength regime crucial for studying phenomena obscured by dust at optical wavelengths, probing cooler objects like protostars, planets, and molecular clouds, and observing highly redshifted light from the early Universe. We begin by discussing the challenges of **IR observations**, contrasting ground-based facilities (which must contend with significant atmospheric absorption and thermal emission, often requiring specialized observing techniques) with space-based telescopes (like Spitzer, Herschel, WISE, and the transformative James Webb Space Telescope - JWST) that provide uninterrupted access to the full IR spectrum. Standard **data reduction** steps specific to IR detectors (often HgCdTe or InSb arrays) are outlined, including non-linearity corrections, reference pixel subtraction, flat-fielding, background subtraction (a major challenge due to bright, variable sky/telescope emission), and correction for detector artifacts like persistence or "snowballs". We review common **IR data formats** (typically FITS images and spectral cubes) and major **archives** like IRSA (Infrared Science Archive). Key **analysis techniques** prominent in IR astronomy are explored, including performing accurate **photometry** in potentially crowded fields or regions with complex backgrounds, **image subtraction** for discovering transients or variable sources, constructing and fitting **Spectral Energy Distributions (SEDs)** using models that incorporate both stellar emission and thermal **dust emission**, analyzing **IR spectral features** (atomic lines, molecular bands like H₂ or CO, polycyclic aromatic hydrocarbons - PAHs, dust silicate/ice features), and **mapping emission lines** using integral field unit (IFU) data from instruments like JWST's NIRSpec and MIRI. Finally, relevant **Python tools** are highlighted, including `Astropy`, `photutils`, instrument-specific pipelines and analysis tools (e.g., the `jwst` pipeline), and specialized **SED fitting codes** (like `Prospector`, `BAGPIPES`, `CIGALE`).

**53.1 IR Observations: Ground vs. Space, Atmospheric Effects**

Infrared astronomy opens a unique window on the Universe, allowing us to observe phenomena often hidden at optical wavelengths. IR radiation penetrates dust clouds more effectively than visible light, revealing obscured star-forming regions, the centers of dusty galaxies, and embedded active galactic nuclei (AGN). It also traces cooler objects whose thermal emission peaks in the infrared, such as planets, brown dwarfs, protostars, debris disks, and interstellar dust grains. Furthermore, due to cosmological redshift, light from the most distant galaxies formed in the early Universe is shifted from the ultraviolet/optical into the near- and mid-infrared, making IR observations essential for studying galaxy formation and cosmic dawn.

Observing in the infrared, however, presents significant challenges, particularly from the ground. The Earth's atmosphere contains molecules (primarily water vapor H₂O and carbon dioxide CO₂) that strongly **absorb** IR radiation at specific wavelengths, creating opaque bands that block extraterrestrial signals. Observations are only possible through specific atmospheric "windows" in the near-IR (e.g., J, H, K bands ~1-2.5 μm), mid-IR (e.g., L, M, N, Q bands ~3-25 μm, requiring high, dry sites), and some sub-millimeter windows. Even within these windows, absorption lines can contaminate spectra.

Furthermore, both the atmosphere and the telescope itself are typically at temperatures around ~270-300 K and therefore **emit strongly** in the infrared via thermal blackbody radiation. This emission creates a bright, often rapidly fluctuating, background against which faint astronomical sources must be detected. The background brightness increases dramatically at longer wavelengths (mid-IR and far-IR), making ground-based observations beyond ~25 μm essentially impossible.

To mitigate these effects, ground-based IR observations employ specialized techniques. Telescopes are often located at high, dry sites (like Mauna Kea, Cerro Paranal, Chajnantor Plateau) to minimize water vapor absorption. Observing techniques like **chopping** (rapidly switching the telescope's secondary mirror between the source and a nearby blank sky position) and **nodding** (moving the entire telescope between source and blank sky positions) are used to subtract the bright, fluctuating sky and telescope background emission in near real-time. Careful calibration and data reduction are needed to remove residual atmospheric effects. Adaptive optics are also crucial at near-IR wavelengths on large ground-based telescopes to correct for atmospheric turbulence (seeing) and achieve high angular resolution.

**Space-based IR observatories** entirely overcome the limitations of atmospheric absorption and emission by operating above the atmosphere. Missions like IRAS, ISO, Spitzer Space Telescope, Herschel Space Observatory, Wide-field Infrared Survey Explorer (WISE), and now the James Webb Space Telescope (JWST) have revolutionized infrared astronomy. To minimize their own thermal emission, these telescopes and their instruments must often be **cryogenically cooled** to very low temperatures (tens of Kelvin or even lower). This allows for unprecedented sensitivity, particularly in the mid- and far-infrared, opening up vast new areas of discovery space, from exoplanet atmospheres to the first galaxies.

Detectors used in IR astronomy also differ from optical CCDs. Near-IR instruments (roughly 1-5 μm) commonly use **Mercury Cadmium Telluride (HgCdTe)** arrays, while mid-IR and far-IR instruments might use **Indium Antimonide (InSb)**, **Arsenic-doped Silicon (Si:As)**, or bolometer arrays. These detectors have different operating principles, noise characteristics (read noise, dark current often highly dependent on temperature), cosmetic defects (bad pixels), and sometimes exhibit non-linear behavior or persistence effects (latent images) that require specific calibration procedures (Sec 53.2).

The choice between ground-based and space-based IR observations depends on the desired wavelength coverage, sensitivity, and angular resolution. Ground-based telescopes offer larger aperture sizes (enabling higher theoretical resolution with adaptive optics) and more flexible instrumentation, but are limited by atmospheric windows and background emission. Space-based observatories provide access to the entire IR spectrum with much higher sensitivity (due to cold optics and no atmospheric background) but are typically smaller and have fixed instrumentation with finite lifetimes (often limited by cryogen coolant). Both modalities require sophisticated computational techniques for data reduction and analysis, tailored to the specific challenges of infrared observations.

**53.2 Data Reduction: Background Subtraction, Flat Fielding, Artifacts**

Processing raw data from infrared instruments to produce scientifically usable images or spectra involves several specific calibration steps, some similar to optical CCD processing (App 61), but others unique to the challenges of IR observations and detector technologies. Robust data reduction pipelines, often provided by observatories or instrument teams (e.g., the JWST pipeline), are essential for handling these complexities, frequently implemented using Python and libraries like Astropy.

**Basic CCD-like Steps:** Many initial steps resemble optical CCD reduction:
*   **Bias/Dark Subtraction:** While true bias frames (zero exposure) might not always be taken, subtracting a baseline signal level (often derived from reference pixels or dark current measurements) is necessary. Dark current, the signal generated thermally within the detector, can be significant, especially for non-cryogenic instruments or long exposures, and requires subtraction using appropriately scaled **dark frames** (long exposures taken with a blank shutter closed).
*   **Non-linearity Correction:** IR detectors can exhibit non-linear response, where the measured signal is not directly proportional to the incoming photon count, especially at high flux levels. Pipelines apply corrections based on laboratory characterization data to linearize the response.
*   **Flat Fielding:** Corrects for pixel-to-pixel variations in quantum efficiency and illumination patterns. **Flat field frames** are obtained by observing a uniformly illuminated source (like the twilight sky, a dome screen, or the telescope background itself). Master flats are created (often by combining many exposures) and divided into the science data after other basic calibrations. Obtaining high-quality flat fields, especially across wide wavelength ranges or for complex instruments like IFUs, can be challenging.

**Background Subtraction:** This is often the most critical and complex step in IR data reduction, especially for ground-based data or space observations dominated by zodiacal light or telescope thermal emission. The bright, often spatially and temporally variable background must be accurately subtracted to reveal faint astronomical sources.
*   **Chopping/Nodding Subtraction (Ground):** Data taken using chopping/nodding techniques inherently subtracts much of the background during observation or early processing steps by differencing on-source and off-source measurements.
*   **Sky Frame Subtraction:** For imaging, dedicated sky exposures taken in nearby "empty" fields can be scaled (if necessary) and subtracted.
*   **Self-Subtraction / Dithering:** Observations are often taken with small spatial offsets (**dithering**). By combining multiple dithered exposures, statistical methods (like median filtering over time for each pixel, assuming the source moves between pixels) can be used to estimate and subtract the background, which is assumed to be more stable or spatially smoother than the compact astronomical sources.
*   **Model Fitting:** Fitting analytical functions (e.g., low-order polynomials or splines) to the background variations across the image, while masking out detected sources, is another common technique, especially for complex backgrounds. `photutils.background` provides tools for this. Accurate background subtraction is crucial for reliable photometry.

**Detector Artifact Correction:** IR arrays can suffer from various cosmetic defects and instrumental effects:
*   **Bad Pixels/Hot Pixels/Dead Pixels:** Pixels with anomalous response need to be identified (e.g., from calibration data) and masked out or corrected by interpolation from neighbors.
*   **Persistence:** Previous bright sources can leave temporary latent images on some IR detectors (especially HgCdTe). Pipelines attempt to model and subtract this persistence signal based on illumination history.
*   **Cosmic Rays:** Similar to optical CCDs, cosmic rays hit the detector and need to be identified and removed, often using algorithms that look for sharp spikes in single exposures or outliers across multiple dithered exposures (`astropy.stats.sigma_clip`, `ccdproc`).
*   **Other Effects:** Instrument-specific effects like "snowballs" or "showers" (large charge diffusion events), column/row crosstalk, or electronic artifacts might require specialized correction algorithms developed by the instrument teams.

**Flux Calibration and WCS:** After basic reduction, images/spectra need to be flux calibrated (converting counts/s or DN/s into physical flux units like Jy or erg/s/cm²/Å, using observations of standard stars or internal calibration sources) and have an accurate World Coordinate System (WCS) solution attached (linking pixels to sky coordinates).

Modern data reduction pipelines (like the `jwst` pipeline) are typically implemented in Python, leveraging libraries like `astropy`, `numpy`, `scipy`, `photutils`, and specialized instrument modeling code. They often involve many sequential steps with numerous configuration parameters. Understanding the key reduction steps applied to a given IR dataset and the associated uncertainties or potential residuals is essential for performing reliable scientific analysis on the final data products.

**(Code examples usually involve calling functions from specific pipelines like `jwst` or `ccdproc`, which have complex setup. Illustrating basic steps like median combining conceptually):**
```python
# --- Code Example: Conceptual Master Bias/Flat Creation ---
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats # For robust combining

print("Conceptual Master Bias and Flat Creation:")

# Assume bias_frames is a list of 2D NumPy arrays (raw bias images)
# Assume flat_frames is a list of 2D NumPy arrays (raw flat images for one filter)
# Assume master_bias_data is a 2D NumPy array (previously computed)

# --- Master Bias ---
# Stack frames into a 3D cube
# bias_cube = np.stack(bias_frames, axis=0) 
# Calculate median along the time axis (axis=0)
# master_bias_data = np.median(bias_cube, axis=0) 
# More robust: Sigma-clipped median
# _, median_bias, _ = sigma_clipped_stats(bias_cube, axis=0, sigma=3.0, maxiters=5)
# master_bias_data = median_bias
print("\nConceptual Master Bias: Median combine raw bias frames.")
# hdu_bias = fits.PrimaryHDU(master_bias_data)
# hdu_bias.writeto('master_bias.fits', overwrite=True)

# --- Master Flat ---
processed_flats = []
# for raw_flat in flat_frames:
#     # Assume master_bias_data is loaded
#     bias_subtracted_flat = raw_flat - master_bias_data 
#     processed_flats.append(bias_subtracted_flat)
# flat_cube = np.stack(processed_flats, axis=0)
# combined_flat = np.median(flat_cube, axis=0) # Or sigma-clipped median
# Normalize the flat (e.g., divide by its median value)
# flat_median = np.median(combined_flat)
# master_flat_data = combined_flat / flat_median
print("\nConceptual Master Flat:")
print("  - Subtract master bias from each raw flat.")
print("  - Median combine bias-subtracted flats.")
print("  - Normalize combined flat by its median value.")
# hdu_flat = fits.PrimaryHDU(master_flat_data)
# hdu_flat.writeto('master_flat_R.fits', overwrite=True)

print("-" * 20)

# Explanation: This conceptually outlines the steps for master bias/flat creation.
# 1. For master bias, it conceptually stacks raw bias frames and calculates the median 
#    (or sigma-clipped median for robustness) across the stack.
# 2. For master flat, it subtracts the master bias from each raw flat, combines the 
#    results using a median, and then normalizes the combined flat by its overall 
#    median value to create a relative sensitivity map.
# In practice, libraries like `ccdproc` provide functions to perform these steps robustly.
```

**53.3 IR Data Formats and Archives (e.g., IRSA)**

Infrared astronomical data, like data at other wavelengths, are predominantly stored and distributed using the **Flexible Image Transport System (FITS)** format (Sec 1.5). FITS files provide a standardized way to store image arrays, data tables, and crucial metadata within headers. However, the specific structure and content can vary depending on the instrument and processing level.

**Common Data Products:**
*   **Raw Data:** Often multi-extension FITS files containing raw detector readouts (e.g., individual integrations or "ramps" for non-destructive read detectors like those on JWST). Headers contain essential telemetry and configuration information.
*   **Calibrated Images:** Typically single or multi-extension FITS files containing processed image data (bias/dark subtracted, flat-fielded, flux calibrated) with associated WCS information in the header. Units might be counts/s, MJy/sr, or other flux density units. Uncertainty maps and data quality masks are often provided as separate FITS extensions.
*   **Spectral Cubes:** Data from Integral Field Units (IFUs) or scanned Fabry-Perot instruments are often stored as 3D FITS image HDUs (Axis 1: RA/X, Axis 2: Dec/Y, Axis 3: Wavelength/Frequency). The FITS header contains WCS information defining all three axes.
*   **Catalogs:** Source catalogs derived from images (positions, fluxes, magnitudes, morphological parameters) are usually stored as FITS binary table HDUs (`BinTableHDU`, Sec 1.6 extension concept).
*   **Spectra (1D):** Extracted 1D spectra might be stored in FITS binary tables (e.g., wavelength, flux, error columns) or sometimes as multi-dimensional FITS image arrays (e.g., for echelle spectra).

**Key Metadata:** FITS headers in IR data contain essential information similar to optical data (WCS coordinates, exposure time, observation date, target name, PI name, proposal ID) but also include keywords specific to IR observations and instruments:
*   **Filter:** Name of the infrared filter used.
*   **Detector Information:** Detector name, readout mode, gain settings, temperature.
*   **Observation Technique:** Keywords indicating if chopping/nodding was used (ground-based).
*   **Calibration Status:** Keywords indicating which processing steps (flat, dark, linearity) have been applied.
*   **Photometric Zeropoint/Flux Conversion:** Keywords needed to convert data values (DN/s) to physical flux units (Jy, mag).

**Major Infrared Archives:** Access to IR data from major missions is primarily through dedicated online archives:
*   **IRSA (Infrared Science Archive):** Hosted by IPAC at Caltech, IRSA is a crucial archive for many NASA infrared missions, including Spitzer, WISE, Herschel (US data), SOFIA, 2MASS, and others. It provides web-based search tools, data download capabilities, and often VO services (SIA, TAP, SCS) for programmatic access. `astroquery.irsa` provides a Python interface.
*   **MAST (Mikulski Archive for Space Telescopes):** Hosted by STScI, MAST is the primary archive for Hubble Space Telescope (HST) and James Webb Space Telescope (JWST) data, both of which have significant near-infrared capabilities (WFC3/IR, NICMOS for HST; NIRCam, NIRSpec, NIRISS, MIRI for JWST). It also hosts other missions like TESS and Kepler. MAST offers web portals, VO services, and programmatic access via `astroquery.mast`.
*   **ESA Archives (ESASky, Herschel Science Archive, ISO Data Archive):** The European Space Agency archives data from its missions, including Herschel (full archive), ISO, and potentially JWST data (depending on proposal origin). ESASky provides a multi-mission exploration interface, often with VO compatibility. `astroquery.esa.hubble`, `astroquery.esa.iso`, `astroquery.esa.herschel` provide interfaces.
*   **Ground-Based Observatory Archives:** Major ground-based observatories with IR capabilities (VLT, Keck, Gemini, Subaru) maintain their own archives, often accessible via web portals and sometimes VO TAP services (e.g., ESO Archive Science Portal).

Accessing data from these archives programmatically usually involves using `astroquery` submodules specific to the archive (e.g., `astroquery.irsa`, `astroquery.mast`) or using general VO query tools (`pyvo` with service URLs obtained from registries or documentation) if the archive provides standard VO interfaces (SIA, SSA, TAP). The downloaded data will typically be in FITS format, requiring `astropy.io.fits` for reading and analysis within Python. Understanding the specific data products, file structures, and header keywords provided by different IR instruments and archives is essential for effective data utilization.

**(No code examples needed here as it describes formats and archives.)**

**53.4 IR Photometry and Image Analysis (Crowding, Subtraction)**

Extracting quantitative measurements like fluxes or magnitudes (**photometry**) and analyzing source structures (**morphology**) from infrared images involves techniques similar to optical analysis but often faces specific challenges related to higher backgrounds, potentially larger PSFs (especially at longer wavelengths or from ground without AO), and potentially very crowded fields (e.g., near the Galactic Center or in dense star-forming regions).

**Aperture Photometry:** Measuring flux within a circular or elliptical aperture centered on a source is a basic technique. `photutils.aperture_photometry` provides tools for this. However, accurate background subtraction is critical in the IR. Using local background estimation via annuli around the source (`photutils.aperture.CircularAnnulus`) is common, but requires care in crowded fields where annuli might be contaminated by neighbors. More robust background estimation using 2D models (`photutils.background.Background2D`) applied to the image before photometry is often preferred (App 53.B). Correctly choosing the aperture size (balancing enclosed flux vs. noise/contamination) and applying aperture corrections (to account for flux outside the finite aperture, often derived from PSF models) are important for accurate total flux measurements.

**PSF Photometry:** In crowded fields where sources overlap significantly, aperture photometry becomes unreliable. **Point Spread Function (PSF) photometry** provides a more robust solution by modeling the shape of the instrumental point spread function (how a point source appears after blurring by telescope optics, atmosphere, and detector) and simultaneously fitting this model to multiple overlapping sources in the image.
1.  **Build PSF Model:** Obtain a model for the PSF, either from theoretical calculations (less common), simulations (e.g., WebbPSF for JWST), or empirically by analyzing bright, isolated stars in the image itself (`photutils.psf.EPSFBuilder`). The PSF can vary across the detector.
2.  **Fit PSF Model:** Use algorithms (available in `photutils.psf` or specialized packages like DAOphot/ALLSTAR concepts, DOLPHOT) that position the PSF model at the location of detected sources (or assumed locations) and fit for the amplitude (flux) of each source simultaneously, often iterating between finding sources, subtracting neighbors, and fitting fluxes. This can disentangle flux from blended sources much better than aperture photometry. PSF photometry requires a well-characterized PSF model and careful source detection.

**Image Subtraction:** Detecting variable sources or transients (like supernovae, variable YSOs, microlensing events) in sequences of IR images often relies on **image subtraction** techniques. The goal is to subtract a deep "template" or reference image from individual science images taken at different epochs, leaving behind only sources that have changed in brightness or position. Algorithms like **Difference Image Analysis (DIA)** (e.g., implementing variants of the Alard & Lupton algorithm) are commonly used. These involve:
1.  **Aligning Images:** Precisely registering images taken at different times using WCS or cross-correlation.
2.  **Matching PSFs:** Modeling the PSF variation between the target image and the reference image and deriving a convolution kernel that matches the reference PSF to the target PSF.
3.  **Convolution and Subtraction:** Convolve the reference image with the matching kernel and subtract it from the target image (or vice-versa, or use symmetric approaches). Ideally, constant sources subtract out, leaving residuals only where variability occurred.
4.  **Source Detection on Difference Image:** Detect significant positive or negative residuals on the difference image, corresponding to sources that brightened or faded.
Libraries dedicated to difference imaging exist, and `astropy.convolution` provides tools for the convolution step. Accurate PSF matching is critical for clean subtractions.

**Morphological Analysis:** Characterizing the shapes and structures of resolved sources (like galaxies or nebulae) uses techniques similar to optical analysis but adapted for IR data characteristics. Tools like `photutils.segmentation` can identify source pixels, and `photutils.morphology` (or external packages like `statmorph`) can calculate non-parametric morphology indicators like concentration, asymmetry, smoothness (CAS parameters), Gini coefficient, M₂₀ statistic, half-light radius, etc. These measurements help classify galaxy types or quantify structural properties, often requiring careful background subtraction and PSF deconvolution beforehand if high fidelity is needed.

Analyzing IR images often involves adapting standard optical techniques to handle higher backgrounds, potentially larger/more variable PSFs, and specific detector artifacts. Python libraries like `Astropy` and especially `photutils` provide many of the core building blocks for performing photometry, source detection, background estimation, and basic morphological analysis on IR FITS images. Specialized instrument pipelines often build upon these foundations.

```python
# --- Code Example: Conceptual Aperture Photometry with Background Subtraction ---
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import SigmaClip
# Assume photutils is installed for actual implementation
# from photutils.detection import DAOStarFinder
# from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
# from photutils.background import Background2D, MedianBackground

print("Conceptual Aperture Photometry with Background Subtraction:")

# --- Assume 'image_data' is a 2D NumPy array loaded from FITS ---
# Simulate image with sources and background gradient
npix = 100
y_coords, x_coords = np.indices((npix, npix))
background = 50 + 0.1 * x_coords + 0.1 * y_coords # Simple gradient
# Add sources (simple Gaussians)
src1_pos = (30.5, 40.5); src1_amp = 500; src1_sig = 3.0
src2_pos = (65.2, 55.8); src2_amp = 300; src2_sig = 4.0
dist1_sq = (x_coords - src1_pos[0])**2 + (y_coords - src1_pos[1])**2
dist2_sq = (x_coords - src2_pos[0])**2 + (y_coords - src2_pos[1])**2
image_data = background + \
             src1_amp * np.exp(-dist1_sq / (2 * src1_sig**2)) + \
             src2_amp * np.exp(-dist2_sq / (2 * src2_sig**2))
# Add noise
image_data += np.random.normal(0, np.sqrt(background), size=image_data.shape)
print("\nGenerated dummy image data.")

# --- Step 1: Estimate Background (using Background2D conceptually) ---
# sigma_clip = SigmaClip(sigma=3.0)
# bkg_estimator = MedianBackground()
# bkg = Background2D(image_data, (10, 10), filter_size=(3, 3), # Box size, filter size
#                    sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
# background_map = bkg.background
# background_rms = bkg.background_rms
# image_bkg_subtracted = image_data - background_map
print("\nConceptual Background Estimation (e.g., Background2D):")
print("  - Creates 2D background map based on sigma-clipped stats in boxes.")
# Simulate subtraction for plotting
background_map_sim = 50 + 0.1 * x_coords + 0.1 * y_coords 
image_bkg_subtracted = image_data - background_map_sim

# --- Step 2: Detect Sources (using DAOStarFinder conceptually) ---
# mean, median, std = sigma_clipped_stats(image_bkg_subtracted, sigma=3.0)
# threshold = median + (5.0 * std) # 5-sigma threshold above background
# daofind = DAOStarFinder(fwhm=3.0, threshold=threshold) # Assume FWHM ~ 3 pixels
# sources_table = daofind(image_bkg_subtracted) # Returns astropy Table
print("\nConceptual Source Detection (e.g., DAOStarFinder):")
print("  - Finds sources above threshold in background-subtracted image.")
# Simulate finding the two sources
sources_found = [(src1_pos[0], src1_pos[1]), (src2_pos[0], src2_pos[1])]
print(f"  Simulated source positions found: {sources_found}")

# --- Step 3: Perform Aperture Photometry ---
# Define apertures
# source_positions = np.transpose((sources_table['xcentroid'], sources_table['ycentroid']))
source_positions = sources_found
aperture_radius = 4.0
source_apertures = CircularAperture(source_positions, r=aperture_radius)
# Define background annulus (optional if background already subtracted)
# annulus_apertures = CircularAnnulus(source_positions, r_in=6.0, r_out=8.0)
print(f"\nConceptual Aperture Photometry (radius={aperture_radius}):")
# Perform photometry on the background-subtracted image
# phot_table = aperture_photometry(image_bkg_subtracted, source_apertures, 
#                                  error=np.sqrt(background_rms**2 + image_data/gain), # Example error
#                                 ) 
print("  - Sums flux within circular apertures on background-subtracted image.")
# Simulate getting flux sums
flux_src1_ap = np.sum(image_bkg_subtracted[dist1_sq < aperture_radius**2])
flux_src2_ap = np.sum(image_bkg_subtracted[dist2_sq < aperture_radius**2])
print(f"  Simulated aperture fluxes: Src1={flux_src1_ap:.1f}, Src2={flux_src2_ap:.1f}")

# --- Visualize ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].imshow(image_data, origin='lower', cmap='viridis', vmax=np.percentile(image_data, 99))
axes[0].set_title("Original Image")
axes[1].imshow(background_map_sim, origin='lower', cmap='viridis')
axes[1].set_title("Estimated Background")
im2=axes[2].imshow(image_bkg_subtracted, origin='lower', cmap='viridis', vmax=np.percentile(image_bkg_subtracted, 99))
source_apertures.plot(axes=axes[2], color='red', lw=1.0) # Plot apertures
axes[2].set_title("Background Subtracted + Apertures")
fig.colorbar(im2, ax=axes[2], label="Flux")
for ax in axes: ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
# plt.show()
plt.close(fig)

print("-" * 20)

# Explanation: This conceptual code illustrates the workflow for aperture photometry.
# 1. Simulates an IR image with background gradient, sources, and noise.
# 2. Conceptually outlines background estimation using `photutils.background.Background2D` 
#    and subtracts a simulated background map.
# 3. Conceptually outlines source detection using `photutils.detection.DAOStarFinder` 
#    on the background-subtracted image. Uses simulated source positions.
# 4. Defines circular apertures around the source positions using `photutils.aperture.CircularAperture`.
# 5. Conceptually describes calling `photutils.aperture.aperture_photometry` on the 
#    background-subtracted image with these apertures to get fluxes. Simulates the flux sum.
# 6. Visualizes the original image, estimated background, and background-subtracted 
#    image with apertures overlaid. This workflow is standard for basic source photometry.
```

**53.5 Spectral Energy Distribution (SED) Fitting (incl. Dust)**

A powerful technique for understanding the physical properties of astrophysical objects, particularly galaxies and young stellar objects, is **Spectral Energy Distribution (SED) fitting**. An SED plots the energy output (flux density or luminosity) of an object as a function of wavelength or frequency, typically compiled from photometric measurements across a wide range of bands, often including optical, near-infrared, mid-infrared, far-infrared, and sometimes sub-millimeter/radio data. The shape of the SED encodes crucial information about the object's constituent components and physical conditions.

For galaxies, the SED shape reveals contributions from:
*   **Stellar Populations:** Emission peaks in the UV/optical/near-IR, depending on the ages, metallicities, and initial mass function (IMF) of the stars. The shape constrains the galaxy's stellar mass (M<0xE2><0x82><0x9B>) and star formation history (SFH).
*   **Interstellar Dust:** Dust grains absorb shorter-wavelength (UV/optical) starlight and re-emit this energy thermally at longer infrared wavelengths. This thermal dust emission peaks typically in the mid-to-far infrared (tens to hundreds of microns), depending on dust temperature and quantity. The amount of IR emission relative to the UV/optical provides information on dust obscuration, total infrared luminosity (L<0xE1><0xB5><0xA2><0xE1><0xB5><0xA3>), and often correlates with the star formation rate (SFR). Dust emission models often involve modified blackbodies (Planck function multiplied by emissivity κ<0xE1><0xB5><0x88> ∝ ν<0xE1><0xB5><0xAE>) or more complex templates including contributions from polycyclic aromatic hydrocarbons (PAHs).
*   **Active Galactic Nuclei (AGN):** Accretion onto a central supermassive black hole can produce strong emission across the spectrum, including a characteristic "torus" emission component peaking in the mid-infrared due to heated dust.

**SED fitting** involves comparing the observed photometric data points (flux densities F<0xE1><0xB5><0x92><0xE1><0xB5><0x87><0xE2><0x82><0x9B>ᵢ at wavelengths λᵢ) to theoretical or empirical **template models** for the SED. The goal is to find the combination of model components (e.g., stellar populations, dust components, AGN) and their parameters (e.g., stellar mass, age, metallicity, dust mass, dust temperature, AGN fraction) that best reproduces the observed data points, usually by minimizing a Chi-squared statistic or maximizing a likelihood function.

Computational approaches to SED fitting include:
*   **Template Fitting:** Comparing the observed photometry to large libraries of pre-computed template SEDs (e.g., from SPS models with different SFHs, metallicities, dust attenuation levels) and finding the best match using Chi-squared minimization. Simple and fast but relies on the completeness and appropriateness of the template library.
*   **Energy Balance Modeling:** More sophisticated methods attempt to model the stellar populations and dust components self-consistently, ensuring energy balance (energy absorbed by dust in UV/optical equals energy re-emitted in IR). This often involves:
    *   Using SPS models to generate intrinsic stellar SEDs based on assumed SFHs and metallicities.
    *   Applying a dust attenuation model (e.g., Calzetti law) to the stellar SED.
    *   Modeling the dust emission using physically motivated templates (e.g., Draine & Li models) or simplified modified blackbodies.
    *   Ensuring energy absorbed = energy emitted.
    *   Fitting the combined model (attenuated stars + dust emission) to the observed photometry using statistical inference methods (MLE or Bayesian MCMC/nested sampling) to constrain parameters like M<0xE2><0x82><0x9B>, SFR, dust mass, attenuation level, etc.

Infrared data, particularly from Spitzer, Herschel, and WISE, are absolutely crucial for SED fitting because they directly probe the re-processed starlight emitted by dust. Without IR data, estimates of total SFR (from L<0xE1><0xB5><0xA2><0xE1><0xB5><0xA3>) and dust properties are impossible, and estimates of stellar mass can be significantly biased by unaccounted-for dust attenuation. Including mid-IR and far-IR photometry provides essential constraints on the dust component and energy balance.

Several powerful Python packages facilitate complex SED fitting:
*   **CIGALE:** ([https://cigale.lam.fr/](https://cigale.lam.fr/)) A popular code performing energy-balance SED fitting using large pre-computed libraries of SPS and dust models, employing a Bayesian-like likelihood analysis. Configured via parameter files.
*   **Prospector:** ([https://prospect.readthedocs.io/](https://prospect.readthedocs.io/)) A flexible Bayesian SED fitting code using MCMC (via `emcee`) or nested sampling (`dynesty`) coupled with the `python-fsps` SPS library, allowing for complex SFHs and detailed modeling of stellar populations and dust.
*   **BAGPIPES:** ([https://bagpipes.readthedocs.io/](https://bagpipes.readthedocs.io/)) Another Bayesian SED fitting code designed for flexible SFH modeling and integration with MCMC/nested sampling.
*   `astropy.modeling`: Can be used to build simpler composite SED models (e.g., blackbody + power law + modified blackbody) for fitting with standard fitters, but lacks the sophistication of dedicated energy-balance codes.

Performing SED fitting involves gathering multi-wavelength photometry (UV, optical, near-IR, mid-IR, far-IR), choosing an appropriate fitting code and model library/parameterization, running the fit (which can be computationally intensive, especially for Bayesian methods), and carefully interpreting the resulting parameter constraints (e.g., posterior distributions, degeneracies visible in corner plots). Infrared data is a vital ingredient for obtaining robust estimates of key galaxy properties like SFR and stellar mass from SED fitting.

**(No simple code example can capture the complexity of full SED fitting; it relies on using one of the dedicated packages mentioned above.)**

**53.6 IR Spectroscopy: Feature Analysis, Line Mapping (JWST)**

Infrared spectroscopy provides powerful diagnostics of physical conditions, chemical composition, and kinematics in ways complementary to imaging or SED fitting. The near- and mid-infrared spectral region (roughly 1-30 μm), now accessible with unprecedented sensitivity and resolution by the James Webb Space Telescope (JWST) instruments (NIRSpec, MIRI), is particularly rich in features tracing gas, dust, and ice.

Key spectral features observable in the IR include:
*   **Atomic Recombination Lines:** Lines from hydrogen (e.g., Paschen series, Brackett series) and helium trace ionized gas (HII regions) associated with young, massive stars. Their fluxes can measure star formation rates, and their ratios can probe extinction.
*   **Molecular Hydrogen (H₂) Lines:** Rotational and ro-vibrational lines of H₂ (particularly in the near-IR around 2 μm) trace warm molecular gas (T ~ 100s-1000s K), often excited by shocks (e.g., in outflows or cloud collisions) or UV fluorescence near star-forming regions.
*   **Other Molecular Lines:** Rotational lines of CO, H₂O, HCN, etc., particularly in the mid-to-far IR, trace the density, temperature, and kinematics of dense molecular clouds where stars form. ALMA dominates at longer sub-mm wavelengths for many of these.
*   **Fine Structure Lines:** Emission lines from ions like [NeII] 12.8μm, [NeIII] 15.6μm, [SIII] 18.7/33.5μm, [OIV] 25.9μm trace highly ionized gas (AGN, intense star formation) and can be used to measure metallicity and ionization state.
*   **Polycyclic Aromatic Hydrocarbon (PAH) Features:** Broad emission features (e.g., at 3.3, 6.2, 7.7, 8.6, 11.3, 12.7 μm) attributed to vibrational modes of large organic molecules (PAHs) excited by UV photons in photodissociation regions (PDRs). They are prominent in star-forming galaxies and the ISM.
*   **Dust Continuum Emission:** Broad thermal emission from dust grains, peaking in the mid-to-far IR, provides information on dust temperature and mass.
*   **Solid-State (Dust/Ice) Features:** Absorption or emission features from dust grains (e.g., silicates near 9.7 and 18 μm) or ice mantles on dust grains (e.g., H₂O ice at 3.0 μm, CO₂ ice at 4.27 μm, CO ice at 4.67 μm) provide unique probes of the composition and processing of solids in dense clouds, protostellar envelopes, and protoplanetary disks.

Analyzing IR spectra involves several computational steps:
*   **Data Reduction:** Processing raw detector data (e.g., from JWST IFUs) using instrument pipelines (`jwst` pipeline) to produce calibrated spectra or spectral cubes, including wavelength calibration, flat-fielding, background subtraction, and flux calibration.
*   **Continuum Fitting/Subtraction:** Modeling and potentially subtracting the underlying stellar or dust continuum emission to isolate spectral lines or features.
*   **Line Fitting:** Fitting profiles (often Gaussian or Voigt) to emission or absorption lines to measure their flux, central wavelength (velocity/redshift), and width (velocity dispersion). `astropy.modeling` and `specutils` (Sec 54.6) provide tools.
*   **Feature Analysis:** Measuring the strength of broad features like PAHs or dust/ice absorption bands, often by defining specific integration windows or fitting specialized feature models.
*   **Diagnostic Diagrams:** Using ratios of different emission lines (e.g., BPT diagrams extended to IR lines) to diagnose ionization mechanisms (star formation vs. AGN) or physical conditions (density, temperature).
*   **Kinematic Analysis:** Measuring Doppler shifts and line widths across spatial regions (from spectral cubes) to map velocity fields and dispersion, tracing rotation, inflows, outflows, or turbulence. Moment map analysis (App 52.B) using `spectral-cube` is applicable.

The arrival of JWST, with its high sensitivity and spatial/spectral resolution across the near- and mid-infrared provided by instruments like NIRCam (imaging), NIRSpec (multi-object and IFU spectroscopy), MIRI (mid-IR imaging and spectroscopy), and NIRISS (imaging, spectroscopy), has revolutionized infrared spectroscopy. Analyzing the complex 3D spectral cubes produced by NIRSpec and MIRI IFUs heavily relies on Python tools.

Python libraries play a central role:
*   The `jwst` pipeline itself is Python-based, using `astropy` and related packages.
*   `astropy.io.fits`, `astropy.wcs`, `astropy.table`, `astropy.units` are fundamental for data I/O and handling.
*   `specutils` provides core objects and functions for representing and analyzing 1D spectra.
*   `spectral-cube` is essential for handling and analyzing 3D spectral cubes from IFUs.
*   `astropy.modeling` and `scipy.optimize` are used for fitting continuum and line profiles.
*   `photutils` and `regions` might be used for defining extraction regions on IFU data.
*   Specialized tools might exist for analyzing specific features like PAHs or ice bands.

Infrared spectroscopy, particularly with JWST, provides a wealth of diagnostic information about obscured regions, cool gas and dust, and the high-redshift Universe. Computational analysis, heavily reliant on the scientific Python ecosystem, is key to extracting physical insights from these rich datasets, involving tasks from basic line fitting to complex analysis of 3D spectral cubes.

**(Code examples typically involve using `specutils` or `spectral-cube` for analysis, similar to App 52.B but potentially with different line fitting models via `astropy.modeling`.)**

**53.7 Python Tools: Astropy, Photutils, JWST Tools, SED Fitting Codes**

The analysis of infrared astronomical data leverages many of the core scientific Python libraries used across wavelengths, but also relies on specialized tools tailored for IR data characteristics, specific instruments (especially JWST), and common analysis tasks like SED fitting.

**Core Libraries (Astropy Ecosystem):**
*   **`astropy`:** Fundamental for almost all tasks. `astropy.io.fits` reads standard data formats. `astropy.wcs` handles coordinate systems in images and cubes. `astropy.table` manages catalogs. `astropy.units` and `astropy.constants` ensure physical consistency. `astropy.stats` provides robust statistics (sigma clipping). `astropy.convolution` handles PSF effects. `astropy.modeling` defines and fits models (e.g., blackbodies, modified blackbodies, line profiles).
*   **`numpy` & `scipy`:** Provide the underlying array computation, numerical integration, optimization (`curve_fit`, `minimize`), interpolation, and signal processing capabilities.
*   **`matplotlib` & `seaborn`:** Used for visualizing images, spectra, SEDs, parameter distributions, etc. WCSAxes (part of `astropy.visualization`) is helpful for plotting images/cubes with correct sky coordinates.
*   **`photutils`:** An Astropy affiliated package crucial for source detection (`DAOStarFinder`, segmentation), aperture photometry (`aperture_photometry`), PSF photometry (`EPSFBuilder`, fitting classes), and particularly important for IR, robust **background estimation** (`Background2D`).
*   **`regions`:** For defining, manipulating, reading/writing spatial regions (circles, polygons, etc.) used for photometry, spectral extraction, or analysis masks.
*   **`specutils` & `spectral-cube`:** Essential for handling 1D spectra (`Spectrum1D` object) and 3D spectral data cubes (`SpectralCube` object), respectively, providing tools for reading data, unit/wavelength conversion, slicing, dicing, calculating moments, smoothing, and basic fitting.

**Instrument-Specific Tools (especially JWST):** Major space missions often have dedicated software pipelines and analysis tools, frequently built on Python and Astropy.
*   **`jwst` Pipeline:** The official pipeline for processing JWST data from all instruments (NIRCam, NIRSpec, MIRI, NIRISS). It's Python-based and handles stages from raw data calibration to higher-level products like mosaics and extracted spectra. Understanding how to run or interact with stages of this pipeline is often necessary. ([https://jwst-pipeline.readthedocs.io/](https://jwst-pipeline.readthedocs.io/))
*   **`jwst` Data Models:** Defines Python classes representing the complex data structures (including multi-dimensional arrays, metadata, WCS, error/DQ arrays) stored in JWST FITS files, providing a convenient way to access different data components. ([https://jwst-pipeline.readthedocs.io/en/latest/jwst/datamodels/index.html](https://jwst-pipeline.readthedocs.io/en/latest/jwst/datamodels/index.html))
*   **`WebbPSF`:** Generates simulated Point Spread Functions (PSFs) for JWST instruments based on detailed optical models. ([https://webbpsf.readthedocs.io/](https://webbpsf.readthedocs.io/))
*   Similar tools exist for other missions like Spitzer (`cubism` for spectral cubes) or WISE (`wise_tools`).

**Spectral Energy Distribution (SED) Fitting Codes:** As discussed in Sec 53.5, fitting models to multi-wavelength photometry, especially including IR data, requires specialized codes:
*   **`CIGALE`:** ([https://cigale.lam.fr/](https://cigale.lam.fr/)) Popular code using pre-computed modules (SPS, dust emission, AGN) and Bayesian-like analysis. Python interface exists.
*   **`Prospector`:** ([https://prospect.readthedocs.io/](https://prospect.readthedocs.io/)) Flexible Bayesian fitting code using MCMC/nested sampling, `python-fsps` for stellar populations, and customizable dust/AGN models. Python based.
*   **`BAGPIPES`:** ([https://bagpipes.readthedocs.io/](https://bagpipes.readthedocs.io/)) Another Bayesian code focusing on flexible star formation history modeling. Python based.
*   **`EAZY`:** While primarily for photometric redshifts using template fitting, often used to derive basic stellar population properties. Python interfaces exist (`eazy-py`).

**Other Potentially Relevant Tools:**
*   **Radiative Transfer Codes (with Python interfaces):** `RADMC-3D`, `SKIRT`, `Powderday` for modeling dust emission/absorption and generating mock observations including RT effects.
*   **Image Subtraction Libraries:** Packages implementing DIA algorithms might be available (e.g., `sfft`, `properimage`).

The ecosystem for IR data analysis heavily relies on the core Astropy stack, augmented by `photutils` for image analysis, `specutils`/`spectral-cube` for spectroscopy, dedicated instrument pipelines/tools (especially crucial for JWST), and specialized SED fitting codes. Researchers often combine these tools within Python scripts and Jupyter notebooks to build custom analysis workflows tailored to their specific scientific goals and datasets obtained from archives like IRSA and MAST.

---
**Application 53.A: Simple IR SED Fitting with Dust Models**

**(Paragraph 1)** **Objective:** This application demonstrates a simplified approach to Spectral Energy Distribution (SED) fitting, focusing on combining stellar and dust emission components to fit mock infrared photometric data points using Python libraries like `astropy.modeling` and `scipy.optimize`. Reinforces Sec 53.5.

**(Paragraph 2)** **Astrophysical Context:** The SED of dusty star-forming galaxies shows a characteristic dip in the optical/UV due to dust absorption and a corresponding peak in the mid-to-far infrared due to thermal re-emission by heated dust grains. Modeling both components simultaneously allows astronomers to estimate intrinsic stellar properties (like stellar mass) corrected for attenuation, and properties of the dust component (like dust mass, temperature, and by proxy, the obscured star formation rate). This requires fitting models across a wide wavelength range, crucially including IR data points.

**(Paragraph 3)** **Data Source/Model:**
    *   **Simulated Data:** Generate mock photometric data points (`wavelengths`, `fluxes`, `flux_errors`) covering optical to far-IR wavelengths (e.g., ugriz, JHK, WISE W1-W4, Herschel PACS/SPIRE bands). Simulate these data based on a known underlying SED model plus realistic noise.
    *   **Model:** A composite model built using `astropy.modeling`.
        *   Stellar Component: A simple representation like a `BlackBody1D` model (parameter: Temperature T_star) or ideally, loading a template spectrum (e.g., from an SPS library like FSPS, conceptually). Needs a normalization parameter related to stellar mass.
        *   Dust Component: A `ModifiedBlackBody1D` model (parameters: Temperature T_dust, emissivity index β (often fixed), normalization related to dust mass).
        *   Total Model: `model = stellar_norm * StellarModel() + dust_norm * DustModel()`. Parameters to fit: `stellar_norm`, `T_star` (or fixed), `dust_norm`, `T_dust`.

**(Paragraph 4)** **Modules Used:** `astropy.modeling` (for defining `BlackBody1D`, `ModifiedBlackBody1D`, `CompoundModel`), `astropy.units` (for wavelengths, fluxes, temperatures), `astropy.constants`, `numpy`, `scipy.optimize.curve_fit` (or `astropy.modeling.fitting`), `matplotlib.pyplot`.

**(Paragraph 5)** **Technique Focus:** Building composite models with `astropy.modeling`. Defining model components with parameters. Simulating data based on a "true" model. Using `curve_fit` (or `LevMarLSQFitter`) to fit the composite model to the noisy data points, providing initial parameter guesses and data uncertainties (`sigma`). Extracting best-fit parameters and potentially their uncertainties from the covariance matrix. Plotting the data, individual model components, and the total best-fit model.

**(Paragraph 6)** **Processing Step 1: Simulate Photometric Data:** Define wavelengths corresponding to common filters (optical to far-IR). Define "true" parameters for a stellar blackbody (T_star, norm_star) and a modified blackbody dust component (T_dust, beta_dust, norm_dust). Create the true composite model using `astropy.modeling`. Evaluate the true model at the filter wavelengths. Add realistic Gaussian noise based on assumed flux errors to generate `observed_fluxes` and `flux_errors`.

**(Paragraph 7)** **Processing Step 2: Define Fit Model:** Create the `astropy.modeling` composite model to be fitted (e.g., `fit_model = N_star * BlackBody1D(temperature=T_star_guess) + N_dust * ModifiedBlackBody1D(temperature=T_dust_guess, beta=beta_fixed)`). Define `N_star`, `T_star`, `N_dust`, `T_dust` as parameters, possibly fixing `beta`. Provide initial guesses (`p0`) for the parameters.

**(Paragraph 8)** **Processing Step 3: Perform Fit:** Use `scipy.optimize.curve_fit`. Define a wrapper function that takes wavelength and fit parameters (`N_star, T_star, N_dust, T_dust`) as input and returns the `fit_model` evaluated at those wavelengths. Call `popt, pcov = curve_fit(wrapper_func, wavelengths, observed_fluxes, p0=initial_guesses, sigma=flux_errors, absolute_sigma=True)`. `popt` contains best-fit parameters, `pcov` the covariance matrix. (Alternatively use `astropy.modeling.fitting.LevMarLSQFitter`).

**(Paragraph 9)** **Processing Step 4: Analyze Results:** Extract best-fit parameter values `popt` and their 1-sigma uncertainties `perr = np.sqrt(np.diag(pcov))`. Print the best-fit parameters and errors, comparing them to the "true" values used for simulation.

**(Paragraph 10)** **Processing Step 5: Visualize Fit:** Plot the simulated photometric data points with error bars. Overplot the best-fit total model curve evaluated over a wide wavelength range. Optionally, also plot the individual best-fit stellar and dust components separately to show their relative contributions. Use log-log scales for the SED plot.

**Output, Testing, and Extension:** Output includes the best-fit parameter values and uncertainties, and the plot showing the data and the fitted SED model. **Testing:** Verify the fitter converges. Check if the best-fit parameters are close to the true values used in the simulation. Examine the plot to see if the model provides a good visual fit to the data points. Calculate reduced Chi-squared for goodness-of-fit. **Extensions:** (1) Use a more realistic stellar template from an SPS library instead of a blackbody. (2) Include dust attenuation affecting the stellar component, potentially linking the absorbed energy to the emitted dust luminosity (energy balance, though hard without dedicated codes). (3) Add an AGN torus component (e.g., another modified blackbody or template) to the model. (4) Use Bayesian fitting with `emcee` or `dynesty` instead of `curve_fit` to get full posterior distributions and evidence values. (5) Apply the fitting procedure to real archival multi-wavelength galaxy photometry.

```python
# --- Code Example: Application 53.A ---
# Note: Simplified SED fitting using astropy.modeling. Requires astropy>=5.0 for ModifiedBlackBody1D.
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from astropy import units as u
from astropy import constants as const
from scipy.optimize import curve_fit 

print("Simple IR SED Fitting (Stellar + Dust Blackbodies):")

# Step 1: Simulate Photometric Data
# Define filter wavelengths (log scale often useful for SEDs)
lambda_um = np.logspace(np.log10(0.4), np.log10(1000), 15) * u.micron 
freq_hz = lambda_um.to(u.Hz, equivalencies=u.spectral())

# True parameters (Example)
T_star_true = 5000 * u.K
T_dust_true = 35 * u.K
beta_dust = 1.8 # Fixed emissivity index
# Normalizations (arbitrary units for shape fitting)
norm_star = 1e-12 
norm_dust = 5e-10 

# Create true model components
star_bb_true = models.BlackBody(temperature=T_star_true, scale=norm_star)
# Note: scale is tricky, better to fit normalization parameter later
# Use ModifiedBlackBody1D (requires Astropy >= 5.0)
# dust_modbb_true = models.ModifiedBlackBody1D(temperature=T_dust_true, beta=beta_dust, scale=norm_dust)
# Simplified: Use normal blackbody for dust conceptually for broader compatibility
dust_bb_true = models.BlackBody(temperature=T_dust_true, scale=norm_dust)

# Evaluate true flux density (F_nu) at filter frequencies
flux_star_true = star_bb_true(freq_hz) 
flux_dust_true = dust_bb_true(freq_hz) # Replace with ModifiedBlackBody if using
flux_total_true = flux_star_true + flux_dust_true

# Add noise (e.g., 10% relative error)
flux_errors = flux_total_true * 0.10
noise = np.random.normal(0.0, flux_errors.value) * flux_errors.unit
observed_fluxes = flux_total_true + noise
print(f"\nGenerated {len(lambda_um)} mock photometric points.")

# Step 2: Define Fit Model
# Fit Normalizations and Temperatures (Fix beta)
# Use CompoundModel or direct addition if models support it
# Using curve_fit requires a function wrapper
def sed_model_func(frequency_hz, norm_s, temp_s, norm_d, temp_d):
    """Wrapper function for curve_fit combining stellar and dust blackbodies."""
    # Ensure inputs are frequencies
    freq = frequency_hz * u.Hz
    # Create models with current parameters
    star_bb = models.BlackBody(temperature=temp_s * u.K)
    dust_bb = models.BlackBody(temperature=temp_d * u.K) # Use ModifiedBlackBody1D if desired
    # Evaluate and sum (need to handle units carefully, return values)
    # Note: Fitting normalization requires care with units, often fit log(Norm)
    # Simplified: Assume norms are multiplicative factors on unit blackbodies
    flux_s = norm_s * star_bb(freq) / star_bb(freq).max() # Normalize component maybe?
    flux_d = norm_d * dust_bb(freq) / dust_bb(freq).max()
    # This simple normalization is likely incorrect for real fitting!
    # Let's fit log(Norm) instead and use scale in BlackBody?
    # Proper SED fitting packages handle this better.
    # Sticking to simple conceptual fit:
    f_star = models.BlackBody(temperature=temp_s*u.K)(freq).value
    f_dust = models.BlackBody(temperature=temp_d*u.K)(freq).value
    return norm_s * f_star + norm_d * f_dust

# Step 3: Perform Fit using curve_fit
initial_guesses = [norm_star*0.8, T_star_true.value*0.9, 
                   norm_dust*1.2, T_dust_true.value*1.1] # Norm_s, Temp_s, Norm_d, Temp_d
print(f"\nFitting model with initial guesses: {initial_guesses}")

try:
    popt, pcov = curve_fit(sed_model_func, freq_hz.value, observed_fluxes.value,
                           p0=initial_guesses, sigma=flux_errors.value,
                           absolute_sigma=True, maxfev=5000)
    fit_successful = True
except Exception as e_fit:
    print(f"  Curve_fit failed: {e_fit}")
    popt, pcov = initial_guesses, None # Use initial guesses if fit fails
    fit_successful = False

# Step 4: Analyze Results
if fit_successful:
    print("\nFit converged.")
    param_names = ['Norm_Star', 'Temp_Star (K)', 'Norm_Dust', 'Temp_Dust (K)']
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros_like(popt)
    print("Best-fit parameters (Value +/- Error):")
    for name, val, err in zip(param_names, popt, perr):
        print(f"  {name:<15}: {val:.3e} +/- {err:.2e}")
else:
    print("\nFit did not converge, using initial guesses.")
    popt = initial_guesses # Use initial guesses if fit failed

# Step 5: Visualize Fit
print("Generating SED plot...")
fig, ax = plt.subplots(figsize=(8, 6))

# Plot data points
ax.errorbar(lambda_um.value, observed_fluxes.value, yerr=flux_errors.value,
            fmt='o', color='black', label='Simulated Data')

# Plot best-fit model components and total
plot_lam_um = np.logspace(np.log10(0.1), np.log10(2000), 200) * u.micron
plot_freq_hz = plot_lam_um.to(u.Hz, equivalencies=u.spectral())

fit_norm_s, fit_temp_s, fit_norm_d, fit_temp_d = popt
flux_fit_star = fit_norm_s * models.BlackBody(temperature=fit_temp_s*u.K)(plot_freq_hz).value
flux_fit_dust = fit_norm_d * models.BlackBody(temperature=fit_temp_d*u.K)(plot_freq_hz).value
flux_fit_total = flux_fit_star + flux_fit_dust

ax.loglog(plot_lam_um.value, flux_fit_total, 'r-', label='Total Fit')
ax.loglog(plot_lam_um.value, flux_fit_star, 'b:', label='Stellar Component (Fit)')
ax.loglog(plot_lam_um.value, flux_fit_dust, 'g--', label='Dust Component (Fit)')

ax.set_xlabel("Wavelength (micron)")
ax.set_ylabel("Flux Density (Arbitrary Units)")
ax.set_title("Simple SED Fit (Stellar + Dust Blackbodies)")
ax.set_ylim(bottom=observed_fluxes.value.min()*0.1, top=observed_fluxes.value.max()*5) # Adjust Y limits
ax.set_xlim(lambda_um.value.min()*0.5, lambda_um.value.max()*2)
ax.legend()
ax.grid(True, which='both', alpha=0.4)
fig.tight_layout()
# plt.show()
print("Plot generated.")
plt.close(fig)

print("-" * 20)
```

**Application 53.B: Background Estimation for Simulated JWST/MIRI Image**

**(Paragraph 1)** **Objective:** Apply and compare different background estimation techniques available in the `photutils` library (Sec 53.7) to a simulated JWST Mid-Infrared Instrument (MIRI) image patch, assessing their ability to model potentially complex or spatially varying IR backgrounds in the presence of faint sources and noise. Reinforces Sec 53.2, 53.4.

**(Paragraph 2)** **Astrophysical Context:** MIRI observes at mid-infrared wavelengths (5-28 μm) where thermal background emission from the zodiacal cloud and the telescope itself can be significant and structured. Accurate subtraction of this background is essential for detecting and measuring faint sources, such as high-redshift galaxies or faint protostars. Standard techniques involve estimating the 2D background variation across the image, often using methods robust against outliers (real sources).

**(Paragraph 3)** **Data Source/Model:** A simulated 2D NumPy array representing a small region of a MIRI image. This simulated image should include:
    *   A spatially varying background (e.g., combining a constant level + gradient + perhaps smoother large-scale fluctuations).
    *   Several faint, PSF-convolved point sources or small galaxies.
    *   Realistic noise (Poisson noise from background/source + read noise).
    Alternatively, a cutout from actual public MIRI data could be used.

**(Paragraph 4)** **Modules Used:** `numpy` (for simulation/arrays), `matplotlib.pyplot` (for visualization), `photutils.background` (for `Background2D`, estimators like `MedianBackground`, `MMMBackground`), `astropy.stats` (for `SigmaClip`), `astropy.visualization` (for display scaling).

**(Paragraph 5)** **Technique Focus:** Using `photutils` for 2D background estimation. (1) Simulating a realistic MIRI image patch with background structure and sources. (2) Creating instances of different background estimators provided by `photutils` (e.g., `MedianBackground`, `ModeEstimatorBackground`, `MMMBackground`). (3) Creating `SigmaClip` instances for outlier rejection during estimation. (4) Using `Background2D` to compute the background map and background RMS map, potentially experimenting with different box sizes (`box_size`) and filter sizes (`filter_size`) for the estimation grid, and different estimators/clipping. (5) Visualizing the original image, the estimated background map, the background RMS map, and the final background-subtracted image for different methods. (6) Qualitatively comparing how well each method models the background without biasing by sources.

**(Paragraph 6)** **Processing Step 1: Simulate Image:** Create `sim_image` array (e.g., 200x200 pixels). Add background gradient/structure. Add faint Gaussian sources using `astropy.modeling.models.Gaussian2D`. Add Poisson noise `np.random.poisson(image_ideal)` and read noise `np.random.normal(0, read_noise_level)`.

**(Paragraph 7)** **Processing Step 2: Setup Background Estimation:** Import necessary modules. Define `SigmaClip` instance (e.g., `sigma_clip = SigmaClip(sigma=3.0, maxiters=5)`). Define box size and filter size for `Background2D` (e.g., `box_size = (20, 20)`, `filter_size = (3, 3)`).

**(Paragraph 8)** **Processing Step 3: Compute Background (Method 1 - Median):** Create estimator `bkg_estimator_med = MedianBackground()`. Create `bkg_med = Background2D(sim_image, box_size, filter_size=filter_size, sigma_clip=sigma_clip, bkg_estimator=bkg_estimator_med)`. Access `bkg_med.background` and `bkg_med.background_rms`.

**(Paragraph 9)** **Processing Step 4: Compute Background (Method 2 - MMM):** Create estimator `bkg_estimator_mmm = MMMBackground()`. Create `bkg_mmm = Background2D(sim_image, box_size, filter_size=filter_size, sigma_clip=sigma_clip, bkg_estimator=bkg_estimator_mmm)`. Access `bkg_mmm.background` and `bkg_mmm.background_rms`.

**(Paragraph 10)** **Processing Step 5: Visualize and Compare:** Create a multi-panel plot using `matplotlib.pyplot`. Show: (a) Original simulated image. (b) Background map from Median method. (c) Background map from MMM method. (d) Background-subtracted image (Original - Median Background). (e) Background-subtracted image (Original - MMM Background). (f) Background RMS map (from either method). Use consistent scaling (e.g., `simple_norm`) to compare visually how well each method captured the background structure and subtracted it, especially around the faint sources.

**Output, Testing, and Extension:** Output consists of the generated plots comparing the different background estimation methods. **Testing:** Visually inspect the background maps - do they capture the simulated gradient/structure without being overly influenced by the sources? Visually inspect the background-subtracted images - are the sources clearly visible with a relatively flat residual background? Check the RMS map for reasonableness. **Extensions:** (1) Try different `box_size` and `filter_size` values in `Background2D` and see how they affect the result (trade-off between smoothness and ability to follow small-scale variations). (2) Use source masking: detect sources first (e.g., using `photutils.segmentation.detect_sources`) and provide the mask to `Background2D` via the `mask` argument to explicitly exclude source pixels from the background estimation. (3) Implement background fitting using `astropy.modeling` with 2D polynomial or spline models combined with robust fitters. (4) Apply these techniques to a real MIRI image cutout downloaded from MAST.

```python
# --- Code Example: Application 53.B ---
# Note: Requires photutils, astropy, matplotlib, numpy
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip
from astropy.visualization import simple_norm
try:
    from photutils.background import Background2D, MedianBackground, MMMBackground
    photutils_ok = True
except ImportError:
     photutils_ok = False
     print("NOTE: photutils not installed. Skipping application.")

print("Comparing Background Estimation Methods for Simulated IR Image:")

# Step 1: Simulate Image
npix_x, npix_y = 150, 200
yy, xx = np.indices((npix_y, npix_x))
# Background: Constant + Gradient + gentle wave
background = 80.0 + 0.15 * xx + 0.05 * yy + 15 * np.sin(xx / 30.0) * np.cos(yy / 40.0)
# Sources: Faint Gaussians
sources = np.zeros_like(background)
pos_amp_sig = [ (50, 30, 50, 2.5), (100, 120, 30, 3.0), (70, 150, 40, 2.8) ]
for x0, y0, amp, sig in pos_amp_sig:
    dist_sq = (xx - x0)**2 + (yy - y0)**2
    sources += amp * np.exp(-dist_sq / (2 * sig**2))
image_ideal = background + sources
# Noise: Poisson + Gaussian read noise
gain = 1.0 # electrons/ADU (example)
read_noise = 5.0 # electrons
image_noisy = np.random.poisson(image_ideal * gain) / gain
image_noisy += np.random.normal(0, read_noise / gain, size=image_ideal.shape)
print(f"\nGenerated simulated image ({npix_y}x{npix_x}) with background and sources.")

if photutils_ok:
    # Step 2: Setup Background Estimation
    sigma_clip = SigmaClip(sigma=3.0, maxiters=5)
    box_size = (15, 15) # Size of boxes for stats
    filter_size = (3, 3) # Median filter size for smoothing bkg map
    print(f"\nUsing sigma_clip={sigma_clip}, box_size={box_size}, filter_size={filter_size}")

    # Step 3: Compute Background (Method 1 - Median)
    print("Calculating Median Background...")
    start_med = time.time()
    bkg_estimator_med = MedianBackground()
    try:
        bkg_med = Background2D(image_noisy, box_size, filter_size=filter_size, 
                               sigma_clip=sigma_clip, bkg_estimator=bkg_estimator_med)
        med_bkg_map = bkg_med.background
        med_rms_map = bkg_med.background_rms
        med_sub_image = image_noisy - med_bkg_map
        print(f"  Median Background calculation time: {time.time() - start_med:.3f}s")
        med_ok = True
    except Exception as e_med:
         print(f"  MedianBackground failed: {e_med}")
         med_ok = False

    # Step 4: Compute Background (Method 2 - MMM)
    print("\nCalculating MMM Background...")
    start_mmm = time.time()
    bkg_estimator_mmm = MMMBackground() # Mean, Median, Mode estimator
    try:
        bkg_mmm = Background2D(image_noisy, box_size, filter_size=filter_size, 
                               sigma_clip=sigma_clip, bkg_estimator=bkg_estimator_mmm)
        mmm_bkg_map = bkg_mmm.background
        mmm_rms_map = bkg_mmm.background_rms # Often similar to median one
        mmm_sub_image = image_noisy - mmm_bkg_map
        print(f"  MMM Background calculation time: {time.time() - start_mmm:.3f}s")
        mmm_ok = True
    except Exception as e_mmm:
         print(f"  MMMBackground failed: {e_mmm}")
         mmm_ok = False

    # Step 5: Visualize and Compare
    print("\nGenerating comparison plots...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel() # Flatten axes array

    # Use robust normalization across panels
    norm = simple_norm(image_noisy, stretch='asinh', percent=99)
    norm_sub = simple_norm(med_sub_image if med_ok else mmm_sub_image, stretch='asinh', percent=99)
    norm_bkg = simple_norm(med_bkg_map if med_ok else mmm_bkg_map, stretch='linear', percent=99)
    norm_rms = simple_norm(med_rms_map if med_ok else mmm_rms_map, stretch='linear', percent=99)

    im0 = axes[0].imshow(image_noisy, origin='lower', cmap='viridis', norm=norm)
    axes[0].set_title("Original Noisy Image"); fig.colorbar(im0, ax=axes[0])
    
    if med_ok:
        im1 = axes[1].imshow(med_bkg_map, origin='lower', cmap='viridis', norm=norm_bkg)
        axes[1].set_title("Median Background Map"); fig.colorbar(im1, ax=axes[1])
        im3 = axes[3].imshow(med_sub_image, origin='lower', cmap='viridis', norm=norm_sub)
        axes[3].set_title("Subtracted (Median)"); fig.colorbar(im3, ax=axes[3])
    else: axes[1].set_title("Median Failed"); axes[3].set_title("Median Failed")
        
    if mmm_ok:
        im2 = axes[4].imshow(mmm_bkg_map, origin='lower', cmap='viridis', norm=norm_bkg)
        axes[4].set_title("MMM Background Map"); fig.colorbar(im2, ax=axes[4])
        im5 = axes[5].imshow(mmm_sub_image, origin='lower', cmap='viridis', norm=norm_sub)
        axes[5].set_title("Subtracted (MMM)"); fig.colorbar(im5, ax=axes[5])
    else: axes[4].set_title("MMM Failed"); axes[5].set_title("MMM Failed")
        
    # Show RMS map (from one method)
    if med_ok:
        im_rms = axes[2].imshow(med_rms_map, origin='lower', cmap='plasma', norm=norm_rms)
        axes[2].set_title("Background RMS (Median)"); fig.colorbar(im_rms, ax=axes[2])
    elif mmm_ok:
         im_rms = axes[2].imshow(mmm_rms_map, origin='lower', cmap='plasma', norm=norm_rms)
         axes[2].set_title("Background RMS (MMM)"); fig.colorbar(im_rms, ax=axes[2])
    else: axes[2].set_title("RMS Failed")


    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Comparison of Background Estimation Methods (photutils)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()
    print("Plots generated.")
    plt.close(fig)

else:
    print("Skipping execution as photutils is not installed.")

print("-" * 20)
```

**Chapter 53 Summary**

This chapter surveyed the computational techniques essential for **infrared (IR) astronomy**, a field vital for studying cool objects, dust-obscured regions, and the high-redshift Universe. It contrasted the challenges of **ground-based IR observations** (atmospheric absorption windows, bright thermal background necessitating chopping/nodding) with the advantages of **space-based facilities** like Spitzer, Herschel, WISE, and JWST (full spectral access, high sensitivity due to cryogenic cooling). Standard **IR data reduction** steps were reviewed, including basic calibrations (dark subtraction, flat-fielding, non-linearity correction) and the critical, often complex, process of **background estimation and subtraction**. Correction for specific IR **detector artifacts** like persistence was also mentioned. Common **data formats** (primarily FITS images, cubes, tables) and major **IR archives** (IRSA, MAST, ESA archives) providing access to mission data were identified.

Key IR **analysis techniques** were explored, including **photometry** (aperture and PSF methods, adapted for high backgrounds and crowding), **image subtraction** for transient searches, and the crucial technique of multi-wavelength **Spectral Energy Distribution (SED) fitting**. SED fitting, which models contributions from stellar populations and thermal dust emission (probed directly by IR data), allows robust estimation of parameters like stellar mass, star formation rate, and dust properties, often using sophisticated energy-balance modeling codes. The chapter also covered **IR spectroscopy**, highlighting key spectral features (atomic/molecular lines, PAH bands, solid-state features) used as diagnostics, particularly with the advent of JWST's NIRSpec and MIRI instruments, and the analysis of spectral cubes using tools like `spectral-cube`. Finally, relevant **Python tools** were listed, emphasizing the core `Astropy` ecosystem (`photutils`, `specutils`, `spectral-cube`, `regions`), instrument-specific pipelines (`jwst`), and powerful **SED fitting packages** (`CIGALE`, `Prospector`, `BAGPIPES`). Two applications illustrated fitting a simple stellar+dust SED model to mock photometry and comparing background estimation techniques on a simulated MIRI image using `photutils`.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Rieke, G. H. (2003).** *Detection of Light: From the Lidar to the Hubble Space Telescope* (2nd ed.). Cambridge University Press. [https://doi.org/10.1017/CBO9781139165479](https://doi.org/10.1017/CBO9781139165479)
    *(Provides detailed background on detector physics, including infrared detectors, noise sources, and calibration concepts relevant to Sec 53.1, 53.2.)*

2.  **JWST User Documentation (JDox). (n.d.).** *JWST Documentation*. STScI. Retrieved January 16, 2024, from [https://jwst-docs.stsci.edu/](https://jwst-docs.stsci.edu/)
    *(The primary source for information on JWST instruments (NIRCam, MIRI, NIRSpec, NIRISS), data products, the `jwst` processing pipeline, and analysis tools, essential context for Sec 53.1, 53.2, 53.6, 53.7.)* (Similar documentation exists for Spitzer, Herschel, etc., often via IRSA).

3.  **Photutils Developers. (n.d.).** *Photutils Documentation*. Read the Docs. Retrieved January 16, 2024, from [https://photutils.readthedocs.io/en/stable/](https://photutils.readthedocs.io/en/stable/)
    *(Official documentation for the Astropy-affiliated `photutils` package, covering background estimation (`Background2D`), source detection, aperture photometry, and PSF photometry techniques discussed in Sec 53.4 and Application 53.B.)*

4.  **Conroy, C. (2013).** Modeling the Panchromatic Spectral Energy Distributions of Galaxies. *Annual Review of Astronomy and Astrophysics*, *51*, 393–455. [https://doi.org/10.1146/annurev-astro-082812-141017](https://doi.org/10.1146/annurev-astro-082812-141017)
    *(A comprehensive review of the physics and techniques involved in SED modeling and fitting for galaxies, covering stellar populations and dust emission crucial for Sec 53.5.)*

5.  **Salim, S., & Narayanan, D. (2020).** The Cosmic Baryon Cycle and Galaxy Evolution: An Observer’s Perspective. *Annual Review of Astronomy and Astrophysics*, *58*, 419-461. [https://doi.org/10.1146/annurev-astro-032620-021934](https://doi.org/10.1146/annurev-astro-032620-021934)
    *(Reviews observational probes of galaxy evolution, including extensive discussion of SED fitting techniques and diagnostics derived from multi-wavelength data, providing scientific context for Sec 53.5.)* (See also documentation for specific SED codes like CIGALE, Prospector, BAGPIPES referenced in Sec 53.7).
