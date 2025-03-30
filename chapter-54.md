**Chapter 54: Computational Techniques in Optical/UV Astronomy**

This chapter delves into the computational methods and data analysis techniques fundamental to **optical and ultraviolet (UV) astronomy**, wavelengths that trace stellar populations, ionized gas emission, and hot astrophysical phenomena, but which are also significantly affected by interstellar dust extinction and atmospheric effects (for ground-based observations). We begin by contrasting observations from ground-based optical telescopes, emphasizing the challenges posed by **atmospheric seeing** and transparency variations, with the high spatial resolution and UV access provided by **space-based facilities** like the Hubble Space Telescope (HST) and GALEX. We then detail the standard **image processing steps** crucial for calibrating raw data from optical/UV CCD detectors, including bias and dark subtraction, flat-fielding, cosmic ray removal, fringing correction (especially near-IR/red optical), geometric distortion correction, and establishing accurate **astrometric solutions** (WCS). Methods for **photometric calibration**, determining instrumental zeropoints and color terms to convert instrumental counts into physical magnitudes or fluxes, are discussed. We cover common **data formats** (primarily FITS images and tables) and major **archives** providing optical/UV data (MAST, NOIRLab Astro Data Archive, SDSS, DES, Pan-STARRS, LSST precursors). Key analysis techniques are then explored, focusing on robust **source detection**, accurate **photometry** in both sparse and crowded fields (aperture vs. PSF photometry), and quantitative **morphological analysis** of galaxies and resolved sources. Finally, we briefly introduce typical approaches for analyzing **stellar and nebular spectra** in the optical/UV range (classification, kinematics, line measurements) and highlight essential **Python tools**, particularly the extensive functionalities within `Astropy` (core, `ccdproc`, `photutils`, `specutils`, `modeling`, `wcs`), the `SEP` library (Source Extractor Python binding), and libraries used for fitting stellar population models to optical/UV data.

**54.1 Optical/UV Observations: Ground (Seeing) vs. Space (HST, UV)**

Optical astronomy, covering roughly 350 nm to 1000 nm (1 μm), is the oldest branch and continues to be a cornerstone of astrophysical research, probing stellar photospheres, ionized gas nebulae (like HII regions and planetary nebulae via lines like Hα, Hβ, [OIII], [NII], [SII]), and the integrated light of distant galaxies. Ultraviolet (UV) astronomy (< 350 nm) accesses hotter phenomena, including young massive stars, accretion disks, stellar chromospheres/transition regions, and the hottest phases of the interstellar medium, but is largely inaccessible from the ground due to atmospheric absorption (primarily by ozone below ~300 nm).

**Ground-based optical observations** are performed using a vast array of telescopes, from small university instruments to large 8-10 meter class facilities (VLT, Keck, Gemini, Subaru) and wide-field survey telescopes (ZTF, Pan-STARRS, DES, Rubin/LSST). Their primary limitation is the Earth's atmosphere, which causes:
*   **Seeing:** Turbulence in the atmosphere blurs incoming light, smearing point sources (like stars) into "seeing disks" typically 0.5-2.0 arcseconds across (or worse). This limits the achievable angular resolution, making it difficult to resolve fine details in distant galaxies or separate stars in crowded fields. Adaptive Optics (AO) systems on large telescopes can partially correct for seeing, achieving much sharper images, especially in the near-infrared.
*   **Atmospheric Extinction:** The atmosphere absorbs and scatters starlight, reducing the observed flux, particularly towards the blue/UV end of the optical spectrum and varying with airmass (the path length through the atmosphere, depending on the source's elevation). Accurate photometry requires correcting for this extinction.
*   **Sky Background:** Scattering of moonlight, artificial light pollution, and airglow contribute a significant background signal, limiting the achievable depth for faint source detection. Observing from dark sites is crucial.
*   **Weather and Transparency:** Clouds, humidity, and dust affect atmospheric transparency, which can vary significantly during an observation, complicating photometric calibration.
Despite these challenges, ground-based telescopes provide crucial access to large collecting areas (enabling observation of faint objects), wide fields of view (for surveys), and flexible instrumentation (imagers, spectrographs, polarimeters).

**Space-based optical/UV observations** overcome atmospheric limitations, offering significant advantages:
*   **High Angular Resolution:** Operating above the atmosphere allows telescopes like the **Hubble Space Telescope (HST)** to achieve diffraction-limited resolution (typically ~0.05-0.1 arcseconds in optical/UV), revealing fine details in galaxies, nebulae, and stellar systems impossible to see clearly from the ground without advanced AO.
*   **UV Access:** Space telescopes provide access to the entire ultraviolet spectrum, crucial for studying hot stars, quasars, interstellar absorption lines, and processes obscured at longer wavelengths. Missions like IUE, GALEX, and HST (with UV instruments like ACS/SBC, WFC3/UVIS, COS, STIS) have revolutionized UV astronomy.
*   **Stable Background and PSF:** The space environment offers a much darker, more stable sky background and a predictable, stable Point Spread Function (PSF), facilitating detection of faint sources and precise photometry/morphometry.
However, space missions are vastly more expensive than ground-based telescopes, have limited lifetimes, often smaller apertures (limiting light-gathering power compared to the largest ground telescopes), and typically narrower fields of view.

Both ground- and space-based observations produce raw data, usually from **Charge-Coupled Devices (CCDs)** or similar detectors (like CMOS), that requires significant processing and calibration (Sec 54.2, 54.3) before scientific analysis can be performed. The specific processing steps might differ slightly depending on whether the data is from ground or space (e.g., ground-based data requires more complex atmospheric extinction correction, while space-based data might need specialized handling for cosmic rays or detector artifacts specific to the space environment). Computational techniques and Python tools are essential for performing these calibrations and analyzing the final data products from both types of facilities.

**54.2 Image Processing: Calibration (Bias, Dark, Flat), CR Rejection, WCS**

Raw images obtained from optical or UV CCD detectors suffer from various instrumental signatures and artifacts that must be removed through a series of **calibration steps** before accurate scientific measurements can be made. This process, often termed **data reduction**, is a fundamental computational workflow in observational astronomy. Python libraries like `Astropy` (particularly the `ccdproc` affiliated package) provide tools to perform these steps.

**1. Bias Subtraction:** CCDs have a baseline electronic offset or **bias level** that is present even in zero-second exposures. This pattern needs to be subtracted. Typically, multiple zero-second exposures (**bias frames**) are taken during calibration. These are combined (usually median-combined to reject outliers) into a **master bias frame**. This master bias is then subtracted from all science images (and flat/dark frames). `ccdproc.Combiner` can be used for combining, and simple array subtraction performs the correction.

**2. Dark Subtraction (Optional but recommended for long exposures):** Detectors generate **dark current**, a signal accumulating over time due to thermal effects, even in the absence of light. This dark current can also have a spatial pattern. Multiple long exposures taken with the shutter closed (**dark frames**) matching the science exposure times are combined into a **master dark frame** (after bias subtraction, and often scaled by exposure time). This master dark is then subtracted from the (bias-subtracted) science images. For many modern, cold detectors, dark current might be negligible for short exposures, but crucial for long ones or warmer detectors.

**3. Flat Fielding:** Pixels across the CCD do not have perfectly uniform sensitivity. Additionally, illumination patterns (vignetting) from the telescope optics can cause large-scale variations. **Flat fielding** corrects for these pixel-to-pixel sensitivity variations and illumination effects. Exposures of a uniformly illuminated source (**flat field frames**) are taken, typically through each filter used for science observations. Common sources are twilight sky flats, dome flats (illuminated screen inside the dome), or stellar flats. Multiple raw flat frames per filter are taken, bias-subtracted (and potentially dark-subtracted), combined (median or average) into a **master flat field** for that filter, and then **normalized** (usually by dividing by its median or mean value). The science image (after bias/dark subtraction) is then **divided** by the corresponding normalized master flat field. This corrects the pixel values to what they would be if the detector were perfectly uniform. `ccdproc` provides tools for combining and processing calibration frames.

**4. Cosmic Ray Rejection:** Energetic cosmic rays hitting the detector can create sharp, localized spikes or trails in images, especially in long exposures or space-based data. These need to be identified and masked or removed. Algorithms typically look for pixels with values significantly higher than their neighbors. `ccdproc.cosmicray_lacosmic` (based on the LA Cosmic algorithm) or similar functions in `astropy.stats` or `scipy.ndimage` can be used to detect and create masks for cosmic ray hits. Interpolating over masked pixels or using statistical combination during dithering can mitigate their impact.

**5. Fringing Correction (Near-IR/Red Optical):** In images taken at red optical or near-infrared wavelengths with thinned CCDs, interference patterns (**fringing**) caused by internal reflections of night sky emission lines can appear. Removing these often requires specialized techniques, potentially involving constructing a fringe map from science images themselves and subtracting it.

**6. Astrometric Calibration (WCS Fitting):** Raw images usually only have pixel coordinates. To relate pixel positions to sky coordinates (RA, Dec), an **astrometric calibration** or **World Coordinate System (WCS)** solution is needed. This involves identifying known astronomical sources (stars) from reference catalogs (like Gaia, USNO, 2MASS) within the image, measuring their pixel coordinates (e.g., using source detection algorithms), and then fitting a mathematical transformation (defined by the FITS WCS standard, involving projection types like TAN, SIN, and distortion polynomials) that maps pixel coordinates (x, y) to sky coordinates (RA, Dec). This fit determines the WCS parameters (like reference pixel coordinates CRPIX, reference sky coordinates CRVAL, rotation matrix PC or CD, and distortion coefficients) which are then stored in the FITS header. Libraries like `astropy.wcs` handle reading/writing WCS headers and performing the transformations. Tools like `Astrometry.net` provide robust automated WCS solving.

These calibration steps transform raw detector readouts into scientifically usable images with known sky coordinates and corrected instrumental signatures. Automating this pipeline using Python scripts leveraging `ccdproc`, `astropy.io.fits`, `astropy.wcs`, and `numpy` is standard practice for handling data from optical/UV observations. Workflow management systems (Chapter 66) are often used to orchestrate these multi-step calibration pipelines.

**54.3 Photometric Calibration: Zeropoints, Color Terms, Extinction**

After basic image processing (bias subtraction, flat fielding, etc.), the pixel values in an optical/UV image are typically in instrumental units (like counts or Analog-to-Digital Units - ADU) proportional to the incident flux. **Photometric calibration** is the process of converting these instrumental measurements into physically meaningful units, usually standard astronomical **magnitudes** (like AB or Vega magnitudes) or physical **fluxes** (like erg/s/cm²/Å or Janskys). This requires determining the relationship between instrumental counts and standard magnitudes/fluxes for each filter used.

The fundamental relationship often used is:
m<0xE2><0x82><0x9B><0xE1><0xB5><0x97<0xE2><0x82><0x8A> = m<0xE2><0x82><0x9E><0xE2><0x82><0x99><0xE2><0x82><0x9B>ₜ<0xE1><0xB5><0xA3><0xE1><0xB5><0x98> - 2.5 * log₁₀(Counts / t<0xE1><0xB5><0x8A><0xE1><0xB5><0x99><0xE1><0xB5><0x96>) + ZP + CT * (Color) - k * (Airmass)
where:
*   m<0xE2><0x82><0x9B><0xE1><0xB5><0x97<0xE2><0x82><0x8A> is the standard apparent magnitude in a given band (e.g., V band).
*   m<0xE2><0x82><0x9E><0xE2><0x82><0x99><0xE2><0x82><0x9B>ₜ<0xE1><0xB5><0xA3><0xE1><0xB5><0x98> = -2.5 * log₁₀(Counts / t<0xE1><0xB5><0x8A><0xE1><0xB5><0x99><0xE1><0xB5><0x96>) is the instrumental magnitude, derived from the measured counts (from aperture or PSF photometry, Sec 54.5) and exposure time t<0xE1><0xB5><0x8A><0xE1><0xB5><0x99><0xE1><0xB5><0x96>.
*   **ZP** is the **photometric zeropoint**, the magnitude corresponding to 1 count per second. This is the primary calibration factor determined for each filter/instrument combination.
*   **CT** is the **color term**. Because filters have finite bandwidths and detector efficiencies vary with wavelength, the relationship between instrumental and standard magnitudes can depend slightly on the object's intrinsic color (e.g., B-V). The color term accounts for this effect. It's often small for standard filter systems closely matching the instrumental system. `Color` is an appropriate color index (e.g., B-V, g-r).
*   **k** is the **atmospheric extinction coefficient** (magnitudes per airmass). It quantifies how much starlight is dimmed when passing through the atmosphere. It depends on wavelength (larger at bluer wavelengths) and atmospheric conditions.
*   **Airmass** is the relative path length through the atmosphere towards the source (approximately sec(z), where z is the zenith angle). Airmass=1 at zenith, increasing towards the horizon. This term is primarily relevant for ground-based observations.

Determining these calibration coefficients (ZP, CT, k) requires observing **standard stars** – stars with precisely known, stable standard magnitudes and colors in the desired photometric system (e.g., Landolt standards, SDSS standard fields). During a photometric night, observers typically measure several standard stars across a range of airmasses and colors in each filter used for science observations.

The calibration process then involves:
1.  Performing instrumental photometry (measuring `Counts / t_exp`) for the observed standard stars.
2.  Calculating their instrumental magnitudes `m_inst`.
3.  Obtaining their known standard magnitudes `m_std` and colors `Color` from catalogs.
4.  Calculating the airmass for each standard star observation based on its time and position.
5.  Performing a **linear least-squares fit** to the equation:
    `m_std - m_inst = ZP + CT * Color - k * Airmass`
    This fit, performed across all standard star observations in a given filter, simultaneously solves for the zeropoint (ZP), color term (CT), and extinction coefficient (k) for that night and filter. `scipy.optimize.curve_fit` or linear regression tools (`LinearRegression`, `astropy.modeling`) can be used.

Once the calibration coefficients (ZP, CT, k) are determined, the calibration equation can be rearranged to convert instrumental magnitudes of science targets `m_inst_sci` (measured in the same way as standards) into standard magnitudes `m_std_sci`. This requires knowing the science target's airmass during observation and estimating its color (which might involve an iterative process if the color itself is derived from the calibrated magnitudes).

For **space-based observations** (like HST), atmospheric extinction (k) is zero, simplifying the equation. Zeropoints (and sometimes time-dependent color terms) are often well-calibrated by the instrument teams and provided directly in the FITS header (e.g., PHOTFLAM, PHOTZPT keywords) or documentation, allowing direct conversion from counts/sec to fluxes or standard magnitudes using provided formulas, although checks using observed standards are still good practice.

Accurate photometric calibration is essential for comparing measurements across different instruments or epochs, constructing SEDs, deriving physical properties based on brightness, and contributing data to large surveys. Python tools are used for performing the photometry on standards, managing standard star catalogs, performing the least-squares fits to determine calibration coefficients, and applying the calibration equation to science targets. Libraries like `astroplan` (Astropy affiliated) can help calculate airmass.

**54.4 Data Formats and Archives (FITS, MAST, SDSS, LSST)**

Optical and ultraviolet astronomical data, both from ground-based and space-based facilities, are overwhelmingly stored and distributed using the **Flexible Image Transport System (FITS)** format (Sec 1.5). FITS provides a standardized way to store not only the primary data (images, tables, spectra) but also crucial metadata in human-readable headers.

**Images:** Calibrated optical/UV images are typically stored as 2D arrays in the primary HDU or image extensions (`IMAGEHDU`) of FITS files. The header contains essential information:
*   **WCS (World Coordinate System):** Keywords (like `CRVAL`, `CRPIX`, `CD` or `PC` matrix, `CTYPE`, distortion parameters like `SIP` or `TPV`) defining the mapping between pixel coordinates (X, Y) and sky coordinates (RA, Dec) (Sec 5.6). `astropy.wcs` reads this information.
*   **Photometric Calibration:** Keywords defining the zeropoint (e.g., `PHOTZP`, `MAGZERO`) and potentially other calibration factors (`PHOTFLAM`, `PHOTPLAM`) needed to convert pixel values (counts or counts/sec) to standard magnitudes or physical fluxes. Units are often specified in `BUNIT`.
*   **Observational Metadata:** Keywords detailing the telescope (`TELESCOP`), instrument (`INSTRUME`), filter (`FILTER`), exposure time (`EXPTIME`), observation date/time (`DATE-OBS`, `MJD-OBS`), airmass (`AIRMASS`), target (`OBJECT`), observer (`OBSERVER`), proposal ID (`PROPID`), etc.
*   **Processing History:** `HISTORY` and `COMMENT` cards often provide valuable information about the calibration steps applied.
Some missions or pipelines might store additional data, like weight maps, data quality masks, or uncertainty images, as separate FITS extensions within the same file.

**Catalogs:** Source lists derived from images (positions, magnitudes, morphologies) or spectroscopic surveys (redshifts, line measurements, classifications) are often stored as binary tables (`BinTableHDU`) within FITS files. Each row corresponds to an object, and columns represent different measured or derived properties. The table header describes the columns (names `TTYPE`, formats `TFORM`, units `TUNIT`, descriptions). `astropy.table.Table.read()` with `format='fits'` is used to read these tables (Sec 2.3). Large surveys might distribute catalogs across multiple FITS files or provide access via database queries (e.g., TAP services, Sec 9.3).

**Spectra:** 1D optical/UV spectra are also commonly stored in FITS files, either as 1D arrays in image HDUs or, more often, as binary tables where columns represent wavelength, flux, error, and masks. Multi-object spectrographs might produce multi-extension FITS files where each extension contains the spectrum for a different object. Integral Field Unit (IFU) data results in 3D data cubes (two spatial, one spectral axis), often stored as a 3D array in a FITS image HDU with a WCS defining all three axes. `astropy.io.fits`, `astropy.table`, `astropy.wcs`, and `specutils` (Sec A.II) provide tools for reading and handling these spectral formats.

**Major Archives:** Access to optical/UV data is primarily through large institutional or mission-specific archives:
*   **MAST (Mikulski Archive for Space Telescopes):** Hosted by STScI, MAST is the primary archive for Hubble Space Telescope (HST), GALEX, TESS, Kepler/K2, and will host JWST data. It provides web interfaces (portal, CasJobs) and programmatic access via `astroquery.mast` and VO protocols (TAP, SIA, SCS).
*   **NOIRLab Astro Data Archive:** Provides access to data from telescopes operated by NOIRLab (e.g., Blanco/DECam, Mayall/Mosaic, SOAR, Gemini). Offers web search and programmatic access, including TAP services.
*   **ESA Archives (ESASky, Gaia Archive, HST Archive):** European Space Agency archives host data from ESA missions like Gaia, XMM-Newton, Herschel, and the European copy of the HST archive. Accessible via web portals and VO protocols (`astroquery.esa`, `astroquery.gaia`).
*   **SDSS (Sloan Digital Sky Survey):** Provides its vast imaging and spectroscopic datasets via web interfaces (SkyServer), CasJobs (SQL database), and direct file downloads (Science Archive Server - SAS). `astroquery.sdss` facilitates access.
*   **Survey-Specific Archives:** Large surveys like Pan-STARRS, DES (Dark Energy Survey), ZTF often have their own dedicated archives and data access portals or services. Rubin Observatory/LSST will have a major new platform.
*   **CDS (Centre de Données astronomiques de Strasbourg):** Hosts SIMBAD (object database), VizieR (catalog service), and Aladin (sky atlas), providing essential catalog cross-matching and data discovery tools accessible via `astroquery.simbad`, `astroquery.vizier`, and VO protocols.

Navigating these archives and understanding their specific data products and access methods is a key skill. Programmatic access using Python libraries like `astroquery`, `pyvo`, `lightkurve`, combined with `astropy` for reading the standard FITS formats, provides the foundation for efficiently retrieving and working with the wealth of optical and UV data available to researchers.

**54.5 Source Detection, Photometry (Aperture, PSF), Morphology**

Extracting quantitative information about astronomical sources (stars, galaxies, nebulae) from calibrated optical/UV images involves several standard analysis techniques: source detection, photometry (measuring brightness), and morphology analysis (measuring shape and structure). Python libraries, especially `Astropy` affiliated packages like `photutils` and `SEP`, provide robust tools for these tasks.

**Source Detection:** The first step is often to identify statistically significant sources above the background noise level. Common algorithms include:
*   **Thresholding:** Identifying contiguous pixels (segments) above a certain threshold (e.g., 3-sigma above the estimated background noise). `photutils.segmentation.detect_sources` implements this based on the SExtractor segmentation approach.
*   **Peak Finding:** Locating local maxima in the image, often after smoothing to reduce noise sensitivity. `photutils.detection.find_peaks` or `DAOStarFinder`/`IRAFStarFinder` (which combine peak finding with filtering) can be used.
The output is typically a list or table of detected source pixel coordinates (centroids) and potentially basic shape information from segmentation maps. Careful background estimation (Sec 53/App 53.B) is crucial prior to detection. **`SEP`** (`pip install sep`), a Python wrapper around the widely used **SExtractor** code, provides a fast and robust alternative for source detection and basic aperture photometry, often used in survey pipelines.

**Photometry:** Measuring the brightness of detected sources. Two main methods are used:
*   **Aperture Photometry:** Sums the pixel values within a defined aperture (usually circular, sometimes elliptical or rectangular) centered on the source, after subtracting an estimated background level measured from a surrounding annulus or local region. `photutils.aperture.CircularAperture`, `photutils.aperture.aperture_photometry` perform this. Simple and robust for isolated sources, but susceptible to contamination in crowded fields or complex backgrounds. Choosing the aperture size and background estimation method requires care.
*   **PSF Photometry:** Assumes sources can be modeled by the telescope/instrument's Point Spread Function (PSF). It involves fitting the PSF model (either analytical like Gaussian/Moffat or empirical derived from image stars) simultaneously to the pixel data for one or multiple potentially overlapping sources. The normalization (amplitude) of the best-fit PSF model for each source gives its flux. More complex but provides more accurate results in crowded fields and can simultaneously fit position. `photutils.psf` provides tools for building PSFs and performing PSF photometry (e.g., `BasicPSFPhotometry`, `IterativelySubtractedPSFPhotometry`).

**Morphology Analysis:** Quantifying the shape and structure of resolved sources, particularly galaxies. Common measures include:
*   **Size:** Effective radius (half-light radius), Petrosian radius, Kron radius.
*   **Ellipticity and Position Angle:** Derived from fitting elliptical apertures or calculating second moments of the light distribution.
*   **Concentration Index:** Ratio of light within two different aperture radii (e.g., C = R₉₀/R₅₀), sensitive to the prominence of a central bulge.
*   **Asymmetry:** Measures rotational asymmetry, indicating disturbances or mergers.
*   **Smoothness/Clumpiness (CAS system):** Quantitative measures of morphological features.
*   **Sérsic Profile Fitting:** Fitting a Sérsic profile (I(r) ∝ exp[-(r/R<0xE2><0x82><0x9A>)¹ᐟⁿ]) to the surface brightness profile using `astropy.modeling` to measure effective radius R<0xE2><0x82><0x9A> and Sérsic index `n` (n=1 for exponential disk, n=4 for de Vaucouleurs bulge).
Libraries like `photutils` provide tools for calculating some basic shape parameters (e.g., from segmentation maps). `astropy.modeling` is used for profile fitting. Specialized galaxy morphology packages might offer more advanced measures. Machine learning (Chapter 22) is also increasingly used for morphological classification based on image features.

```python
# --- Code Example 1: Basic Source Detection and Aperture Photometry ---
# Note: Requires photutils, astropy
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus
import matplotlib.pyplot as plt # For visualization check

print("Source Detection and Aperture Photometry using photutils:")

# --- Simulate or Load Image ---
# Create dummy image with noise and a few sources
npix = 200
mean_bkg, median_bkg, std_bkg = 50, 50, 5 # Background stats
data = np.random.normal(mean_bkg, std_bkg, size=(npix, npix))
# Add sources (simple Gaussians)
sources_pos = [(50, 50), (100, 130), (150, 80)]
sources_amp = [500, 300, 400]
sources_stddev = [3.0, 4.0, 2.5]
yy, xx = np.mgrid[:npix, :npix]
for i, pos in enumerate(sources_pos):
    g = sources_amp[i] * np.exp(-((xx-pos[0])**2 + (yy-pos[1])**2) / (2*sources_stddev[i]**2))
    data += g
print(f"\nGenerated dummy image ({data.shape}) with 3 sources.")

# --- Source Detection ---
# Estimate background and noise robustly
# mean, median, std = sigma_clipped_stats(data, sigma=3.0) # Using known values here
print(f"Assuming background median={median_bkg}, stddev={std_bkg}")
# Use DAOStarFinder to find sources ~5-sigma above background
daofind = DAOStarFinder(fwhm=3.0 * 2.355, threshold=5.0 * std_bkg) # fwhm ~ 2.355*sigma
# Subtract median background for detection
sources_tbl = daofind(data - median_bkg) 
if sources_tbl:
    print(f"\nDAOStarFinder found {len(sources_tbl)} sources.")
    # Print first few rows (columns: id, xcentroid, ycentroid, sharpness, roundness, flux, mag)
    sources_tbl['xcentroid'].format = '.2f' # Format output
    sources_tbl['ycentroid'].format = '.2f'
    sources_tbl['flux'].format = '.2f'
    print(sources_tbl)
else:
    print("\nNo sources found by DAOStarFinder.")
    exit() # Stop if no sources

# --- Aperture Photometry ---
print("\nPerforming aperture photometry...")
positions = np.transpose((sources_tbl['xcentroid'], sources_tbl['ycentroid']))
# Define apertures (e.g., circular with radius 4 pixels)
apertures = CircularAperture(positions, r=4.0)
# Define background annulus (inner radius 6, outer radius 8)
annulus_apertures = CircularAnnulus(positions, r_in=6.0, r_out=8.0)

# Calculate median background per source in annulus
# Requires creating annulus mask and calculating stats within mask
# Simpler photutils approach: bkg_median = MedianBackground() ... 
# Or use aperture_photometry with background subtraction methods later
# Let's assume constant background subtraction for simplicity here
print(f"  Using apertures r=4, background annulus r=6-8 (median={median_bkg})")

# Perform photometry
# This sums counts within the aperture
phot_table = aperture_photometry(data, apertures) 
# Add background subtraction estimate (value * area)
aperture_area = apertures.area_overlap(data) # Or just pi*r^2 if no overlap/masking
phot_table['background_subtracted_flux'] = phot_table['aperture_sum'] - median_bkg * aperture_area
phot_table['background_subtracted_flux'].format = '.2f'
print("\nPhotometry Table:")
print(phot_table['id', 'xcenter', 'ycenter', 'aperture_sum', 'background_subtracted_flux'])

# --- Visualize (Optional) ---
print("\nGenerating visualization (image + apertures)...")
plt.figure(figsize=(7, 7))
norm = simple_norm(data, stretch='log', percent=99)
plt.imshow(data, origin='lower', cmap='viridis', norm=norm)
apertures.plot(color='white', lw=1.5, alpha=0.7)
annulus_apertures.plot(color='red', linestyle='--', lw=1.0, alpha=0.7)
plt.colorbar(label='Log Flux')
plt.title("Detected Sources with Apertures")
# plt.show()
print("Plot generated.")
plt.close()

print("-" * 20)

# Explanation:
# 1. Simulates a 2D image with background noise and three Gaussian sources.
# 2. Estimates background properties (using known values here for simplicity, usually 
#    `sigma_clipped_stats` is used).
# 3. Uses `photutils.detection.DAOStarFinder` to detect sources significantly above 
#    the background, requiring estimates of FWHM and a detection threshold. It returns 
#    an Astropy Table `sources_tbl` with source properties.
# 4. Defines circular apertures (`CircularAperture`) and background annuli 
#    (`CircularAnnulus`) centered on the detected source positions.
# 5. Performs aperture photometry using `photutils.aperture.aperture_photometry`, 
#    which sums pixel values within each aperture.
# 6. Performs simple background subtraction by multiplying the estimated median background 
#    by the aperture area and subtracting from the aperture sum. (More robust background 
#    estimation methods exist in `photutils`).
# 7. Prints the resulting photometry table.
# 8. Creates a plot showing the image with the apertures and annuli overlaid.
# This illustrates a basic source detection and aperture photometry workflow. PSF 
# photometry (using `photutils.psf`) would be needed for crowded fields.
```

These computational techniques are applied routinely to vast datasets from optical/UV surveys using automated Python pipelines. Libraries like `photutils`, `SEP`, and `astropy.modeling` provide the core functionalities for extracting scientifically valuable measurements of source brightness, position, and structure from calibrated images.

**54.7 Python Tools: Astropy (ccdproc, photutils, specutils), SEP, SPS fitters**

The analysis of optical and UV astronomical data relies heavily on a rich ecosystem of Python packages, largely built around the core `Astropy` project, providing specialized tools for various stages of data reduction and scientific analysis.

**`Astropy` Core and Affiliated Packages:**
*   **`astropy.io.fits` & `astropy.table`:** Essential for reading/writing the ubiquitous FITS image and table formats (Sec 1.5, 1.6, 2.3).
*   **`astropy.wcs`:** Handles World Coordinate System information in FITS headers, enabling conversion between pixel and sky coordinates (Sec 5.6, 5.7).
*   **`astropy.units` & `astropy.constants`:** Crucial for managing physical units and constants correctly throughout calculations (Chapter 3).
*   **`astropy.coordinates`:** Represents and transforms celestial coordinates between different frames (Sec 5.1-5.5).
*   **`astropy.stats`:** Provides statistical functions useful for astronomy, including robust statistics (`sigma_clipped_stats`), histogram tools, and Bayesian blocks (Sec 13).
*   **`astropy.modeling`:** A powerful framework for defining, fitting (using wrappers around `scipy.optimize` or other fitters), and evaluating mathematical models, including PSF models (Gaussian, Moffat), profile models (Sérsic), spectral line models, and composite models (Sec 14.6).
*   **`ccdproc`:** An Astropy affiliated package specifically designed for basic CCD data reduction. It provides classes and functions for managing image data with uncertainties and masks (`CCDData`), combining calibration frames (`Combiner`), performing bias/dark/flat correction, cosmic ray rejection (`cosmicray_lacosmic`), and managing processing metadata. Streamlines the calibration workflow described in Sec 54.2.
*   **`photutils`:** An Astropy affiliated package focused on source detection and photometry in images (Sec 54.5). Includes tools for background estimation (`Background2D`), aperture photometry (`aperture_photometry`), PSF photometry (building models with `EPSFBuilder`, fitting with `BasicPSFPhotometry`), and segmentation-based source detection (`detect_sources`).
*   **`specutils`:** An Astropy affiliated package for reading, representing, and analyzing 1D spectra (Sec A.II). Defines the `Spectrum1D` object and provides tools for continuum subtraction, line fitting, equivalent width measurement, spectral smoothing, and resampling.

**Other Key Libraries:**
*   **`SEP` (`pip install sep`):** A Python wrapper around the widely used **SExtractor** code (written in C by E. Bertin). SExtractor is renowned for its speed and robustness in detecting and performing basic photometry (aperture magnitudes, shape parameters) on sources in astronomical images. `SEP` provides a convenient Python interface to SExtractor's core algorithms (background estimation, source extraction) directly on NumPy arrays, often significantly faster than `photutils` for large images or complex segmentation tasks.
*   **Stellar Population Synthesis (SPS) Fitting Codes:** Analyzing the integrated light of galaxies (photometry or spectra) to infer properties like stellar mass, age distribution, star formation history, metallicity, and dust content requires fitting SPS models. Several powerful Python packages facilitate this:
    *   **`Prospector`:** Flexible Bayesian inference framework using MCMC or nested sampling, interfacing with `python-fsps` for SPS models. Highly customizable models and priors. ([https://prospect.readthedocs.io/](https://prospect.readthedocs.io/))
    *   **`BAGPIPES`:** Another Bayesian tool, also often using FSPS, designed for fitting galaxy SEDs and spectra efficiently. ([https://bagpipes.readthedocs.io/](https://bagpipes.readthedocs.io/))
    *   **`CIGALE` / `pcigale`:** Fits galaxy SEDs from UV to radio using energy balance arguments and pre-computed libraries of stellar, nebular, dust, and AGN emission models. Python version `pcigale`. ([https://cigale.lam.fr/](https://cigale.lam.fr/))
    *   **(Directly using `python-fsps`):** For simpler SPS calculations or integration into custom fitting routines. ([https://dfm.io/python-fsps/current/](https://dfm.io/python-fsps/current/))
*   **Libraries from Specific Surveys/Missions:** Large surveys or missions often provide dedicated Python packages for accessing and working with their specific data products (e.g., `sdss-access`, `desiutil`, packages within the LSST Science Pipelines, TESS tools like `lightkurve`).

This ecosystem, centered around Astropy and NumPy/SciPy/Matplotlib, provides a comprehensive suite of tools for nearly all aspects of optical and UV data reduction, analysis, and interpretation within a unified Python environment. Familiarity with the core Astropy submodules (`io.fits`, `table`, `wcs`, `units`, `coordinates`, `modeling`), specialized tools like `ccdproc` and `photutils` (or `SEP`), and potentially SPS fitting codes is essential for most computational work with optical/UV astronomical data.

**Application 54.A: PSF Photometry on a Crowded Stellar Field Image**

**(Paragraph 1)** **Objective:** This application provides a practical demonstration of performing **Point Spread Function (PSF) photometry** on a simulated astronomical image containing potentially overlapping (crowded) stellar sources. It utilizes the `photutils` package (Sec 54.5, 54.7) to first build a model of the instrument's PSF from isolated stars and then fits this model to all detected stars simultaneously to estimate their fluxes and precise positions.

**(Paragraph 2)** **Astrophysical Context:** Accurate photometry in crowded stellar fields, such as globular clusters, open clusters, or dense regions of nearby galaxies, is crucial for constructing precise color-magnitude diagrams (CMDs), studying stellar populations, and searching for variable stars. Simple aperture photometry fails when light from neighboring stars contaminates the measurement aperture. PSF photometry overcomes this by modeling the expected shape of a star's image (the PSF) and fitting this model to the pixel data, effectively de-blending overlapping sources.

**(Paragraph 3)** **Data Source:** A simulated 2D FITS image (`crowded_field.fits`) representing a crowded stellar field. The simulation should include stars with varying brightness placed at close proximity, convolved with a known or realistic PSF (e.g., Gaussian or Moffat), and with added background noise (e.g., Gaussian read noise + Poisson noise). We will simulate creating such an image.

**(Paragraph 4)** **Modules Used:** `photutils` (specifically `detection.DAOStarFinder`, `psf.EPSFBuilder`, `psf.BasicPSFPhotometry` or `DAOPhotPSFPhotometry`), `astropy.table`, `astropy.modeling.models` (for PSF if analytical), `astropy.stats`, `numpy`, `matplotlib.pyplot`.

**(Paragraph 5)** **Technique Focus:** The workflow involves several `photutils` components: (1) **Source Detection:** Using `DAOStarFinder` (or similar) to get initial estimates of star positions. (2) **PSF Modeling:** Selecting bright, relatively isolated stars from the detected list. Extracting cutouts around these stars. Using `EPSFBuilder` to iteratively build an empirical PSF model (an 'effective' or 'empirical' PSF, stored as an image or model) by aligning and combining the selected star images, rejecting outliers. Alternatively, fitting an analytical model (e.g., `Gaussian2D`) to these stars. (3) **PSF Fitting:** Creating a photometry object (e.g., `BasicPSFPhotometry` or the more robust `DAOPhotPSFPhotometry` which combines detection and fitting). Providing the PSF model, source positions (or running detection internally), and potentially fitting parameters (like fit shape, aperture size for initial guess). Running the fitter on the image data. (4) **Analyzing Results:** Examining the output table containing fitted positions (`x_fit`, `y_fit`), fluxes (`flux_fit`), uncertainties (`flux_unc`), and potentially flags indicating fit quality or crowding.

**(Paragraph 6)** **Processing Step 1: Simulate/Load Image and Detect Sources:** Create or load the crowded field image `data`. Estimate background `bkg` and noise `std` using `sigma_clipped_stats`. Detect initial source positions using `DAOStarFinder(threshold=..., fwhm=...)` on the background-subtracted image, obtaining `sources_init_tbl`.

**(Paragraph 7)** **Processing Step 2: Build Empirical PSF (EPSF):** Select bright, isolated stars from `sources_init_tbl` based on flux and proximity to neighbors (requires calculating distances between sources). Extract image cutouts (e.g., 25x25 pixels) centered on these selected stars using `photutils.datasets.extract_stars`. Create an `EPSFBuilder` instance with desired oversampling, smoothing, and recentering parameters. Call `builder(star_cutouts)` to build the empirical PSF model `epsf_model`. Visualize the resulting PSF (`plt.imshow(epsf_model.data)`).

**(Paragraph 8)** **Processing Step 3: Perform PSF Photometry:** Choose a photometry class, e.g., `DAOPhotPSFPhotometry` which handles finding/fitting iteratively. Provide parameters: `crit_separation` (minimum distance between stars for grouping), `threshold` (detection threshold for fitter), `fwhm` (PSF estimate), `psf_model=epsf_model`, `fitshape` (size of region around each star to fit). Run the photometer: `phot_results_tbl = photometer(data - bkg)`.

**(Paragraph 9)** **Processing Step 4: Analyze Output Table:** Inspect the `phot_results_tbl`. It contains columns like `x_fit`, `y_fit`, `flux_fit`, `flux_unc`, `id`, `group_id`, potentially sharpness/roundness, quality flags. Compare fitted fluxes (`flux_fit`) to initial estimates or known input fluxes (if simulated). Check uncertainties. Identify potentially problematic fits flagged by the photometry routine.

**(Paragraph 10)** **Processing Step 5: Visualize Residuals:** To assess fit quality, create a model image by placing the fitted PSF model scaled by `flux_fit` at each `(x_fit, y_fit)` position. Subtract this model image from the original (background-subtracted) data image. Display the residual image (`data - bkg - model`). Ideally, residuals should look like background noise with no significant structure remaining at the positions of fitted stars, indicating a good PSF fit.

**Output, Testing, and Extension:** Output includes the table of PSF photometry results and potentially visualizations of the built PSF and the residual image after subtracting fitted models. **Testing:** Compare fitted fluxes and positions to input values if using simulated data. Check if blended sources are reasonably de-blended in the residual image. Verify uncertainties seem appropriate. Test with different PSF models (empirical vs. analytical Gaussian/Moffat). **Extensions:** (1) Use more sophisticated group finding or iterative PSF fitting approaches available in `photutils` for very dense crowding. (2) Incorporate PSF spatial variations by building different PSFs for different image regions or using `photutils` tools that model PSF variability. (3) Convert fitted fluxes to magnitudes using photometric calibration (Sec 54.3). (4) Use the results to create a color-magnitude diagram for the crowded field.

```python
# --- Code Example: Application 54.A ---
# Note: Requires photutils, astropy, matplotlib, scipy
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.modeling.models import Gaussian2D
from photutils.detection import DAOStarFinder
from photutils.psf import extract_stars, EPSFBuilder, BasicPSFPhotometry # Example fitter
from astropy.visualization import simple_norm
import time

print("PSF Photometry on Simulated Crowded Field:")

# Step 1: Simulate Crowded Image
npix = 256
n_stars_sim = 500
bkg_level = 50.0
noise_std = 3.0

print(f"\nSimulating {npix}x{npix} image with {n_stars_sim} stars...")
# Simulate star positions (more clustered near center)
pos_xy = np.random.normal(loc=npix/2, scale=npix/5, size=(n_stars_sim, 2))
pos_xy = np.clip(pos_xy, 10, npix-11) # Keep away from edges
# Simulate fluxes (power law)
fluxes = 1000 * (np.random.power(a=1.5, size=n_stars_sim))**(-1) + 50 

# Simulate PSF (Gaussian)
psf_fwhm_pix = 3.5
psf_sigma = psf_fwhm_pix / 2.3548
yy, xx = np.mgrid[:npix, :npix]
image = np.random.normal(bkg_level, noise_std, size=(npix, npix)) # Add background+noise
for i in range(n_stars_sim):
    g = fluxes[i] * (1.0/(2*np.pi*psf_sigma**2)) * \
        np.exp(-((xx-pos_xy[i,0])**2 + (yy-pos_xy[i,1])**2) / (2*psf_sigma**2))
    image += g
print("Image simulation complete.")

# --- Detect Sources (Initial Guess) ---
mean, median, std = sigma_clipped_stats(image, sigma=3.0)
print(f"\nBackground stats: mean={mean:.2f}, median={median:.2f}, std={std:.2f}")
daofind = DAOStarFinder(fwhm=psf_fwhm_pix, threshold=5.0 * std)
# It's better to subtract background *before* detection
sources = daofind(image - median) 
if not sources: 
    print("Error: No sources detected by DAOStarFinder!")
    exit()
print(f"DAOStarFinder found {len(sources)} initial sources.")
# Add buffer around image for cutout extraction
from photutils.datasets import make_noise_image # Make matching noise for buffer
from astropy.nddata import NDData
img_buffered = np.pad(image, 15, mode='constant', constant_values=median)
sources['xcentroid'] += 15 # Adjust coordinates for buffer
sources['ycentroid'] += 15

# Step 2: Build Empirical PSF (Select bright, isolated stars)
print("\nBuilding empirical PSF (ePSF)...")
# Crude isolation criteria: distance to nearest neighbor > N*FWHM
# More robust methods exist, e.g., checking segmentation maps
from scipy.spatial import KDTree
star_coords = np.vstack((sources['xcentroid'], sources['ycentroid'])).T
tree = KDTree(star_coords)
dist, _ = tree.query(star_coords, k=2) # Dist to nearest neighbor (excluding self)
nearest_dist = dist[:, 1]
flux_sorted = np.argsort(sources['flux'])[::-1] # Indices sorted by flux high->low
bright_mask = flux_sorted[:int(0.2*len(sources))] # Top 20% brightest
isolated_mask = nearest_dist[bright_mask] > 3 * psf_fwhm_pix # Isolated among bright stars

stars_for_psf = sources[bright_mask][isolated_mask]
print(f"Selected {len(stars_for_psf)} bright, isolated stars for ePSF.")

if len(stars_for_psf) < 5:
     print("Warning: Too few stars to build reliable ePSF. Using Gaussian model instead.")
     psf_model = Gaussian2D(amplitude=1.0, x_mean=0, y_mean=0, 
                            x_stddev=psf_sigma, y_stddev=psf_sigma)
     # Need to set fixed parameters for fitting shape
     psf_model.amplitude.fixed = True
     psf_model.x_mean.fixed = True; psf_model.y_mean.fixed = True
else:
     # Extract cutouts around selected stars
     extract_size = 25 # Size of cutout box
     nddata = NDData(data=img_buffered) # Use NDData for extract_stars
     stars_tbl = Table()
     stars_tbl['x'] = stars_for_psf['xcentroid']
     stars_tbl['y'] = stars_for_psf['ycentroid']
     star_cutouts = extract_stars(nddata, stars_tbl, size=extract_size)
     
     # Build ePSF
     epsf_builder = EPSFBuilder(oversampling=4, maxiters=5, progress_bar=False)
     # This step can take time
     psf_model, fitted_stars = epsf_builder(star_cutouts)
     print("ePSF built.")
     # Visualize PSF
     plt.figure(); plt.imshow(psf_model.data, origin='lower', cmap='viridis'); plt.title("Built ePSF Model"); plt.colorbar(); plt.close()
     print("ePSF plot generated (not shown).")

# Step 3 & 4: Perform PSF Photometry
print("\nPerforming PSF photometry...")
# Use BasicPSFPhotometry (simpler fitter for demo)
# Requires fitter, psf_model, and initial guesses for positions
from astropy.modeling.fitting import LevMarLSQFitter
# Use DAOStarFinder results as initial guesses, need Table for BasicPSFPhotometry
init_guess_tbl = Table()
init_guess_tbl['x_0'] = sources['xcentroid']
init_guess_tbl['y_0'] = sources['ycentroid']
init_guess_tbl['flux_0'] = sources['flux'] # DAO flux is approximate

# Subtract background BEFORE fitting
image_sub = img_buffered - median 
photometry = BasicPSFPhotometry(
                 finder=None, # Use provided initial guesses
                 group_maker=None, # No grouping for BasicPSFPhotometry
                 bkg_estimator=None, # Background already subtracted
                 psf_model=psf_model, 
                 fitter=LevMarLSQFitter(),
                 fitshape=(11, 11)) # Fit within 11x11 box around source

start_phot = time.time()
phot_results_tbl = photometry(image=image_sub, init_guesses=init_guess_tbl)
end_phot = time.time()
print(f"Photometry finished. Time: {end_phot - start_phot:.2f}s")

# Adjust coordinates back by removing buffer
phot_results_tbl['x_fit'] -= 15
phot_results_tbl['y_fit'] -= 15
# Print first few rows
print("\nPSF Photometry Results Table (first 5 rows):")
phot_results_tbl['x_fit'].format = '.2f'
phot_results_tbl['y_fit'].format = '.2f'
phot_results_tbl['flux_fit'].format = '.2f'
phot_results_tbl['flux_unc'].format = '.2f'
print(phot_results_tbl[:5])

# Step 5: Visualize Residuals (Conceptual)
print("\nVisualizing Residuals (Conceptual)...")
# Create model image: loop through phot_results_tbl, place psf_model at (x_fit, y_fit) scaled by flux_fit
# Subtract model image from data-bkg
# Display residual image
print("  (Skipping residual image generation for brevity)")

print("-" * 20)
```

**Application 54.B: Fitting Emission Lines in a Simulated Galaxy Spectrum**

**(Paragraph 1)** **Objective:** Demonstrate the process of fitting multiple Gaussian profiles to emission lines superimposed on a continuum in a simulated 1D optical galaxy spectrum using the `astropy.modeling` framework (Sec 54.7) and its associated fitters. This allows measuring key line properties like flux, central wavelength (for redshift/kinematics), and line width (for velocity dispersion).

**(Paragraph 2)** **Astrophysical Context:** Optical spectra of star-forming galaxies, HII regions, and active galactic nuclei (AGN) are rich in emission lines (e.g., Balmer lines Hα, Hβ; forbidden lines [OIII]λλ4959,5007, [NII]λλ6548,6583, [SII]λλ6716,6731). Measuring the properties of these lines is fundamental for diagnosing the physical conditions of the emitting gas (temperature, density, metallicity, ionization state), star formation rates, AGN activity, and gas kinematics (rotation, outflows, turbulence). Accurate measurement often requires de-blending nearby lines (like Hα and [NII]) and subtracting the underlying stellar continuum, typically achieved by fitting models (like Gaussians) to the line profiles.

**(Paragraph 3)** **Data Source:** A simulated 1D optical spectrum, represented as two NumPy arrays: `wavelength` (Angstroms) and `flux` (e.g., erg/s/cm²/Å). The simulated spectrum should include a smooth continuum, realistic noise, and several emission lines (e.g., Hα, [NII] doublet, [SII] doublet) modeled as Gaussian profiles at known (or slightly offset) wavelengths.

**(Paragraph 4)** **Modules Used:** `numpy` (for data, arrays), `astropy.modeling.models` (for `Gaussian1D`, `Const1D` or `Polynomial1D`), `astropy.modeling.fitting` (e.g., `LevMarLSQFitter`), `astropy.units` (for wavelength/flux units), `matplotlib.pyplot`. `specutils` could also be used for higher-level spectral analysis, but we focus on `astropy.modeling` here.

**(Paragraph 5)** **Technique Focus:** Model fitting with `astropy.modeling`. (1) Simulating realistic spectral data. (2) Defining individual model components (`Gaussian1D` for each emission line, `Const1D` or `Polynomial1D` for the continuum). (3) Combining these into a composite model using arithmetic operators (`+`). (4) Providing reasonable initial guesses for all model parameters (amplitudes, means/centers, stddevs, continuum coefficients), potentially guided by visual inspection or prior knowledge. (5) Instantiating a fitter object (`LevMarLSQFitter`). (6) Calling the fitter (`fitter(composite_model, wavelength, flux)`) to find the best-fit parameters. (7) Accessing the parameters and uncertainties (if fitter provides covariance matrix) of the best-fit model. (8) Plotting the data, the individual fitted line components, the fitted continuum, and the total fitted model.

**(Paragraph 6)** **Processing Step 1: Simulate Spectrum:** Create a `wavelength` array (e.g., 6500-6800 Å). Define true parameters for continuum and emission lines (Hα λ6563, [NII] λ6583, [NII] λ6548, [SII] λ6716, [SII] λ6731). Create the true model flux by summing continuum and Gaussian profiles. Add Gaussian noise to simulate observed `flux`.

**(Paragraph 7)** **Processing Step 2: Define Astropy Model:** Import models. Create instances: `continuum = Const1D(...)`. `gauss_Ha = Gaussian1D(amplitude=..., mean=..., stddev=...)`. Similarly for [NII] and [SII] lines. Combine them: `full_model = continuum + gauss_Ha + gauss_NII1 + gauss_NII2 + gauss_SII1 + gauss_SII2`. Provide reasonable initial guesses (e.g., based on visual inspection or known line ratios/positions) for all parameters. Can also set bounds or fix parameters if needed (`parameter.bounds`, `parameter.fixed=True`).

**(Paragraph 8)** **Processing Step 3: Fit Model:** Instantiate the fitter: `fitter = LevMarLSQFitter()`. Fit the model: `fit_result_model = fitter(full_model, wavelength, flux, weights=1/uncertainty_array)` (include uncertainties if available for weighted fit). Check `fitter.fit_info['message']` for convergence status.

**(Paragraph 9)** **Processing Step 4: Access Results:** The `fit_result_model` object holds the best-fit parameters. Access them via `fit_result_model.parameters` or by name (e.g., `fit_result_model.amplitude_1.value` for Hα amplitude). If the fitter provides covariance information (`fitter.fit_info['param_cov']`), estimate parameter uncertainties. Calculate line fluxes (Flux ≈ Amplitude * Stddev * sqrt(2π)). Calculate line ratios (e.g., [NII]/Hα). Calculate velocity offsets from rest wavelengths using `(lambda_fit - lambda_rest)/lambda_rest * c`.

**(Paragraph 10)** **Processing Step 5: Visualize Fit:** Plot the original `flux` vs `wavelength`. Overplot the total `fit_result_model(wavelength)`. Optionally, plot the individual fitted Gaussian components (`fit_result_model[component_name](wavelength)`) and the fitted continuum (`fit_result_model['Const1D_0'](wavelength)`) to visualize the decomposition. Plot residuals (`flux - fit_result_model(wavelength)`).

**Output, Testing, and Extension:** Output includes the best-fit parameter values (line centers, widths, amplitudes/fluxes, continuum level) and their uncertainties. Plots showing the data, full fit, individual components, and residuals. **Testing:** Compare fitted parameters with the input "true" values used in the simulation to assess accuracy and bias. Check the quality of the fit visually and potentially via goodness-of-fit statistics (e.g., reduced Chi-squared if uncertainties were used). Verify derived quantities (line ratios, velocities) are reasonable. **Extensions:** (1) Fit more complex line profiles (e.g., Lorentz1D, Voigt1D). (2) Fit spectra with blended lines where constraints are needed (e.g., fixing wavelength separation or flux ratio of [NII] doublet lines). (3) Use Bayesian fitting methods (e.g., `emcee` with `astropy.modeling`) to obtain full posterior distributions for parameters. (4) Apply the fitting procedure to real observational spectra from FITS files, potentially using `specutils` for initial loading and continuum estimation.

```python
# --- Code Example: Application 54.B ---
# Note: Requires numpy, astropy, scipy, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from astropy import units as u 
from astropy import constants as const

print("Fitting Emission Lines in Simulated Spectrum:")

# Step 1: Simulate Spectrum (H-alpha + [NII] doublet)
wavelength = np.linspace(6500, 6650, 300) * u.AA # Wavelength in Angstroms
# Continuum
cont_level = 1.5
continuum = np.full_like(wavelength.value, cont_level)
# True Line Parameters
amp_ha = 5.0; center_ha = 6563.0; stddev_ha = 2.0 # H-alpha
amp_nii1 = 3.0; center_nii1 = 6583.0; stddev_nii1 = 2.0 # [NII] 6583
amp_nii2 = 1.0; center_nii2 = 6548.0; stddev_nii2 = 2.0 # [NII] 6548 (approx 1/3 of 6583)
# Generate model flux
flux_model = (models.Const1D(cont_level)(wavelength) +
              models.Gaussian1D(amp_ha, center_ha, stddev_ha)(wavelength) +
              models.Gaussian1D(amp_nii1, center_nii1, stddev_nii1)(wavelength) +
              models.Gaussian1D(amp_nii2, center_nii2, stddev_nii2)(wavelength))
# Add noise
noise_std = 0.3
noise = np.random.normal(0, noise_std, size=wavelength.shape)
flux_observed = flux_model + noise
# Assume constant uncertainty for simplicity in fitting
flux_uncertainty = np.full_like(flux_observed, noise_std)
print("\nSimulated spectrum created.")

# Step 2: Define Astropy Model with Initial Guesses
# Provide guesses close to true values, but not exact
cont_init = models.Const1D(amplitude=1.6)
g_ha_init = models.Gaussian1D(amplitude=4.5, mean=6564, stddev=2.5, name='Halpha')
g_nii1_init = models.Gaussian1D(amplitude=2.8, mean=6584, stddev=2.5, name='NII_6583')
g_nii2_init = models.Gaussian1D(amplitude=0.9, mean=6549, stddev=2.5, name='NII_6548')

# Combine models
full_model_init = cont_init + g_ha_init + g_nii1_init + g_nii2_init
print("Initial composite model defined.")

# Step 3: Fit Model
print("\nFitting model to data...")
# Use Levenberg-Marquardt fitter
fitter = fitting.LevMarLSQFitter()
# Fit using observed flux and weights=1/uncertainty
# Note: weights in astropy fitting are 1/sigma^2 or 1/sigma depending on context, check docs.
# Assuming weights = 1/sigma here
weights = 1.0 / flux_uncertainty 
fit_model = fitter(full_model_init, wavelength, flux_observed, weights=weights, maxiter=500) 
# Access fit info
print(f"Fit converged: {fitter.fit_info['message']}")

# Step 4: Access Results
print("\nBest-fit Parameters:")
print(fit_model) # Prints summary of fitted parameters

# Example: Get H-alpha parameters
ha_model_fit = fit_model['Halpha'] # Access component by name
amp_ha_fit = ha_model_fit.amplitude.value
cen_ha_fit = ha_model_fit.mean.value
std_ha_fit = ha_model_fit.stddev.value
# Calculate flux (approx for Gaussian = amplitude * stddev * sqrt(2*pi))
flux_ha_fit = amp_ha_fit * std_ha_fit * np.sqrt(2*np.pi)
print(f"\nFitted H-alpha: Amp={amp_ha_fit:.2f}, Cen={cen_ha_fit:.2f}, Stddev={std_ha_fit:.2f}, Flux={flux_ha_fit:.2f}")

# Get covariance matrix if available (for uncertainties)
if fitter.fit_info['param_cov'] is not None:
    param_cov = fitter.fit_info['param_cov']
    # Extract uncertainties (sqrt of diagonal elements)
    param_unc = np.sqrt(np.diag(param_cov))
    # Map unc back to parameters (order matters!)
    # print(f"Parameter uncertainties: {param_unc}") # Needs careful mapping
else:
    print("\nCovariance matrix not available from fitter.")

# Step 5: Visualize Fit
print("\nGenerating fit visualization...")
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True, 
                        gridspec_kw={'height_ratios': [3, 1]})

# Top panel: Data and Fit
axs[0].errorbar(wavelength.value, flux_observed, yerr=flux_uncertainty, fmt='.', 
                color='black', label='Data', markersize=4, elinewidth=1, alpha=0.6)
axs[0].plot(wavelength.value, fit_model(wavelength), 'r-', label='Total Fit')
# Plot individual components
axs[0].plot(wavelength.value, fit_model['Const1D_0'](wavelength), 'b--', label='Continuum')
axs[0].plot(wavelength.value, fit_model['Halpha'](wavelength) + fit_model['Const1D_0'](wavelength), 
            'g:', label='Halpha Fit')
axs[0].plot(wavelength.value, fit_model['NII_6583'](wavelength) + fit_model['Const1D_0'](wavelength), 
            'm:', label='[NII]6583 Fit')
axs[0].set_ylabel("Flux (Arbitrary Units)")
axs[0].legend(fontsize=9)
axs[0].set_title("Emission Line Fitting with Astropy Modeling")

# Bottom panel: Residuals
residuals = flux_observed - fit_model(wavelength.value)
axs[1].plot(wavelength.value, residuals, 'k.')
axs[1].axhline(0, color='red', linestyle='--')
axs[1].set_xlabel(f"Wavelength ({wavelength.unit})")
axs[1].set_ylabel("Residuals")
axs[1].grid(True, alpha=0.5)

fig.tight_layout()
# plt.show()
print("Plot generated.")
plt.close(fig)

print("-" * 20)
```

**Chapter 54 Summary**

This chapter surveyed the computational landscape of **optical and ultraviolet (UV) astronomy**, covering data acquisition, processing, and analysis techniques common for these wavelength regimes. It contrasted ground-based observations, limited by **atmospheric seeing** and extinction, with the high-resolution and UV access provided by space-based facilities like HST. Standard **CCD image processing** steps were detailed, including bias/dark subtraction, flat-fielding, cosmic ray rejection, and the crucial **astrometric calibration** to obtain accurate World Coordinate System (WCS) information, highlighting tools within `Astropy` like `ccdproc` and `wcs`. **Photometric calibration**, the conversion of instrumental counts to standard magnitudes or fluxes using observations of standard stars to determine zeropoints, color terms, and (for ground data) atmospheric extinction coefficients, was explained. Common data formats (FITS images and tables) and major data archives (MAST, NOIRLab, ESA, SDSS) were reviewed.

Key scientific analysis techniques were then discussed, focusing on computational implementation. **Source detection** methods (thresholding, peak finding) and the distinction between **aperture photometry** (simpler, for isolated sources) and **PSF photometry** (model fitting, better for crowded fields) were covered, mentioning the capabilities of `photutils` and `SEP`. Basic **galaxy morphology** measurements (size, shape, concentration, Sérsic profiles) were introduced. Analysis approaches for **stellar and nebular spectroscopy** in the optical/UV, including classification, kinematics, and emission line fitting for physical diagnostics, were briefly outlined. Throughout, the importance of the **Python ecosystem**, particularly core `Astropy` packages (`io.fits`, `table`, `wcs`, `units`, `coordinates`, `modeling`, `stats`), affiliated packages (`ccdproc`, `photutils`, `specutils`), and specialized tools (`SEP`, SPS fitting codes like `Prospector` or `BAGPIPES`), was emphasized as providing the essential toolkit for modern optical/UV data analysis. Two applications illustrated PSF photometry using `photutils` and fitting emission lines using `astropy.modeling`.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Astropy Collaboration, Price-Whelan, A. M., et al. (2018).** The Astropy Project: Building an Open-science Project and Status of the v2.0 Core Package. *The Astronomical Journal*, *156*(3), 123. [https://doi.org/10.3847/1538-3881/aabc4f](https://doi.org/10.3847/1538-3881/aabc4f) (See also current Astropy Docs: [https://docs.astropy.org/en/stable/](https://docs.astropy.org/en/stable/))
    *(Provides an overview of the Astropy project and its core functionalities extensively used in optical/UV analysis (WCS, Table, FITS, Units, Coords, Modeling, Stats). The documentation is essential.)*

2.  **Bradley, L., et al. (2022).** *Photutils: An Astropy package for detection and photometry of astronomical sources*. Zenodo. [https://doi.org/10.5281/zenodo.6825092](https://doi.org/10.5281/zenodo.6825092) (See also Documentation: [https://photutils.readthedocs.io/en/stable/](https://photutils.readthedocs.io/en/stable/))
    *(Citation for the `photutils` package, covering the source detection, aperture photometry, and PSF photometry techniques discussed in Sec 54.5 and Application 54.A.)*

3.  **Barbary, K. (2016).** SEP: Source Extractor as a library. *Journal of Open Source Software*, *1*(6), 58. [https://doi.org/10.21105/joss.00058](https://doi.org/10.21105/joss.00058) (See also Documentation: [https://sep.readthedocs.io/en/latest/](https://sep.readthedocs.io/en/latest/))
    *(Paper and documentation for `SEP`, the Python binding for SExtractor, a widely used alternative/complement to `photutils` for fast source detection and basic photometry, relevant to Sec 54.5, 54.7.)*

4.  **Howell, S. B. (2006).** *Handbook of CCD Astronomy* (2nd ed.). Cambridge University Press. [https://doi.org/10.1017/CBO9780511808000](https://doi.org/10.1017/CBO9780511808000)
    *(A classic text detailing the principles of CCD detectors and the standard calibration steps (bias, dark, flat) discussed in Sec 54.2.)* (Note: `ccdproc` documentation often provides more modern implementation details).

5.  **Conroy, C. (2013).** Modeling the Panchromatic Spectral Energy Distributions of Galaxies. *Annual Review of Astronomy and Astrophysics*, *51*, 393–455. [https://doi.org/10.1146/annurev-astro-082812-141017](https://doi.org/10.1146/annurev-astro-082812-141017)
    *(Review covering SPS models crucial for interpreting galaxy photometry and spectra in the optical/UV/IR, relevant context for Sec 54.5, 54.6, 54.7 and related fitting packages.)*
