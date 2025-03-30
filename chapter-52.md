**Chapter 52: Computational Techniques in Radio Astronomy**

This chapter focuses on the computational methods central to modern **radio astronomy**, a field that probes the Universe through long-wavelength electromagnetic radiation (from meters down to sub-millimeter). Radio waves trace a wide range of astrophysical phenomena, including cool neutral hydrogen gas (HI), molecular clouds (via molecular lines like CO), synchrotron emission from relativistic electrons in magnetic fields (jets, supernova remnants, pulsars), thermal emission from HII regions and dust, and the faint signals from the Cosmic Microwave Background. We begin by contrasting data acquisition with **single dishes** versus **radio interferometers** (like VLA, ALMA, MeerKAT, LOFAR, SKA precursors), emphasizing that interferometers measure **complex visibilities** in the Fourier domain (uv-plane) rather than directly imaging the sky. The core computational task of **interferometric imaging** will be detailed, covering the conceptual steps of **gridding** visibilities (placing samples onto a regular grid), performing the **Fast Fourier Transform (FFT)** to create a "dirty" image (true sky convolved with the instrument's synthesized beam), and the crucial, computationally intensive non-linear process of **deconvolution** (using algorithms like **CLEAN** or Maximum Entropy Methods - MEM) to remove artifacts caused by sparse uv-coverage and approximate the true sky brightness distribution. We will introduce common data formats like **FITS images**, **Measurement Sets (MS)** (the standard for storing visibility data), and **CASA tables**. Key data analysis techniques specific to radio astronomy, such as analyzing **spectral line data cubes** (creating moment maps, position-velocity diagrams), handling **polarization data** (Stokes parameters, Rotation Measure synthesis), analyzing **pulsar timing** data, and searching for **Fast Radio Bursts (FRBs)** (including de-dispersion and candidate searching), will be discussed. Finally, we highlight essential Python libraries used in radio astronomy, particularly **Astropy** (for core functionalities like FITS I/O, WCS, coordinates, units), the **CASA** (Common Astronomy Software Applications) package (often used via its Python `casatools` and `casatasks` interfaces for interferometric data reduction and imaging), and potentially more specialized libraries like `radio-beam` or `spectral-cube`.

**52.1 Radio Data: Single Dish vs. Interferometry, Visibilities**

Radio astronomy utilizes telescopes ranging from large single parabolic dishes (like Green Bank Telescope, Effelsberg, Arecibo historically) to vast arrays of smaller antennas connected electronically, known as **interferometers** (like the Very Large Array (VLA), Atacama Large Millimeter/submillimeter Array (ALMA), LOFAR, MeerKAT, ASKAP). The nature of the data collected and the subsequent processing differ significantly between these two types of instruments.

**Single-dish radio telescopes** operate similarly in principle to optical reflecting telescopes. The large dish focuses incoming radio waves from a specific direction onto a receiver located at the focal point. The receiver detects and records the power (intensity or temperature) of the radio emission, often as a function of frequency (spectroscopy) or time (pulsar timing). Single dishes typically have a relatively large **primary beam** on the sky, determined by the diffraction limit (beam size θ ≈ λ/D, where λ is the wavelength and D is the dish diameter). They measure the sky brightness distribution convolved with this beam. Mapping large areas involves scanning the dish across the sky. Single dishes excel at detecting faint, extended, low surface brightness emission due to their large collecting area and sensitivity to all spatial scales within the primary beam. Data reduction often involves calibration (gain, bandpass), baseline subtraction, and potentially gridding if creating maps from scans.

**Radio interferometers**, conversely, achieve high angular resolution by combining signals from multiple, widely separated antennas. Instead of forming an image directly, an interferometer measures the **complex visibility**, V(u,v), for pairs of antennas (baselines). The visibility is essentially a measure of the coherence of the radio waves arriving at the two antennas forming a baseline. According to the **van Cittert-Zernike theorem**, under certain conditions, the complex visibility V(u,v) measured by a baseline is related to the sky brightness distribution I(l,m) via a **Fourier transform**:
V(u, v) ∝ ∫∫ I(l, m) * exp[-2πi (ul + vm)] dl dm
Here, (l, m) are direction cosines on the sky (related to RA, Dec offsets), and (u, v) are the components of the baseline vector (separation between the two antennas projected onto the plane perpendicular to the source direction), measured in units of wavelength.

This fundamental relationship means that an interferometer **samples the Fourier transform** of the sky brightness distribution at specific points (u,v) determined by the geometric layout of the antennas and the direction of observation as the Earth rotates. Each baseline (pair of antennas) provides one sample V(u,v) at a specific (u,v) coordinate at a given time. Over the course of an observation, as the Earth rotates, the projected baseline lengths and orientations change, allowing the interferometer to sample many different (u,v) points, tracing out tracks or arcs in the **uv-plane**.

The goal of interferometric imaging is to reconstruct the sky image I(l,m) from the sparsely sampled visibility data V(u,v). This involves performing an inverse Fourier transform, but since the uv-plane is incompletely sampled, specialized imaging and deconvolution techniques are required (Sec 52.2, 52.3).

Interferometers offer several advantages over single dishes:
*   **High Angular Resolution:** The resolution is determined by the *longest* baseline (B<0xE1><0xB5><0x8D><0xE1><0xB5><0x8A><0xE1><0xB5><0x82>) in the array (θ ≈ λ/B<0xE1><0xB5><0x8D><0xE1><0xB5><0x8A><0xE1><0xB5><0x82>), allowing much finer detail to be resolved than with single dishes of practical size.
*   **Precise Astrometry:** The long baselines allow for very precise measurements of source positions.
*   **Spatial Filtering:** Interferometers are less sensitive to very large-scale, smooth emission (corresponding to very short baselines near the uv-plane origin), which can help filter out atmospheric or Galactic foreground contamination in some cases.

However, interferometers also have drawbacks:
*   **Poor Sensitivity to Extended Emission:** The lack of short baselines means interferometers can "resolve out" large, smooth structures, making them less suitable for detecting faint, extended emission compared to single dishes (though combining interferometer and single-dish data is possible).
*   **Complex Calibration:** Correcting for atmospheric and instrumental effects on the phase and amplitude of the visibilities measured by each baseline is a complex calibration process.
*   **Imaging Complexity:** Reconstructing an image from sparsely sampled Fourier data requires sophisticated algorithms (deconvolution).

Understanding the concept of visibilities as Fourier components of the sky brightness and the nature of uv-coverage determined by the array configuration is fundamental to comprehending interferometric data and the subsequent imaging process. Data from interferometers are fundamentally different from direct images obtained by single dishes or optical telescopes.

**52.2 Interferometric Imaging: Gridding, FFT, Dirty Image**

The primary computational task for radio interferometers is to reconstruct an image of the sky brightness distribution I(l,m) from the measured complex visibilities V(u,v). As established by the van Cittert-Zernike theorem, these two quantities are related by a Fourier transform. Therefore, the naive approach to imaging is to perform an inverse Fourier transform on the measured visibilities.

However, a direct inverse FFT requires the visibility data V(u,v) to be sampled on a regular grid in the uv-plane. The interferometer antennas, through Earth rotation synthesis, trace out tracks and arcs, providing samples at irregular (u,v) locations. The first step in standard imaging pipelines is therefore **gridding**: interpolating or averaging the irregularly sampled visibilities onto a regular 2D grid suitable for FFTs.

Gridding involves defining a 2D array representing the uv-plane. Each measured visibility V(u,v) is assigned a weight (often related to data quality or sampling density) and its value (or weighted value) is accumulated onto nearby grid cells using a **convolution function** (also called a gridding kernel or prolate spheroidal function). This convolution step is crucial for minimizing aliasing artifacts that would arise from simple nearest-neighbor or bilinear interpolation onto the grid, effectively performing an anti-aliasing filter in the Fourier domain. The choice of gridding kernel and grid cell size involves trade-offs between computational cost, resolution, and aliasing suppression.

Once the visibilities have been gridded onto a regular uv-grid `V_grid(u,v)`, the next step is to perform a **2D Fast Fourier Transform (FFT)** on this grid. The inverse FFT relationship means that the FFT of the gridded visibilities yields an estimate of the sky brightness distribution:
I<0xE1><0xB5><0x87><0xE1><0xB5><0xA2><0xE1><0xB5><0xA3><0xE1><0xB5><0x97><0xE1><0xB5><0xA7>(l, m) ≈ FFT⁻¹[ V_grid(u, v) ] (or FFT, depending on convention)
This resulting image is known as the **dirty image**.

The dirty image is *not* the true sky brightness distribution. Because the interferometer only samples a finite, often sparse, set of (u,v) points, the uv-coverage is incomplete. Mathematically, the measured visibilities V_meas can be thought of as the true visibilities V_true multiplied by a **sampling function** S(u,v) (which is 1 where data was taken and 0 elsewhere): V_meas = V_true * S. According to the convolution theorem, the inverse Fourier transform (the imaging process) yields:
I_dirty = FFT⁻¹[V_meas] = FFT⁻¹[V_true * S] = FFT⁻¹[V_true] ⊗ FFT⁻¹[S] = I_true ⊗ B_dirty
where ⊗ denotes convolution. The inverse Fourier transform of the sampling function S(u,v) is called the **dirty beam** or **Point Spread Function (PSF)** of the interferometer. It represents how a single point source on the sky would appear in the dirty image, reflecting the pattern of the uv-coverage.

The dirty beam typically has a central peak (whose width determines the image resolution) surrounded by **sidelobes** – complex positive and negative patterns extending across the image. These sidelobes are caused by the gaps and edges in the uv-coverage. Consequently, the dirty image `I_dirty` is the true sky brightness distribution `I_true` convolved with this often complicated dirty beam `B_dirty`. Strong sources in the sky produce prominent sidelobe patterns in the dirty image, which can obscure faint emission or create artifacts that mimic real structures.

Therefore, creating the dirty image via gridding and FFT is only the first step. A crucial subsequent step, **deconvolution** (Sec 52.3), is required to remove the effects of the dirty beam and estimate the true sky brightness distribution `I_true`.

Python libraries like NumPy (`numpy.fft.fft2`, `numpy.fft.ifft2`, `numpy.fft.fftshift`) provide the core FFT capabilities needed. Gridding itself requires careful implementation, often found within specialized radio astronomy packages like CASA or developed using libraries like `scipy.interpolate` or custom code. Visualizing the uv-coverage (scatter plot of (u,v) points), the dirty beam (FFT of uv-coverage), and the dirty image (`matplotlib.pyplot.imshow`) are essential diagnostic steps.

```python
# --- Code Example: Conceptual Gridding and FFT for Dirty Image ---
# Highly simplified: Demonstrates FFT relation, NOT proper gridding.
import numpy as np
import matplotlib.pyplot as plt

print("Conceptual Interferometric Imaging: FFT and Dirty Image:")

# --- 1. Simulate Visibilities and uv-coverage ---
# Assume we have visibility samples V at locations (u, v)
# Let's simulate sparse samples for a point source at center I(l,m)=delta(l,m)
# True visibility V_true(u,v) = const * exp(-2pi*i*(u*0 + v*0)) = const
np.random.seed(0)
n_vis = 500
# Simulate random uv points (sparse coverage)
u_coords = np.random.randn(n_vis) * 50 
v_coords = np.random.randn(n_vis) * 50
visibilities = np.ones(n_vis, dtype=complex) + \
               np.random.normal(0, 0.1, (n_vis, 2)).view(complex).flatten() # Add noise
print(f"\nGenerated {n_vis} visibility samples V(u,v).")

# --- 2. Gridding (Highly Simplified: Nearest Neighbor Assignment) ---
# Define image/grid parameters
npix_image = 256 # Size of output image
fov_arcsec = 10.0 # Field of view in arcsec
pixel_scale_arcsec = fov_arcsec / npix_image
# Corresponding uv grid parameters
uv_max = 1.0 / (pixel_scale_arcsec * u.arcsec.to(u.rad)) / 2.0 # Max u/v coord (in wavelengths)
uv_cell_size = uv_max * 2 / npix_image
print(f"Image: {npix_image}x{npix_image} pixels, FoV={fov_arcsec} arcsec")
print(f"UV Grid: Max u/v ~ {uv_max:.0f}, Cell ~ {uv_cell_size:.1f}")

uv_grid = np.zeros((npix_image, npix_image), dtype=complex)
uv_sampling = np.zeros((npix_image, npix_image)) # For dirty beam

# Simple gridding: Find nearest grid cell for each visibility
# This is INEFFICIENT and inaccurate (no convolution kernel)
u_indices = np.clip(np.round(u_coords / uv_cell_size + npix_image/2).astype(int), 0, npix_image-1)
v_indices = np.clip(np.round(v_coords / uv_cell_size + npix_image/2).astype(int), 0, npix_image-1)
# Add visibilities to grid (simple average if multiple fall in same cell - not done here)
# Use np.add.at for potentially faster accumulation if needed
for i in range(n_vis):
    uv_grid[v_indices[i], u_indices[i]] += visibilities[i] # Sum into cell (better: weighted average)
    uv_sampling[v_indices[i], u_indices[i]] = 1 # Mark sampled cells
print("Performed simple gridding (nearest neighbor).")

# --- 3. FFT to get Dirty Image and Dirty Beam ---
print("Performing FFT...")
# Need fftshift to center frequencies before FFT, then inverse shift after
# Inverse FFT (or FFT depending on convention) of visibilities gives dirty image
dirty_image = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(uv_grid))).real 
# FFT of sampling function gives dirty beam (un-normalized PSF)
dirty_beam = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(uv_sampling))).real
print("FFTs complete.")

# --- 4. Visualize ---
print("Generating plots...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
# uv-coverage
axes[0].plot(u_coords, v_coords, '.', markersize=1, alpha=0.5)
axes[0].set_xlabel("u (wavelengths)"); axes[0].set_ylabel("v (wavelengths)")
axes[0].set_title("uv-coverage (Samples)")
axes[0].set_aspect('equal')
# Dirty Beam (PSF) - show central part logarithmically
beam_center_pix = npix_image // 2
beam_cutoff = 50
beam_slice = dirty_beam[beam_center_pix-beam_cutoff : beam_center_pix+beam_cutoff,
                        beam_center_pix-beam_cutoff : beam_center_pix+beam_cutoff]
im1 = axes[1].imshow(np.log10(np.abs(beam_slice)+1e-3), cmap='gray_r', 
                     extent=[-beam_cutoff, beam_cutoff, -beam_cutoff, beam_cutoff])
axes[1].set_title("Dirty Beam (PSF) Center (Log Scale)")
axes[1].set_xlabel("Image Pixels"); axes[1].set_ylabel("Image Pixels")
# Dirty Image
im2 = axes[2].imshow(dirty_image, cmap='gray_r', vmax=np.percentile(dirty_image, 99.5))
axes[2].set_title("Dirty Image")
axes[2].set_xlabel("Image Pixels"); axes[2].set_ylabel("Image Pixels")

fig.tight_layout()
# plt.show()
print("Plots generated.")
plt.close(fig)
print("-" * 20)

# Explanation: This code conceptually demonstrates interferometric imaging.
# 1. Simulates sparse, noisy visibility samples `V(u,v)` for a point source at the center.
# 2. Performs highly simplified 'gridding' by assigning each visibility to the nearest 
#    cell in a `uv_grid` array. A `uv_sampling` array marks which cells received data. 
#    **Proper gridding uses convolution kernels.**
# 3. Performs a 2D FFT on the `uv_grid` (after appropriate shifts) to get the `dirty_image`. 
#    It also FFTs the `uv_sampling` grid to get the `dirty_beam` (PSF).
# 4. Visualizes the input uv-coverage, the central part of the resulting dirty beam 
#    (showing the central peak and sidelobes characteristic of the sparse uv-coverage), 
#    and the dirty image (which shows the central source convolved with these sidelobes).
# This illustrates the relationship V(u,v) <-> I(l,m) via FFT and how incomplete uv-sampling 
# leads to artifacts (sidelobes) in the dirty image, motivating deconvolution.
```

**52.3 Deconvolution: CLEAN Algorithm, MEM**

The dirty image produced by FFTing the gridded visibilities is a convolution of the true sky brightness with the interferometer's dirty beam (PSF), I<0xE1><0xB5><0x87><0xE1><0xB5><0xA2><0xE1><0xB5><0xA3><0xE1><0xB5><0x97><0xE1><0xB5><0xA7> = I<0xE1><0xB5><0x97><0xE1><0xB5><0xA3><0xE1><0xB5><0x98><0xE1><0xB5><0x8A> ⊗ B<0xE1><0xB5><0x87><0xE1><0xB5><0xA2><0xE1><0xB5><0xA3><0xE1><0xB5><0x97><0xE1><0xB5><0xA7>. The goal of **deconvolution** is to estimate the true sky brightness I<0xE1><0xB5><0x97><0xE1><0xB5><0xA3><0xE1><0xB5><0x98><0xE1><0xB5><0x8A> given the dirty image I<0xE1><0xB5><0x87><0xE1><0xB5><0xA2><0xE1><0xB5><0xA3><0xE1><0xB5><0x97><0xE1><0xB5><0xA7> and knowledge of the dirty beam B<0xE1><0xB5><0x87><0xE1><0xB5><0xA2><0xE1><0xB5><0xA3><0xE1><0xB5><0x97><0xE1><0xB5><0xA7> (which is the FFT of the uv-sampling function). Since the uv-coverage is incomplete, this is an ill-posed inverse problem, and additional assumptions or constraints are needed to obtain a reasonable solution. Two widely used classes of deconvolution algorithms in radio astronomy are CLEAN and Maximum Entropy Methods (MEM).

The **CLEAN algorithm** (developed by Högbom and refined by Clark, Cotton-Schwab, and others) is the most common deconvolution method, particularly effective for images dominated by discrete point sources or moderately resolved structures. It works iteratively based on the assumption that the sky can be reasonably represented by a collection of point sources.
The basic CLEAN loop proceeds as follows:
1.  **Find Peak:** Identify the location (l, m) and flux density (I<0xE1><0xB5><0x96><0xE1><0xB5><0x8A><0xE1><0xB5><0x8A><0xE1><0xB5><0x8C>) of the brightest pixel remaining in the current **residual image** (initially, the residual is the dirty image).
2.  **Subtract Beam:** Subtract a scaled version of the *dirty beam* centered at the peak location from the residual image. The scaling factor is typically the peak flux multiplied by a small **loop gain** (γ, e.g., 0.1-0.3): Residual<0xE1><0xB5><0x8B><0xE1><0xB5><0x8A><0xE1><0xB5><0x98> = Residual<0xE1><0xB5><0x92><0xE1><0xB5><0x8A><0xE1><0xB5><0x87> - γ * I<0xE1><0xB5><0x96><0xE1><0xB5><0x8A><0xE1><0xB5><0x8A><0xE1><0xB5><0x8C> * B<0xE1><0xB5><0x87><0xE1><0xB5><0xA2><0xE1><0xB5><0xA3><0xE1><0xB5><0x97><0xE1><0xB5><0xA7>(l, m).
3.  **Store Component:** Record the position (l, m) and subtracted flux (γ * I<0xE1><0xB5><0x96><0xE1><0xB5><0x8A><0xE1><0xB5><0x8A><0xE1><0xB5><0x8C>) as a "CLEAN component" (a delta function representing a point source).
4.  **Repeat:** Continue iterating (finding peak, subtracting beam, storing component) until a stopping criterion is met, such as the maximum residual falling below a noise threshold (e.g., a few times the RMS noise in the residual map) or a maximum number of iterations is reached.
5.  **Restore Image:** Create the final "CLEAN image" by taking the accumulated list of CLEAN components (delta functions), convolving them with a "clean beam" (typically a 2D Gaussian fitted to the central peak of the dirty beam, representing the effective resolution), and adding back the final residual map (containing remaining noise and low-level emission not captured by the point source model).

CLEAN effectively decomposes the dirty image into a set of point sources and a residual noise map, removing the prominent sidelobes associated with bright sources. Various implementations exist (e.g., Högbom CLEAN works entirely in the image plane, Clark CLEAN performs major/minor cycles potentially using FFTs for subtractions, Cotton-Schwab includes self-calibration). Multi-scale CLEAN variants attempt to model emission on different spatial scales simultaneously, improving performance for somewhat extended sources. CLEAN is implemented as standard tasks in CASA (`tclean`) and other radio astronomy packages.

**Maximum Entropy Methods (MEM):** MEM offers an alternative deconvolution philosophy. Instead of assuming the sky consists of point sources, MEM seeks the sky image I<0xE1><0xB5><0x97><0xE1><0xB5><0xA3><0xE1><0xB5><0x98><0xE1><0xB5><0x8A> that is *consistent* with the observed visibilities (within the noise) *and* simultaneously maximizes a measure of "entropy" (often defined as S = -Σ I<0xE1><0xB5><0xA2> log(I<0xE1><0xB5><0xA2>/M<0xE1><0xB5><0xA2>), where M is a prior map or default image). Maximizing entropy tends to produce the "smoothest" or "simplest" image consistent with the data, implicitly regularizing the ill-posed deconvolution problem. MEM is often considered better than standard CLEAN for reconstructing smooth, extended emission, but can be computationally more intensive and sometimes sensitive to the choice of prior map and entropy definition. MEM implementations are also available in packages like CASA (`tclean` with `deconvolver='mem'`).

Both CLEAN and MEM are computationally intensive non-linear algorithms. Their implementation requires careful handling of FFTs, image manipulation, and iterative convergence. While core algorithms are usually implemented in compiled code within packages like CASA, Python scripts are used to drive these tasks, set parameters (number of iterations, gain, threshold for CLEAN; constraints for MEM), and analyze the resulting deconvolved images. Understanding the assumptions and limitations of the chosen deconvolution method is crucial for interpreting the final radio image. CLEAN images represent a *model* of the sky (point sources convolved with clean beam + residuals), not a direct reconstruction, especially for complex extended structures.

**(Implementing CLEAN or MEM from scratch is complex. Code examples typically involve calling existing functions from CASA or related libraries.)**

```python
# --- Code Example: Conceptual Call to CASA tclean task ---
# Note: Requires CASA installation and casatasks/casatools. 
# This is conceptual syntax for running a CASA task from Python.

print("Conceptual call to CASA tclean for deconvolution:")

try:
    # Import CASA tasks if running in CASA environment or via casatasks
    # from casatasks import tclean
    print("\n(Assuming CASA environment or 'casatasks' available)")
    
    # Define parameters for tclean
    vis_file = 'my_observation.ms' # Input Measurement Set
    image_name_base = 'output_image' # Basename for output images (.image, .residual, .psf, etc.)
    num_iter = 1000             # Max number of CLEAN iterations
    threshold_val = '0.1mJy'    # CLEAN threshold (e.g., 3-5 sigma noise)
    deconvolver_alg = 'clark'   # CLEAN algorithm (e.g., 'hogbom', 'clark', 'multiscale')
    # Gridder parameters (e.g., 'standard', 'wproject')
    # Image size, cell size, weighting scheme (natural, uniform, Briggs)
    # ... many other parameters ...
    
    print("\nConceptual tclean parameters:")
    print(f"  vis = '{vis_file}'")
    print(f"  imagename = '{image_name_base}'")
    print(f"  niter = {num_iter}")
    print(f"  threshold = '{threshold_val}'")
    print(f"  deconvolver = '{deconvolver_alg}'")
    print("  (Plus parameters for gridder, imsize, cell, weighting, etc.)")
    
    # --- Conceptual Call to tclean ---
    print("\nConceptual call: tclean(...)")
    # tclean(
    #     vis=vis_file,
    #     imagename=image_name_base,
    #     imsize=[512, 512], # Example image size
    #     cell=['1.0arcsec', '1.0arcsec'], # Example pixel size
    #     specmode='mfs', # Multi-frequency synthesis (continuum)
    #     gridder='standard',
    #     weighting='natural',
    #     deconvolver=deconvolver_alg,
    #     niter=num_iter,
    #     threshold=threshold_val,
    #     interactive=False # Run non-interactively
    # )
    print("  (This would run the imaging and deconvolution process)")
    print("  Outputs: output_image.image.fits (Clean image), .residual, .psf, .model etc.")
    
except ImportError:
    print("\nNOTE: casatasks not found. Cannot execute CASA command.")
except Exception as e:
     print(f"\nAn error occurred (conceptual): {e}")

print("-" * 20)

# Explanation: This code conceptually illustrates how the CASA `tclean` task, which 
# performs both imaging (gridding, FFT) and deconvolution (e.g., CLEAN), might be 
# called from within a Python script using the `casatasks` interface.
# 1. It defines example input parameters: the visibility data file (`vis_file`, typically 
#    a Measurement Set), the base name for output images (`imagename`), the number of 
#    CLEAN iterations (`niter`), the CLEAN stopping threshold (`threshold`), and the 
#    choice of deconvolver (`deconvolver='clark'`). 
# 2. It conceptually calls the `tclean` function with these parameters, plus others 
#    needed to define the imaging grid (imsize, cell) and data weighting.
# 3. It notes the typical output files generated by `tclean`, including the final 
#    restored CLEAN image (`.image.fits`), the residual map (`.residual`), the dirty 
#    beam (`.psf`), and the CLEAN component model (`.model`).
# This shows how Python is used to *script* complex radio astronomy processing tasks 
# provided by dedicated packages like CASA.
```

**52.4 Radio Data Formats (FITS, Measurement Set, CASA Tables)**

Radio astronomy utilizes several data formats for storing raw visibility data, calibrated data, images, and ancillary information. Familiarity with these formats is necessary for accessing and processing radio data.

**FITS (Flexible Image Transport System):** While originally developed for optical/IR images, FITS (Sec 1.5) remains widely used in radio astronomy, particularly for storing **final image products** (dirty images, CLEAN component models, restored CLEAN images, residual maps, spectral cubes) generated by imaging pipelines. Radio images are often stored as standard FITS image HDUs. Spectral cubes (Intensity vs. RA, Dec, Frequency/Velocity) are typically stored either as 3D FITS image HDUs or sometimes as FITS binary tables. FITS headers contain crucial WCS information defining the spatial and spectral coordinates, as well as metadata about the observation, processing steps (`HISTORY` cards), and potentially information about the synthesized beam (stored in header keywords or a separate `BEAM` table extension). Libraries like `astropy.io.fits` (Sec 1.6) are used to read and write radio FITS images and tables.

**Measurement Set (MS):** This is the standard format for storing **raw and calibrated interferometric visibility data**, originating from the AIPS++ / CASA lineage. An MS is not a single file, but a **directory structure** containing multiple tables managed by the **CASACORE** table system. Key tables within the MS directory include:
*   `MAIN`: Stores the visibility data itself (complex values, weights, flags) along with time and baseline (antenna pair) information. This is usually the largest table.
*   `ANTENNA`: Antenna positions and metadata.
*   `FIELD`: Source field positions and names.
*   `SPECTRAL_WINDOW`: Frequency setup information (channel frequencies, widths).
*   `SOURCE`: Source properties.
*   `POLARIZATION`: Polarization setup (e.g., RR, LL, RL, LR or XX, YY, XY, YX).
*   Various calibration tables (`POINTING`, `CALIBRATION`, etc.) might also be included or stored separately.
The MS format is highly flexible, capable of storing multi-frequency, multi-polarization data with complex flagging and calibration information. It is the primary input format for CASA tasks (`tclean`, `applycal`, `uvcontsub`, etc.) and can also be read/written by other tools and libraries that link against CASACORE libraries (though direct manipulation outside CASA can be complex).

**CASA Image Format / Tables:** Within the CASA ecosystem, images are often stored in a specific **CASA image format**, which is also a directory structure containing data arrays and metadata managed by the CASACORE table system. CASA also uses its own table format (again, directories containing files) for various purposes, including storing CLEAN component models or calibration solutions. While CASA provides tasks (`exportfits`) to convert its internal image and table formats to FITS for broader compatibility, analysis within the CASA environment often operates directly on these native formats. Python access is primarily through `casatools` (e.g., the `image` tool or `table` tool).

**Other Formats:**
*   **SIGPROC Filterbank (`.fil`):** A common format for storing dynamic spectra (frequency vs. time) from single-dish surveys or pulsar observations, often used as input for pulsar/FRB search pipelines. Libraries like `your` (App II) or custom readers can parse this format.
*   **PSRFITS:** A FITS-based standard specifically designed for storing pulsar data (folded profiles, search-mode data, timing information). `astropy.io.fits` can often read PSRFITS files, but specialized pulsar libraries (`PINT`, `libstempo`) provide higher-level tools for analysis.
*   **UVFITS:** An older FITS-based standard for storing visibility data, sometimes still encountered but largely superseded by the Measurement Set for modern interferometers like VLA and ALMA. `astropy.io.fits` might handle some variants.

Working with radio data often requires familiarity with multiple formats. FITS is common for final image products and accessible via `astropy`. The Measurement Set is standard for visibility data and primarily handled by CASA (or tools linking CASACORE). Understanding which format contains which type of data (visibilities vs. images) and the primary software used to interact with it (CASA vs. Astropy) is essential. Python scripts often serve as the interface layer, calling CASA tasks via `casatasks`/`casatools` to process MS data or using `astropy.io.fits` to analyze the final FITS images or cubes.

**(No code examples needed here as it describes formats rather than operations.)**

**52.5 Data Analysis: Spectral Cubes, Polarization, Pulsars, FRBs**

Beyond the core task of imaging continuum sources, radio astronomy involves diverse analysis techniques tailored to specific scientific goals and data types, often requiring specialized computational methods implemented in Python or dedicated packages.

**Spectral Line Cube Analysis:** Interferometers and single dishes often observe spectral lines (like the 21cm HI line, CO rotational transitions, recombination lines) producing 3D data cubes (RA, Dec, Frequency/Velocity). Analyzing these cubes involves:
*   **Visualization:** Creating channel maps (images at specific velocity channels), position-velocity (PV) diagrams (slices through the cube showing velocity structure along a spatial axis), and potentially 3D rendering.
*   **Moment Maps:** Calculating moments of the spectrum along the line of sight (velocity axis) for each spatial pixel. Moment 0 gives integrated intensity (gas distribution), Moment 1 gives intensity-weighted velocity (velocity field, revealing rotation/infall/outflow), Moment 2 gives velocity dispersion (line width, related to turbulence or unresolved kinematics).
*   **Spectral Fitting:** Fitting profiles (e.g., Gaussians) to spectra extracted from specific locations or regions to measure line properties (intensity, velocity, width, multiple components).
The `spectral-cube` Astropy affiliated package provides powerful tools for reading, manipulating, analyzing, and visualizing spectral cubes in Python.

**Polarization Analysis:** Radio emission, particularly synchrotron radiation, is often linearly polarized, providing information about magnetic field structure and plasma properties. Interferometers measure polarization correlations (e.g., RR, LL, RL, LR or XX, YY, XY, YX) which can be combined to form **Stokes parameters** (I, Q, U, V). Analyzing polarization data involves:
*   **Calibration:** Correcting for instrumental polarization leakage and calibrating polarization angles.
*   **Imaging Stokes Parameters:** Creating images of Stokes I (total intensity), Q and U (linear polarization), and V (circular polarization).
*   **Calculating Polarization Angle (χ) and Fraction (p):** `p = sqrt(Q² + U²) / I`, `χ = 0.5 * arctan2(U, Q)`.
*   **Faraday Rotation Measure (RM) Synthesis:** Analyzing the change in polarization angle (χ ∝ λ²) across multiple frequency channels to determine the Rotation Measure (RM ∝ ∫ n<0xE1><0xB5><0x8A> B<0xE2><0x82><0x96><0xE2><0x82><0x96> ds), which probes the line-of-sight magnetic field strength and electron density.
CASA provides tasks for polarization calibration and imaging. Specialized Python tools might be used for RM synthesis.

**Pulsar Timing:** Pulsars are rapidly rotating neutron stars emitting beamed radio waves. Their highly regular pulses act as precise clocks. **Pulsar timing** involves measuring the arrival times of pulses over long periods (years to decades) with high precision (microseconds or better). Analyzing these times of arrival (TOAs) allows:
*   **Measuring Pulsar Parameters:** Determining spin period (P), period derivative (Ṗ), position, proper motion, binary orbital parameters (if applicable) by fitting a detailed timing model to the TOAs.
*   **Testing General Relativity:** Using binary pulsars to test predictions of GR (e.g., orbital decay due to GW emission).
*   **Detecting Gravitational Waves:** Using Pulsar Timing Arrays (PTAs) to search for correlated variations in TOAs from many pulsars caused by nano-Hertz GWs.
Software packages like **TEMPO/TEMPO2** (primarily Fortran) and modern Python libraries like **PINT** (`pint-pulsar`) and **libstempo** provide tools for reading pulsar data formats (e.g., TOA files, PSRFITS), defining timing models, fitting models to data (often using least squares or MCMC), and analyzing timing residuals.

**Fast Radio Burst (FRB) Searches:** FRBs are enigmatic, bright, millisecond-duration radio bursts originating from extragalactic distances. Searching for them involves processing large volumes of radio survey data (often filterbank format):
*   **De-dispersion:** Correcting for interstellar/intergalactic dispersion (Sec 51.1, App 41.A) is crucial. Since the DM is unknown, this requires trying many different trial DM values, often performed using FFT-based de-dispersion ("coherent") or simpler time-domain shifting ("incoherent"). This is computationally intensive.
*   **Candidate Searching:** Applying algorithms (like matched filtering with template pulse shapes or peak detection algorithms like Heimdall/AMBER) to the de-dispersed time series for each trial DM to identify potential burst candidates exceeding a significance threshold.
*   **Candidate Classification/Vetting:** Using signal properties (SNR, width, DM consistency, frequency structure) and potentially machine learning classifiers to distinguish real FRBs from radio frequency interference (RFI) and noise fluctuations.
Dedicated software packages (often C/C++/CUDA based for performance, e.g., `PRESTO`, `heimdall`, `peasoup`) are used for large-scale searches, often orchestrated by Python scripts or workflow managers.

These diverse analysis areas demonstrate the breadth of computational techniques employed in radio astronomy beyond basic imaging. Python, combined with core libraries like Astropy/NumPy/SciPy and specialized packages (CASA tools, spectral-cube, PINT, Gammapy for related high-energy aspects), provides a powerful environment for performing many of these analyses, often serving as the interface to underlying compiled code for performance-critical steps like deconvolution, de-dispersion, or timing model fitting.

**(No single code example captures all these diverse topics well. App 52.B focuses on spectral cubes.)**

**52.6 Python Tools: Astropy, CASA Tools, Specialised Libraries**

The Python ecosystem provides a rich set of libraries for performing various tasks in radio astronomy data analysis, ranging from fundamental operations to specialized techniques. While some core processing steps for interferometry (like calibration and deconvolution) are still often dominated by dedicated packages like CASA, Python serves as the primary language for scripting, higher-level analysis, visualization, and integration with other tools.

**Astropy and Affiliated Packages:** The core `astropy` package is indispensable:
*   `astropy.io.fits`: Reading and writing FITS images and tables, including radio images and spectral cubes.
*   `astropy.wcs`: Handling World Coordinate System information essential for interpreting spatial and spectral axes in FITS files.
*   `astropy.coordinates`: Representing and transforming sky coordinates.
*   `astropy.units` / `astropy.constants`: Managing physical units and constants.
*   `astropy.convolution`: Applying kernels for smoothing or PSF convolution.
*   `astropy.stats`: Statistical functions (sigma clipping, etc.).
*   `spectral-cube` (Affiliated Package): Provides the `SpectralCube` class for reading, manipulating, analyzing (moments, masking, smoothing), and visualizing FITS spectral cubes (Sec 52.5, App 52.B). Highly recommended for spectral line work.
*   `radio-beam` (Affiliated Package): Specifically designed for handling and manipulating synthesized beam information (PSFs) associated with radio interferometric images, including reading/writing beam info from/to FITS headers and performing beam convolutions/deconvolutions.

**CASA Integration (`casatools`, `casatasks`):** The Common Astronomy Software Applications (CASA) package is the standard for calibration and imaging of data from major interferometers like VLA and ALMA. While CASA itself is a large monolithic application often run interactively or via scripts in its own environment, modern versions expose much of their functionality through Python modules:
*   `casatasks`: Provides Python functions corresponding to standard CASA tasks (e.g., `tclean`, `applycal`, `gaincal`, `bandpass`, `uvcontsub`, `imstat`, `exportfits`). These can be imported and called directly from a Python script (if run within a CASA-aware environment or if `casatasks` is installed standalone, which can be complex).
*   `casatools`: Provides lower-level access to CASA tools (objects with methods) for more fine-grained control over data manipulation (e.g., `image` tool (`ia`), `table` tool (`tb`), `ms` tool (`ms`), `calibrater` tool (`cb`)).
Using these modules allows scripting complex CASA reduction and imaging workflows entirely within Python, facilitating automation, parameterization, and integration with other Python-based analysis. Requires a compatible CASA installation.

**Specialized Libraries:**
*   **Visibility Data:** While CASA is standard for MS files, libraries like `python-casacore` (bindings to CASACORE table system) or potentially interfaces within `pyradiosky` (part of `pyuvdata` ecosystem) might offer alternative ways to read MS data, though often with less functionality than CASA tools. `pyuvdata` focuses more on formats used in EoR/cosmology radio experiments (UVFITS, MIRIAD, UVH5).
*   **Pulsar Timing:** `PINT` (`pint-pulsar`) is a modern, Astropy-affiliated package for high-precision pulsar timing analysis, providing tools for reading TOAs/ephemerides, defining timing models, performing fits, and analyzing residuals. `libstempo` offers bindings to TEMPO2. `your` reads various pulsar data formats.
*   **FRB/Pulsar Search:** While core search algorithms are often in compiled code (`PRESTO`, `heimdall`), Python libraries might be used for pipeline orchestration, candidate visualization (`waterfaller`), or interfacing with search outputs. Libraries for de-dispersion might exist (e.g., `dedisp`).
*   **Single Dish:** Packages specific to certain telescopes might exist, or analysis might rely on combinations of `astropy`, `spectral-cube`, and custom scripts.

**General Scientific Python:** Libraries covered extensively elsewhere in the book remain crucial:
*   `numpy`: For all array manipulations.
*   `scipy`: For numerical integration, optimization, interpolation, signal processing (FFTs, filters).
*   `matplotlib`: For visualization of images, spectra, cubes, uv-coverage, etc.
*   `pandas`: For handling catalogs or metadata tables.
*   `h5py`: If dealing with data stored in HDF5.

```python
# --- Code Example: Basic Interaction with CASA tools (Conceptual) ---
# Note: Requires working CASA installation with casatools/tasks accessible in Python.
# This illustrates the TYPE of interaction, syntax details depend on CASA version.

print("Conceptual Interaction with CASA Tools from Python:")

try:
    # Import specific tools needed
    from casatools import image as iatool 
    # from casatasks import imstat # Import specific task if needed
    
    casa_ok = True
    print("\n(Conceptual: casatools imported successfully)")
    
    # Create an image tool instance
    ia = iatool()
    
    # --- Example: Open a CASA image and get statistics ---
    image_name = 'my_casa_image.image' # CASA image format (directory)
    print(f"\nConceptually opening CASA image: {image_name}")
    # Assume image exists. Need to use 'open' method.
    # if ia.open(image_name):
    #     print("  Image opened.")
    #     # Get image summary
    #     summary_dict = ia.summary(list=False) # Get dict, not printed list
    #     print(f"  Image Shape: {summary_dict.get('shape', 'N/A')}")
    #     # Calculate statistics using image tool methods or imstat task
    #     stats = ia.statistics(list=False) # Example method call
    #     print(f"  Image Stats (e.g., Mean): {stats.get('mean', 'N/A')}")
    #     # Close the image tool
    #     ia.close()
    #     print("  Image closed.")
    # else:
    #     print(f"  Error: Failed to open image {image_name}")

    print("\n(Actual CASA tool/task usage requires specific CASA environment and methods)")

except ImportError:
    casa_ok = False
    print("\nNOTE: casatools/casatasks not found. Skipping CASA example.")
except Exception as e:
    casa_ok = False
    print(f"\nAn error occurred (check CASA setup): {e}")

print("-" * 20)

# Explanation: This code *conceptually* illustrates interacting with CASA from Python.
# 1. It imports the `image` tool from `casatools`.
# 2. It creates an instance `ia` of the image tool.
# 3. It conceptually shows opening a CASA format image using `ia.open()`.
# 4. It conceptually calls methods like `ia.summary()` or `ia.statistics()` to get 
#    information or perform calculations on the open image.
# 5. It conceptually closes the image tool using `ia.close()`.
# **Important:** The exact method names, arguments, and return values depend heavily 
# on the specific CASA version and tool being used. This only shows the general pattern 
# of importing tools/tasks and calling their methods/functions from Python when 
# working within a CASA-enabled environment. Consult CASA documentation for actual usage.
```

The Python ecosystem provides a powerful and flexible environment for radio astronomy analysis. While specialized packages like CASA remain central for low-level interferometric data reduction and imaging due to their optimized, compiled algorithms, Python serves as the overarching language for scripting these tasks, performing higher-level analysis on the resulting data products (FITS images/cubes, catalogs) using libraries like Astropy, spectral-cube, NumPy/SciPy, and visualizing results with Matplotlib. For specialized domains like pulsar timing or FRB searches, dedicated Python or Python-interfaced libraries are also crucial components of the radio astronomer's computational toolkit.

---
**Application 52.A: Simulating and Imaging Interferometer Visibilities**

**(Paragraph 1)** **Objective:** Use fundamental Python libraries (`numpy`, `astropy`) to simulate simplified radio interferometer **visibility data** for a basic sky model (e.g., two point sources) and then perform the core **imaging steps** conceptually outlined in Sec 52.2: gridding the visibilities and applying a Fast Fourier Transform (FFT) to generate the dirty image and dirty beam. This application illustrates the Fourier relationship between sky brightness and visibilities and the effect of sparse uv-sampling.

**(Paragraph 2)** **Astrophysical Context:** Radio interferometers measure samples of the Fourier transform (visibilities) of the sky brightness distribution at specific spatial frequencies (uv-coordinates) determined by the antenna baseline projections. To create an image, these irregularly sampled visibilities must be placed onto a regular grid (gridding) before an inverse FFT can be applied. The incompleteness of the uv-sampling results in a 'dirty' image contaminated by sidelobes (the dirty beam pattern). This simulation helps visualize this fundamental process.

**(Paragraph 3)** **Data Source/Model:**
    *   **Sky Model:** Define a simple model, e.g., two point sources with given positions (l₁, m₁), (l₂, m₂) relative to the image center and fluxes S₁, S₂.
    *   **uv-coverage:** Simulate sampling the uv-plane. Generate a set of random or patterned (u, v) coordinates representing the baseline samples obtained during an observation (units of wavelength).
    *   **Visibilities:** Calculate the complex visibility V(u,v) for each sampled (u,v) point based on the sky model using the Fourier transform relation: V(u,v) = Σᵢ Sᵢ * exp[-2πi (u lᵢ + v mᵢ)]. Add some Gaussian noise to the visibilities.

**(Paragraph 4)** **Modules Used:** `numpy` (for arrays, FFTs, complex numbers), `astropy.constants` (optional, for physical scales), `astropy.units` (optional, for angles), `matplotlib.pyplot` (for visualization).

**(Paragraph 5)** **Technique Focus:** Simulating the core interferometry concepts. (1) Calculating complex visibilities from a sky model using the discrete Fourier transform sum. (2) Implementing a simplified gridding algorithm (e.g., nearest neighbor assignment) to place visibilities onto a 2D NumPy array representing the uv-grid. (3) Creating a corresponding uv-sampling grid (mask). (4) Using `numpy.fft.fft2` (and `fftshift`/`ifftshift`) to compute the inverse transform of the gridded visibilities (dirty image) and the transform of the sampling grid (dirty beam). (5) Visualizing the uv-coverage, dirty beam (PSF), and dirty image.

**(Paragraph 6)** **Processing Step 1: Define Sky Model and uv-Coverage:** Define source parameters (positions l₁, m₁; l₂, m₂ in radians relative to center; fluxes S₁, S₂). Generate `n_vis` random `u_coords`, `v_coords` (in wavelengths).

**(Paragraph 7)** **Processing Step 2: Calculate Visibilities:** Loop through uv-points. For each (u, v), calculate the complex visibility `V = S₁*exp(-2pi*1j*(u*l₁+v*m₁)) + S₂*exp(-2pi*1j*(u*l₂+v*m₂))`. Add complex Gaussian noise `np.random.normal(0, noise_sigma, 2).view(complex)`. Store `V` values.

**(Paragraph 8)** **Processing Step 3: Gridding:** Define image size `npix`, FoV, pixel scale, and corresponding `uv_max`, `uv_cell_size`. Create empty complex `uv_grid` and real `uv_sampling` arrays (size `npix` x `npix`). Implement simple gridding: loop through visibilities, calculate corresponding grid indices `(v_idx, u_idx)` from `(u, v)` coordinates, add `V` to `uv_grid[v_idx, u_idx]` (or average), and set `uv_sampling[v_idx, u_idx] = 1`.

**(Paragraph 9)** **Processing Step 4: FFT and Image Formation:** Apply necessary FFT shifts (`np.fft.ifftshift`) to center the DC component for the FFT. Calculate `dirty_image = np.fft.fftshift(np.fft.fft2(shifted_uv_grid)).real`. Calculate `dirty_beam = np.fft.fftshift(np.fft.fft2(shifted_uv_sampling)).real`.

**(Paragraph 10)** **Processing Step 5: Visualize:** Create a 3-panel plot: (a) Scatter plot of `u_coords`, `v_coords` (uv-coverage). (b) `imshow` of the central part of `dirty_beam` (log scale often helps) showing the central peak and sidelobes. (c) `imshow` of the `dirty_image` showing the two point sources convolved with the dirty beam sidelobes.

**Output, Testing, and Extension:** Output includes the plots of uv-coverage, dirty beam, and dirty image. **Testing:** Verify the dirty image shows peaks at the locations corresponding to the input source positions (l, m). Check that the dirty beam has a central peak and surrounding sidelobe structure related to the uv-coverage pattern. Increase visibility noise and observe its effect on the dirty image. **Extensions:** (1) Implement a proper gridding function using a convolution kernel (e.g., Gaussian or boxcar). (2) Simulate uv-tracks from Earth rotation synthesis instead of random points. (3) Implement a basic CLEAN algorithm (Sec 52.3) iteratively subtracting the dirty beam scaled by the peak residual from the dirty image to deconvolve it. (4) Use a more complex sky model (e.g., a Gaussian source). (5) Explore different visibility weighting schemes (natural, uniform) during gridding and their effect on the beam/image.

```python
# --- Code Example: Application 52.A ---
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from numpy.fft import fft2, ifftshift, fftshift

print("Simulating and Imaging Simple Interferometer Visibilities:")

# Step 1: Define Sky Model and uv-Coverage
# Two point sources
sources = [
    {'l': 0.0 * u.arcsec.to(u.rad), 'm': 0.0 * u.arcsec.to(u.rad), 'flux': 1.0},
    {'l': 2.0 * u.arcsec.to(u.rad), 'm': 1.0 * u.arcsec.to(u.rad), 'flux': 0.5}
]
n_vis = 1000
uv_max_sim = 50e3 # Max baseline in wavelengths (e.g., 50 kLambda)
np.random.seed(1)
# Simulate random uv points (more dense near center typically)
uv_dist = np.random.chisquare(df=2, size=n_vis) * uv_max_sim / 5 
uv_angle = np.random.uniform(0, 2*np.pi, n_vis)
u_coords = uv_dist * np.cos(uv_angle)
v_coords = uv_dist * np.sin(uv_angle)
print(f"\nDefined sky model ({len(sources)} sources) and uv coverage ({n_vis} points).")

# Step 2: Calculate Visibilities
visibilities = np.zeros(n_vis, dtype=complex)
noise_std = 0.1 # Visibility noise level
for i in range(n_vis):
    u, v = u_coords[i], v_coords[i]
    v_signal = 0j
    for src in sources:
        v_signal += src['flux'] * np.exp(-2j * np.pi * (u * src['l'] + v * src['m']))
    visibilities[i] = v_signal + np.random.normal(0, noise_std, 2).view(complex)
print("Calculated complex visibilities with noise.")

# Step 3: Gridding (Simplified Nearest Neighbor)
npix = 256
# Determine uv cell size needed to avoid aliasing image FoV
# FoV ~ 1 / uv_cell_size (in radians)
# Image pixel scale ~ 1 / (2 * uv_max) (in radians)
# Choose image pixel scale, determine uv_max needed, then uv_cell
pixel_scale_rad = (1.0 * u.arcsec).to(u.rad).value # Example: target 1 arcsec resolution -> uv_max ~ 1/(2*pix_scale)
uv_max_grid = 1.0 / (2.0 * pixel_scale_rad) 
uv_cell_size = (2.0 * uv_max_grid) / npix
print(f"Image pixel scale ~ {pixel_scale_rad*u.rad.to(u.arcsec):.2f} arcsec")
print(f"UV grid extent +/-{uv_max_grid:.0f}, cell size {uv_cell_size:.0f}")

uv_grid = np.zeros((npix, npix), dtype=complex)
uv_sampling = np.zeros((npix, npix), dtype=float)

# Grid visibilities using nearest grid point (highly simplified)
u_indices = np.clip(np.round(u_coords / uv_cell_size + npix/2).astype(int), 0, npix-1)
v_indices = np.clip(np.round(v_coords / uv_cell_size + npix/2).astype(int), 0, npix-1)

# Average visibilities landing in the same cell (simple mean)
counts = np.zeros_like(uv_grid, dtype=int)
np.add.at(uv_grid, (v_indices, u_indices), visibilities) # Sum visibilities
np.add.at(counts, (v_indices, u_indices), 1) # Count hits per cell
non_zero = counts > 0
uv_grid[non_zero] /= counts[non_zero] # Average
uv_sampling[non_zero] = 1.0 # Mark sampled cells
print("Performed simplified gridding (nearest grid point, averaged).")

# Step 4: FFT and Image Formation
print("Performing FFTs...")
# Ensure DC component (u=v=0) is at the center for FFT using ifftshift
shifted_uv_grid = ifftshift(uv_grid)
shifted_uv_sampling = ifftshift(uv_sampling)
# FFT gives image; use fftshift to put DC (center of image) in middle
dirty_image = fftshift(fft2(shifted_uv_grid)).real 
# FFT of sampling function gives dirty beam
dirty_beam = fftshift(fft2(shifted_uv_sampling)).real
print("FFTs complete.")

# Step 5: Visualize
print("Generating plots...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# uv-coverage
axes[0].plot(u_coords/1e3, v_coords/1e3, '.', markersize=1, alpha=0.3)
axes[0].set_xlabel("u (kλ)"); axes[0].set_ylabel("v (kλ)")
axes[0].set_title("uv-coverage")
axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)
# Dirty Beam (PSF) - log scale, central part
beam_zoom = npix // 8 # Show central +/- beam_zoom pixels
beam_slice = dirty_beam[npix//2-beam_zoom:npix//2+beam_zoom, npix//2-beam_zoom:npix//2+beam_zoom]
beam_norm = beam_slice / np.max(beam_slice) # Normalize peak to 1
im1 = axes[1].imshow(np.log10(np.abs(beam_norm) + 1e-4), cmap='gray_r', interpolation='nearest',
                     extent=[-beam_zoom, beam_zoom, -beam_zoom, beam_zoom])
axes[1].set_title("Dirty Beam Center (Log Scale)")
axes[1].set_xlabel("Image Pixels offset"); axes[1].set_ylabel("Image Pixels offset")
# Dirty Image
img_zoom = npix // 4 # Show central part of image
img_slice = dirty_image[npix//2-img_zoom:npix//2+img_zoom, npix//2-img_zoom:npix//2+img_zoom]
img_max = np.percentile(img_slice, 99.9)
im2 = axes[2].imshow(img_slice, cmap='magma', origin='lower', interpolation='nearest',
                     vmin=-0.1*img_max, vmax=img_max, 
                     extent=[-img_zoom*pixel_scale_rad*u.rad.to(u.arcsec), 
                              img_zoom*pixel_scale_rad*u.rad.to(u.arcsec)]*2)
axes[2].set_title("Dirty Image (Central Region)")
axes[2].set_xlabel("Offset (arcsec)"); axes[2].set_ylabel("Offset (arcsec)")

fig.tight_layout()
# plt.show()
print("Plots generated.")
plt.close(fig)

print("-" * 20)
```

**Application 52.B: Basic Spectral Line Cube Analysis**

**(Paragraph 1)** **Objective:** This application demonstrates essential analysis steps for a radio spectral line data cube (Intensity vs. RA, Dec, Velocity/Frequency) using the `spectral-cube` package (Sec 52.6). We will load a FITS cube, calculate and visualize **moment maps** (integrated intensity, velocity field, dispersion), extract a spectrum from a specific spatial region, and create a **Position-Velocity (PV) diagram** along a spatial slice.

**(Paragraph 2)** **Astrophysical Context:** Observing spectral lines (like HI 21cm or CO rotational lines) in radio astronomy provides crucial information about the distribution, kinematics, and physical conditions of gas in galaxies and the interstellar medium. Data is often obtained as 3D cubes. Moment maps provide concise 2D summaries: moment 0 traces gas column density/distribution, moment 1 traces the large-scale velocity field (e.g., galactic rotation), and moment 2 traces velocity dispersion (turbulence, beam smearing, unresolved motions). Spectra from specific locations probe conditions locally, while PV diagrams reveal kinematic structure along chosen spatial axes (e.g., rotation curve along major axis).

**(Paragraph 3)** **Data Source:** A FITS file (`spectral_cube.fits`) containing a spectral line data cube. This could be HI data for a nearby galaxy, CO data for a molecular cloud, etc. The FITS header must contain valid WCS information for all three axes (two spatial, one spectral usually in frequency or velocity). We can use sample data available online or simulate a simple cube.

**(Paragraph 4)** **Modules Used:** `spectral_cube.SpectralCube` (requires `pip install spectral-cube`), `astropy.io.fits` (used internally by spectral-cube), `astropy.units` (for handling units), `numpy`, `matplotlib.pyplot`.

**(Paragraph 5)** **Technique Focus:** Using the `spectral-cube` library for cube analysis. (1) Loading the FITS cube into a `SpectralCube` object using `SpectralCube.read()`. (2) Accessing cube properties (shape, WCS, spectral axis units). (3) Calculating moment maps using the `.moment(order=...)` method. (4) Visualizing moment maps using Matplotlib, potentially using WCSAxes for correct coordinate display. (5) Extracting a spectrum corresponding to a spatial pixel or region using slicing or helper methods (`.sum` or `.mean` over spatial axes). (6) Creating a PV diagram by extracting a spatial slice (e.g., `cube.subcube_from_ds9region(...)` or slicing along one spatial dimension) and then visualizing the resulting 2D slice (Velocity vs. Position) using `imshow` or `pcolormesh`.

**(Paragraph 6)** **Processing Step 1: Load Spectral Cube:** Import `SpectralCube` and `u`. Load the cube: `cube = SpectralCube.read('spectral_cube.fits')`. Print basic info: `print(cube)`. Check spectral axis units and potentially convert to velocity: `cube = cube.with_spectral_unit(u.km/u.s, velocity_convention='radio')`.

**(Paragraph 7)** **Processing Step 2: Calculate Moment Maps:** Calculate moment 0 (integrated intensity), moment 1 (velocity field), and moment 2 (velocity dispersion): `mom0 = cube.moment(order=0)`, `mom1 = cube.moment(order=1)`, `mom2 = cube.moment(order=2)`. These return Astropy `Quantity` objects with spatial dimensions.

**(Paragraph 8)** **Processing Step 3: Visualize Moment Maps:** Use Matplotlib's `imshow` (or better, `WCSAxes` projection from `astropy.visualization`) to display each moment map. For `mom0`, show intensity. For `mom1`, use a diverging colormap centered on systemic velocity to show rotation. For `mom2`, show line width. Add colorbars with correct units. Save plots.

**(Paragraph 9)** **Processing Step 4: Extract Spectrum:** Define spatial coordinates (pixel or sky coordinates). Extract the 1D spectrum at that location: `spectrum = cube[:, y_pix, x_pix]` (using pixel indices) or use `cube.subcube_from_ds9region(...)` for regions. Plot the extracted spectrum `spectrum.value` vs `cube.spectral_axis.value`.

**(Paragraph 10)** **Processing Step 5: Create PV Diagram:** Define a slice (e.g., along the major axis of a galaxy). Use `cube.spatial_slab()` or slice notation to extract a 2D slice (Position vs. Velocity). Determine appropriate axis labels and units. Use `imshow` or `pcolormesh` to visualize the PV diagram, showing velocity structure as a function of position along the slice. Save plot.

**Output, Testing, and Extension:** Output includes saved FITS files or plots for the moment maps, extracted spectrum, and PV diagram. **Testing:** Verify the units of the moment maps and spectral axes are correct. Check if the velocity field (moment 1) shows expected patterns (e.g., rotation). Check if the extracted spectrum and PV diagram show plausible line emission features. **Extensions:** (1) Apply spectral smoothing or baseline subtraction to the cube before analysis. (2) Fit Gaussian profiles to the extracted spectrum using `astropy.modeling` or `specutils`. (3) Perform moment calculations only on channels with significant signal (using a mask generated by `cube.sigma_threshold_mask()` or similar). (4) Use more sophisticated region definitions (e.g., from `regions` package) for extracting spectra or calculating moments over specific areas. (5) Collapse the cube spatially to get the integrated spectrum of the entire source.

```python
# --- Code Example: Application 52.B ---
# Note: Requires spectral-cube, astropy, matplotlib. 
# Needs a FITS spectral cube file - we simulate one here.

try:
    from spectral_cube import SpectralCube
    from astropy.io import fits
    from astropy import units as u
    from astropy.wcs import WCS
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    spec_cube_ok = True
except ImportError:
    spec_cube_ok = False
    print("NOTE: spectral-cube, astropy, or matplotlib not installed. Skipping.")

print("Basic Spectral Line Cube Analysis using spectral-cube:")

# Step 1: Simulate and Load Spectral Cube
cube_filename = "sim_co_cube.fits"

def create_dummy_cube(filename):
    """Creates a simple FITS cube with a rotating disk pattern."""
    print(f"Creating dummy FITS cube: {filename}")
    nx, ny, nv = 50, 40, 30 # Spatial x, Spatial y, Velocity
    # Create data (e.g., Gaussian disk with linear rotation)
    data = np.zeros((nv, ny, nx))
    v_axis = np.linspace(-150, 150, nv) # km/s
    x_coords, y_coords = np.meshgrid(np.arange(nx), np.arange(ny))
    center_x, center_y = nx // 2, ny // 2
    radius = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    # Velocity field (simple solid body rotation + systemic)
    v_sys = 0.0
    v_rot_slope = 5.0 # km/s per pixel
    # Angle for rotation axis (e.g. along y-axis)
    pos_angle = np.arctan2(y_coords - center_y, x_coords - center_x)
    v_field = v_sys + v_rot_slope * (x_coords - center_x) # Simple rotation gradient
    
    sigma_v = 20.0 # Velocity dispersion km/s
    amplitude = np.exp(-radius**2 / (2 * (nx/5)**2)) # Gaussian spatial profile
    
    for i in range(nv):
        vel = v_axis[i]
        # Gaussian profile centered on v_field, height=amplitude
        data[i, :, :] = amplitude * np.exp(-(vel - v_field)**2 / (2 * sigma_v**2))
        
    data += np.random.normal(0, 0.05, data.shape) # Add noise
    
    # Create basic WCS header
    w = WCS(naxis=3)
    w.wcs.crpix = [center_x + 1, center_y + 1, nv // 2 + 1] # FITS 1-based center pixel
    w.wcs.cdelt = np.array([-0.5, 0.5, (v_axis[1]-v_axis[0])]) # Pixel scale (arcsec, km/s)
    w.wcs.crval = [266.0, -29.0, v_sys] # RA, Dec (deg), Velocity (km/s) at center
    w.wcs.ctype = ["RA---TAN", "DEC--TAN", "VELO-LSR"]
    w.wcs.cunit = ['deg', 'deg', 'km/s']
    header = w.to_header()
    header['BUNIT'] = 'K' # Brightness unit (e.g., Kelvin)
    
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(filename, overwrite=True)
    print("Dummy cube created.")

if spec_cube_ok:
    # Create dummy file if it doesn't exist
    if not os.path.exists(cube_filename): create_dummy_cube(cube_filename)

    try:
        # Load cube
        print(f"\nLoading cube: {cube_filename}")
        cube = SpectralCube.read(cube_filename)
        print(cube) # Display basic info

        # Optional: Convert spectral axis to km/s if needed (already is here)
        # cube = cube.with_spectral_unit(u.km/u.s, velocity_convention='radio')
        
        # Step 2: Calculate Moment Maps
        print("\nCalculating moment maps...")
        # Calculate moment 0 only over channels with signal > threshold (e.g. 3*sigma)
        noise_sigma = 0.05 # Assumed noise level
        mask = cube > noise_sigma * 3 # Create boolean mask
        masked_cube = cube.with_mask(mask) # Apply mask

        mom0 = masked_cube.moment(order=0) # Integrated intensity
        mom1 = masked_cube.moment(order=1) # Intensity-weighted velocity
        mom2 = masked_cube.moment(order=2) # Velocity dispersion
        print("Moment maps calculated.")
        
        # Step 3: Visualize Moment Maps
        print("Generating moment map plots...")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), subplot_kw={'projection': mom0.wcs})
        
        im0 = axes[0].imshow(mom0.value, origin='lower', cmap='viridis')
        axes[0].set_title("Moment 0 (Integrated Intensity)"); fig.colorbar(im0, ax=axes[0], label=mom0.unit.to_string())
        
        im1 = axes[1].imshow(mom1.value, origin='lower', cmap='coolwarm', vmin=-80, vmax=80) # Center cmap
        axes[1].set_title("Moment 1 (Velocity Field)"); fig.colorbar(im1, ax=axes[1], label=mom1.unit.to_string())
        
        im2 = axes[2].imshow(mom2.value, origin='lower', cmap='plasma', vmin=0)
        axes[2].set_title("Moment 2 (Velocity Dispersion)"); fig.colorbar(im2, ax=axes[2], label=mom2.unit.to_string())
        
        for ax in axes: ax.coords.grid(True, color='grey', ls=':')
        fig.suptitle("Spectral Cube Moment Maps")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.show()
        print("Moment map plots generated.")
        plt.close(fig)

        # Step 4: Extract Spectrum
        center_x_pix, center_y_pix = cube.shape[2]//2, cube.shape[1]//2
        print(f"\nExtracting spectrum at pixel ({center_x_pix}, {center_y_pix})...")
        center_spectrum = cube[:, center_y_pix, center_x_pix]
        print("Spectrum extracted.")
        
        plt.figure(figsize=(8, 4))
        plt.plot(center_spectrum.spectral_axis.value, center_spectrum.value)
        plt.xlabel(f"Velocity ({center_spectrum.spectral_axis.unit})")
        plt.ylabel(f"Intensity ({center_spectrum.unit})")
        plt.title(f"Spectrum at Center Pixel")
        plt.grid(True, alpha=0.4)
        # plt.show()
        print("Spectrum plot generated.")
        plt.close()

        # Step 5: Create PV Diagram (Slice along central x-axis, y=center_y_pix)
        print("\nCreating Position-Velocity diagram along central row...")
        pv_slice = cube[:, center_y_pix, :] # Shape (n_vel, n_x)
        
        plt.figure(figsize=(8, 5))
        # extent=[x_min, x_max, v_min, v_max]
        # Need physical coordinates for x-axis
        x_coords_physical = cube.world[:, center_y_pix, :][0].ra.deg # Get RA values
        v_coords_physical = cube.spectral_axis.value
        plt.imshow(pv_slice.value, aspect='auto', origin='lower', cmap='Blues',
                   extent=[x_coords_physical[0], x_coords_physical[-1], 
                           v_coords_physical[0], v_coords_physical[-1]])
        plt.xlabel("RA (deg)") # Use WCS for proper labels if needed
        plt.ylabel(f"Velocity ({cube.spectral_axis.unit})")
        plt.title(f"PV Diagram along y={center_y_pix}")
        plt.colorbar(label=f"Intensity ({cube.unit})")
        # plt.show()
        print("PV diagram generated.")
        plt.close()

    except Exception as e:
        print(f"An error occurred: {e}")

    # Cleanup dummy file
    finally:
        if os.path.exists(cube_filename): os.remove(cube_filename)
        print(f"\nCleaned up {cube_filename}")
else:
    print("Skipping spectral cube execution.")

print("-" * 20)
```

**Chapter 52 Summary**

This chapter focused on the computational techniques specific to **radio astronomy**, covering both single-dish and interferometric observations. It emphasized the fundamental difference that **interferometers** measure **complex visibilities** in the uv-plane (Fourier domain), rather than direct images. The core process of **interferometric imaging** was detailed: **gridding** the irregularly sampled visibilities onto a regular grid (often using convolution kernels), applying an **FFT** to produce the **dirty image** (true sky convolved with the dirty beam/PSF), and the necessity of **deconvolution** algorithms like **CLEAN** (iteratively subtracting scaled dirty beams from peaks) or Maximum Entropy Methods (MEM) to remove sidelobe artifacts and estimate the true sky brightness. Common data formats were reviewed, including **FITS** (for images/cubes) and the standard **Measurement Set (MS)** directory structure (for visibility data, used by CASA).

Key radio data analysis techniques were discussed, including **spectral line cube analysis** (moment maps, PV diagrams, spectral fitting), handling **polarization data** (Stokes parameters, Faraday Rotation Measure synthesis), **pulsar timing** (fitting timing models to pulse arrival times), and searching for **Fast Radio Bursts (FRBs)** (requiring intensive de-dispersion and candidate searching). Essential Python tools were highlighted, notably **Astropy** and its affiliated packages (`spectral-cube`, `radio-beam`), and the **CASA** package accessed via its Python interface (`casatools`, `casatasks`) which remains central for low-level interferometric calibration and imaging. Two applications provided practical illustrations: conceptually simulating visibilities and performing the FFT to visualize uv-coverage, the dirty beam, and the dirty image; and using the `spectral-cube` package to load a FITS spectral cube, calculate and display moment maps, extract spectra, and create a Position-Velocity diagram.

---


**References for Further Reading (APA Format, 7th Edition):**

1.  **Thompson, A. R., Moran, J. M., & Swenson, G. W., Jr. (2017).** *Interferometry and Synthesis in Radio Astronomy* (3rd ed.). Springer. [https://doi.org/10.1007/978-3-319-44431-4](https://doi.org/10.1007/978-3-319-44431-4)
    *(Remains the essential, comprehensive textbook covering the fundamental principles of radio interferometry, visibility data, uv-coverage, calibration, imaging theory, and deconvolution algorithms like CLEAN, providing the theoretical underpinnings for Sec 52.1-52.3.)*

2.  **McMullin, J. P., Waters, B., Schiebel, D., Young, W., & Golap, K. (2007).** CASA Architecture and Applications. In D. C. Hines, A. H. Bridle, J. B. Tilanus, & R. D. Sramek (Eds.), *Astronomical Data Analysis Software and Systems XVI (ADASS XVI)* (ASP Conference Series, Vol. 376, p. 127). Astronomical Society of the Pacific. ([Link via ADS](https://ui.adsabs.harvard.edu/abs/2007ASPC..376..127M/abstract)) (See also CASA Documentation: [https://casadocs.readthedocs.io/en/stable/](https://casadocs.readthedocs.io/en/stable/))
    *(Describes the architecture of CASA, the standard software package for VLA/ALMA data reduction and analysis. The linked documentation is crucial for understanding Measurement Sets, CASA tasks like `tclean`, and the Python interface via `casatools`/`casatasks`, relevant to Sec 52.3, 52.4, 52.6.)*

3.  **Ginsburg, A., et al. (2019).** The Spectral Cube Toolbox. *Zenodo*. [https://doi.org/10.5281/zenodo.3557471](https://doi.org/10.5281/zenodo.3557471) (See also `spectral-cube` Documentation: [https://spectral-cube.readthedocs.io/en/latest/](https://spectral-cube.readthedocs.io/en/latest/))
    *(Zenodo link points to a specific version/workshop, but references the `spectral-cube` package introduced in the paper Ginsburg et al. 2018, JOSS, 3(31), 1001. The linked documentation is the primary resource for using this Astropy-affiliated package for spectral cube analysis discussed in Sec 52.5 and Application 52.B.)*

4.  **Lorimer, D. R., & Kramer, M. (2012).** *Handbook of Pulsar Astronomy*. Cambridge University Press. [https://doi.org/10.1017/CBO9780511808000](https://doi.org/10.1017/CBO9780511808000)
    *(A comprehensive handbook covering pulsar physics, observational techniques, data analysis including de-dispersion, searching, timing models, and relevant software (like TEMPO/PRESTO), providing context for Sec 52.5.)* (See also PINT documentation for modern Python tools: [https://nanograv-pint.readthedocs.io/en/latest/](https://nanograv-pint.readthedocs.io/en/latest/))

5.  **Petroff, E., Hessels, J. W. T., & Lorimer, D. R. (2019).** Fast radio bursts. *Astronomy and Astrophysics Review*, *27*(1), 4. [https://doi.org/10.1007/s00159-019-0116-6](https://doi.org/10.1007/s00159-019-0116-6)
    *(A review article summarizing the state of Fast Radio Burst research, including observational properties, theoretical models, and crucially, the search techniques involving de-dispersion and candidate identification discussed computationally in Sec 52.5.)*
