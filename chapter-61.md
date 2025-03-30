**Chapter 61: AstroOps: Instrument Calibration Workflows**

Building upon the digital telescope twin developed in Chapter 60, this chapter integrates it into a simulated operational workflow, focusing specifically on automating the crucial process of **instrument calibration** within an **AstroOps** framework (introduced conceptually in Chapter 59). Reliable scientific measurements require removing instrumental signatures and atmospheric effects (for ground-based data), a process achieved through calibration observations like bias frames, dark frames, and flat fields. This chapter explores how to define, simulate the acquisition of, process, and manage these essential calibration frames using Python scripts interacting with our telescope twin. We will discuss how to represent **calibration requirements** programmatically, define functions or methods to **simulate taking calibration exposures** (e.g., zero-second exposures for biases, long dark exposures, twilight or dome flats) using the twin's `expose` method with appropriate inputs. We then implement the core logic for **processing raw calibration frames** to create **master calibration files** (e.g., median-combining biases, scaling darks, combining and normalizing flats). Finally, strategies for **storing and managing** these master calibration products (e.g., naming conventions, associating them with specific dates or instrument configurations) are discussed, emphasizing how the AstroOps approach aims to automate and standardize these fundamental steps for ensuring data quality and reproducibility.

**61.1 Defining Calibration Needs (Bias, Dark, Flat)**

Before scientific observations can yield accurate measurements, instrumental artifacts and detector signatures must be carefully removed. Astronomical detectors like CCDs or IR arrays introduce systematic effects that need characterization through dedicated **calibration observations**. Establishing a robust calibration strategy, defining what calibrations are needed and how often, is a fundamental requirement for any observatory operation and a key component of an AstroOps workflow. The primary calibration frames for imaging data include bias frames, dark frames, and flat fields.

**Bias Frames:** These capture the baseline electronic signal or **offset** present in a detector readout even with zero illumination and zero exposure time. This "bias level" can vary across the detector, sometimes exhibiting fixed patterns (e.g., column variations, amplifier glow). A high signal-to-noise **master bias** frame, typically created by median-combining many (e.g., 10-20) individual zero-second exposures, represents this stable baseline pattern. Subtracting the master bias from all other image types (science, darks, flats) is usually the very first step in data reduction, removing the offset and pattern noise. The *need* for bias frames is generally constant as long as the detector readout mode and temperature are stable; observatory policies often dictate taking a set of bias frames daily or at the start/end of the night.

**Dark Frames:** These measure the signal generated within detector pixels due purely to **thermal effects** over time, even in complete darkness (shutter closed). This **dark current** accumulates electrons in pixels, and the rate is highly dependent on detector temperature and pixel characteristics (some pixels might be "hot"). Dark frames are taken with exposure times matching (or scaling to) typical science exposures. A **master dark** frame (for a specific exposure time) is created by combining multiple bias-subtracted dark frames. This master dark is then scaled to the science exposure time and subtracted from the (bias-subtracted) science frame to remove the accumulated dark current pattern. The importance of dark subtraction varies: it is critical for IR detectors and warmer or uncooled CCDs, or for very long exposures. For cryogenically cooled optical CCDs used in short exposures, dark current might be negligible compared to other noise sources, potentially making separate dark frames unnecessary (bias subtraction might suffice). The calibration *need* might be defined based on instrument temperature stability and science exposure times (e.g., "obtain 5x300s darks if temperature stable and science exposures > 60s").

**Flat Field Frames:** These correct for variations in response across the detector area. This includes both pixel-to-pixel differences in **quantum efficiency (QE)** and larger-scale **illumination patterns** caused by the telescope optics (like vignetting) or obstructions (like dust particles on filters or windows). Flats are obtained by observing a source expected to be uniformly illuminated across the detector field of view. Common methods include observing the twilight sky (**twilight flats**) or an illuminated screen inside the dome (**dome flats**). Since both QE and illumination patterns depend on wavelength, flats are **filter-dependent**, requiring separate sets for each filter used. A **master flat** for a given filter is created by combining multiple bias-subtracted and dark-subtracted raw flat frames, then **normalizing** the result (e.g., dividing by its median or mean value) so it represents the *relative* sensitivity variations (values typically cluster around 1.0). Dividing the science image (after bias/dark subtraction) by the appropriate normalized master flat corrects for these variations, enabling accurate photometry across the entire image. The *need* for flats often depends on filter changes and instrument stability; nightly dome flats per filter or twilight flats might be standard policy.

Defining these calibration needs within an AstroOps system requires specifying the type, frequency, quantity, and configuration (exposure time, filter) for each required calibration frame. This plan serves as input to the observation scheduling and execution system. Our simulated system will need to represent these policies to trigger the appropriate simulated observations via the digital twin.

**61.2 Requesting and Simulating Calibration Observations**

Once calibration needs are defined (Sec 61.1), the AstroOps system must schedule and execute the acquisition of the raw calibration frames. This involves generating specific observation requests and commanding the digital telescope twin (Chapter 60) to simulate these non-science exposures.

**Representing Calibration Requests:** We can extend the concept of an observation request (potentially used for science targets in Chapter 62) to include calibration types. A request might be structured as a Python dictionary or object containing:
*   `request_id`: Unique identifier.
*   `request_type`: 'BIAS', 'DARK', 'FLAT'.
*   `num_frames`: Number of exposures needed.
*   `exposure_time`: Target exposure time (e.g., `0*u.s` for bias, `300*u.s` for dark, `5*u.s` for flat).
*   `instrument_config`: Required settings, e.g., `{'filter': 'R', 'shutter_state': 'Closed'}`. For flats, this includes the filter. For darks/bias, the shutter state is implicitly closed.
*   `target_info`: For flats, this might specify pointing direction (e.g., 'TWILIGHT_WEST', 'DOME_SCREEN'). For bias/darks, it's irrelevant.
*   `priority`: Usually high for standard calibrations.
*   `status`: 'QUEUED', 'RUNNING', 'DONE', 'ERROR'.

These requests would be generated based on observatory calibration plans or policies (e.g., "At start of night, queue 10 BIAS requests"). The scheduler (Chapter 62) would handle interspersing these calibration requests with science requests according to timing constraints (e.g., twilight flats) and priorities.

**Simulating Acquisition via Digital Twin:** The execution engine (Chapter 63), upon receiving a scheduled calibration request, commands the `Telescope` twin:
1.  **Configuration:** Set the required instrument state using `telescope.configure_instrument(...)`. For darks/biases, ensure the simulation logic knows the shutter is closed (no sky signal). For flats, point the telescope conceptually towards the uniform source (`telescope.point(flat_target)`).
2.  **Exposure:** Call `telescope.expose(exposure_time, sky_model)`.
    *   For **BIAS:** `exposure_time = 0 * u.s`, `sky_model = None`. The `expose` method should primarily simulate the readout process, adding the bias pattern (if modeled beyond a simple offset) and read noise from the `Camera` model.
    *   For **DARK:** `exposure_time = specified_dark_time`, `sky_model = None`. The `expose` method simulates no incoming light but includes the accumulation of dark current signal and associated Poisson noise over the exposure time, plus the bias pattern and read noise, according to the `Camera` model.
    *   For **FLAT:** `exposure_time = specified_flat_time`. The `sky_model` needs to represent a high signal level, spatially uniform source. Inside `expose`, this uniform illumination should be multiplied by a conceptual pixel-to-pixel QE map (if modeled, otherwise assume uniform sensitivity initially) and potentially a vignetting pattern (from the `Optics` model). Then, standard detector noise (dark, read, shot noise on the high flat signal) is added.
3.  **Data Retrieval:** The `expose` method returns the simulated raw calibration frame (e.g., a FITS HDU object containing the data array and header).
4.  **Saving Raw Frame:** The execution engine saves this raw frame to a designated directory (e.g., `raw_calibs/bias/`, `raw_calibs/flat_R/`) using a consistent naming convention including type, date, time, and sequence number. The header should contain all relevant metadata (`OBSTYPE`, `EXPTIME`, `FILTER`, simulated `DATE-OBS`, etc.).

```python
# --- Code Example 1: Function to Simulate Taking Biases ---
# Assumes Telescope class with expose method exists and is instantiated as 'tele_twin'
# Assumes expose(0*u.s) simulates a bias frame correctly.

import os
import time
from astropy.io import fits
from astropy import units as u
from astropy.time import Time
import shutil # For cleanup

# --- Assume Telescope class is defined and available ---
# Minimal placeholder for running the example structure:
class Telescope: 
    def __init__(self): self.camera=type('obj', (object,), {'shape':(100,100)})() # Dummy camera
    def expose(self, exposure_time, sky_model=None): 
        print(f"    Simulating exposure: {exposure_time}")
        exp_time_s = exposure_time.to(u.s).value if hasattr(exposure_time, 'unit') else 0.0
        # Simulate basic bias + read noise for zero exposure
        bias_level = 1000; read_noise = 5
        adu_data = np.random.normal(bias_level, read_noise, self.camera.shape).astype(np.int16)
        hdr = fits.Header({'EXPTIME': exp_time_s, 'OBSTYPE':'BIAS'})
        hdr['DATE-OBS'] = Time.now().isot
        return fits.PrimaryHDU(data=adu_data, header=hdr)
    def configure_instrument(self, **kwargs): pass # Dummy configure
    def point(self, target): pass # Dummy point

# --- Function to orchestrate bias acquisition ---
def acquire_calibration_frames(telescope, calib_type, num_frames, output_dir_base, 
                               exptime=0*u.s, filter_name=None):
    """Simulates acquiring a set of calibration frames."""
    
    if not telescope: 
        print("Error: Telescope object not provided.")
        return []
        
    calib_type = calib_type.upper()
    output_dir = os.path.join(output_dir_base, calib_type.lower())
    if filter_name:
         output_dir = os.path.join(output_dir, filter_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nAcquiring {num_frames} frames of type '{calib_type}'...")
    if filter_name: print(f"  Filter: {filter_name}")
    if exptime > 0*u.s: print(f"  Exposure Time: {exptime}")

    filenames = []
    for i in range(num_frames):
        frame_num = i + 1
        print(f"  Taking frame {frame_num}/{num_frames}...")
        
        # --- Configure and Expose ---
        sky = None # Assume dark/bias unless flat
        if calib_type == 'FLAT':
            telescope.configure_instrument(filter=filter_name)
            # telescope.point(flat_source_location) # Point conceptually
            # Simulate flat field illumination source (high constant signal)
            # This signal level should be calculated based on target ADU and gain
            flat_signal_electrons = 40000 
            sky = np.ones(telescope.camera.shape) * flat_signal_electrons
        elif calib_type == 'DARK':
            # telescope.configure_instrument(shutter='closed') # Conceptual
            pass # Sky remains None
        elif calib_type == 'BIAS':
            exptime = 0 * u.s # Ensure zero exposure
            pass # Sky remains None
            
        try:
            hdu = telescope.expose(exposure_time=exptime, sky_model=sky) 
            if hdu is None: raise ValueError("expose() returned None")
            
            # --- Add/Update Header ---
            hdu.header['OBSTYPE'] = calib_type
            if filter_name: hdu.header['FILTER'] = filter_name
            # Add a simulated timestamp
            # hdu.header['DATE-OBS'] = (Time.now() - i*u.s).isot # Simulate sequence time
            
            # --- Save File ---
            timestamp = Time.now().strftime('%Y%m%dT%H%M%S')
            fname_base = f"{calib_type.lower()}_{filter_name if filter_name else ''}_{timestamp}_{i:03d}.fits".replace("__","_")
            fname = os.path.join(output_dir, fname_base)
            hdu.writeto(fname, overwrite=True)
            filenames.append(fname)
            print(f"    Saved {fname}")
            time.sleep(0.05) # Simulate small delay
            
        except Exception as e:
            print(f"    Error acquiring frame {frame_num}: {e}")
            
    print(f"Finished acquisition. Acquired {len(filenames)} frames.")
    return filenames

# --- Example Usage ---
print("\n--- Example: Simulating Bias Acquisition ---")
# Instantiate the placeholder telescope
tele_sim = Telescope() 
raw_calib_dir = "simulated_raw_calibs"
try:
    bias_files_list = acquire_calibration_frames(tele_sim, 'BIAS', 5, raw_calib_dir)
    print(f"Generated bias files: {bias_files_list}")
    
    # Simulate taking R band flats
    flat_files_list_R = acquire_calibration_frames(tele_sim, 'FLAT', 3, raw_calib_dir, 
                                                   exptime=5*u.s, filter_name='R')
    print(f"Generated R flat files: {flat_files_list_R}")
finally: # Cleanup
    if os.path.exists(raw_calib_dir):
         print(f"\nCleaning up {raw_calib_dir}...")
         shutil.rmtree(raw_calib_dir)

print("-" * 20)

# Explanation:
# 1. Defines a reusable function `acquire_calibration_frames` that takes the telescope 
#    object, calibration type, number of frames, output directory, and optional 
#    exposure time/filter.
# 2. It creates the appropriate output subdirectory.
# 3. It loops `num_frames` times. Inside the loop:
#    - It configures the telescope (setting filter for flats).
#    - It prepares the `sky_model` input for `.expose()` (None for bias/dark, uniform 
#      high signal for flats).
#    - It calls `telescope.expose()` with the correct exposure time.
#    - It updates the returned FITS header with OBSTYPE and filter.
#    - It saves the simulated raw frame with a descriptive, unique filename.
#    - Includes basic error handling and simulated delay.
# 4. Returns a list of filenames for the acquired raw calibration frames.
# 5. Example usage calls this function to simulate acquiring 5 bias frames and 3 R-band flats.
```

This function, integrated into the AstroOps execution engine, allows the system to automatically "take" the required raw calibration frames using the digital twin based on scheduled calibration requests, producing the necessary input data for the master calibration processing steps.

**61.4 Creating Master Calibration Files (Bias, Dark, Flat)**

This section focuses on the computational step of processing the raw calibration frames (simulated in Sec 61.2) into the final **master calibration files** (master bias, master dark rate/frame, master flats per filter). This involves implementing the combination and arithmetic operations described conceptually in Sec 61.3. These functions take lists of raw calibration filenames as input and produce the master FITS files as output.

**Master Bias Creation:**
1.  **Input:** List of raw bias filenames (`raw_bias_files`), output master filename (`output_master_bias`).
2.  **Load Data:** Loop through `raw_bias_files`, open each FITS file, read the data into a NumPy array (e.g., `float32`), and append to a list. Handle file reading errors.
3.  **Stack Data:** Convert the list of 2D arrays into a 3D NumPy array (`bias_stack`). Check if enough valid frames were loaded.
4.  **Combine:** Calculate the median along the first axis: `master_bias_data = np.median(bias_stack, axis=0)`. Or use `astropy.stats.sigma_clipped_stats` for a robust mean.
5.  **Create Header:** Create a new `fits.Header`. Add keywords: `OBSTYPE='MASTER_BIAS'`, `NCOMBINE=len(raw_bias_list)`, `COMBTYPE='MEDIAN'`, creation date, software info, potentially list input filenames in `HISTORY` cards. Copy essential detector/instrument keywords from the first raw bias header if desired.
6.  **Save Output:** Create a `fits.PrimaryHDU` with `master_bias_data` and the created header. Save using `hdu.writeto(output_master_bias, overwrite=True)`.

**Master Dark Rate Creation (Recommended approach):**
1.  **Input:** List of raw dark filenames (`raw_dark_files`) all taken with the *same* exposure time `t_dark`, path to the `master_bias_file`, output master dark rate filename (`output_master_dark_rate`).
2.  **Load Master Bias:** Read `master_bias_data`.
3.  **Load and Process Raw Darks:** Loop through `raw_dark_files`, load data, subtract `master_bias_data`. Store bias-subtracted darks in a list.
4.  **Stack and Combine:** Stack the bias-subtracted darks into a 3D array. Combine using median or sigma-clipped mean: `combined_dark_texp = combine_method(...)`.
5.  **Calculate Rate:** Divide the combined dark frame by the exposure time: `master_dark_rate_data = combined_dark_texp / t_dark`. Ensure `t_dark` > 0.
6.  **Create Header:** Create header. Add `OBSTYPE='MASTER_DARK_RATE'`, `NCOMBINE`, `COMBTYPE`, `BUNIT='electron/s'` (if data converted to electrons) or 'ADU/s', `BIASFILE=basename(master_bias_file)`, `SRC_EXP=t_dark` (original exposure time), creation date, etc.
7.  **Save Output:** Create HDU and save to `output_master_dark_rate`.

**Master Flat Creation:**
1.  **Input:** List of raw flat filenames for a specific filter (`raw_flat_files_filter`), `master_bias_file`, `master_dark_rate_file` (optional), filter name (`filter_id`), output master flat filename (`output_master_flat`).
2.  **Load Master Bias & Dark Rate:** Read `master_bias_data` and `master_dark_rate_data`. Get the exposure time `t_flat` from the first raw flat header.
3.  **Load and Process Raw Flats:** Loop through `raw_flat_files_filter`. Load data. Subtract master bias. Calculate and subtract scaled dark current: `dark_to_subtract = master_dark_rate_data * t_flat`. Store the fully calibrated raw flats.
4.  **Stack and Combine:** Stack the processed flats. Combine using median or sigma-clipped mean: `combined_flat = combine_method(...)`.
5.  **Normalize:** Calculate the normalization factor (e.g., `norm_value = np.median(combined_flat[center_region])`). Divide the combined flat by this value: `master_flat_norm = combined_flat / norm_value`. Handle potential zero `norm_value`.
6.  **Create Header:** Create header. Add `OBSTYPE='MASTER_FLAT'`, `FILTER=filter_id`, `NCOMBINE`, `COMBTYPE`, `NORMVAL=norm_value`, `BIASFILE`, `DARKFILE` (if used), creation date, etc.
7.  **Save Output:** Create HDU with `master_flat_norm` and save to `output_master_flat`.

These processing steps are often implemented as separate functions within a calibration pipeline module. The AstroOps workflow manager (like Snakemake or just Python scripting) would call the acquisition function (Sec 61.2) first, then pass the resulting raw filenames to these processing functions to generate the master calibration files.

```python
# --- Code Example 1: Function to Create Master Bias from Files ---
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
import os

print("Function to Create Master Bias from Files:")

def create_master_bias_from_files(raw_bias_files, output_filename, 
                                  combine_method='median', sigma=3.0, overwrite=True):
    """Combines raw bias FITS files into a master bias FITS file."""
    
    if not raw_bias_files:
        print("Error: No raw bias files provided.")
        return False
        
    print(f"\nCombining {len(raw_bias_files)} raw bias frames (method={combine_method})...")
    bias_stack = []
    header0 = None # Store header from first file
    loaded_count = 0
    
    # Load data from files
    for i, fname in enumerate(raw_bias_files):
        try:
            with fits.open(fname) as hdul:
                if i == 0: header0 = hdul[0].header 
                bias_stack.append(hdul[0].data.astype(np.float32)) # Use float
                loaded_count += 1
        except FileNotFoundError: print(f"Warning: Bias file not found: {fname}")
        except Exception as e: print(f"Warning: Error reading bias file {fname}: {e}")

    if loaded_count < 2: # Need at least 2 frames to combine meaningfully
        print(f"Error: Not enough valid bias frames loaded ({loaded_count}).")
        return False
        
    # Stack into 3D numpy array
    bias_cube = np.array(bias_stack)
    print(f"  Created data cube of shape: {bias_cube.shape}")
    
    # Combine
    if combine_method.lower() == 'median':
        print("  Calculating median combination...")
        master_bias_data = np.median(bias_cube, axis=0)
        comb_type_hdr = 'MEDIAN'
    elif combine_method.lower() == 'sigmaclip':
        print(f"  Calculating sigma-clipped mean (sigma={sigma})...")
        # axis=0 combines along the file dimension
        mean, _, _ = sigma_clipped_stats(bias_cube, sigma=sigma, axis=0) 
        master_bias_data = mean 
        comb_type_hdr = 'SIGMACLIP'
    else:
        print(f"Error: Unknown combine_method '{combine_method}'. Using median.")
        master_bias_data = np.median(bias_cube, axis=0)
        comb_type_hdr = 'MEDIAN'
        
    print("  Combination complete.")
    
    # Create Header
    hdr = fits.Header()
    hdr['OBSTYPE'] = ('MASTER_BIAS', 'Observation type')
    hdr['NCOMBINE'] = (loaded_count, 'Number of raw frames combined')
    hdr['COMBTYPE'] = (comb_type_hdr, 'Combining method used')
    if header0: # Copy some basic instrument info if available
        for key in ['INSTRUME', 'DETECTOR', 'GAIN', 'RDNOISE']:
             if key in header0: hdr[key] = header0[key]
    hdr['DATE'] = (Time.now().isot, 'File creation date')
    hdr['SOFTWARE'] = ('AstroOps Calib v0.2', 'Processing software')
    # Add HISTORY entries for input files
    hdr['HISTORY'] = f"Master Bias created from {loaded_count} files:"
    for i, fname in enumerate(raw_bias_files):
        if i < 15: # Limit number of history lines
             hdr['HISTORY'] = f" Input: {os.path.basename(fname)}"
        elif i == 15:
             hdr['HISTORY'] = " ... and more."
             break
             
    # Save the master bias
    print(f"  Saving master bias to {output_filename}...")
    master_hdu = fits.PrimaryHDU(data=master_bias_data.astype(np.float32))
    master_hdu.header = hdr # Assign header
    try:
        master_hdu.writeto(output_filename, overwrite=overwrite)
        print("  Master bias saved successfully.")
        return True
    except Exception as e:
        print(f"  Error saving master bias: {e}")
        return False

# --- Example Usage (Conceptual - requires dummy raw files) ---
# Assume 'bias_files_list' contains paths like 'raw_calibs/bias/bias_000.fits', etc.
# bias_files_list = acquire_calibration_frames(...) # From previous step
# master_bias_path = "master_calibs/master_bias_test.fits"
# success = create_master_bias_from_files(bias_files_list, master_bias_path)
# if success: print(f"\nSuccessfully created {master_bias_path}")
# if os.path.exists(master_bias_path): os.remove(master_bias_path) # Cleanup test file

print("\n(Defined function `create_master_bias_from_files`)")
print("-" * 20)
```

Functions like `create_master_bias_from_files` (and similar ones developed for darks and flats) form the core processing logic of the calibration pipeline. They take raw calibration filenames as input, perform the necessary image arithmetic and combinations using NumPy/Astropy, and save the master calibration product as a FITS file with informative metadata, ready for storage and management.

**61.5 Storing and Managing Calibration Products**

Effective management of master calibration files is crucial for the smooth operation of an automated data reduction pipeline. Once master bias, dark, and flat frames are created (Sec 61.4), they need to be stored systematically and their associated metadata tracked so the pipeline can easily find and apply the *correct* calibration file for any given science observation. In real observatories, this is handled by sophisticated calibration management systems, often linked to archive databases. For our simulated AstroOps workflow, we need to implement simpler but effective strategies.

**1. Consistent File Naming and Directory Structure:** As mentioned previously, adopting a strict convention for naming master calibration files and organizing them into a logical directory structure is the first step. The filename should clearly indicate the calibration type, relevant parameters (filter, exposure time), and the date/time of creation or validity. The directory structure should facilitate easy browsing and programmatic access (e.g., `/master_calibs/INSTRUMENT/TYPE/FILTER/YYYYMMDD/file.fits`). The functions creating the master files should adhere to this structure.

**2. Metadata Database:** Relying solely on filenames and directory structures becomes fragile as the number of calibration files grows or if complex validity rules are needed. A **metadata database** (e.g., an SQLite database managed via Python's `sqlite3` module, Sec 12.4) provides a much more robust solution. When a master calibration file is created, its key metadata should be inserted as a new row into a dedicated database table (e.g., `master_calibrations`).

Essential columns in this database table would include:
*   `filepath`: The full path to the saved FITS file.
*   `obstype`: 'MASTER_BIAS', 'MASTER_DARK', 'MASTER_FLAT' (or 'MASTER_DARK_RATE').
*   `instrument`: Name of the instrument/detector.
*   `filter`: Filter name (primarily for flats). Stored as `NULL` or 'N/A' for bias/dark.
*   `exptime`: Exposure time associated with the master dark (could be `NULL` for bias/flat/dark_rate).
*   `date_valid_start`: ISO timestamp indicating the beginning of the period for which this calibration is valid (e.g., the time it was created or the start of the night it applies to).
*   `date_valid_end`: ISO timestamp indicating when this calibration expires (e.g., when the next calibration of the same type/config was taken, or a predefined validity duration).
*   `date_created`: ISO timestamp when the master file was generated.
*   `ncombine`: Number of raw frames combined.
*   `quality_flag`: A flag indicating quality (e.g., 'GOOD', 'WARN', 'BAD') based on QC checks (see below).
*   `source_files_ref`: Reference to the raw input files (e.g., path to a list file, or comma-separated IDs).

The functions creating master files (`create_master_bias_auto`, etc.) should be updated to not only save the FITS file but also connect to this database and insert a new record with the relevant metadata upon successful creation.

**3. Quality Control (QC) Integration:** The process of creating master calibrations should include basic quality checks. For example, check the standard deviation of the master bias, the median level and spatial uniformity of the master flat, or the dark current level and number of hot pixels in the master dark. These QC metrics can be stored in the database alongside the file path. Simple thresholds can be applied to set the `quality_flag` (e.g., 'BAD' if flat uniformity is poor). The science pipeline can then use this flag to select only 'GOOD' quality calibrations.

**4. Retrieval Function:** A dedicated function is needed to query this database and retrieve the best calibration file path for a given science observation. This function (`find_best_calib` in the conceptual example below) takes metadata from the science frame header (instrument, filter, exposure time, observation date) as input. It then executes SQL `SELECT` queries against the calibration database:
*   Find `MASTER_BIAS` matching instrument, where `date_valid_start <= obs_date < date_valid_end`, quality='GOOD', ordered by `date_valid_start` descending, limit 1.
*   Find `MASTER_DARK` (or `DARK_RATE`) matching instrument, exposure time (if applicable), where `date_valid_start <= obs_date < date_valid_end`, quality='GOOD', ordered by `date_valid_start` descending, limit 1.
*   Find `MASTER_FLAT` matching instrument and filter, where `date_valid_start <= obs_date < date_valid_end`, quality='GOOD', ordered by `date_valid_start` descending, limit 1.
This function returns the file paths of the selected best calibration files.

**5. Versioning and Updates:** The database naturally handles versioning through the validity dates. When a new master calibration (e.g., a new nightly bias) is added, its `date_valid_start` marks the beginning of its validity, and ideally, the `date_valid_end` of the *previous* calibration of the same type/config should be updated to this new start time. This ensures queries always pick the most recent valid file.

**6. Storage Location:** Master calibration files are crucial reference data and should be stored in a reliable, backed-up location (e.g., project space, dedicated calibration archive) accessible by the processing pipeline, not temporary scratch space.

```python
# --- Code Example 1: Function to Add Master Calib Entry to SQLite DB ---
import sqlite3
from datetime import datetime
import os

print("Conceptual Function to Log Master Calibration to SQLite DB:")

db_filename = 'calibration_log.db' # Database file name

def setup_calib_db(db_file):
    """Creates the master_calibrations table if it doesn't exist."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS master_calibrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT UNIQUE NOT NULL,
                obstype TEXT NOT NULL,
                instrument TEXT,
                filter TEXT,
                exptime REAL,
                date_valid_start TEXT NOT NULL,
                date_valid_end TEXT NOT NULL,
                date_created TEXT NOT NULL,
                ncombine INTEGER,
                quality_flag TEXT DEFAULT 'UNKNOWN',
                source_ref TEXT 
            )
        """)
        conn.commit()
        print(f"Database table 'master_calibrations' ensured in {db_file}")
    except sqlite3.Error as e:
        print(f"Database Error during setup: {e}")
    finally:
        if conn: conn.close()

def log_master_calib_to_db(db_file, filepath, obstype, valid_start_dt, valid_end_dt, 
                           ncombine, instrument='SimCam', filter_name=None, 
                           exptime=None, quality='GOOD', source_ref=None):
    """Adds an entry for a newly created master calibration file to the DB."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Convert datetimes to ISO strings for SQLite
        valid_start_iso = valid_start_dt.isoformat()
        valid_end_iso = valid_end_dt.isoformat()
        created_iso = datetime.utcnow().isoformat()
        
        # Prepare data tuple
        data_tuple = (filepath, obstype, instrument, filter_name, exptime, 
                      valid_start_iso, valid_end_iso, created_iso, 
                      ncombine, quality, source_ref)
                      
        # Insert into database
        cursor.execute("""
            INSERT INTO master_calibrations 
            (filepath, obstype, instrument, filter, exptime, 
             date_valid_start, date_valid_end, date_created, 
             ncombine, quality_flag, source_ref)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data_tuple)
        
        conn.commit()
        print(f"  Logged {os.path.basename(filepath)} ({obstype}) to database.")
        return True
        
    except sqlite3.Error as e:
        print(f"Database Error logging {filepath}: {e}")
        return False
    finally:
        if conn: conn.close()

# --- Conceptual Usage (after creating e.g., a master bias) ---
# master_bias_path = "master_calibs/bias/master_bias_20240115.fits"
# num_combined = 10
# Assume validity starts now, ends in 1 day for example
# validity_start = datetime.utcnow()
# validity_end = validity_start + timedelta(days=1)
# input_files_str = ",".join(['bias1.fits', 'bias2.fits', '...']) # String reference
# 
# setup_calib_db(db_filename) # Ensure table exists
# success_log = log_master_calib_to_db(db_filename, master_bias_path, 'MASTER_BIAS',
#                                      validity_start, validity_end, num_combined, 
#                                      source_ref=input_files_str)
# if success_log: print("Successfully logged master bias.")
# Cleanup dummy DB
if os.path.exists(db_filename): os.remove(db_filename) 

print("\n(Defined conceptual setup_calib_db and log_master_calib_to_db functions)")
print("-" * 20)
```

This database-driven approach provides a scalable and queryable system for managing calibration products, ensuring the AstroOps pipeline can reliably find and apply the correct files during science data reduction.

**61.6 Linking Calibrations to Science Observations**

The culmination of the calibration workflow is applying the appropriate master calibration files to the raw science observation data. This step bridges the calibration generation process (Sec 61.1-61.5) with the science data reduction pipeline. Within an AstroOps framework, this linking and application process should be automated and traceable.

**1. Retrieving Calibration Files:** As outlined in Sec 61.5, the first step when processing a specific raw science FITS file is to determine which master calibration files are required.
*   Read the science header (`fits.getheader(science_filename)`).
*   Extract key metadata: `DATE-OBS`, `INSTRUME`, `FILTER`, `EXPTIME`. Convert `DATE-OBS` to a standard format (e.g., `astropy.time.Time` object or `datetime` object).
*   Query the calibration management system (e.g., the SQLite database from Sec 61.5) using a dedicated function like `find_best_calib(obs_time, instrument, filter, exptime, db_cursor)`.
*   This query searches the database for master files (`obstype` = 'MASTER_BIAS', 'MASTER_DARK'/'DARK_RATE', 'MASTER_FLAT') that match the instrument, filter (for flats), exposure time (for specific darks, if used), have a validity period (`date_valid_start` to `date_valid_end`) encompassing the science observation time, and ideally have a 'GOOD' quality flag. The query typically selects the *most recently created* valid file for each type.
*   The function returns the file paths (strings) to the selected master bias, dark, and flat files. Handle cases where a suitable calibration file is not found (e.g., raise an error or issue a warning and proceed without that step).

**2. Applying Calibrations:** Once the paths to the correct master files are known, the standard calibration sequence is applied to the raw science data (loaded as a NumPy array `raw_data`):
*   Load `master_bias_data = fits.getdata(bias_filepath)`.
*   Perform bias subtraction: `sci_proc1 = raw_data.astype(np.float32) - master_bias_data`.
*   Load `master_dark_data = fits.getdata(dark_filepath)`. Get `master_dark_exptime` from its header. If `dark_filepath` corresponds to a dark rate frame, adjust units accordingly.
*   Perform dark subtraction (scaling if necessary based on `science_exptime` vs `master_dark_exptime`): `sci_proc2 = sci_proc1 - dark_scaled`. Handle cases where dark subtraction is skipped.
*   Load `master_flat_data = fits.getdata(flat_filepath)`. Ensure it's normalized (values around 1.0).
*   Perform flat fielding: `sci_calibrated = sci_proc2 / master_flat_data`. Implement safeguards against division by zero or near-zero values in the flat field (e.g., replace problematic flat pixels with 1.0 or NaN, or use masks).
*   The `sci_calibrated` array now represents the science data corrected for basic instrumental signatures.

**3. Provenance Tracking:** This is absolutely crucial for reproducibility. The FITS header of the **output calibrated science file** must record which specific master calibration files were used in the process. Before saving `sci_calibrated`, update its associated header (`science_header`) with keywords like:
*   `HISTORY Calibration performed: Bias subtracted, Dark subtracted, Flat fielded.`
*   `CAL_BIAS = ('master_bias_YYYYMMDD.fits', 'Master Bias file used')`
*   `CAL_DARK = ('master_dark_60s_YYYYMMDD.fits', 'Master Dark file used')` (or rate file)
*   `CAL_FLAT = ('master_flat_R_YYYYMMDD.fits', 'Master Flat file used')`
*   Include version information for the calibration software/pipeline itself (`CAL_SOFT = 'MyPipeline v1.1'`).

**4. Saving Calibrated Data:** Save the final `sci_calibrated` NumPy array along with the updated `science_header` to a new output FITS file (e.g., `science_image_cal.fits`) using `fits.writeto()` or creating an `HDUList`.

This automated sequence – metadata extraction -> database query -> calibration file retrieval -> calibration application -> provenance recording -> saving output – forms the core of a robust calibration pipeline within an AstroOps framework. It ensures consistency, traceability, and efficient processing of science data using the best available calibration products managed by the system. Libraries like `astropy.ccdproc` provide higher-level tools that encapsulate many of these steps (loading calibrations, processing steps with metadata updates), simplifying the implementation compared to raw NumPy operations.

```python
# --- Code Example 1: Conceptual Science Frame Calibration Function ---
import numpy as np
from astropy.io import fits
from astropy.time import Time
import sqlite3 # For conceptual DB query function call
import os

print("Conceptual function to calibrate a science frame:")

# Assume find_best_calib function exists (from Sec 61.5)
# def find_best_calib(science_obs_time_iso, science_filter, science_exptime, db_cursor):
#     # ... (implementation as shown previously) ...
#     # Returns dict: {'bias': path, 'dark': path, 'flat': path}
#     # Placeholder implementation for demo:
def find_best_calib(science_obs_time_iso, science_filter, science_exptime, db_cursor):
    print(f"  Querying DB for calibs for T={science_obs_time_iso}, F={science_filter}, E={science_exptime}s")
    # Simulate finding files
    bias_f = f"master_calibs/master_bias_{science_obs_time_iso[:10].replace('-','')}.fits"
    dark_f = f"master_calibs/master_dark_{int(science_exptime)}s_{science_obs_time_iso[:10].replace('-','')}.fits"
    flat_f = f"master_calibs/flat/{science_filter}/master_flat_{science_filter}_{science_obs_time_iso[:10].replace('-','')}.fits"
    # Simulate only finding bias and flat for demo
    return {'bias': bias_f, 'dark': None, 'flat': flat_f}
# Assume calibration DB exists ('calibration_log.db')

def calibrate_science_frame(raw_science_file, calib_db_path, output_file):
    """Loads raw science, finds best calibs from DB, applies them, saves output."""
    
    print(f"\nCalibrating science file: {raw_science_file}")
    master_files = {}
    sci_header = None
    raw_data = None
    
    # 1. Extract Metadata from Science Header
    try:
        with fits.open(raw_science_file) as hdul:
            sci_header = hdul[0].header
            raw_data = hdul[0].data.astype(np.float32) # Work with floats
            
        obs_time_iso = sci_header['DATE-OBS']
        obs_filter = sci_header.get('FILTER', 'UNKNOWN')
        obs_exptime = float(sci_header.get('EXPTIME', 0))
        instrument = sci_header.get('INSTRUME', 'UNKNOWN')
        print(f"  Extracted Metadata: Time={obs_time_iso}, Filter={obs_filter}, Exp={obs_exptime}s")
        
    except Exception as e:
        print(f"Error reading science file or header: {e}")
        return False

    # 2. Query Calibration Database
    db_conn = None
    try:
        if not os.path.exists(calib_db_path): raise FileNotFoundError("Calib DB not found")
        db_conn = sqlite3.connect(calib_db_path)
        db_cursor = db_conn.cursor()
        master_files = find_best_calib(obs_time_iso, obs_filter, obs_exptime, db_cursor)
    except Exception as e:
        print(f"Error querying calibration database: {e}")
        # Continue without calibration? Or fail? Let's fail here.
        if db_conn: db_conn.close()
        return False
    finally:
        if db_conn: db_conn.close()

    # 3. Apply Calibrations
    calibrated_data = raw_data.copy()
    applied_calibs = [] # Keep track of applied steps
    
    # Bias Subtraction
    bias_path = master_files.get('bias')
    if bias_path and os.path.exists(bias_path): # Check if file actually exists too
        try:
            master_bias = fits.getdata(bias_path).astype(np.float32)
            calibrated_data -= master_bias
            applied_calibs.append('Bias')
            sci_header['CAL_BIAS'] = (os.path.basename(bias_path), 'Master Bias file applied')
            print(f"  Applied Bias: {os.path.basename(bias_path)}")
        except Exception as e: print(f"  Error applying bias {bias_path}: {e}")
    else: print("  Warning: Master Bias not found or specified. Skipping.")

    # Dark Subtraction (assuming dark is not rate frame here for simplicity)
    dark_path = master_files.get('dark')
    if dark_path and os.path.exists(dark_path):
        try:
            master_dark = fits.getdata(dark_path).astype(np.float32)
            dark_hdr = fits.getheader(dark_path)
            dark_exp = float(dark_hdr.get('EXPTIME', -1))
            if dark_exp <= 0: raise ValueError("Invalid dark exposure time in header")
            
            if np.isclose(obs_exptime, dark_exp):
                 calibrated_data -= master_dark
                 scale_factor = 1.0
            else:
                 scale_factor = obs_exptime / dark_exp
                 calibrated_data -= (master_dark * scale_factor)
                 
            applied_calibs.append('Dark')
            sci_header['CAL_DARK'] = (os.path.basename(dark_path), f'Master Dark (scaled {scale_factor:.2f})')
            print(f"  Applied Dark: {os.path.basename(dark_path)} (Scale={scale_factor:.2f})")
        except Exception as e: print(f"  Error applying dark {dark_path}: {e}")
    else: print("  Warning: Master Dark not found or specified. Skipping.")
    
    # Flat Fielding
    flat_path = master_files.get('flat')
    if flat_path and os.path.exists(flat_path):
        try:
            master_flat = fits.getdata(flat_path).astype(np.float32)
            # Ensure flat is normalized (median ~ 1) and handle zeros
            flat_median = np.median(master_flat[master_flat > 0.1]) # Robust median
            if flat_median > 0.1 and not np.isclose(flat_median, 1.0):
                 print(f"  Warning: Flat field median ({flat_median:.3f}) is not close to 1. Normalizing again.")
                 master_flat /= flat_median
                 
            master_flat[master_flat < 0.01] = 1.0 # Replace low/zero values
            calibrated_data /= master_flat
            applied_calibs.append('Flat')
            sci_header['CAL_FLAT'] = (os.path.basename(flat_path), 'Master Flat file applied')
            print(f"  Applied Flat: {os.path.basename(flat_path)}")
        except Exception as e: print(f"  Error applying flat {flat_path}: {e}")
    else: print("  Warning: Master Flat not found or specified. Skipping.")

    # 4. Save Calibrated Data
    print(f"\nSaving calibrated data to {output_file}...")
    sci_header['HISTORY'] = f"Calibrated by AstroOps pipeline on {Time.now().iso}"
    sci_header['HISTORY'] = f" Steps performed: {', '.join(applied_calibs)}"
    try:
        fits.writeto(output_file, calibrated_data.astype(np.float32), header=sci_header, overwrite=True)
        print("  Save successful.")
        return True
    except Exception as e:
        print(f"  Error saving calibrated file: {e}")
        return False

# --- Example Usage (Conceptual - requires dummy files/db) ---
# Assume 'raw_sci.fits' and 'calibration_log.db' exist and are populated
# success = calibrate_science_frame('raw_sci.fits', 'calibration_log.db', 'calibrated_sci.fits')
# if success: print("\nCalibration complete.")

print("\n(Defined conceptual calibrate_science_frame function)")
print("-" * 20)
```

---

**Application 61.A: Implementing Automated Master Bias Creation**

**(Paragraph 1)** **Objective:** Develop and test a Python function as part of a simulated AstroOps calibration workflow (Sec 61) that automates the creation of a master bias frame. This function will command the digital telescope twin (Chapter 60) to acquire multiple simulated bias frames and then process these frames (using median or sigma-clipped combination) to produce and save the final master bias FITS file with appropriate header information. Reinforces Sec 61.2, 61.3, 61.4.

**(Paragraph 2)** **Astrophysical Context:** All CCD imaging data contains a baseline electronic offset and readout pattern, the "bias". This must be accurately removed for quantitative analysis. Creating a high signal-to-noise master bias frame by combining many individual zero-second exposures is the standard first step in CCD data reduction pipelines. Automating the acquisition and processing of these frames ensures consistency and efficiency in observatory operations or large survey data processing.

**(Paragraph 3)** **Data Source:** No external data files are initially needed. The function interacts with the `Telescope` digital twin object (defined conceptually based on Chapter 60) to *generate* simulated raw bias frames. Input parameters are the number of bias frames to simulate and the desired output filename.

**(Paragraph 4)** **Modules Used:** Custom AstroOps functions, the `Telescope` twin class (including its `.expose()` method), `numpy` (for stacking and median/mean calculation), `astropy.io.fits` (for saving the output FITS file), `astropy.stats` (optional, for `sigma_clipped_stats`), `astropy.time`, `os`.

**(Paragraph 5)** **Technique Focus:** Integrating simulation (twin interaction) with data processing within a single automated function. (1) Defining a function `create_master_bias_auto(telescope_twin, num_frames, output_filename, combine_method='median', sigma=3.0)`. (2) Inside the function, looping `num_frames` times, calling `telescope_twin.expose(exposure_time=0*u.s)` in each iteration to simulate taking a raw bias frame. (3) Collecting the returned data arrays (e.g., in a Python list). (4) Stacking the list of 2D arrays into a 3D NumPy array. (5) Combining the stack along the first axis using either `np.median()` or `astropy.stats.sigma_clipped_stats` (taking the `mean` of the non-clipped values) based on the `combine_method` argument. (6) Creating a FITS header containing relevant metadata (OBSTYPE, NCOMBINE, COMBTYPE, DATE, software info). (7) Creating a `fits.PrimaryHDU` with the combined data and header. (8) Saving the HDU to the specified `output_filename` using `.writeto()`. Including error handling.

**(Paragraph 6)** **Processing Step 1: Define Function Signature:** Create `def create_master_bias_auto(telescope_twin, num_frames, output_filename, combine_method='median', sigma=3.0):`. Include docstrings explaining arguments and return value (e.g., boolean success flag). Ensure the function handles potential `telescope_twin` being `None`.

**(Paragraph 7)** **Processing Step 2: Simulate Acquisition Loop:** Initialize an empty list `raw_bias_list`. Start a loop `for i in range(num_frames):`. Inside the loop, print status (`Taking bias {i+1}...`). Call `hdu = telescope_twin.expose(0*u.s)`. Check if `hdu` is valid and `hdu.data` is not None. If valid, append `hdu.data` to `raw_bias_list`. Add a small `time.sleep(0.1)` to simulate readout overhead. Include a `try...except` block around the `expose` call to catch potential simulation errors and continue to the next frame if needed, logging a warning.

**(Paragraph 8)** **Processing Step 3: Combine Frames:** After the loop, check if `len(raw_bias_list)` is sufficient (e.g., > 1). If not, print an error and return `False`. Convert `raw_bias_list` to a 3D NumPy array `bias_stack`. Use a `try...except` block for the combination step. Implement logic for `combine_method == 'median'` using `np.median(bias_stack, axis=0)` and for `'sigmaclip'` using `sigma_clipped_stats(..., axis=0)` and taking the `mean`. Cast the result `master_bias_data` to `np.float32`.

**(Paragraph 9)** **Processing Step 4: Create Header and Save:** Create an `astropy.io.fits.Header` object. Add standard keywords: `OBSTYPE = 'MASTER_BIAS'`, `NCOMBINE = len(raw_bias_list)`, `COMBTYPE = combine_method.upper()`, `DATE = Time.now().iso`, `SOFTWARE = 'AstroOps_App61A_v1.0'`. Try to copy essential instrument/detector keywords from the header of the *first* successfully read raw bias frame (if available). Create `hdu_out = fits.PrimaryHDU(data=master_bias_data, header=header)`. Save using `hdu_out.writeto(output_filename, overwrite=True)` inside a `try...except` block.

**(Paragraph 10)** **Processing Step 5: Return Status and Logging:** Print a success message including the output filename. Return `True`. If any step failed, print informative error messages and return `False`. Consider adding more sophisticated logging using Python's `logging` module instead of just print statements.

**Output, Testing, and Extension:** The primary output is the `master_bias.fits` file created in the specified location. Console output indicates progress and success/failure. **Testing:** Call the function with a mock `Telescope` object. Verify the FITS file is created. Use `astropy.io.fits` to open the file and check header keywords (`OBSTYPE`, `NCOMBINE`, `COMBTYPE`). Inspect the master bias image data visually - it should look like low-noise data centered around a bias level, with any strong outliers (like simulated cosmic rays in raw frames) removed by the median/clipping. Test with `combine_method='median'` and `'sigmaclip'`. Test edge cases like `num_frames=1` or failure of the `expose` method. **Extensions:** (1) Add option to pass a list of existing raw bias filenames instead of simulating acquisition. (2) Implement dark frame processing (`create_master_dark_auto`) following similar logic (bias subtraction, scaling by exposure time). (3) Store provenance (list of input raw filenames) in the output header `HISTORY` cards. (4) Add quality checks (e.g., measure read noise from the master bias) and return QC metrics. (5) Integrate this function with the calibration database logging from App 61.B.

```python
# --- Code Example: Application 61.A ---
# Note: Requires numpy, astropy. Assumes Telescope class definition available.

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
import os
import time
import shutil # For cleanup

# --- Placeholder Telescope Class (Minimal for Bias Simulation) ---
class Telescope: 
    def __init__(self): 
        self.camera = type('obj', (object,), {'shape':(100,100), 'gain':1.5*u.electron/u.adu})() 
        self.read_noise_adu = 5.0 / self.camera.gain.value # Read noise in ADU
        self.bias_level = 1000 # ADU
        print("Minimal Telescope Initialized.")
    def expose(self, exposure_time, sky_model=None): 
        if exposure_time == 0*u.s: # Bias frame
            # Simulate bias + read noise
            data = np.random.normal(self.bias_level, self.read_noise_adu, self.camera.shape)
            # Add a few "cosmic rays" (hot pixels) to test rejection
            if np.random.rand() < 0.3: # 30% chance per frame
                 nx, ny = self.camera.shape
                 cr_x, cr_y = np.random.randint(0, nx), np.random.randint(0, ny)
                 data[cr_y, cr_x] += np.random.uniform(500, 2000) # Add counts
            hdr = fits.Header({'EXPTIME': 0.0, 'OBSTYPE':'BIAS', 'INSTRUME':'SimCam'})
            hdr['DATE-OBS'] = Time.now().isot # Approximate time
            return fits.PrimaryHDU(data=data.astype(np.int16), header=hdr)
        else: # Not handling other types here
            return None
# --- End Placeholder ---

print("Implementing Automated Master Bias Creation:")

def create_master_bias_auto(telescope_twin, num_frames, output_filename, 
                            combine_method='median', sigma=3.0, overwrite=True):
    """Simulates taking bias frames and creates a master bias FITS file."""
    
    if not telescope_twin: 
        print("Error: Invalid telescope_twin object provided.")
        return False
        
    print(f"\nStarting master bias creation: {num_frames} frames requested, combine='{combine_method}'")
    raw_bias_list = []
    raw_headers = [] # Store headers to copy keywords later

    # Simulate Acquisition Loop
    print("  Acquiring raw bias frames...")
    for i in range(num_frames):
        # print(f"    Taking bias {i+1}/{num_frames}...") # Verbose
        try:
            hdu = telescope_twin.expose(exposure_time=0*u.s) 
            if hdu is None or hdu.data is None: 
                print(f"    Warning: expose() returned invalid HDU for frame {i+1}.")
                continue
            raw_bias_list.append(hdu.data)
            if not raw_headers: raw_headers.append(hdu.header) # Store first header
            # Simulate readout time
            # time.sleep(0.05) 
            
        except Exception as e:
            print(f"    Error taking bias frame {i+1}: {e}")
            
    actual_num_frames = len(raw_bias_list)
    if actual_num_frames < 2: # Need at least 2 frames to combine
        print(f"Error: Not enough valid bias frames acquired ({actual_num_frames}). Cannot combine.")
        return False
    print(f"  Successfully acquired {actual_num_frames} raw bias frames.")

    # Combine Frames
    try:
        bias_stack = np.stack(raw_bias_list, axis=0).astype(np.float32) # Use float for calculation
        print(f"  Combining frames (shape: {bias_stack.shape})...")
        if combine_method.lower() == 'median':
            master_bias_data = np.median(bias_stack, axis=0)
            comb_type_hdr = 'MEDIAN'
        elif combine_method.lower() == 'sigmaclip':
            mean, _, _ = sigma_clipped_stats(bias_stack, sigma=sigma, maxiters=5, axis=0) 
            master_bias_data = mean 
            comb_type_hdr = 'SIGMACLIP'
        else:
            print(f"Error: Unknown combine_method '{combine_method}'.")
            return False
        print(f"  Combination complete using {comb_type_hdr}.")
    except Exception as e:
        print(f"Error combining bias frames: {e}")
        return False

    # Create Header and Save
    try:
        print(f"  Saving master bias to {output_filename}...")
        hdr_out = fits.Header()
        # Copy essential keywords from first raw header if available
        if raw_headers:
            for key in ['INSTRUME', 'DETECTOR', 'GAIN', 'RDNOISE', 'NAXIS1', 'NAXIS2']: # Example keys
                if key in raw_headers[0]: hdr_out[key] = raw_headers[0][key]
                
        hdr_out['OBSTYPE'] = ('MASTER_BIAS', 'Master calibration type')
        hdr_out['NCOMBINE'] = (actual_num_frames, 'Number of raw frames combined')
        hdr_out['COMBTYPE'] = (comb_type_hdr, 'Pixel combination method')
        hdr_out['DATE'] = (Time.now().iso, 'File creation date (UTC)')
        hdr_out['SOFTWARE'] = ('AstroOps App 61A', 'Processing software')
        hdr_out['HISTORY'] = f"Created from {actual_num_frames} raw bias frames."
        
        master_hdu = fits.PrimaryHDU(data=master_bias_data.astype(np.float32), header=hdr_out)
        master_hdu.writeto(output_filename, overwrite=overwrite)
        print("  Master bias saved successfully.")
        return True
    except Exception as e:
        print(f"  Error saving master bias FITS file: {e}")
        return False

# --- Example Usage ---
print("\n--- Example: Creating Master Bias ---")
tele_instance = Telescope() # Use the placeholder
num_bias_to_take = 9
output_master_bias = "master_bias_demo.fits"
success = create_master_bias_auto(tele_instance, num_bias_to_take, output_master_bias, 
                                  combine_method='median') 

if success:
    print(f"\nMaster bias created: {output_master_bias}")
    # Inspect header (optional)
    # with fits.open(output_master_bias) as hdul: print(hdul[0].header)
    if os.path.exists(output_master_bias): os.remove(output_master_bias) # Cleanup
else:
    print("\nMaster bias creation failed.")

print("-" * 20)
```

**Application 61.B: Simulating Flat Field Calibration and Quality Tracking**

**(Paragraph 1)** **Objective:** Extend the AstroOps calibration workflow simulation by implementing functions to acquire simulated flat field frames using the digital twin (Sec 61.2), process them into a master flat (including bias subtraction, Sec 61.3), save the master flat (Sec 61.4), log it to a conceptual calibration database (Sec 61.5), and perform a rudimentary quality check (e.g., checking mean level or uniformity).

**(Paragraph 2)** **Astrophysical Context:** Flat fielding is essential for correcting pixel sensitivity variations and achieving accurate photometry. Generating high-quality master flat fields requires acquiring multiple raw flat frames (e.g., dome or twilight flats), processing them correctly (bias and potentially dark subtraction), combining them, and normalizing the result. Observatories routinely perform quality checks on master flats (e.g., ensuring sufficient signal level, checking for excessive gradients or dust donut stability) before certifying them for use in science pipelines. This application simulates this end-to-end flat field generation and basic QC process.

**(Paragraph 3)** **Data Source:** Requires the `Telescope` digital twin object (from Chapter 60, including the `.expose()` method capable of simulating a flat field source conceptually). Requires the path to a valid `master_bias.fits` file (e.g., generated by App 61.A). Inputs include the filter name, number of flats to simulate, desired flat exposure time, and output filename. Assumes a conceptual calibration database exists (like in App 61.5).

**(Paragraph 4)** **Modules Used:** AstroOps functions (including `acquire_calibration_frames` conceptually from Sec 61.2 or App 61.A), `numpy`, `astropy.io.fits`, `astropy.stats`, `astropy.time`, `os`, `sqlite3` (conceptual, for logging).

**(Paragraph 5)** **Technique Focus:** Orchestrating a multi-step calibration workflow involving simulation, processing, saving, logging, and QC. (1) Modifying or using `acquire_calibration_frames` to simulate taking raw flat fields for a specific filter. (2) Implementing a function `create_master_flat(raw_flat_files, master_bias_file, filter_name, output_filename)` that performs bias subtraction, combination (median/sigmaclip), and normalization (dividing by median). (3) Implementing basic quality checks within `create_master_flat` (e.g., check median level is within expected range, check standard deviation in a central box). (4) Saving the normalized master flat FITS file with appropriate header keywords (OBSTYPE, FILTER, NCOMBINE, NORMVAL, QCFLAG). (5) Calling a conceptual function `log_master_calib_to_db` (from App 61.5) to record the metadata and QC flag of the generated master flat into the calibration database.

**(Paragraph 6)** **Processing Step 1: Acquire Raw Flats:** Call a function like `acquire_calibration_frames(tele_twin, 'FLAT', num_flats, raw_dir, exptime=flat_exptime, filter_name=my_filter)` to simulate taking the raw flats using the digital twin and save them to a temporary directory.

**(Paragraph 7)** **Processing Step 2: Create Master Flat Function:** Define `create_master_flat(...)` as described above. Inside: load raw flats, load master bias, subtract bias, combine frames (median/sigmaclip), calculate normalization factor (e.g., median of combined frame), normalize the frame.

**(Paragraph 8)** **Processing Step 3: Quality Control:** Within `create_master_flat` after normalization:
    *   Calculate the median (`med_norm`) and standard deviation (`std_norm`) of the *normalized* master flat (values should be near 1.0).
    *   Define acceptable ranges (e.g., `0.95 < med_norm < 1.05`, `std_norm < 0.05`).
    *   Set a `qc_flag` string ('GOOD' or 'WARN'/'BAD') based on whether the metrics fall within the acceptable ranges.
    *   Print QC results and the flag.

**(Paragraph 9)** **Processing Step 4: Save Master Flat with QC:** Create FITS header, add standard keywords (`OBSTYPE`, `FILTER`, `NCOMBINE`, etc.) *plus* the calculated QC metrics and the final `QCFLAG`. Save the normalized master flat data and the header to the output FITS file.

**(Paragraph 10)** **Processing Step 5: Log to Database:** Call the conceptual `log_master_calib_to_db(...)` function, passing the output filename, metadata (type, filter, dates, ncombine), and the determined `qc_flag` to record the calibration product in the management system. Return success/failure status based on processing and QC.

**Output, Testing, and Extension:** Output includes the master flat FITS file, console messages detailing processing steps and QC results, and conceptual logging to a database. **Testing:** Verify the master flat file is created with correct header keywords (including FILTER and QCFLAG). Check the data values are normalized around 1.0. Visually inspect the flat for reasonable structure (e.g., lower values near edges if vignetting was simulated). Test the QC checks by simulating raw flats with unusually low signal or high noise. **Extensions:** (1) Implement dark subtraction in the flat processing. (2) Add more sophisticated QC metrics (e.g., checking for dust donut stability by comparing with previous master flats, measuring large-scale gradients). (3) Implement the SQLite database logging fully (App 61.5). (4) Integrate the `create_master_bias_auto` and `create_master_flat` functions into a Snakemake or Parsl workflow (Chapter 66).

```python
# --- Code Example: Application 61.B ---
# Note: Builds on previous examples. Requires numpy, astropy. Assumes Telescope/Camera.
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
import os
import time
import shutil
import sqlite3 # For conceptual logging
from datetime import datetime, timedelta

# --- Reuse/Include Placeholders/Functions ---
# Placeholder Telescope/Camera needed by acquire_calibration_frames
class Camera: 
    def __init__(self, shape=(100, 100), read_noise_e=5.0, dark_rate_e_s=0.01, gain_e_adu=1.5, pixel_scale=0.5*u.arcsec/u.pixel):
        self.shape = shape; self.read_noise = read_noise_e; self.dark_rate = dark_rate_e_s; self.gain = gain_e_adu; self.pixel_scale = pixel_scale
    def generate_noise(self, image_e, exp_time_s): 
        noisy = image_e + np.random.normal(0, self.read_noise, self.shape)
        # Add crude flat field effect (QE variation, vignetting) ONLY for flat simulation
        yy, xx = np.indices(self.shape)
        radius_pix = np.sqrt((xx-self.shape[1]/2)**2 + (yy-self.shape[0]/2)**2)
        vignette = 1.0 - 0.1 * (radius_pix / (max(self.shape)/2))**2 # Simple vignetting
        pixel_qe = np.random.normal(1.0, 0.03, self.shape) # Pixel variations
        noisy *= (pixel_qe * vignette)
        return noisy
    def convert_to_adu(self, image_e): return (image_e / self.gain).astype(np.int16)
class Telescope:
     def __init__(self): self.camera = Camera(); self.current_pointing=True; self.is_tracking=True; self.instrument_config={'filter':'R'}
     def expose(self, exposure_time, sky_model=None): 
          exp_time_s = exposure_time.to(u.s).value if hasattr(exposure_time, 'unit') else 0.0
          if sky_model is None: # Bias/Dark
               signal_e = 0 # No sky
          else: signal_e = sky_model # Assume sky_model is already signal in electrons/pixel
          # Simulate detector noise + effects
          noisy_e = self.camera.generate_noise(signal_e, exp_time_s)
          adu_data = self.camera.convert_to_adu(noisy_e)
          hdr = fits.Header({'EXPTIME': exp_time_s, 'OBSTYPE':'FLAT' if sky_model is not None else 'BIAS'})
          hdr['DATE-OBS'] = Time.now().isot; hdr['FILTER']=self.instrument_config['filter']
          return fits.PrimaryHDU(data=adu_data, header=hdr)
     def configure_instrument(self, filter=None, **kwargs): 
          if filter: self.instrument_config['filter'] = filter; print(f"  INFO: Filter set to {filter}")
     def point(self, target): pass # Dummy

# Assume acquire_calibration_frames function from App 61.A exists
def acquire_calibration_frames(telescope, calib_type, num_frames, output_dir_base, 
                               exptime=0*u.s, filter_name=None, flat_signal=30000):
    filenames = []
    # ... (Implementation similar to App 61.A, using flat_signal for FLAT sky_model) ...
    # Ensure it saves files to appropriate subdirectories
    output_dir = os.path.join(output_dir_base, calib_type.lower())
    if filter_name: output_dir = os.path.join(output_dir, filter_name)
    os.makedirs(output_dir, exist_ok=True)
    # --- Simulation Loop ---
    sky = None
    if calib_type == 'FLAT':
        telescope.configure_instrument(filter=filter_name)
        sky = np.ones(telescope.camera.shape) * flat_signal
    for i in range(num_frames):
        try:
            hdu = telescope.expose(exposure_time=exptime, sky_model=sky)
            if hdu is None: continue
            hdu.header['OBSTYPE'] = calib_type
            if filter_name: hdu.header['FILTER'] = filter_name
            ts = Time.now()
            # hdu.header['DATE-OBS'] = ts.isot # Keep this unique? Or set once?
            fname_base = f"{calib_type.lower()}_{filter_name if filter_name else ''}_{ts.strftime('%Y%m%dT%H%M%S')}_{i:03d}.fits".replace("__","_")
            fname = os.path.join(output_dir, fname_base)
            hdu.writeto(fname, overwrite=True)
            filenames.append(fname)
            time.sleep(0.01)
        except Exception as e: print(f"Error acquiring {calib_type} frame: {e}")
    print(f"  Acquired {len(filenames)} raw {calib_type} frames.")
    return filenames
    
# Assume log_master_calib_to_db function from App 61.5 exists conceptually
def log_master_calib_to_db(db_file, filepath, obstype, valid_start_dt, valid_end_dt, 
                           ncombine, quality, filter_name=None, exptime=None, source_ref=None):
     print(f"  CONCEPT: Logging {os.path.basename(filepath)} ({obstype}, Q={quality}) to DB {db_file}")
     return True # Simulate success

# --- Main Function to Create Master Flat ---
def create_master_flat(raw_flat_files, master_bias_file, filter_name, output_filename, 
                       combine_method='median', sigma=3.0, norm_region_frac=0.5, 
                       qc_limits={'mean_min': 0.9, 'mean_max': 1.1, 'std_max': 0.05},
                       calib_db=None, validity_days=7, overwrite=True):
    """Creates a master flat field from raw flats, including QC and logging."""
    
    print(f"\nCreating Master Flat for Filter '{filter_name}'...")
    
    # --- Load Master Bias ---
    try:
        master_bias_data = fits.getdata(master_bias_file).astype(np.float32)
        print(f"  Loaded Master Bias: {os.path.basename(master_bias_file)}")
    except Exception as e:
        print(f"Error loading master bias {master_bias_file}: {e}")
        return False
        
    # --- Load Raw Flats and Subtract Bias ---
    flat_stack = []
    header0 = None
    loaded_count = 0
    t_flat = None # Get exposure time from first flat
    
    for i, fname in enumerate(raw_flat_files):
        try:
            with fits.open(fname) as hdul:
                if i == 0: 
                    header0 = hdul[0].header
                    t_flat = float(header0.get('EXPTIME', -1))
                # Check if exposure times match if needed
                flat_data = hdul[0].data.astype(np.float32)
                flat_stack.append(flat_data - master_bias_data) # Bias subtract
                loaded_count += 1
        except Exception as e: print(f"Warning: Error reading/processing flat {fname}: {e}")

    if loaded_count < 2:
        print(f"Error: Not enough valid flat frames loaded ({loaded_count}).")
        return False
        
    # --- Combine Flats ---
    flat_cube = np.array(flat_stack)
    print(f"  Combining {loaded_count} bias-subtracted flats...")
    if combine_method.lower() == 'median':
        combined_flat = np.median(flat_cube, axis=0)
        comb_type_hdr = 'MEDIAN'
    elif combine_method.lower() == 'sigmaclip':
        mean, _, _ = sigma_clipped_stats(flat_cube, sigma=sigma, axis=0) 
        combined_flat = mean
        comb_type_hdr = 'SIGMACLIP'
    else: # Default median
        combined_flat = np.median(flat_cube, axis=0)
        comb_type_hdr = 'MEDIAN'
    print(f"  Combined using {comb_type_hdr}.")

    # --- Normalize ---
    print("  Normalizing combined flat...")
    ny, nx = combined_flat.shape
    half_h = int(ny * norm_region_frac / 2)
    half_w = int(nx * norm_region_frac / 2)
    center_slice = (slice(ny//2 - half_h, ny//2 + half_h), 
                    slice(nx//2 - half_w, nx//2 + half_w))
    norm_value = np.median(combined_flat[center_slice])
    
    if norm_value <= 0: 
        print(f"Error: Flat normalization factor is zero or negative ({norm_value:.2f}).")
        return False
        
    master_flat_norm = combined_flat / norm_value
    print(f"  Normalized by median value: {norm_value:.3f}")

    # --- Quality Control ---
    print("  Performing basic Quality Control...")
    mean_final = np.mean(master_flat_norm)
    std_final = np.std(master_flat_norm)
    qc_flag = 'GOOD'
    qc_msg = f"Mean={mean_final:.4f}, StdDev={std_final:.4f}."
    if not (qc_limits['mean_min'] < mean_final < qc_limits['mean_max']):
        qc_flag = 'WARN'
        qc_msg += f" Mean outside range [{qc_limits['mean_min']},{qc_limits['mean_max']}]."
    if std_final > qc_limits['std_max']:
        qc_flag = 'WARN' if qc_flag == 'GOOD' else 'BAD'
        qc_msg += f" StdDev > {qc_limits['std_max']}."
    print(f"  QC Result: {qc_flag}. {qc_msg}")

    # --- Save Master Flat ---
    print(f"  Saving master flat to {output_filename}...")
    hdr_out = fits.Header()
    if header0: # Copy relevant keys
        for key in ['INSTRUME', 'DETECTOR', 'GAIN', 'RDNOISE', 'NAXIS1', 'NAXIS2']:
            if key in header0: hdr_out[key] = header0[key]
    hdr_out['OBSTYPE'] = ('MASTER_FLAT', 'Master calibration type')
    hdr_out['FILTER'] = (filter_name, 'Filter name')
    hdr_out['NCOMBINE'] = (loaded_count, 'Number of raw frames combined')
    hdr_out['COMBTYPE'] = (comb_type_hdr, 'Pixel combination method')
    hdr_out['NORMVAL'] = (norm_value, 'Value used for normalization')
    hdr_out['BIASFILE'] = (os.path.basename(master_bias_file), 'Master bias file used')
    hdr_out['QCFLAG'] = (qc_flag, 'Quality Control flag')
    hdr_out['QCMETRIC'] = (qc_msg, 'Quality Control metrics/notes')
    hdr_out['DATE'] = (Time.now().iso, 'File creation date (UTC)')
    hdr_out['SOFTWARE'] = ('AstroOps App 61B', 'Processing software')
    master_hdu = fits.PrimaryHDU(data=master_flat_norm.astype(np.float32), header=hdr_out)
    try:
        master_hdu.writeto(output_filename, overwrite=overwrite)
        print("  Master flat saved successfully.")
        
        # --- Log to Database (Conceptual Call) ---
        if calib_db:
            valid_start = datetime.utcnow()
            valid_end = valid_start + timedelta(days=validity_days)
            log_master_calib_to_db(calib_db, output_filename, 'MASTER_FLAT', 
                                   valid_start, valid_end, loaded_count, qc_flag, 
                                   filter_name=filter_name)
        return True
        
    except Exception as e:
        print(f"  Error saving/logging master flat: {e}")
        return False

# --- Example Usage ---
print("\n--- Example: Creating Master Flat ---")
# Assume tele_sim exists
# Assume master_bias_file exists (from App 61.A)
# Assume raw flat files exist (e.g., from acquire_calibration_frames)
master_bias_file_path = "master_calibs/master_bias.fits" # Assume exists
raw_flat_dir = "simulated_raw_calibs"
filter_to_process = 'R'
num_raw_flats = 3
flat_exp = 5.0 * u.s
raw_flat_files = []

# Need to simulate acquisition and master bias creation first
try:
    # 1. Create master bias (reuse function from 61.A)
    master_bias_dir = "master_calibs"
    os.makedirs(master_bias_dir, exist_ok=True)
    os.makedirs(os.path.join(raw_calib_dir, 'bias'), exist_ok=True)
    tele_instance_b = Telescope() 
    bias_files = acquire_calibration_frames(tele_instance_b, 'BIAS', 5, raw_calib_dir)
    create_master_bias_from_files(bias_files, master_bias_file_path) # Use the function from 61.A

    # 2. Acquire raw flats
    tele_instance_f = Telescope() # Fresh instance?
    raw_flat_files = acquire_calibration_frames(tele_instance_f, 'FLAT', num_raw_flats, 
                                                raw_calib_dir, exptime=flat_exp, 
                                                filter_name=filter_to_process, 
                                                flat_signal=30000) # Target signal level
    
    # 3. Create master flat
    if os.path.exists(master_bias_file_path) and raw_flat_files:
         master_flat_file_path = os.path.join(master_bias_dir, f"master_flat_{filter_to_process}.fits")
         success_flat = create_master_flat(raw_flat_files, master_bias_file_path, 
                                          filter_to_process, master_flat_file_path, 
                                          calib_db='calibration_log.db') # Conceptual DB log
         if success_flat: print(f"\nMaster Flat created and logged: {master_flat_file_path}")
    else:
         print("\nSkipping master flat creation (missing inputs).")

finally: # Cleanup
    if os.path.exists(raw_calib_dir): shutil.rmtree(raw_calib_dir)
    if os.path.exists(master_bias_dir): shutil.rmtree(master_bias_dir)
    db_filename = 'calibration_log.db'
    if os.path.exists(db_filename): os.remove(db_filename)

print("-" * 20)
```

**Chapter 61 Summary**

This chapter focused on incorporating **instrument calibration workflows** into the simulated AstroOps framework, using the previously developed digital telescope twin. It began by defining the need for standard **calibration frames** in optical/IR imaging: **bias frames** (zero-second exposures capturing readout pattern/offset), **dark frames** (long exposures with shutter closed capturing thermal dark current), and **flat fields** (observing a uniform source through specific filters to capture pixel sensitivity and illumination variations). The process of **simulating the acquisition** of these raw frames was detailed, involving commanding the digital twin's `expose` method with appropriate parameters (zero exposure time for bias, shutter closed/no sky model for dark, uniform illumination source for flat) and saving the simulated raw FITS files with correct metadata.

The core logic for **processing these raw calibration frames** into master calibration files was then implemented. This included functions for combining multiple raw bias frames (using median or sigma-clipped mean) to create a `master_bias`; processing raw darks (bias subtraction, combining) to create a `master_dark` (often scaled to a rate, e.g., e⁻/s); and processing raw flats (bias subtraction, potentially dark subtraction, combining, and crucial **normalization** typically by the median value) to create a filter-specific `master_flat`. Strategies for **storing and managing** these master calibration products were discussed, emphasizing consistent file naming, logical directory structures, and ideally using a **metadata database** (e.g., SQLite) to track file paths, types, configurations (filter, exptime), validity dates, quality flags, and provenance. Finally, the chapter addressed how to **link the correct master calibration files to science observations** by querying the management system (database) based on the science frame's metadata (date, instrument, filter, exptime) and recording the filenames of the applied master files in the calibrated science frame's FITS header for provenance. Two applications demonstrated implementing automated master bias creation and simulating the flat field workflow including basic quality checks and conceptual database logging.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Howell, S. B. (2006).** *Handbook of CCD Astronomy* (2nd ed.). Cambridge University Press. [https://doi.org/10.1017/CBO9780511536170](https://doi.org/10.1017/CBO9780511536170)
    *(Provides detailed explanations of CCD characteristics and standard calibration procedures like bias subtraction, dark correction, and flat fielding, covering the physics and rationale behind the steps implemented in this chapter.)*

2.  **Astropy Collaboration. (n.d.).** *Astropy Documentation: Image Reduction (`astropy.ccdproc`)*. Astropy Documentation. Retrieved January 16, 2024, from [https://docs.astropy.org/en/stable/ccdproc/](https://docs.astropy.org/en/stable/ccdproc/)
    *(Documentation for the `ccdproc` package, which provides tools and classes specifically designed for standard CCD data calibration steps (combining images, subtracting bias/dark, dividing flat), offering a higher-level alternative to manual NumPy operations.)*

3.  **Massey, P., & Hanson, M. M. (2010).** Basic Optical and Infrared Calibration. In P. Massey, & C. Pilachowski (Eds.), *Basic Optical and Infrared Calibration for the Advanced Observer*. ([Link via NOAO/NOIRLab resources or search for title](https://noirlab.edu/science/programs/observing-resources)). *(Specific workshop notes might be hard to find long term, but search for similar calibration guides from major observatories).*
    *(Observatory user manuals and calibration guides often provide practical details on recommended calibration procedures, combination methods, and quality checks for specific instruments.)*

4.  **Valdes, F. G. (1992).** Reduction of Single-Slit Long Slit Spectroscopic Data with IRAF. In D. M. Worrall, C. Biemesderfer, & J. Barnes (Eds.), *Astronomical Data Analysis Software and Systems I (ADASS I)* (ASP Conference Series, Vol. 25, p. 417). Astronomical Society of the Pacific. ([Link via ADS](https://ui.adsabs.harvard.edu/abs/1992ASPC...25..417V/abstract)) (See also modern pipeline documentation like DRAGONS for Gemini).
    *(While focused on spectroscopy and IRAF, papers describing observatory pipelines illustrate the structure and necessity of managing calibration data and associating it correctly with science data, relevant to Sec 61.5, 61.6.)*

5.  **Jenness, T., et al. (2022).** The James Webb Space Telescope calibration pipeline. *Proceedings of the SPIE*, *12180*, 121800X. [https://doi.org/10.1117/12.2628061](https://doi.org/10.1117/12.2628061)
    *(Describes the complex calibration pipeline for JWST, showcasing the modern approach to automated processing, calibration reference file management, and provenance tracking in a large observatory context.)*
