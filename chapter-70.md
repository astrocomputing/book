**Chapter 70: Case Study: End-to-End TESS Transit Search Workflow**

This concluding chapter of Part XI brings together the principles and tools discussed previously – workflow structure, parameterization, automation, environment management, and parallel execution – into a comprehensive **case study**: building and executing an **end-to-end workflow for searching for transiting exoplanets in TESS data**. Using `lightkurve` for core astronomical tasks and choosing either **Snakemake** or **Dask** as the orchestration framework, we will construct a pipeline that starts from a list of target TIC IDs, retrieves the necessary TESS data, performs aperture photometry, cleans and detrends the light curves, runs a Box Least Squares (BLS) transit search, and generates summary plots and candidate lists. We begin by clearly defining the **scientific goals** and outlining the required **input data and parameters**. Each logical step of the workflow – **data retrieval/preparation**, **detrending**, **transit search**, and **candidate vetting/output generation** – will be detailed, highlighting the specific `lightkurve` functions or algorithms involved. We will then show concrete (though potentially simplified for brevity) implementations of the complete workflow using both **Snakemake** (rule-based, file-centric) and **Dask** (task-based, Python-centric) approaches, illustrating how each framework manages dependencies and parallelizes the processing of multiple targets. Finally, we discuss practical considerations for **running the workflow**, monitoring its progress, interpreting the outputs, and ensuring the **reproducibility** of the transit search results using environment management (e.g., Conda or containers).

**70.1 Workflow Goals and Input Data**

The primary scientific goal of this case study workflow is to perform a systematic search for periodic transit-like signals in the light curves of a predefined list of target stars observed by the TESS mission. The desired output is a list of potential planet candidates, characterized by their detected period, duration, depth, and statistical significance, along with diagnostic plots for vetting.

Achieving this requires a multi-step computational process starting from archived TESS data. The workflow needs to handle potentially hundreds or thousands of target stars, possibly observed across multiple TESS sectors, making automation and parallel execution highly desirable.

The main **inputs** required for this workflow are:
1.  **Target List:** A list of target star identifiers, typically TESS Input Catalog (TIC) IDs. This could be provided as a simple text file, a CSV file, or defined directly within a configuration file. For each target, we might also need to specify which TESS sector(s) to process.
2.  **Configuration Parameters:** Settings that control the behavior of the workflow steps, such as:
    *   Data source parameters (e.g., data author like 'SPOC', quality bitmask).
    *   Photometry parameters (e.g., aperture definition method or size).
    *   Detrending parameters (e.g., method 'flatten' or 'pld', window length for flatten, number of PLD regressors).
    *   BLS search parameters (e.g., minimum/maximum period range, trial transit durations).
    *   Candidate selection criteria (e.g., minimum signal-to-noise ratio or BLS power threshold).
    *   Output directory paths.
    *   Execution parameters (e.g., number of cores for local run).
    These parameters are best managed via a configuration file (e.g., YAML) as discussed in Sec 65.4 and 69.3.

The workflow will access TESS data products (TPFs or potentially pre-processed light curves) from the MAST archive programmatically using `lightkurve`. No large initial data download is required locally, as the workflow itself handles the retrieval step.

The expected **final outputs** include:
1.  A summary table or file listing potential transit candidates found, including target ID, sector, detected period, duration, depth, epoch, and signal significance (e.g., BLS power).
2.  Diagnostic plots for each promising candidate, typically showing the detrended light curve, the BLS periodogram, and the light curve folded at the detected period to visually verify the transit shape.
3.  **(Optional)** Intermediate data products like the downloaded TPFs, raw extracted light curves, and flattened light curves might be saved for debugging or further analysis, though they can consume significant disk space.

Defining these inputs, outputs, and the overall goal clearly is the essential first step before designing the specific tasks and their sequence within the workflow structure. This case study aims to automate the process from target list to candidate list and vetting plots.

**70.2 Step 1: Data Retrieval and Preparation**

The first stage of our TESS transit search workflow involves acquiring the necessary observational data for each target star and sector and performing initial preparation steps like aperture photometry if starting from pixel-level data. This stage translates the input target list into usable light curve data objects.

**Task:** For a given TIC ID and Sector number:
1.  Search the MAST archive for available TESS data products using `lightkurve`. We typically prioritize Target Pixel Files (TPFs) if we want control over photometry, or potentially pre-processed PDCSAP light curves if relying on the pipeline's detrending. We focus on products from the primary SPOC pipeline (`author='SPOC'` or `'TESS-SPOC'`).
2.  Download the selected data product (e.g., the TPF FITS file for the specified sector). Handle cases where data for a given target/sector combination might not be available.
3.  If starting from a TPF:
    *   Load the TPF data using `lk.read(tpf_filename)`.
    *   Define an optimal aperture mask. This can be done programmatically using thresholding (`tpf.create_threshold_mask()`) or by selecting pixels based on the Gaia catalog overlay (more advanced). Choosing a good aperture is critical for minimizing noise and systematics.
    *   Perform Simple Aperture Photometry (SAP) by summing flux within the aperture at each cadence: `lc_raw = tpf.to_lightcurve(aperture_mask=...)`. This produces a `LightCurve` object containing `time` and raw `flux`.
4.  Perform initial data cleaning on the raw light curve:
    *   Remove cadences flagged with bad quality bits (using the `quality` column or the `quality_bitmask` during download).
    *   Remove any NaN values in flux or time: `lc_clean = lc_raw.remove_nans()`.
5.  Save the resulting "cleaned" raw light curve to an intermediate file (e.g., FITS table format using `lc_clean.to_fits()`) for the next stage.

**Inputs:** TIC ID, Sector number, Photometry parameters (e.g., aperture definition strategy).
**Outputs:** A FITS file containing the cleaned, raw SAP light curve (Time, Flux, Flux Error, Quality columns).

**Implementation:** This logic would be encapsulated in one or more Python functions or a script (e.g., `prepare_lc.py`). This script would take TIC, sector, and output filename as inputs (perhaps via `argparse` or called by a WMS). It uses `lightkurve` for search, download, aperture selection, and photometry.

**Workflow Integration:**
*   **Snakemake:** A rule `prepare_raw_lc` would take `{tic}`, `{sector}` as wildcards, have output `WORKDIR + "/raw_lc/tic{tic}_s{sector}_rawlc.fits"`, and its `shell:` or `run:` section would execute the `prepare_lc.py` script with appropriate arguments. This rule might implicitly depend on a (possibly empty) input file list or directly use the wildcards as parameters.
*   **Dask:** A function `prepare_lc_dask(tic, sector)` would be defined (potentially decorated with `@delayed` or mapped over a bag of (tic, sector) tuples). It would perform the steps and return the path to the saved raw light curve file (or potentially the `LightCurve` object itself if memory allows and subsequent steps are also Dask tasks).

This initial stage is often I/O intensive (downloading TPFs) and might involve some computationally intensive steps (if complex aperture selection or background modeling is done). Handling potential download failures or lack of data for certain targets gracefully is important within the workflow. Saving the intermediate raw light curve ensures this step doesn't need repeating if later stages fail.

```python
# --- Code Example: Conceptual Function for Step 1 ---
# (Assumes lightkurve is imported as lk, os, Table, fits used)

def prepare_raw_lightcurve(tic_id, sector, output_filename, 
                            aperture_threshold=3.0, download_dir="tpf_tmp"):
    """Downloads TPF, performs photometry, cleans, saves raw LC."""
    print(f"  Starting Step 1 for TIC {tic_id} Sector {sector}")
    lc_raw = None
    tpf = None # Initialize
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    try:
        # 1a. Search (Prioritize SPOC 2-min data if available)
        search_term = f"TIC {tic_id}"
        search_result_tpf = lk.search_targetpixelfile(search_term, sector=sector, author="SPOC", cadence="short")
        if not search_result_tpf: # Fallback to other authors/cadences if needed
             search_result_tpf = lk.search_targetpixelfile(search_term, sector=sector)
             if not search_result_tpf:
                  print(f"    No TPF found.")
                  return None # Indicate failure

        # 1b. Download
        print(f"    Downloading TPF for {search_result_tpf[0].mission[0]}...")
        tpf = search_result_tpf[0].download(download_dir=download_dir, quality_bitmask='default')
        if tpf is None: raise IOError("Download failed or returned None.")

        # 2. Define Aperture
        print(f"    Creating aperture mask (threshold={aperture_threshold} sigma)...")
        aperture_mask = tpf.create_threshold_mask(threshold=aperture_threshold, reference_pixel='center')
        # Optional: Plot aperture for diagnostics
        # tpf.plot(aperture_mask=aperture_mask) 
        
        # 3. Perform Aperture Photometry
        print(f"    Performing aperture photometry...")
        lc_raw = tpf.to_lightcurve(aperture_mask=aperture_mask)
        
        # 4. Clean Light Curve (NaNs, basic flags already applied via quality_bitmask usually)
        print(f"    Cleaning light curve (removing NaNs)...")
        lc_clean = lc_raw.remove_nans()
        if len(lc_clean) == 0:
             raise ValueError("Light curve empty after cleaning.")

        # 5. Save Cleaned Raw Light Curve
        print(f"    Saving cleaned raw light curve to {output_filename}...")
        lc_clean.to_fits(output_filename, overwrite=True)
        print(f"  Step 1 Finished for TIC {tic_id} Sector {sector}")
        return output_filename # Return success indicator (filename)

    except Exception as e:
        print(f"  ERROR in Step 1 for TIC {tic_id} Sector {sector}: {e}")
        return None # Indicate failure
    finally:
         # Clean up downloaded TPF? Optional.
         if tpf and hasattr(tpf, 'path') and os.path.exists(tpf.path):
              # Careful with deleting files if TPF object is needed elsewhere
              pass # os.remove(tpf.path) 
         pass

# --- Conceptual Usage ---
# filename = prepare_raw_lightcurve(261136679, 1, "output/raw_lc/tic261136679_s1_rawlc.fits")
# if filename: print(f"Successfully created {filename}")
```

**70.3 Step 2: Detrending and Cleaning Light Curves**

Once the raw light curve (SAP flux) has been extracted and basic quality cleaning performed, the next crucial step is to remove or correct for instrumental systematic trends. TESS light curves, especially from the SAP pipeline, often exhibit significant trends related to spacecraft pointing jitter, thermal variations, and other instrumental effects that can obscure or mimic transit signals. **Detrending** or **flattening** aims to remove these systematics while preserving the astrophysical signal.

**Task:** For a given cleaned raw light curve file:
1.  **Load Light Curve:** Read the intermediate FITS file containing the cleaned raw light curve (time, flux, flux_err) into a `lightkurve.LightCurve` object.
2.  **Apply Detrending Method:** Choose and apply a detrending algorithm. Common `lightkurve` options include:
    *   **`.flatten(window_length=..., polyorder=..., break_tolerance=...)`:** Uses a Savitzky-Golay filter (a type of moving polynomial filter) to estimate and remove the low-frequency variability. Requires careful selection of `window_length` – too short can distort transits, too long may leave systematics. `break_tolerance` helps handle gaps.
    *   **`.remove_outliers(sigma=...)`:** Can be applied again after initial flattening if needed.
    *   **Using `RegressionCorrector` or `PLDCorrector` (more advanced):** If starting from TPFs or having appropriate regressors (like CBVs or pixel data), these classes model systematics using linear regression and subtract the model. PLD (Pixel Level Decorrelation) uses the pixel data from the TPF itself to build regressors, often effective for TESS systematics.
3.  **Normalize:** Ensure the final light curve is normalized, typically by dividing by its median flux, so that the baseline flux is centered around 1.0. `.flatten()` usually does this.
4.  **Save Flattened Light Curve:** Save the resulting detrended, normalized `LightCurve` object to a new intermediate FITS file.

**Inputs:** Filename of the cleaned raw light curve FITS file. Detrending parameters (method choice, window length, etc.).
**Outputs:** A FITS file containing the flattened light curve (Time, Flux, Flux Error columns, with Flux typically normalized around 1).

**Implementation:** This logic is encapsulated in a function or script (e.g., `detrend_lc.py`). It takes input/output filenames and detrending parameters, uses `lk.read()` to load the input, applies the chosen `lightkurve` detrending method (e.g., `lc.flatten(...)`), and saves the result using `lc_flat.to_fits()`.

**Workflow Integration:**
*   **Snakemake:** A `detrend_lc` rule takes the raw LC filename (from the previous rule's output) as input and produces the flattened LC filename as output, calling the `detrend_lc.py` script in its `shell:` section. Parameters like `window_length` can be passed from the `config` dictionary via the `params:` section of the rule.
*   **Dask:** A `detrend_lc_dask(raw_lc_filename)` function (potentially decorated) performs the detrending and returns the output filename or object. In a Dask workflow (Bag or Delayed), this function would be called after the step that generates the raw light curve filename/object.

Choosing the appropriate detrending method and its parameters is often critical and may require experimentation or optimization depending on the target star and the nature of the systematics. Saving the flattened light curve allows the computationally intensive transit search to operate on this cleaner data product.

```python
# --- Code Example: Conceptual Function for Step 2 ---
import lightkurve as lk
import numpy as np
import os

def detrend_lightcurve_func(raw_lc_filename, flat_lc_filename, 
                           method='flatten', window_length_factor=3.0, sigma=5.0):
    """Loads raw LC, detrends/flattens it, and saves the result."""
    print(f"  Starting Step 2 for {os.path.basename(raw_lc_filename)} -> {os.path.basename(flat_lc_filename)}")
    lc_flat = None
    try:
        lc = lk.read(raw_lc_filename)
        
        if method == 'flatten':
            # Calculate window_length relative to expected transit duration? Or trial period?
            # Example: 3 times a typical ~0.1 day transit duration, converted to cadences
            # Needs cadence info (assume ~2 min TESS short cadence)
            cadence_days = np.median(np.diff(lc.time.value)) # Estimate cadence in days
            # Be robust against empty diff or single point
            if cadence_days > 0 and not np.isnan(cadence_days):
                 # Example: Window = 3 * (typical duration ~0.1-0.2d) / cadence
                 # Need odd integer window
                 window_length = int(window_length_factor * (0.15 / cadence_days)) 
                 if window_length % 2 == 0: window_length += 1 # Make odd
                 window_length = max(51, window_length) # Ensure minimum reasonable window
                 print(f"    Using flatten window_length = {window_length} cadences")
                 
                 # Flatten first, then remove outliers? Or vice-versa? Try flatten -> remove
                 lc_flat = lc.flatten(window_length=window_length, polyorder=2, break_tolerance=5)
                 lc_flat = lc_flat.remove_outliers(sigma=sigma)
                 
            else: 
                 print("    Warning: Could not determine cadence, using default flatten.")
                 # Fallback without window length calculation - might be poor
                 lc_flat = lc.flatten().remove_outliers(sigma=sigma)
                 
        # elif method == 'pld':
        #     # Need to load TPF as well and implement PLDCorrector...
        #     print("    PLD detrending not implemented in this example.")
        #     lc_flat = lc # Placeholder
        else:
            raise ValueError(f"Unknown detrending method: {method}")
            
        # Ensure result is valid
        if len(lc_flat) < 10: # Check if too much data removed
             raise ValueError("Light curve has too few points after detrending/cleaning.")

        # Save flattened light curve
        os.makedirs(os.path.dirname(flat_lc_filename), exist_ok=True)
        lc_flat.to_fits(flat_lc_filename, overwrite=True)
        print(f"  Step 2 Finished. Saved {flat_lc_filename}")
        return flat_lc_filename
        
    except Exception as e:
        print(f"  ERROR in Step 2 for {os.path.basename(raw_lc_filename)}: {e}")
        return None

# --- Conceptual Usage ---
# flat_fname = detrend_lightcurve_func("output/raw_lc/tic261136679_s1_rawlc.fits", 
#                                      "output/detrended/tic261136679_s1_flat.fits")
# if flat_fname: print(f"Successfully created {flat_fname}")
```

**70.4 Step 3: Transit Search using BLS**

With a cleaned and detrended light curve, the workflow proceeds to the core task: searching for periodic transit-like signals using the Box Least Squares (BLS) algorithm. BLS is specifically designed to detect the characteristic periodic rectangular dips caused by planets passing in front of their host stars.

**Task:** For a given flattened light curve file:
1.  **Load Flattened Light Curve:** Read the FITS file containing the flattened light curve (time, normalized flux, flux error) into a `lightkurve.LightCurve` object.
2.  **Define Search Parameters:** Specify the range of orbital periods to search (e.g., `period_min`, `period_max`) and the range of possible transit durations (`duration`) appropriate for the target star and expected planet types. Duration is often specified as a list of trial values or a range.
3.  **Run BLS Algorithm:** Use the `lightkurve` method `lc_flat.to_periodogram(method='bls', period=..., duration=...)`. Providing explicit arrays of periods (e.g., `np.linspace(p_min, p_max, n_periods)`) and durations is common. Alternatively, `frequency_factor` can increase sampling density relative to the default grid. This step can be computationally intensive, especially for long light curves or dense period/duration grids.
4.  **Extract Results:** The `.to_periodogram()` method returns a `BoxLeastSquaresPeriodogram` object. Extract key results characterizing the *strongest* detected signal:
    *   Period corresponding to the highest peak in the periodogram: `bls.period_at_max_power`
    *   Transit duration corresponding to the highest peak: `bls.duration_at_max_power`
    *   Transit depth (approximate) at the highest peak: `bls.depth_at_max_power`
    *   Mid-transit time (epoch) of the transit at max power: `bls.transit_time_at_max_power`
    *   Statistical significance or power of the peak: `bls.max_power` (often related to Signal-to-Noise Ratio, SNR, or Signal Detection Efficiency, SDE).
    *   Optionally, extract statistics for multiple significant peaks above a threshold.
5.  **Save BLS Results:** Store the extracted best-fit parameters and significance metrics, typically associating them with the input TIC ID and sector. Saving to a structured format like a CSV row, JSON object, or database entry is common. The full periodogram object itself could also be saved (e.g., using `bls.to_fits()`) for later inspection if needed, but can be large.

**Inputs:** Filename of the flattened light curve FITS file. BLS search parameters (period range, duration range/values).
**Outputs:** Key parameters (period, duration, depth, epoch, power/SNR) of the strongest BLS signal found. Optionally, the saved periodogram object.

**Implementation:** Encapsulate this logic in a function `run_bls_search(flat_lc_filename, output_results_filename, bls_params)` or a script `run_bls.py`. It loads the light curve, defines BLS parameters (potentially read from config), runs `.to_periodogram()`, extracts results, and saves them to the output file. Error handling is important, as BLS might fail or produce non-finite results for noisy or problematic light curves.

**Workflow Integration:**
*   **Snakemake:** A `run_bls` rule takes the flattened LC file as input, produces a BLS result file (e.g., `.txt` or `.json`) as output, and calls the `run_bls.py` script. BLS parameters can come from `config`.
*   **Dask:** A `run_bls_dask(flat_lc_filename)` app function performs the BLS search and returns a results dictionary. This would be mapped over the outputs of the detrending step in the Dask workflow (Bag or Delayed).

This step performs the core signal detection. Subsequent steps (Sec 70.5) focus on vetting the significance and shape of the signals identified here. Parallelizing this step across many light curves using Dask or a WMS is where significant time savings are achieved in large surveys.

```python
# --- Code Example: Conceptual Function for Step 3 ---
import lightkurve as lk
import numpy as np
import os
import json # For saving results
from astropy import units as u # For periods/durations

def run_bls_search_func(flat_lc_filename, output_json_filename, 
                       period_range=(0.5, 20.0), duration_hours=[1, 2, 4, 8, 12]):
    """Loads flattened LC, runs BLS, saves peak results to JSON."""
    print(f"  Starting Step 3 for {os.path.basename(flat_lc_filename)} -> {os.path.basename(output_json_filename)}")
    results = {'filename': os.path.basename(flat_lc_filename), 'error': None} # Initialize result dict
    try:
        lc = lk.read(flat_lc_filename)
        # Convert durations from hours to days for BLS
        duration_days = np.array(duration_hours) / 24.0
        
        print(f"    Running BLS (Periods: {period_range[0]}-{period_range[1]}d, Durations: {duration_hours}hr)...")
        # Use objective='likelihood' for more statistically robust power metric if desired
        bls = lc.to_periodogram(method='bls', 
                               period=np.arange(period_range[0], period_range[1], 0.01)*u.day, 
                               duration=duration_days*u.day)
                               
        # Extract results
        results['bls_period_day'] = bls.period_at_max_power.value
        results['bls_t0_btjd'] = bls.transit_time_at_max_power.value
        results['bls_duration_day'] = bls.duration_at_max_power.value
        results['bls_depth'] = bls.depth_at_max_power[0] # Index 0 if depth uncertainty included
        results['bls_power'] = bls.max_power.value
        results['bls_snr'] = bls.compute_stats(period=results['bls_period_day']*u.day, 
                                               duration=results['bls_duration_day']*u.day, 
                                               transit_time=results['bls_t0_btjd']*u.day)['snr'][0]
        
        print(f"    BLS Finished. Max Power={results['bls_power']:.1f} @ P={results['bls_period_day']:.3f} d")

        # Save results to JSON
        os.makedirs(os.path.dirname(output_json_filename), exist_ok=True)
        with open(output_json_filename, 'w') as f:
            # Convert numpy types to native python types for JSON
            for key, value in results.items():
                 if isinstance(value, np.generic):
                      results[key] = value.item() 
            json.dump(results, f, indent=2)
        print(f"  Step 3 Finished. Saved results to {output_json_filename}")
        return output_json_filename

    except Exception as e:
        print(f"  ERROR in Step 3 for {os.path.basename(flat_lc_filename)}: {e}")
        results['error'] = str(e)
        # Save error state to JSON
        try:
             os.makedirs(os.path.dirname(output_json_filename), exist_ok=True)
             with open(output_json_filename, 'w') as f: json.dump(results, f, indent=2)
        except: pass # Ignore errors during error saving
        return None

# --- Conceptual Usage ---
# bls_results_file = run_bls_search_func("output/detrended/tic261136679_s1_flat.fits", 
#                                         "output/bls/tic261136679_s1_bls.json")
# if bls_results_file: print(f"Successfully created {bls_results_file}")
```

**70.5 Step 4: Candidate Vetting and Output Generation**

The final stage of the basic transit search workflow involves vetting the potential signals identified by the BLS algorithm and generating summary outputs (plots and potentially filtered lists) for human inspection. Raw BLS detections often include false positives caused by stellar variability (like starspots or flares), instrumental effects not perfectly removed by detrending, or inherent statistical noise. Vetting aims to distinguish plausible transit candidates from these contaminants.

**Task:** For each potential candidate identified by BLS (e.g., signals exceeding a certain power or SNR threshold):
1.  **Load Data:** Read the flattened light curve file and the corresponding BLS results file (containing the best period, t0, duration, depth, power).
2.  **Phase-Folding:** Fold the flattened light curve at the best-fit period detected by BLS using `lc_flat.fold(period=bls_period, epoch_time=bls_t0)`.
3.  **Binning:** Bin the folded light curve (e.g., using `lc_folded.bin(time_bin_size=...)`) to improve the visual clarity of the potential transit shape.
4.  **Calculate Vetting Metrics (Optional):** Compute metrics that help distinguish transits from other signals:
    *   Odd/Even Transit Depth Comparison: Check if depths of odd- and even-numbered transits are consistent (eclipsing binaries often show alternating depths). Use `lc_folded.odd_mask` and `lc_folded.even_mask`.
    *   Secondary Eclipse Search: Look for a potential secondary eclipse near phase 0.5.
    *   Transit Shape Metrics: Compare the folded transit shape to a theoretical model (e.g., using `lc_folded.model()`).
5.  **Generate Diagnostic Plots:** Create a multi-panel plot summarizing the results for the candidate:
    *   Panel 1: Full flattened light curve.
    *   Panel 2: BLS Periodogram, highlighting the chosen peak.
    *   Panel 3: Phase-folded and binned light curve, showing the potential transit shape.
    *   Potentially include other plots like odd/even transits overlaid, or pixel-level diagnostics from the TPF.
6.  **Save Outputs:** Save the diagnostic plot (e.g., as PNG). Add the candidate (if it passes basic vetting criteria or power threshold) to a summary candidate list or table, including key parameters and potentially vetting flags or scores.

**Inputs:** Filename of flattened light curve, filename of corresponding BLS results (or the results dictionary), vetting thresholds (e.g., minimum BLS power).
**Outputs:** Diagnostic plot file (PNG) for each candidate passing the threshold. A summary file (e.g., CSV) listing vetted candidates and their properties.

**Implementation:** Encapsulate this logic in a function or script (e.g., `vet_and_plot_candidate.py`). It takes the necessary input filenames and thresholds, performs the folding, binning, metric calculation (optional), generates the plot using Matplotlib (potentially leveraging `lightkurve`'s folded plot methods), and appends candidate info to a summary file.

**Workflow Integration:**
*   **Snakemake:** A `vet_plot` rule could take the flattened LC file and the BLS result file as input, and produce the PNG plot file and potentially update a global candidate CSV file (handling concurrent writes to the CSV might require care or specific Snakemake features like `gather`). The rule would only run if the BLS power (read from input file) exceeds a threshold specified in the config.
*   **Dask:** A `vet_plot_dask(bls_result_dict, flat_lc_filename)` app function could be defined. The main script would first `.compute()` the BLS results bag/delayed objects, then filter this list based on the power threshold. A second Dask computation (`dask.compute` or `client.map`) would then call the `vet_plot_dask` function only for the promising candidates, potentially running the plotting in parallel.

This final stage translates the raw BLS detections into a more manageable set of candidates accompanied by essential diagnostic plots, facilitating the crucial step of human visual inspection and scientific interpretation required before claiming a planet discovery. Automating the generation of these vetting products is essential when dealing with large numbers of potential signals from surveys like TESS.

```python
# --- Code Example: Conceptual Function for Step 4 ---
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd # For potential candidate list output

def vet_and_plot_candidate(flat_lc_filename, bls_results_filename, 
                           output_plot_filename, bls_power_threshold=10.0, 
                           candidate_list_file="candidates.csv"):
    """Loads BLS results, vets based on power, generates diagnostic plot."""
    print(f"  Starting Step 4 (Vetting/Plotting) for {os.path.basename(flat_lc_filename)}")
    
    try:
        # 1. Load BLS Results
        with open(bls_results_filename, 'r') as f:
            bls_results = json.load(f)
        
        if bls_results.get('error') is not None:
            print(f"    Skipping vetting due to BLS error: {bls_results['error']}")
            return False # Indicate failure/skip
            
        # Check threshold
        if bls_results['bls_power'] < bls_power_threshold:
            print(f"    Signal power {bls_results['bls_power']:.1f} below threshold {bls_power_threshold}. Skipping plot.")
            return True # Indicate success (processed, but not plotted)

        print(f"    Signal found above threshold (Power={bls_results['bls_power']:.1f}). Generating vetting plot...")
            
        # 2. Load Flattened Light Curve
        lc_flat = lk.read(flat_lc_filename)
        
        # 3. Phase-Fold
        period = bls_results['bls_period_day']
        t0 = bls_results['bls_t0_btjd']
        duration_d = bls_results['bls_duration_day']
        lc_folded = lc_flat.fold(period=period, epoch_time=t0)

        # 4. Bin Folded Data (e.g., 20 minute bins for 2 min cadence)
        # Adjust bin size as needed
        bin_minutes = 20
        time_bin_size = bin_minutes / (60*24) # Convert minutes to days
        lc_binned = lc_folded.bin(time_bin_size=time_bin_size) 

        # 5. Generate Diagnostic Plot
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
        
        # Panel 1: Folded and binned light curve
        lc_folded.plot(ax=axes[0], marker='.', markersize=1, linestyle='none', color='silver', label='Folded Data')
        lc_binned.plot(ax=axes[0], marker='o', linestyle='none', color='blue', label=f'Binned ({bin_minutes} min)')
        axes[0].set_title(f"TIC {bls_results.get('tic_id','N/A')} S{bls_results.get('sector','N/A')} | P={period:.4f} d | T0={t0:.4f}")
        axes[0].set_xlabel(f"Phase (Period = {period:.4f} d)")
        axes[0].set_ylabel("Normalized Flux")
        # Highlight transit duration (approx)
        half_dur_phase = (duration_d / 2.0) / period
        axes[0].axvspan(-half_dur_phase, half_dur_phase, color='red', alpha=0.1, label='BLS Dur.')
        axes[0].legend(fontsize='small')
        axes[0].set_xlim(-0.25, 0.25) # Zoom on transit phase

        # Panel 2: Full flattened light curve
        lc_flat.plot(ax=axes[1], marker='.', linestyle='none', markersize=1, color='black', label='Flattened LC')
        # Mark transits (optional, needs calculating transit times)
        # transit_times = t0 + np.arange(int((lc_flat.time.max() - t0)/period) + 1) * period
        # for tt in transit_times: axes[1].axvline(tt.value, color='red', alpha=0.3, lw=1)
        axes[1].set_title("Full Flattened Light Curve")
        
        # 6. Save Plot
        os.makedirs(os.path.dirname(output_plot_filename), exist_ok=True)
        plt.savefig(output_plot_filename, dpi=150)
        print(f"    Diagnostic plot saved to {output_plot_filename}")
        plt.close(fig)
        
        # 6b. Append Candidate to List (Simplified - needs file locking for parallel writes)
        candidate_info = bls_results.copy() # Get all results
        candidate_info['tic_id'] = os.path.basename(flat_lc_filename).split('_')[0].replace('tic','') # Extract TIC
        candidate_info['sector'] = os.path.basename(flat_lc_filename).split('_')[1].replace('s','')   # Extract Sector
        
        # Append to CSV (Warning: not parallel-safe without locking!)
        mode = 'a' if os.path.exists(candidate_list_file) else 'w'
        header = mode == 'w'
        pd.DataFrame([candidate_info]).to_csv(candidate_list_file, mode=mode, header=header, index=False)
        print(f"    Appended candidate info to {candidate_list_file}")

        return True # Indicate success

    except Exception as e:
        print(f"  ERROR in Step 4 for {os.path.basename(flat_lc_filename)}: {e}")
        return False

# --- Conceptual Usage ---
# Assume input files exist from previous steps
# success = vet_and_plot_candidate(
#     "output/detrended/tic261136679_s1_flat.fits", 
#     "output/bls/tic261136679_s1_bls.json",
#     "output/plots/tic261136679_s1_vetting.png",
#     bls_power_threshold=12.0
# )
# if success: print("\nVetting/Plotting completed successfully.")
```

Okay, let's significantly expand Section 70.6 and the corresponding Application code examples (App 70.A and 70.B) in Chapter 70 to provide more detailed Python code illustrating the integration with Snakemake and Dask for the TESS transit search workflow.

---

*(Replacing Section 70.6 and Applications 70.A/B in the previous Chapter 70 content)*

---

**70.6 Implementation using Snakemake OR Dask**

Having defined the logical steps of our TESS transit search workflow (retrieve/prepare, detrend, search, vet/plot), we now illustrate how to orchestrate these steps using either the Snakemake Workflow Management System or the Dask parallel computing library. Both approaches automate the execution and handle dependencies, but differ in their syntax and execution model. We assume the existence of helper Python scripts (like `download_tpf.py`, `extract_lc.py`, `detrend_lc.py`, `run_bls.py`, `plot_vetting.py`) that encapsulate the logic for each step and accept command-line arguments for inputs, outputs, and parameters, as developed conceptually in Chapter 68.

**Option 1: Snakemake Implementation:** Snakemake uses a `Snakefile` to define rules based on input/output file patterns. This approach emphasizes file-based dependencies and is well-suited for pipelines involving multiple scripts or command-line tools.

The `Snakefile` would define rules for each step, using wildcards `{tic}` and `{sector}`. The `rule all` specifies the final desired outputs (e.g., vetting plots). Dependencies are implicit: the `input` of one rule matches the `output` of a preceding rule. Parameters can be managed via a `config.yaml` file accessed through the `config` dictionary.

```python
# --- Code Example 1: Detailed Snakemake Structure (Snakefile Content) ---

# (Save as 'Snakefile')
import glob
from pathlib import Path

# --- Load Configuration ---
# Assumes a config.yaml file exists with:
# WORKDIR: 'tess_output_snake'
# TIC_IDS: ["TIC1", "TIC2"] 
# SECTORS: ["10", "11"]
# APERTURE_THRESHOLD: 3.0
# FLATTEN_WINDOW_FACTOR: 3.0
# BLS_PERIOD_RANGE: [0.5, 20.0]
# BLS_DURATION_HOURS: [1, 2, 4, 8]
# BLS_POWER_THRESHOLD: 10.0
# SCRIPTS_DIR: 'scripts' # Path to helper scripts
configfile: "config.yaml" 

# --- Define Input/Output using Config ---
WORKDIR = config["WORKDIR"]
TIC_IDS = config["TIC_IDS"]
SECTORS = config["SECTORS"]
SCRIPTS_DIR = config["SCRIPTS_DIR"]

# Generate target final files (e.g., vetting plots)
FINAL_PLOTS = expand(WORKDIR + "/plots/{tic}_s{sector}_vetting.png", 
                     tic=TIC_IDS, sector=SECTORS)

# --- Rule 'all': Defines final outputs ---
rule all:
    input: FINAL_PLOTS
    run:
        print("Snakemake TESS Workflow Finished.")
        print("Generated vetting plots (if candidates found):")
        for f in input: print(f"  - {f}")

# --- Rule: Generate Vetting Plot ---
# Depends on flat LC and BLS results
rule plot_vetting:
    input:
        flat_lc=WORKDIR + "/detrended/{tic}_s{sector}_flat.fits",
        bls_results=WORKDIR + "/bls/{tic}_s{sector}_bls.json"
    output:
        vet_plot=touch(WORKDIR + "/plots/{tic}_s{sector}_vetting.png") 
    params:
        threshold = config["BLS_POWER_THRESHOLD"]
    log:
        WORKDIR + "/logs/plot/{tic}_s{sector}.log"
    shell:
        "mkdir -p $(dirname {output.vet_plot}); mkdir -p $(dirname {log}); "
        "echo 'Plotting candidate for {wildcards.tic} S{wildcards.sector}...' > {log}; "
        "python {SCRIPTS_DIR}/plot_vetting.py "
        "--lc {input.flat_lc} --bls {input.bls_results} "
        "--output {output.vet_plot} --threshold {params.threshold} &>> {log}"

# --- Rule: Run BLS Search ---
# Depends on detrended LC
rule run_bls:
    input:
        flat_lc=WORKDIR + "/detrended/{tic}_s{sector}_flat.fits"
    output:
        bls_results=touch(WORKDIR + "/bls/{tic}_s{sector}_bls.json")
    log:
        WORKDIR + "/logs/bls/{tic}_s{sector}.log"
    params:
        min_p = config["BLS_PERIOD_RANGE"][0],
        max_p = config["BLS_PERIOD_RANGE"][1],
        durations = config["BLS_DURATION_HOURS"] # Pass list as string? Or handle in script
    # Convert duration list to comma-separated string for shell command
    run:
        dur_str = ",".join(map(str, params.durations))
        shell(
            "mkdir -p $(dirname {output.bls_results}); mkdir -p $(dirname {log}); "
            "echo 'Running BLS on {input.flat_lc}...' > {log}; "
            "python {SCRIPTS_DIR}/run_bls.py "
            "--input {input.flat_lc} --output {output.bls_results} "
            "--min_period {params.min_p} --max_period {params.max_p} "
            "--durations {dur_str} &>> {log}"
        )

# --- Rule: Detrend Light Curve ---
# Depends on raw LC
rule detrend_lc:
    input:
        raw_lc=WORKDIR + "/raw_lc/{tic}_s{sector}_rawlc.fits"
    output:
        flat_lc=touch(WORKDIR + "/detrended/{tic}_s{sector}_flat.fits")
    log:
        WORKDIR + "/logs/detrend/{tic}_s{sector}.log"
    params:
        # Example: Calculate window length based on factor in config
        window_factor = config["FLATTEN_WINDOW_FACTOR"] 
        # Could add more complex logic here if needed, or pass factor to script
    shell:
        "mkdir -p $(dirname {output.flat_lc}); mkdir -p $(dirname {log}); "
        "echo 'Detrending {input.raw_lc}...' > {log}; "
        "python {SCRIPTS_DIR}/detrend_lc.py "
        "--input {input.raw_lc} --output {output.flat_lc} " 
        # Assume script handles window calculation based on factor/default
        # Or pass explicitly: "--window_factor {params.window_factor}"
        "&>> {log}"

# --- Rule: Extract Raw Light Curve ---
# Depends on TPF
rule extract_lc:
    input:
        tpf=WORKDIR + "/tpfs/{tic}_s{sector}_tp.fits"
    output:
        raw_lc=touch(WORKDIR + "/raw_lc/{tic}_s{sector}_rawlc.fits")
    log:
        WORKDIR + "/logs/extract/{tic}_s{sector}.log"
    params:
        ap_thresh = config["APERTURE_THRESHOLD"]
    shell:
        "mkdir -p $(dirname {output.raw_lc}); mkdir -p $(dirname {log}); "
        "echo 'Extracting LC from {input.tpf}...' > {log}; "
        "python {SCRIPTS_DIR}/extract_lc.py "
        "--input {input.tpf} --output {output.raw_lc} --ap_thresh {params.ap_thresh} &>> {log}"

# --- Rule: Download TPF ---
# Starting point, has only output defined via wildcards
rule download_tpf:
    output:
        tpf=touch(WORKDIR + "/tpfs/{tic}_s{sector}_tp.fits")
    log:
        WORKDIR + "/logs/download/{tic}_s{sector}.log"
    params: # Pass wildcards to the script
        tic = "{tic}",
        sector = "{sector}"
    shell:
        "mkdir -p $(dirname {output.tpf}); mkdir -p $(dirname {log}); "
        "echo 'Downloading TPF for TIC {params.tic} Sector {params.sector}...' > {log}; "
        "python {SCRIPTS_DIR}/download_tpf.py --tic {params.tic} --sector {params.sector} --output {output.tpf} &>> {log}"

```

**Option 2: Dask Implementation:** Dask allows defining the workflow within a Python script, often using `dask.bag` or `dask.delayed` for task parallelism across the list of targets/sectors. This approach keeps the workflow logic within Python.

We define Python functions for each step (similar to Sec 68.2). These functions ideally take input arguments (like TIC ID, sector, filenames, parameters) and return output information (like the path to the created file or a dictionary of results). We then use `dask.bag.map` or `dask.delayed` to chain these functions together, creating a task graph for processing all targets.

```python
# --- Code Example 2: Detailed Dask Delayed Structure (Python Script) ---
import dask
from dask import delayed
from dask.diagnostics import ProgressBar
import glob
import os
import yaml # For config
import shutil
import time

# --- Assume Helper Functions Exist (Import or Define) ---
# These functions take inputs and return output filenames or result dicts
# e.g., download_tpf_func(tic, sector, outdir) -> tpf_filename or None
#       extract_lc_func(tpf_filename, outdir, ap_thresh) -> raw_lc_filename or None
#       detrend_lc_func(raw_lc_filename, outdir, window_factor) -> flat_lc_filename or None
#       run_bls_func(flat_lc_filename, outdir, p_min, p_max, durations) -> bls_results_dict or None
#       plot_vetting_func(flat_lc_filename, bls_results_dict, outdir, threshold) -> plot_filename or None
# (These would contain the lightkurve logic)
print("--- Conceptual Dask Delayed Workflow for TESS ---")
print("# (Assuming helper functions like download_tpf_func etc. are defined elsewhere)")
# Example Dummy function for illustration:
@delayed # Decorate with dask.delayed
def dummy_process_step(step_name, input_arg, out_suffix, sleep_time=0.1):
    """Generic dummy function representing a workflow step."""
    print(f"  Running {step_name} for {input_arg}...")
    time.sleep(sleep_time) # Simulate work
    # Assume input_arg might be tic_sector string or filename
    basename = str(input_arg).split('/')[-1].split('.')[0]
    output = f"output/{step_name}/{basename}{out_suffix}"
    # Create dummy output file/dir for demo
    os.makedirs(os.path.dirname(output), exist_ok=True)
    # with open(output,'w') as f: f.write("done") 
    print(f"  Finished {step_name} for {input_arg}")
    return output # Return output filename (or result dict)

# --- Load Config ---
# config_file = 'config.yaml' 
# with open(config_file, 'r') as f: config = yaml.safe_load(f)
# TIC_IDS = config['TIC_IDS']
# SECTORS = config['SECTORS']
# WORKDIR = config['WORKDIR']
# ... etc ...
TIC_IDS = [f"TIC{i:03d}" for i in range(5)] # Shorter list for demo
SECTORS = ["1", "2"]
WORKDIR = "dask_tess_output"
BLS_PARAMS = {'period_range':(0.5,10), 'duration_hours':[2,4]}
print(f"Processing {len(TIC_IDS)} TICs in {len(SECTORS)} Sectors using Dask Delayed.")
os.makedirs(WORKDIR, exist_ok=True)

# --- Build Task Graph using dask.delayed ---
print("\nBuilding Dask task graph...")
final_tasks = [] # List to hold the final delayed object for each target/sector

for tic in TIC_IDS:
    for sec in SECTORS:
        tic_sec = f"{tic}_s{sec}" # Unique identifier
        
        # Wrap each function call. Pass delayed objects as inputs.
        # Using dummy functions here:
        tpf_file_delayed = dummy_process_step("Download", tic_sec, "_tpf.fits", 0.1)
        raw_lc_delayed = dummy_process_step("Extract", tpf_file_delayed, "_rawlc.fits", 0.15)
        flat_lc_delayed = dummy_process_step("Detrend", raw_lc_delayed, "_flat.fits", 0.2)
        # Pass BLS params dictionary conceptually
        bls_result_delayed = dummy_process_step("BLS", flat_lc_delayed, "_bls.json", 0.3) 
        plot_file_delayed = dummy_process_step("Plot", bls_result_delayed, "_plot.png", 0.1)
        
        final_tasks.append(plot_file_delayed) # Collect the final task/output

print(f"Graph built for {len(final_tasks)} target/sector combinations.")

# --- Execute Graph ---
# Configure scheduler (e.g., local processes)
import multiprocessing
n_workers = multiprocessing.cpu_count()
dask.config.set(scheduler='processes', num_workers=n_workers) 
print(f"\nExecuting graph using '{dask.config.get('scheduler')}' scheduler ({n_workers} workers)...")

start_time = time.time()
with ProgressBar():
    # Compute all final tasks in parallel
    final_outputs = dask.compute(*final_tasks) 
end_time = time.time()

print(f"\nWorkflow finished. Time taken: {end_time - start_time:.2f} seconds")
print(f"Generated {len(final_outputs)} output files (conceptual):")
# print(final_outputs[:5]) # Show first few output filenames/results

# --- Cleanup ---
if os.path.exists(WORKDIR): shutil.rmtree(WORKDIR)
print(f"\nCleaned up {WORKDIR}.")
print("-" * 20)
```

**Comparison:**
*   **Snakemake:** Defines workflow via rules and file dependencies. Strong for command-line tools, wildcards, Conda/container integration, robust restarts based on files. Execution managed via `snakemake` command.
*   **Dask (Delayed):** Defines workflow via Python function calls wrapped with `@delayed`. Dependencies are implicit via function arguments (futures). Strong for Python-native workflows, flexible task definition, integrates with Dask collections (Array, DataFrame). Execution managed via `.compute()` and Dask schedulers (local or distributed).

Both systems allow defining the TESS workflow, managing dependencies, and executing steps in parallel. The choice depends on whether a file-centric, rule-based approach (Snakemake) or a Python-centric, task-graph approach (Dask) is preferred for the specific project and team.

**68.7 Running, Monitoring, and Reproducibility Considerations**

Executing the defined TESS workflow, whether using Snakemake or Dask, involves practical considerations for monitoring progress and ensuring reproducibility.

**Running the Workflow:**
*   **Snakemake:** Executed from the command line using `snakemake [options] [targets]`. Key options include `--cores N` (local parallel execution), `--jobs N` (max concurrent jobs for cluster submission), `--profile <name>` (use cluster configuration profile), `--configfile <file>` (specify config), `--use-conda` or `--use-singularity` (activate environment management), `--rerun-incomplete` (force rerunning incomplete steps), `--dryrun` (show plan without executing), `-p` (print shell commands). Specify target rules (like `all`) or specific output files.
*   **Dask:** Execution is triggered within the Python script by calling `.compute()` on a Dask collection or `dask.compute()` on delayed objects. The behavior depends on the configured scheduler. For local execution, configure via `dask.config.set(...)`. For cluster execution, first set up a `dask.distributed.Client` connected to the cluster (e.g., via `dask-jobqueue`), then run the script; computations submitted via the client will run on the cluster workers.

**Monitoring Progress:**
*   **Snakemake:** Provides real-time updates on the command line showing the number of jobs completed out of the total required, and which rules are currently running. Log files generated per rule (if specified using the `log:` directive) provide detailed output and error messages for each step.
*   **Dask:** The `dask.diagnostics.ProgressBar()` context manager provides a simple text-based progress bar during `.compute()` calls. For the distributed scheduler, the **Dask Dashboard** (accessed via web browser, link usually printed when `Client` starts) provides rich, real-time visualization of the task graph, worker status, CPU/memory usage, task stream, and logs, offering invaluable insights into execution progress and potential bottlenecks. Parsl and Nextflow also have their own monitoring and logging mechanisms.

**Ensuring Reproducibility:** Beyond defining the workflow, ensuring others can reproduce the results requires careful management of the entire environment (Chapter 69):
*   **Version Control (Git):** Track the workflow definition file (`Snakefile`, Python script), configuration files, helper scripts, and container definitions (`Dockerfile`). Use Git tags to mark specific versions used for publications.
*   **Environment Management:** Use Conda `environment.yml` files or `requirements.txt` (captured with `pip freeze`) to precisely specify Python and library versions. Integrate these with the WMS (e.g., Snakemake's `--use-conda`, Nextflow's `conda` directive) or build them into containers.
*   **Containerization (Docker/Singularity):** Package the entire software environment, including OS-level dependencies, into container images defined by `Dockerfile` or Singularity `.def` files. Run workflow steps inside these containers using WMS integration (e.g., Snakemake's `container:` directive, Nextflow's `container`, Parsl executor configuration) for maximum portability and reproducibility. Share container images via registries (Docker Hub).
*   **Data Access:** Provide clear instructions or scripts for obtaining the exact input data (e.g., specific TESS sectors, versions of catalogs). If using custom input data, archive it in persistent repositories (Zenodo, Figshare) and reference it using DOIs.
*   **Configuration:** Version control the exact configuration files (`config.yaml`, etc.) used to generate specific results.
*   **Random Seeds:** If any part of the workflow involves random numbers (e.g., in simulations, bootstrapping, some ML aspects), ensure random seeds are either fixed and reported, or the sensitivity to different seeds is assessed.

By combining an explicitly defined workflow (using a WMS or Dask), rigorous environment management (Conda/containers), version control (Git), and clear documentation (README), researchers can create TESS analysis pipelines (and other complex workflows) that are not only automated and scalable but also highly reproducible, adhering to best practices in modern computational science.

---
**Application 70.A: Implementing the TESS Transit Search with Snakemake**

**(Paragraph 1)** **Objective:** This application provides a more concrete implementation sketch of the end-to-end TESS transit search workflow described in the chapter, specifically using **Snakemake** (Sec 66.2, 68.3) for orchestration. It translates the conceptual steps (Download -> Extract -> Detrend -> BLS -> Plot) into interconnected Snakemake rules, emphasizing dependency management via file patterns and execution via shell commands calling helper Python scripts.

**(Paragraph 2)** **Astrophysical Context:** Automating the search for transits in large numbers of TESS light curves requires a robust pipeline. Snakemake provides a structured way to define this pipeline, ensuring steps run in the correct order, handling intermediate files, allowing parallel execution on multiple cores or cluster nodes, and enhancing reproducibility, making it suitable for processing large target lists systematically.

**(Paragraph 3)** **Data Source:** Defined by a `config.yaml` file specifying `TIC_IDS` (list of target IDs), `SECTORS` (list of sectors to process for each TIC), `WORKDIR` (base output directory), and parameters for helper scripts (aperture settings, detrending window, BLS ranges, plotting thresholds). Assumes helper Python scripts exist in `scripts/` directory.

**(Paragraph 4)** **Modules Used:** `snakemake` (command-line tool), Python (for `Snakefile`), helper scripts use `lightkurve`, `numpy`, `astropy`, `argparse`, `json`, `matplotlib`.

**(Paragraph 5)** **Technique Focus:** Detailed `Snakefile` implementation. Using `configfile:` directive. Defining input file lists dynamically using `expand` and potentially helper functions based on config. Writing multiple dependent rules (`download_tpf`, `extract_lc`, `detrend_lc`, `run_bls`, `plot_vetting`). Ensuring correct use of wildcards (`{tic}`, `{sector}`) in `input:`, `output:`, `params:`, and `shell:` sections. Using `log:` directive for logging. Defining the final `rule all` based on desired outputs (e.g., plots or a combined candidate list). Using `touch()` for output files where the script might conditionally not create one (e.g., plot rule if no significant BLS peak).

**(Paragraph 6)** **Processing Step 1: Create `config.yaml`:** Define the configuration file specifying TICs, Sectors, output directory, script parameters (e.g., `BLS_POWER_THRESHOLD: 10`).

**(Paragraph 7)** **Processing Step 2: Write `Snakefile`:** Create the `Snakefile`. Load the config using `configfile: "config.yaml"`. Generate the list of final target files (e.g., plots) using `expand` based on `config["TIC_IDS"]` and `config["SECTORS"]`. Define `rule all` depending on these final files. Define each intermediate rule (`download_tpf`, `extract_lc`, `detrend_lc`, `run_bls`, `plot_vetting`) ensuring:
    *   `output:` uses wildcards (e.g., `config["WORKDIR"]+"/tpfs/tic{tic}_s{sector}_tp.fits"`).
    *   `input:` uses wildcards matching the output of the prerequisite rule(s).
    *   `params:` passes necessary configuration values or wildcards to the script.
    *   `log:` specifies a unique log file per job.
    *   `shell:` constructs the correct command line to call the helper script in `config["SCRIPTS_DIR"]`, passing inputs, outputs, logs, and parameters. Use `mkdir -p` to ensure output directories exist.

**(Paragraph 8)** **Processing Step 3: Create Helper Scripts:** Ensure the Python scripts (`download_tpf.py`, `extract_lc.py`, etc.) exist in the specified `scripts/` directory and correctly parse command-line arguments defined in the `Snakefile`'s `shell` commands using `argparse`. They should perform their respective tasks using `lightkurve` and save outputs to the specified paths. The plotting script should respect the BLS threshold.

**(Paragraph 9)** **Processing Step 4: Execute Snakemake:** Run `snakemake --cores N all`. Snakemake determines the DAG based on file dependencies and executes jobs in parallel, potentially running multiple `download_tpf.py` instances concurrently, followed by `extract_lc.py`, etc., respecting dependencies for each TIC/Sector combination.

**(Paragraph 10)** **Processing Step 5: Inspect Results:** Check the `WORKDIR` for the final plots (only generated for candidates above the threshold) and potentially the BLS results files. Examine Snakemake's output and log files for errors or progress details.

**Output, Testing, and Extension:** Output includes the directory structure defined by `WORKDIR` containing TPFs, raw LCs, detrended LCs, BLS results, logs, and vetting plots for significant candidates. **Testing:** Use `--dryrun -p` extensively. Test with a small list of TICs/Sectors. Verify intermediate and final files are correctly generated. Delete intermediate files and check if Snakemake reruns correctly. Test error handling (e.g., if a download fails or BLS produces NaNs). **Extensions:** (1) Add a final rule that aggregates all individual `bls.json` files into a single master candidate table (e.g., using a Python script triggered by `run:`). (2) Integrate Conda environments per rule using the `conda:` directive. (3) Create a cluster profile (`--profile`) for submitting Snakemake jobs to SLURM/PBS.

```python
# --- Code Example: Application 70.A ---
# Shows the Snakefile and conceptual config/scripts.

print("--- Implementing TESS Transit Search with Snakemake ---")

# --- 1. Conceptual config.yaml ---
config_yaml_content = """
WORKDIR: 'tess_output_snakemake'
SCRIPTS_DIR: 'scripts'
TIC_IDS: ["261136679", "89025133", "142178937"] 
SECTORS: ["1", "27", "54"] # List sectors to try per TIC
APERTURE_THRESHOLD: 3.0
FLATTEN_WINDOW_FACTOR: 3.0 # Factor to multiply typical duration for window size
BLS_PERIOD_RANGE: [0.5, 15.0] # Days
BLS_DURATION_HOURS: [1, 2, 4, 8, 12, 16]
BLS_POWER_THRESHOLD: 10.0 # For generating vetting plots
"""
print("\n--- Conceptual config.yaml ---")
print(config_yaml_content)

# --- 2. Conceptual Helper Scripts (in SCRIPTS_DIR) ---
print("\n--- Conceptual Helper Scripts (in scripts/) ---")
print("# download_tpf.py: Takes --tic, --sector, --output; downloads TPF.")
print("# extract_lc.py: Takes --input (TPF), --output (LC FITS), --ap_thresh; performs photometry.")
print("# detrend_lc.py: Takes --input (LC FITS), --output (flat LC FITS), --window_factor; flattens LC.")
print("# run_bls.py: Takes --input (flat LC FITS), --output (JSON), --min_period, --max_period, --durations (comma-sep); runs BLS, saves peak result.")
print("# plot_vetting.py: Takes --lc (flat), --bls (JSON), --output (PNG), --threshold; creates plot if BLS power > threshold.")

# --- 3. Snakefile Content (Save as 'Snakefile') ---
snakefile_tess_full = """
import glob
from pathlib import Path

configfile: "config.yaml" # Load configuration

WORKDIR = config["WORKDIR"]
TIC_IDS = config["TIC_IDS"]
SECTORS = config["SECTORS"]
SCRIPTS_DIR = config["SCRIPTS_DIR"]

# Generate list of all potential final plots (some might not be created if no peak found)
FINAL_PLOTS = expand(WORKDIR + "/plots/tic{tic}_s{sector}_vetting.png",
                     tic=TIC_IDS, sector=SECTORS)

# Rule 'all': Defines the final desired output (or aggregation target)
rule all:
    input:
        FINAL_PLOTS
        # Could add a final aggregation report here too
        # aggregate_report = WORKDIR + "/final_candidates.csv"
    run:
        print(f"Snakemake TESS search finished. Check {WORKDIR}/plots/ and logs.")

# Rule: Generate Vetting Plot (Only if BLS power high enough)
rule plot_vetting:
    input:
        flat_lc=WORKDIR + "/detrended/tic{tic}_s{sector}_flat.fits",
        bls_results=WORKDIR + "/bls/tic{tic}_s{sector}_bls.json"
    output:
        # Use temp output for conditional creation? Or handle in script.
        # Let script handle threshold check internally for simplicity here.
        vet_plot=touch(WORKDIR + "/plots/tic{tic}_s{sector}_vetting.png")
    params:
        threshold = config["BLS_POWER_THRESHOLD"]
    log:
        WORKDIR + "/logs/plot/tic{tic}_s{sector}.log"
    shell:
        "mkdir -p $(dirname {output.vet_plot}); mkdir -p $(dirname {log}); "
        "echo 'Plotting candidate for {wildcards.tic} S{wildcards.sector}...' > {log}; "
        "python {SCRIPTS_DIR}/plot_vetting.py --lc {input.flat_lc} --bls {input.bls_results} "
        "--output {output.vet_plot} --threshold {params.threshold} &>> {log}"

# Rule: Run BLS Search
rule run_bls:
    input:
        flat_lc=WORKDIR + "/detrended/tic{tic}_s{sector}_flat.fits"
    output:
        bls_results=touch(WORKDIR + "/bls/tic{tic}_s{sector}_bls.json")
    log:
        WORKDIR + "/logs/bls/tic{tic}_s{sector}.log"
    params:
        min_p = config["BLS_PERIOD_RANGE"][0],
        max_p = config["BLS_PERIOD_RANGE"][1],
        durations = ",".join(map(str, config["BLS_DURATION_HOURS"])) # Comma-separated string
    shell:
        "mkdir -p $(dirname {output.bls_results}); mkdir -p $(dirname {log}); "
        "echo 'Running BLS on {input.flat_lc}...' > {log}; "
        "python {SCRIPTS_DIR}/run_bls.py --input {input.flat_lc} --output {output.bls_results} "
        "--min_period {params.min_p} --max_period {params.max_p} --durations '{params.durations}' &>> {log}" 
        # Note quotes around durations if passing list

# Rule: Detrend Light Curve
rule detrend_lc:
    input:
        raw_lc=WORKDIR + "/raw_lc/tic{tic}_s{sector}_rawlc.fits"
    output:
        flat_lc=touch(WORKDIR + "/detrended/tic{tic}_s{sector}_flat.fits")
    log:
        WORKDIR + "/logs/detrend/tic{tic}_s{sector}.log"
    params:
        window_factor = config["FLATTEN_WINDOW_FACTOR"]
    shell:
        "mkdir -p $(dirname {output.flat_lc}); mkdir -p $(dirname {log}); "
        "echo 'Detrending {input.raw_lc}...' > {log}; "
        "python {SCRIPTS_DIR}/detrend_lc.py --input {input.raw_lc} --output {output.flat_lc} " 
        "--window_factor {params.window_factor} &>> {log}"

# Rule: Extract Raw Light Curve
rule extract_lc:
    input:
        # Use ancient to allow download rule to maybe rerun if extract fails? No, simple dependency.
        tpf=WORKDIR + "/tpfs/tic{tic}_s{sector}_tp.fits" 
    output:
        raw_lc=touch(WORKDIR + "/raw_lc/tic{tic}_s{sector}_rawlc.fits")
    log:
        WORKDIR + "/logs/extract/tic{tic}_s{sector}.log"
    params:
        ap_thresh = config["APERTURE_THRESHOLD"]
    shell:
        "mkdir -p $(dirname {output.raw_lc}); mkdir -p $(dirname {log}); "
        "echo 'Extracting LC from {input.tpf}...' > {log}; "
        "python {SCRIPTS_DIR}/extract_lc.py --input {input.tpf} --output {output.raw_lc} --ap_thresh {params.ap_thresh} &>> {log}"

# Rule: Download TPF (handles potential missing data)
# Use checkpoint() for files that might not exist? No, let download script handle failure.
rule download_tpf:
    output:
        # Use touch here so rule completes even if download fails, preventing downstream failures
        # But then need way to signal failure - better to let script return error code?
        # Let's assume script creates file only on success. Use touch just for rule definition.
        # If using actual output: tpf=WORKDIR + "/tpfs/tic{tic}_s{sector}_tp.fits"
        tpf=touch(WORKDIR + "/tpfs/tic{tic}_s{sector}_tp.fits")
    log:
        WORKDIR + "/logs/download/tic{tic}_s{sector}.log"
    params: 
        tic = "{tic}",
        sector = "{sector}"
    shell:
        "mkdir -p $(dirname {output.tpf}); mkdir -p $(dirname {log}); "
        "echo 'Downloading TPF for TIC {params.tic} Sector {params.sector}...' > {log}; "
        # Script should handle "no data found" gracefully (e.g., exit 0 but don't create file)
        # Snakemake will see missing input for next step and stop that branch.
        "python {SCRIPTS_DIR}/download_tpf.py --tic {params.tic} --sector {params.sector} --output {output.tpf} &>> {log}"
"""
print("--- Snakefile Content ---")
print(snakefile_tess_full)

print("\n--- Conceptual Execution Command ---")
print("# (Create config.yaml, scripts/ directory with helper scripts)")
print("# snakemake --cores 8 all")
print("-" * 20)
```

**Application 70.B: Implementing the TESS Transit Search with Dask**

**(Paragraph 1)** **Objective:** This application implements the same end-to-end TESS transit search workflow (Download -> Extract -> Detrend -> BLS -> Plot/Vet) but uses **Dask** (specifically `dask.delayed`, Sec 67.4, 68.4) within a single Python script for orchestrating the parallel execution across multiple local CPU cores or a Dask cluster.

**(Paragraph 2)** **Astrophysical Context:** Similar to App 70.A, the goal is to efficiently process multiple TESS light curves to find transit candidates. Dask offers a Python-native way to achieve this parallelism, appealing to users who prefer to define and manage workflows entirely within Python using familiar function call syntax and letting Dask handle the task graph execution.

**(Paragraph 3)** **Data Source:** A list of TESS Input Catalog IDs and Sector numbers, potentially read from a configuration file (YAML) or defined in the script. Helper functions performing the actual work (download, extract, detrend, bls, plot) using `lightkurve` are assumed to exist.

**(Paragraph 4)** **Modules Used:** `dask` (core), `dask.delayed`, `dask.diagnostics.ProgressBar`, `dask.distributed.Client` (optional, for cluster), Python helper functions using `lightkurve`, `numpy`, `astropy`, `matplotlib`, `yaml`, `os`, `glob`.

**(Paragraph 5)** **Technique Focus:** Building a Dask task graph using `dask.delayed`. (1) Defining standard Python functions for each workflow step (e.g., `download_tpf_task`, `extract_lc_task`, etc.) that take necessary inputs (filenames, parameters) and return outputs (filenames or results). (2) Wrapping calls to these functions with `dask.delayed`. (3) Chaining these delayed calls together by passing the output `Delayed` object from one step as input to the next, creating the dependency graph implicitly for each target/sector combination. (4) Collecting the final `Delayed` objects (e.g., for the vetting plots or BLS results) for all targets into a list. (5) Configuring a Dask scheduler (local processes or a distributed client). (6) Executing the entire computation graph using `dask.compute(*final_tasks)`.

**(Paragraph 6)** **Processing Step 1: Define Helper Functions:** Define or import Python functions for each step (`download_tpf_task`, `extract_lc_task`, `detrend_lc_task`, `run_bls_task`, `plot_vetting_task`). Ensure they handle potential errors and return meaningful values (e.g., output filename upon success, `None` or raise exception upon failure). These functions should *not* be decorated with `@delayed` themselves; the wrapping happens when they are called.

**(Paragraph 7)** **Processing Step 2: Load Config and Inputs:** Read TIC IDs, Sectors, and other parameters from a configuration file (e.g., `config.yaml`) into a Python dictionary `config`.

**(Paragraph 8)** **Processing Step 3: Build Delayed Task Graph:** Create an empty list `final_results_futures`. Loop through each `tic` in `config['TIC_IDS']` and each `sec` in `config['SECTORS']`:
    *   `tpf_file_future = delayed(download_tpf_task)(tic, sec, config['WORKDIR'])`
    *   `raw_lc_future = delayed(extract_lc_task)(tpf_file_future, config['WORKDIR'], config['APERTURE_THRESHOLD'])`
    *   `flat_lc_future = delayed(detrend_lc_task)(raw_lc_future, config['WORKDIR'], config['FLATTEN_WINDOW_FACTOR'])`
    *   `bls_result_future = delayed(run_bls_task)(flat_lc_future, config['WORKDIR'], config['BLS_PERIOD_RANGE'], config['BLS_DURATION_HOURS'])`
    *   `plot_future = delayed(plot_vetting_task)(flat_lc_future, bls_result_future, config['WORKDIR'], config['BLS_POWER_THRESHOLD'])`
    *   Append `plot_future` (or `bls_result_future`) to `final_results_futures`.
    This loop builds the graph lazily without executing anything yet.

**(Paragraph 9)** **Processing Step 4: Configure Scheduler and Execute:**
    *   If running locally: `dask.config.set(scheduler='processes', num_workers=N_CORES)`.
    *   If using a cluster: `client = Client(cluster_address_or_object)`.
    *   Execute the graph: `with ProgressBar(): final_outputs = dask.compute(*final_results_futures)`. `dask.compute` takes any number of delayed objects and returns a tuple of results in the same order.

**(Paragraph 10)** **Processing Step 5: Process Outputs:** The `final_outputs` tuple contains the return values from the final tasks (e.g., paths to plots or BLS result dictionaries). Process this list as needed (e.g., print summary, save candidate list). Remember to close the Dask client if one was created (`client.close()`).

**Output, Testing, and Extension:** Output consists of the final files (plots, candidate lists) generated by the workflow steps, plus Dask's progress bar and potentially logs from the worker functions. **Testing:** Verify that the workflow runs for all targets/sectors and produces the expected output files. Compare results (e.g., detected periods) with a serial execution on a small subset. Test performance scaling by varying the number of workers or using a distributed cluster. Test error handling within the task functions and how Dask reports failures. **Extensions:** (1) Use Dask Bag instead of Delayed for a more functional programming style if the workflow is mostly linear mapping. (2) Implement more sophisticated error handling and result filtering within the Dask graph itself (e.g., using `dask.utils.apply` or conditional logic within delayed functions). (3) Use Dask DataFrames if intermediate results involve large tables that need distributed processing. (4) Optimize data passing between tasks (e.g., using Dask futures holding data in distributed memory instead of just filenames, if appropriate and memory allows).

```python
# --- Code Example: Application 70.B ---
# Shows Dask Delayed implementation structure. Assumes helper functions are defined.
import dask
from dask import delayed
from dask.diagnostics import ProgressBar
# from dask.distributed import Client # For cluster execution
import glob
import os
import yaml
import shutil
import time
import numpy as np # For dummy functions
import pandas as pd # For final results

print("--- Implementing TESS Transit Search with Dask Delayed ---")

# --- Step 1: Define Helper Functions (Dummy versions) ---
# These should perform the actual lightkurve operations and return output paths/results
@delayed
def download_tpf_task(tic, sector, outdir):
    print(f"  DL: TIC {tic} S{sector}")
    time.sleep(0.05); fname=f"{outdir}/tpfs/tic{tic}_s{sector}_tp.fits"; os.makedirs(os.path.dirname(fname),exist_ok=True); open(fname,'w').close(); return fname
@delayed
def extract_lc_task(tpf_filename, outdir, ap_thresh):
    if tpf_filename is None: return None
    print(f"  EX: {os.path.basename(tpf_filename)}")
    time.sleep(0.1); fname=f"{outdir}/raw_lc/" + os.path.basename(tpf_filename).replace('_tp.fits','_rawlc.fits'); os.makedirs(os.path.dirname(fname),exist_ok=True); open(fname,'w').close(); return fname
@delayed
def detrend_lc_task(raw_lc_filename, outdir, window_factor):
    if raw_lc_filename is None: return None
    print(f"  DT: {os.path.basename(raw_lc_filename)}")
    time.sleep(0.15); fname=f"{outdir}/detrended/" + os.path.basename(raw_lc_filename).replace('_rawlc.fits','_flat.fits'); os.makedirs(os.path.dirname(fname),exist_ok=True); open(fname,'w').close(); return fname
@delayed
def run_bls_task(flat_lc_filename, outdir, period_range, duration_hours):
    if flat_lc_filename is None: return None
    print(f"  BLS: {os.path.basename(flat_lc_filename)}")
    time.sleep(0.2)
    # Simulate results
    result = {'filename': os.path.basename(flat_lc_filename), 
              'bls_period': np.random.uniform(period_range[0], period_range[1]), 
              'bls_power': np.random.exponential(10), 'error': None}
    fname=f"{outdir}/bls/" + os.path.basename(flat_lc_filename).replace('_flat.fits','_bls.json')
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    # with open(fname,'w') as f: json.dump(result, f) # Save result
    return result # Return dictionary for easier processing
@delayed
def plot_vetting_task(flat_lc_filename, bls_results_dict, outdir, threshold):
    if flat_lc_filename is None or bls_results_dict is None or bls_results_dict.get('error') is not None: return None
    print(f"  PLOT: {os.path.basename(flat_lc_filename)}")
    time.sleep(0.05)
    if bls_results_dict['bls_power'] > threshold:
         fname=f"{outdir}/plots/" + os.path.basename(flat_lc_filename).replace('_flat.fits','_vetting.png')
         os.makedirs(os.path.dirname(fname), exist_ok=True)
         open(fname,'w').close() # Create dummy plot file
         return fname
    else:
         return None # No plot if below threshold

# --- Step 2: Load Config and Inputs ---
# Create dummy config for demo
config = {
    'WORKDIR': 'dask_tess_output',
    'TIC_IDS': [f"TIC{i:03d}" for i in range(8)], # Small number for demo
    'SECTORS': ["1", "2"],
    'APERTURE_THRESHOLD': 3.0,
    'FLATTEN_WINDOW_FACTOR': 3.0,
    'BLS_PERIOD_RANGE': [0.5, 10.0],
    'BLS_DURATION_HOURS': [1, 2, 4, 8],
    'BLS_POWER_THRESHOLD': 12.0 
}
WORKDIR = config['WORKDIR']
os.makedirs(WORKDIR, exist_ok=True)
print(f"Loaded configuration for {len(config['TIC_IDS'])} TICs and {len(config['SECTORS'])} Sectors.")

# --- Step 3: Build Delayed Task Graph ---
print("\nBuilding Dask task graph using 'delayed'...")
final_results_futures = [] # Collect BLS result futures or plot futures
for tic in config['TIC_IDS']:
    for sec in config['SECTORS']:
        # Chain the delayed function calls
        tpf_fut = download_tpf_task(tic, sec, WORKDIR)
        raw_fut = extract_lc_task(tpf_fut, WORKDIR, config['APERTURE_THRESHOLD'])
        flat_fut = detrend_lc_task(raw_fut, WORKDIR, config['FLATTEN_WINDOW_FACTOR'])
        bls_fut = run_bls_task(flat_fut, WORKDIR, config['BLS_PERIOD_RANGE'], config['BLS_DURATION_HOURS'])
        # Optionally add plotting task dependent on BLS result
        # plot_fut = plot_vetting_task(flat_fut, bls_fut, WORKDIR, config['BLS_POWER_THRESHOLD'])
        # final_results_futures.append(plot_fut)
        final_results_futures.append(bls_fut) # Collect BLS results futures here

print(f"Built graph with {len(final_results_futures)} final tasks.")

# --- Step 4: Configure Scheduler and Execute ---
n_workers = min(os.cpu_count(), 4) # Limit workers for demo
dask.config.set(scheduler='processes', num_workers=n_workers) 
print(f"\nExecuting graph using '{dask.config.get('scheduler')}' scheduler ({n_workers} workers)...")
start_time = time.time()
with ProgressBar():
    # Compute all BLS result futures
    final_outputs_list = list(dask.compute(*final_results_futures))
end_time = time.time()
print(f"\nWorkflow finished. Time taken: {end_time - start_time:.2f} seconds")

# --- Step 5: Process Outputs ---
if final_outputs_list:
    valid_results = [r for r in final_outputs_list if r and r.get('error') is None]
    if valid_results:
        results_df = pd.DataFrame(valid_results)
        print("\n--- Sample BLS Results (DataFrame) ---")
        print(results_df.head())
        # Filter candidates
        candidates = results_df[results_df['bls_power'] > config['BLS_POWER_THRESHOLD']].sort_values('bls_power', ascending=False)
        print(f"\nFound {len(candidates)} potential candidates (BLS Power > {config['BLS_POWER_THRESHOLD']}):")
        if not candidates.empty: print(candidates[['filename', 'bls_period', 'bls_power']])
    else:
        print("\nNo valid BLS results generated.")
else:
    print("\nComputation returned no results.")

# --- Cleanup ---
finally:
     if os.path.exists(WORKDIR): 
          shutil.rmtree(WORKDIR)
          print(f"\nCleaned up {WORKDIR}.")

print("-" * 20)
```

**Chapter 70 Summary**

This final case study chapter integrated the principles of workflow management and parallel execution by constructing an end-to-end pipeline for **searching for transiting exoplanets in TESS data**, primarily using `lightkurve` for the core astronomical tasks. The workflow's scientific goal (identifying transit candidates from a list of targets) and inputs (TIC IDs, sectors, configuration parameters) were established. The major processing steps were defined: **Step 1 (Data Retrieval/Preparation)** using `lightkurve` to find and download TESS Target Pixel Files (TPFs) and perform aperture photometry to extract raw light curves; **Step 2 (Detrending)** applying methods like `.flatten()` to remove instrumental systematics; **Step 3 (Transit Search)** running the Box Least Squares (BLS) algorithm using `.to_periodogram(method='bls')` to find the best periodic transit-like signal; and **Step 4 (Vetting/Output)** generating diagnostic plots (like phase-folded light curves) and summary tables for promising candidates exceeding a significance threshold.

Two distinct implementation approaches using popular workflow tools were presented conceptually and illustrated with code structure. The first used **Snakemake**, defining the pipeline in a `Snakefile` with distinct **rules** for each step, using **wildcards** (`{tic}`, `{sector}`) to generalize across targets, managing dependencies implicitly through input/output **filename patterns**, and executing helper Python scripts via shell commands. The second approach used **Dask** within a single Python script, defining each step as a Python function and using **`dask.delayed`** to wrap function calls. Dependencies were created implicitly by passing the `Delayed` object (future) output from one step as input to the next, building a task graph for all targets/sectors. Execution was triggered using `dask.compute()` with a parallel scheduler (like `'processes'`). Finally, the importance of robust **running and monitoring** procedures for these potentially long workflows was reiterated, along with the critical need for ensuring **reproducibility** through rigorous environment management (Conda/containers), version control (Git for code, workflow definitions, configs), and clear documentation, encapsulating the best practices discussed throughout Part XI.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Lightkurve Collaboration. (2018).** Lightkurve: Kepler and TESS time series analysis in Python. *Astrophysics Source Code Library*, record ascl:1812.011. ([Link via ADS](https://ui.adsabs.harvard.edu/abs/2018ascl.soft12011L/abstract)) (See also Documentation: [https://docs.lightkurve.org/](https://docs.lightkurve.org/))
    *(Essential reference for the `lightkurve` library used extensively for data retrieval, light curve manipulation, detrending, and BLS periodograms in this chapter's case study.)*

2.  **Kovács, G., Zucker, S., & Mazeh, T. (2002).** A box-fitting algorithm in the search for periodic transits. *Astronomy & Astrophysics*, *391*, 369–377. [https://doi.org/10.1051/0004-6361:20020802](https://doi.org/10.1051/0004-6361:20020802)
    *(The original paper describing the Box Least Squares (BLS) algorithm, the core transit search method implemented in the workflow.)*

3.  **Köster, J. (Ed.). (n.d.).** *Snakemake Documentation*. Snakemake Readthedocs. Retrieved January 16, 2024, from [https://snakemake.readthedocs.io/en/stable/](https://snakemake.readthedocs.io/en/stable/)
    *(Documentation for Snakemake, used for implementing the workflow in Application 70.A.)*

4.  **Dask Development Team. (n.d.).** *Dask Documentation*. Dask. Retrieved January 16, 2024, from [https://docs.dask.org/en/latest/](https://docs.dask.org/en/latest/) (Specifically Delayed: [https://docs.dask.org/en/latest/delayed.html](https://docs.dask.org/en/latest/delayed.html))
    *(Documentation for Dask, particularly Dask Delayed used for implementing the workflow in Application 70.B.)*

5.  **Jenkins, J. M., et al. (2010).** Overview of the Kepler Science Processing Pipeline. *The Astrophysical Journal Letters*, *713*(2), L87–L91. [https://doi.org/10.1088/2041-8205/713/2/L87](https://doi.org/10.1088/2041-8205/713/2/L87) (See also TESS SPOC pipeline documentation: [https://archive.stsci.edu/tess/documentation.html](https://archive.stsci.edu/tess/documentation.html))
    *(Provides context on the complexity and stages involved in large-scale mission pipelines like Kepler's (and similarly TESS's SPOC pipeline), motivating the need for workflow management.)*
