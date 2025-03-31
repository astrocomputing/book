**Chapter 68: Implementing TESS Workflows with Lightkurve and WMS/Dask**

This chapter focuses on applying the workflow management and parallel computing principles discussed previously (Chapters 65, 66, 67) to a concrete astrophysical use case: building and executing analysis pipelines for data from the Transiting Exoplanet Survey Satellite (TESS), primarily using the **`lightkurve`** Python package. Building on the common TESS analysis steps identified earlier, we will demonstrate how to encapsulate these steps into reusable Python functions or scripts suitable for integration into automated workflows. We then illustrate how to define and manage a typical TESS pipeline – involving data retrieval, aperture photometry from Target Pixel Files (TPFs), light curve cleaning, systematic error correction (detrending), and potentially transit searching – using either a dedicated **Workflow Management System (WMS)** like **Snakemake** or by leveraging the parallel processing capabilities of **Dask** (particularly `dask.bag` or `dask.delayed` for task parallelism across many targets). We will address practical issues like managing dependencies between steps (e.g., photometry must precede detrending) and handling intermediate data files within these frameworks. Finally, we discuss considerations for scaling these TESS analysis workflows from processing single targets locally to handling large numbers of targets efficiently on multi-core machines or HPC clusters using the chosen WMS or Dask framework.

**68.1 Recap: Common Lightkurve Analysis Steps for TESS**

The `lightkurve` package (introduced conceptually in App 65.A) provides a high-level, user-friendly interface for common tasks involved in analyzing time-series photometry data, particularly from NASA's Kepler, K2, and TESS missions. Before building automated workflows, it's essential to recall the typical sequence of steps often performed using `lightkurve` when processing TESS data, starting from Target Pixel Files (TPFs).

**1. Data Search and Download:** The process usually begins by identifying the target star (typically via its TIC ID) and the desired TESS sector(s). `lightkurve.search_targetpixelfile(target, sector=..., author=...)` queries the MAST archive for available TPF data. The resulting `SearchResult` object lists available data products, and `.download()` or `.download_all()` retrieves the corresponding TPF FITS file(s). TPF files contain time-series images of a small pixel cutout around the target, along with metadata and quality flags.

**2. Aperture Photometry:** While pre-processed light curves (SAP, PDCSAP) are often available, reprocessing from TPFs allows for custom aperture selection, which can be crucial for crowded fields or mitigating systematic effects. This involves:
    *   Loading the TPF file: `tpf = lk.read(tpf_filename)`.
    *   Defining a pixel aperture mask: This can be done interactively (`tpf.plot(aperture_mask=...)`) or programmatically based on pixel brightness (e.g., `aperture = tpf.create_threshold_mask(threshold=...)`).
    *   Performing Simple Aperture Photometry (SAP): Summing the flux within the chosen aperture mask at each time step using `lc_sap = tpf.to_lightcurve(aperture_mask=aperture)`. This yields a `LightCurve` object containing time and raw SAP flux.

**3. Background Subtraction (Optional but Recommended):** Estimating and subtracting the background flux contribution from the aperture photometry can improve precision. This might involve defining background pixels in the TPF and using methods within `to_lightcurve` or separate background modeling techniques.

**4. Light Curve Cleaning:** Raw light curves often contain outliers or data points affected by instrumental issues. These need to be identified and removed.
    *   Using Quality Flags: TESS data includes quality flags indicating known issues (e.g., cosmic rays, pointing instability). `lc = lc[lc.quality == 0]` or using `quality_bitmask` during download often removes bad cadences.
    *   Sigma Clipping: Removing statistical outliers using functions like `lc_clean = lc.remove_outliers(sigma=...)`.

**5. Systematic Error Correction (Detrending/Flattening):** TESS light curves are affected by systematic trends, primarily due to spacecraft pointing jitter moving the target star across pixels with slightly different sensitivities, as well as thermal variations. Removing these systematics is crucial for detecting shallow transit signals. Common `lightkurve` methods include:
    *   Basic Flattening (`lc_flat = lc_clean.flatten(window_length=...)`): Applies a Savitzky-Golay filter or other smoothing techniques to remove long-term trends. Simple but can distort transit shapes if the window length is poorly chosen.
    *   Pixel Level Decorrelation (PLD): Uses linear regression against the flux variations in individual pixels within the TPF to model and remove instrument systematics. Requires the TPF data. Implemented in `lightkurve`'s `RegressionCorrector` or `PLDCorrector` classes.
    *   Using Cotrending Basis Vectors (CBVs): Uses pre-computed basis vectors (available from MAST) derived from correlated noise across many stars on the same detector, modeling systematics as a linear combination of these CBVs. Handled via `RegressionCorrector`.
The output of this step is typically a flattened, normalized light curve (`LightCurve` object with flux centered around 1 or 0).

**6. Period Searching / Transit Search:** Once detrended, the light curve is searched for periodic signals, often specifically transit-like signals.
    *   Lomb-Scargle Periodogram (`pg = lc_flat.to_periodogram(method='lombscargle')`): For general periodic variability.
    *   Box Least Squares (BLS) Periodogram (`bls = lc_flat.to_periodogram(method='bls', duration=...)`): Optimized for detecting the box-like shape of transits. Requires specifying trial transit durations.
    *   Analysis of the periodogram involves identifying significant peaks (`bls.period_at_max_power`, `bls.power.max()`) to find candidate periods.

**7. Candidate Vetting and Plotting:** Candidate signals identified in the periodogram need vetting. This often involves folding the light curve at the candidate period (`lc_folded = lc_flat.fold(period=...)`) and visually inspecting the folded light curve plot for a convincing transit shape, checking for potential false positives (e.g., eclipsing binaries). Diagnostic plots (`lc.plot()`, `pg.plot()`, `bls.plot()`, `lc_folded.plot()`) are essential throughout the process.

This sequence represents a typical workflow for processing TESS data for transit searches using `lightkurve`. Each step takes specific inputs and produces outputs that feed into the next step, lending itself well to management by workflow systems or parallel execution frameworks like Dask when applied to many targets.

**68.2 Encapsulating Lightkurve Steps into Functions/Scripts**

To integrate the TESS analysis steps described above into an automated workflow managed by a WMS or a parallel framework like Dask, it's essential to encapsulate the logic for each distinct stage into reusable **Python functions** or separate **command-line scripts**. This modular approach, following the principles of Chapter 65, makes the workflow definition cleaner and allows individual steps to be tested, reused, and potentially executed independently or in parallel.

For example, we can define functions corresponding to the major stages:

*   **`download_tpf(tic_id, sector, download_dir)`:** Takes TIC and sector, uses `lightkurve.search_targetpixelfile().download()` and saves the TPF to a specified file path (e.g., `{download_dir}/tic{tic_id}_s{sector}_tp.fits`). Returns the output filename or raises an error.
*   **`extract_sap_lightcurve(tpf_filename, output_lc_filename, aperture_kwargs)`:** Takes the TPF filename, reads it using `lk.read()`, defines an aperture (perhaps based on `aperture_kwargs` like threshold), performs photometry using `.to_lightcurve()`, potentially performs basic quality masking, and saves the resulting raw SAP light curve object or table to `output_lc_filename` (e.g., as a FITS file). Returns the output filename.
*   **`detrend_lightcurve(raw_lc_filename, output_flat_filename, detrend_method='flatten', flatten_kwargs)`:** Takes the raw light curve filename, reads it, applies a specified detrending method (e.g., `.flatten()` with options from `flatten_kwargs`, or perhaps a more complex corrector), and saves the flattened light curve to `output_flat_filename`. Returns the output filename.
*   **`run_bls(flat_lc_filename, output_bls_filename, bls_kwargs)`:** Takes the flattened light curve filename, reads it, computes the BLS periodogram using `.to_periodogram(method='bls', **bls_kwargs)`, extracts key results (period, power, duration, depth), and saves these results (e.g., to a text file, JSON, or FITS header) named `output_bls_filename`.
*   **`plot_summary(tpf_file, raw_lc_file, flat_lc_file, bls_results_file, output_plot_filename)`:** Takes the filenames of intermediate and final products and generates a multi-panel summary plot (e.g., TPF aperture, raw LC, flattened LC, folded LC at best period, BLS periodogram), saving it to `output_plot_filename`.

Each function performs a specific, well-defined task. It takes input file paths (and potentially parameters) and produces output files. This file-based interaction makes the functions suitable for integration into WMSs like Snakemake or Nextflow, which manage dependencies based on file production.

Alternatively, these functions could be designed to return Python objects (like `LightCurve` or BLS result dictionaries) for use in workflows managed by Python libraries like Parsl or Dask (where data might flow in memory between tasks running within the same distributed environment, although saving intermediate files can still be useful for resilience).

Creating separate Python scripts (e.g., `download_tpf.py`, `extract_lc.py`, etc.) that use `argparse` (Sec 65.4) to accept input/output filenames and parameters from the command line is another common approach, especially for use with WMSs like Snakemake or Nextflow which primarily orchestrate shell commands.

```python
# --- Code Example 1: Encapsulating Download Step into Function ---
import lightkurve as lk
import os
import time # For potential retry delays

def download_tpf_func(tic_id, sector, output_dir="tpf_files", author="SPOC", retries=2):
    """Downloads TPF file for a given TIC and sector."""
    target = f"TIC {tic_id}"
    search_term = target + (f" Sector {sector}" if sector else "")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"tic{tic_id}_s{sector}_tpf.fits")
    
    print(f"  Attempting download for {search_term} -> {output_filename}")
    
    # Check if file already exists (basic resume capability)
    if os.path.exists(output_filename):
        print(f"    File already exists: {output_filename}")
        return output_filename

    search_kwargs = {'author': author}
    if sector is not None: search_kwargs['sector'] = sector
    
    current_retry = 0
    while current_retry <= retries:
        try:
            search_result = lk.search_targetpixelfile(target, **search_kwargs)
            if not search_result:
                print(f"    Error: No TPF found for {search_term}")
                return None # Indicate failure
            
            # Select first result (could add more logic)
            tpf_result = search_result[0]
            print(f"    Found TPF: {tpf_result.mission} S{tpf_result.sector}...")
            
            tpf = tpf_result.download(download_dir=output_dir, quality_bitmask='default',
                                      # Ensure filename matches expectation if download returns path
                                      # filename=output_filename # Does download support this? Check docs.
                                      ) 
                                      
            if tpf is None: # Download might fail silently sometimes
                 raise IOError("Download returned None object.")
            
            # Rename if download didn't use exact name (adjust based on lk behavior)
            # downloaded_path = ... # Get actual path from download() return if needed
            # if downloaded_path != output_filename: os.rename(downloaded_path, output_filename)
            
            print(f"    Download successful: {output_filename}")
            return output_filename # Return path on success

        except Exception as e:
            current_retry += 1
            print(f"    Error during search/download (Attempt {current_retry}/{retries}): {e}")
            if current_retry > retries:
                print(f"    Failed to download after {retries} retries.")
                return None # Indicate failure
            print(f"    Waiting {2**current_retry}s before retrying...")
            time.sleep(2**current_retry)

# --- Example Usage ---
# result_path = download_tpf_func(tic_id=261136679, sector=1, output_dir="temp_tpf")
# if result_path: print(f"\nSuccess: Downloaded to {result_path}")
# else: print("\nDownload failed.")
# if os.path.exists("temp_tpf"): shutil.rmtree("temp_tpf") # Cleanup
```
```python
# --- Code Example 2: Defining a Script with argparse ---
# File: scripts/detrend_lc.py (Conceptual)
import lightkurve as lk
import argparse
import os
import sys

def detrend_lightcurve_script(input_lc_path, output_lc_path, method='flatten', window=101):
    """Loads, detrends, and saves a light curve."""
    print(f"Detrending {input_lc_path} -> {output_lc_path} using method '{method}'")
    try:
        lc = lk.read(input_lc_path)
        lc = lc.remove_nans() # Basic cleaning
        
        if method == 'flatten':
            # Use Savitzky-Golay smoother via flatten
            lc_flat = lc.flatten(window_length=window)
        # elif method == 'pld': # Needs TPF access
        #     # Implement PLD correction... more complex
        #     lc_flat = lc # Placeholder
        else:
            print(f"Error: Unknown detrending method '{method}'")
            return False
            
        # Save flattened light curve
        outdir = os.path.dirname(output_lc_path)
        if outdir: os.makedirs(outdir, exist_ok=True)
        lc_flat.to_fits(output_lc_path, overwrite=True)
        print("Detrending and saving complete.")
        return True
        
    except Exception as e:
        print(f"Error during detrending: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detrend a TESS light curve.")
    parser.add_argument("-i", "--input", required=True, help="Input raw light curve FITS file.")
    parser.add_argument("-o", "--output", required=True, help="Output flattened light curve FITS file.")
    parser.add_argument("--method", default="flatten", choices=['flatten'], help="Detrending method ('flatten').")
    parser.add_argument("--window", type=int, default=101, help="Window length for flatten method.")
    
    args = parser.parse_args()
    
    success = detrend_lightcurve_script(args.input, args.output, args.method, args.window)
    
    if not success:
        sys.exit(1) # Exit with error code if failed
```
```python
# --- Python Block showing Conceptual Script Call ---
print("--- Conceptual Command Line Call for Script ---")
print("# python scripts/detrend_lc.py -i tess_output/raw_lc/tic123_s1_rawlc.fits -o tess_output/detrended/tic123_s1_flat.fits --window 801")
print("-" * 20)

# Explanation:
# Example 1: Defines a Python function `download_tpf_func` that encapsulates the 
#            lightkurve search and download logic, including basic error handling, 
#            retries, and checking for existing files. It returns the path to the 
#            downloaded file (or None). This function could be imported and called 
#            by a main workflow script or a WMS.
# Example 2: Shows the structure of a standalone script `scripts/detrend_lc.py` 
#            that uses `argparse` to take input/output filenames and method parameters 
#            from the command line. It defines a core processing function and calls 
#            it in the `if __name__ == "__main__":` block. This script could be 
#            executed directly from the shell or called by a WMS like Snakemake 
#            (as shown conceptually in App 66.A).
```

Encapsulating workflow steps into functions or parameterized scripts is fundamental for building automated, maintainable, and scalable analysis pipelines. It promotes code reuse, simplifies testing, and provides the modular components needed for orchestration by either custom Python scripts or dedicated Workflow Management Systems.

**68.3 Defining a TESS Workflow with Snakemake/Parsl**

Once the individual steps of the TESS analysis pipeline are encapsulated into functions or scripts (Sec 68.2), we can use a Workflow Management System (WMS) like Snakemake (Sec 66.2) or a parallel scripting library like Parsl (Sec 66.4) to define the dependencies and automate the execution for multiple targets and sectors.

**Using Snakemake:** Snakemake is well-suited for file-based workflows. We define rules in a `Snakefile` where each rule corresponds to a processing step (download, extract LC, detrend, BLS search, plot). Wildcards (`{tic}`, `{sector}`) are used extensively in input and output filenames to make the rules general. Snakemake automatically infers dependencies by matching output patterns of one rule to input patterns of another.

The `Snakefile` structure would resemble Application 66.A, but extended:
*   Define lists of `TIC_IDS` and `SECTORS`.
*   Use `expand()` to define the final target output files (e.g., BLS result files or summary plots for all TIC/Sector combinations).
*   Define `rule download_tpf:` with output `WORKDIR + "/tpfs/tic{tic}_s{sector}_tp.fits"`.
*   Define `rule extract_raw_lc:` with input using the TPF output pattern and output `WORKDIR + "/raw_lc/tic{tic}_s{sector}_rawlc.fits"`.
*   Define `rule detrend_cleaned_lc:` with input using the raw LC pattern and output `WORKDIR + "/detrended/tic{tic}_s{sector}_flat.fits"`.
*   Define `rule run_bls:` with input using the flattened LC pattern and output `WORKDIR + "/bls_results/tic{tic}_s{sector}_bls.txt"`.
*   Define `rule plot_summary:` taking multiple inputs (TPF, raw LC, flat LC, BLS results) and producing an output plot `WORKDIR + "/plots/tic{tic}_s{sector}_summary.png"`.
*   The `rule all:` would depend on the final plot files or BLS result files.
Running `snakemake --cores N all` executes the entire pipeline, automatically managing dependencies and parallelizing independent tasks (like processing different TICs or sectors concurrently). Snakemake's integration with Conda/Singularity helps ensure the correct `lightkurve` version and dependencies are used.

**Using Parsl:** Parsl defines workflows within Python by decorating functions as `@python_app` or `@bash_app`. Dependencies are created by passing the output *futures* of one app call as input arguments to the next.

The structure would involve:
*   Configuring Parsl (e.g., `parsl.load(local_htex_config)`).
*   Defining Python functions for each step (`download_tpf_app`, `extract_lc_app`, `detrend_lc_app`, `run_bls_app`, `plot_summary_app`), each decorated with `@python_app`. These functions would typically take filenames as input and return the output filename upon successful completion.
*   The main script would loop through the `TIC_IDS` and `SECTORS`.
*   Inside the loop, chain the app calls, passing the future result (output filename) from one step to the next:
    ```python
    # Conceptual Parsl Chaining
    tpf_future = download_tpf_app(tic, sector, ...)
    raw_lc_future = extract_lc_app(tpf_filename=tpf_future, ...) 
    flat_lc_future = detrend_lc_app(raw_lc_filename=raw_lc_future, ...)
    bls_future = run_bls_app(flat_lc_filename=flat_lc_future, ...)
    plot_future = plot_summary_app(tpf_file=tpf_future, raw_lc_file=raw_lc_future, ..., bls_results_file=bls_future, ...)
    # Store final futures if needed
    ```
*   Parsl automatically builds the dependency graph based on these future objects. Tasks for different TICs/Sectors can run in parallel. A final step collects results by calling `.result()` on the desired output futures (e.g., the `plot_future` objects or `bls_future` objects).

```python
# --- Code Example 1: Conceptual Snakemake Structure (Partial Snakefile) ---
# (Illustrates rules and dependencies for TESS workflow)
conceptual_snakefile = """
# --- Snakemake TESS Workflow ---
WORKDIR = "snakemake_tess_output"
TIC_IDS = ["TIC1", "TIC2"]
SECTORS = ["10", "11"]

FINAL_PLOTS = expand(WORKDIR + "/plots/{tic}_s{sector}_summary.png", tic=TIC_IDS, sector=SECTORS)

rule all:
    input: FINAL_PLOTS

rule plot_summary:
    input:
        tpf=WORKDIR+"/tpfs/{tic}_s{sector}_tp.fits",
        raw=WORKDIR+"/raw_lc/{tic}_s{sector}_rawlc.fits",
        flat=WORKDIR+"/detrended/{tic}_s{sector}_flat.fits",
        bls=WORKDIR+"/bls/{tic}_s{sector}_bls.txt"
    output:
        WORKDIR+"/plots/{tic}_s{sector}_summary.png"
    shell: "python scripts/plot_summary.py --tpf {input.tpf} --raw {input.raw} --flat {input.flat} --bls {input.bls} -o {output}"

rule run_bls:
    input: WORKDIR+"/detrended/{tic}_s{sector}_flat.fits"
    output: WORKDIR+"/bls/{tic}_s{sector}_bls.txt"
    shell: "python scripts/run_bls.py -i {input} -o {output}"

rule detrend_cleaned_lc:
    input: WORKDIR+"/raw_lc/{tic}_s{sector}_rawlc.fits"
    output: WORKDIR+"/detrended/{tic}_s{sector}_flat.fits"
    shell: "python scripts/detrend_lc.py -i {input} -o {output}"

rule extract_raw_lc:
    input: WORKDIR+"/tpfs/{tic}_s{sector}_tp.fits"
    output: WORKDIR+"/raw_lc/{tic}_s{sector}_rawlc.fits"
    shell: "python scripts/extract_lc.py -i {input} -o {output}"

rule download_tpf:
    output: WORKDIR+"/tpfs/{tic}_s{sector}_tp.fits"
    params: tic="{tic}", sector="{sector}"
    shell: "python scripts/download_tpf.py --tic {params.tic} --sector {params.sector} --output {output}"
"""
print("--- Conceptual Snakemake Structure ---")
print(conceptual_snakefile)
print("-" * 20)
```

```python
# --- Code Example 2: Conceptual Parsl Structure (Partial Script) ---
# (Illustrates chaining apps with futures)
print("\n--- Conceptual Parsl Workflow Structure ---")
conceptual_parsl = """
import parsl
from parsl import python_app
# Assume parsl is loaded with a config
# Assume app functions download_tpf_app, extract_lc_app, etc. are defined

tic_ids = ["TIC1", "TIC2"]
sectors = ["10", "11"]

all_final_futures = [] # Store futures for final results

for tic in tic_ids:
    for sec in sectors:
        print(f"Defining tasks for TIC {tic} Sector {sec}")
        
        tpf_fut = download_tpf_app(tic, sec) 
        # Pass future object to next app
        raw_lc_fut = extract_lc_app(tpf_filename=tpf_fut) 
        flat_lc_fut = detrend_lc_app(raw_lc_filename=raw_lc_fut)
        bls_fut = run_bls_app(flat_lc_filename=flat_lc_fut)
        # Plot app depends on multiple intermediate futures
        plot_fut = plot_summary_app(
            tpf_file=tpf_fut, 
            raw_lc_file=raw_lc_fut, 
            flat_lc_file=flat_lc_fut, 
            bls_results_file=bls_fut
        )
        all_final_futures.append(plot_fut) # Collect final plot futures

print("\\nWaiting for all plot futures to complete...")
# Wait for all plots to be generated
# This loop implicitly waits due to .result()
final_plot_files = [f.result() for f in all_final_futures] 

print(f"\\nWorkflow finished. Generated plots: {final_plot_files}")
# parsl.dfk().cleanup() 
"""
print(conceptual_parsl)
print("-" * 20)

# Explanation:
# Example 1 shows the rule structure in a Snakefile. Each rule defines how an output 
# file (e.g., `detrended/..._flat.fits`) depends on input files (e.g., `raw_lc/..._rawlc.fits`) 
# and the shell command to create it. Snakemake resolves the chain automatically.
# Example 2 shows the Parsl approach within Python. Functions decorated with `@python_app` 
# are called. Crucially, the output of one app call (a `future` object) is passed as 
# input to the next app call (e.g., `extract_lc_app(tpf_filename=tpf_fut)`). Parsl uses 
# these future dependencies to build the DAG and schedule execution. The final loop 
# with `.result()` waits for completion.
```

Both Snakemake and Parsl (and Nextflow) provide robust mechanisms for defining and executing the TESS workflow. Snakemake excels with its file-pattern matching and Conda/container integration. Parsl offers a potentially more seamless integration for users primarily working within Python, using function calls and futures to define the workflow structure. The choice depends on user preference and the specific requirements for environment management and execution platform flexibility. Dask (next section) offers another powerful Python-native alternative, especially when computations involve large NumPy arrays or Pandas DataFrames directly.

**68.4 Parallelizing TESS Analysis with Dask Bag / Delayed**

Dask provides several ways to parallelize workflows like the TESS light curve analysis, particularly suitable for **task parallelism** where the same operation is applied independently to many input items (here, TESS targets or light curve files). The most relevant Dask components for this are **Dask Bag** (Sec 40.5, 67.4) and **Dask Delayed** (Sec 40.4).

**Using Dask Bag:** Dask Bag is designed for parallel processing of sequences or collections of arbitrary Python objects. We can represent our list of TESS TIC IDs or input light curve filenames as a Dask Bag and then use the `.map()` operation to apply our analysis functions in parallel.

1.  **Create Bag:** Start with a list of targets (e.g., `tic_list`). Create a bag using `b = db.from_sequence(tic_list, npartitions=...)`. `npartitions` controls the degree of parallelism locally.
2.  **Define Worker Function(s):** Define Python functions for each step (e.g., `download_tpf`, `extract_lc`, `detrend_lc`, `run_bls`) that take the necessary input (like a TIC ID or filename) and return the output (e.g., filename of the processed file, or a results dictionary).
3.  **Chain Map Operations:** Apply the functions sequentially using `.map()`:
    ```python
    b = db.from_sequence(tic_list, npartitions=N)
    tpf_files_bag = b.map(download_tpf_func) 
    raw_lc_files_bag = tpf_files_bag.map(extract_lc_func) # Assumes functions take/return filenames
    flat_lc_files_bag = raw_lc_files_bag.map(detrend_lc_func)
    bls_results_bag = flat_lc_files_bag.map(run_bls_func)
    ```
    Note: This chaining assumes each function simply takes the output filename of the previous step. Error handling (e.g., if a step returns `None`) needs to be incorporated, perhaps using `.filter()`. Alternatively, a single complex worker function can perform the whole chain internally.
4.  **Compute:** Trigger execution using `results = bls_results_bag.compute()`. Dask executes the mapped functions in parallel across partitions using the configured scheduler (threads, processes, or distributed).

**Using Dask Delayed:** Dask Delayed offers a more explicit way to build the task graph by wrapping individual function calls. It's very flexible and allows constructing complex, non-linear workflows.

1.  **Wrap Function Calls:** Import `dask.delayed`. Wrap each call to your processing functions with `delayed()`: `d_tpf = delayed(download_tpf_func)(tic, sec)`. This returns a `Delayed` object representing the task, not the result.
2.  **Define Dependencies:** Pass the `Delayed` object output from one step as input to the next wrapped function call: `d_raw_lc = delayed(extract_lc_func)(d_tpf)`. Dask automatically tracks these dependencies.
3.  **Build Graph for All Targets:** Loop through your list of targets, creating the chain of delayed calls for each one. Store the final `Delayed` objects (e.g., for the BLS results or plots) in a list.
4.  **Compute:** Pass the list of final delayed objects to `dask.compute()`: `results = dask.compute(*list_of_final_delayed_objects)`. Dask analyzes the combined graph for all targets and executes independent tasks in parallel.

```python
# --- Code Example 1: Dask Bag Workflow (Conceptual) ---
import dask.bag as db
import dask
from dask.diagnostics import ProgressBar
import os
import time

# Assume functions download_tpf_func, extract_lc_func, 
# detrend_lc_func, run_bls_func exist (they take/return filenames or results)
# Simplified versions for illustration:
def dummy_download(tic): print(f" DL {tic}"); time.sleep(0.02); return f"tpf_{tic}.fits"
def dummy_extract(fname): print(f" EX {fname}"); time.sleep(0.03); return f"raw_{fname}"
def dummy_detrend(fname): print(f" DT {fname}"); time.sleep(0.05); return f"flat_{fname}"
def dummy_bls(fname): print(f" BLS {fname}"); time.sleep(0.1); return {'file':fname, 'period':np.random.rand()*10}

print("Conceptual TESS Workflow with Dask Bag:")
tic_list = [f"TIC_{i:03d}" for i in range(20)]
n_workers = 4
print(f"\nProcessing {len(tic_list)} TICs using Dask Bag with {n_workers} workers...")

# Create bag and chain map operations
b = db.from_sequence(tic_list, npartitions=n_workers)
tpf_bag = b.map(dummy_download)
raw_bag = tpf_bag.map(dummy_extract)
flat_bag = raw_bag.map(dummy_detrend)
bls_bag = flat_bag.map(dummy_bls)

# Execute
# dask.config.set(scheduler='processes', num_workers=n_workers) # Use processes
print("\nComputing results...")
start_time = time.time()
with ProgressBar():
    final_results = bls_bag.compute() # Returns list of results
end_time = time.time()
print(f"\nComputation finished. Time: {end_time - start_time:.3f}s")
print(f"Results sample (first 3): {final_results[:3]}")
print("-" * 20)
```

```python
# --- Code Example 2: Dask Delayed Workflow (Conceptual) ---
import dask
from dask import delayed
from dask.diagnostics import ProgressBar
import time
import os

# Use same dummy functions as above
print("\nConceptual TESS Workflow with Dask Delayed:")
tic_list = [f"TIC_{i:03d}" for i in range(20)]
n_workers = 4
print(f"\nProcessing {len(tic_list)} TICs using Dask Delayed with {n_workers} workers...")

# Build list of delayed objects representing the final result for each TIC
final_delayed_results = []
for tic in tic_list:
    # Wrap each function call with delayed()
    # Pass the delayed object (dependency) to the next step
    d_tpf = delayed(dummy_download)(tic)
    d_raw = delayed(dummy_extract)(d_tpf)
    d_flat = delayed(dummy_detrend)(d_raw)
    d_bls = delayed(dummy_bls)(d_flat)
    final_delayed_results.append(d_bls) # Collect final task for each TIC

print(f"Built list of {len(final_delayed_results)} delayed objects (task graph).")

# Execute the entire graph
# dask.config.set(scheduler='processes', num_workers=n_workers) # Use processes
print("\nComputing results...")
start_time = time.time()
with ProgressBar():
    # dask.compute executes all delayed objects passed to it
    final_results = dask.compute(*final_delayed_results) # Unpack list into args
end_time = time.time()
print(f"\nComputation finished. Time: {end_time - start_time:.3f}s")
print(f"Results sample (first 3): {final_results[:3]}")
print("-" * 20)

# Explanation:
# Example 1 (Dask Bag): Creates a bag from the `tic_list`. Each `.map(func)` call applies 
# the corresponding dummy processing function lazily to each item passed through the bag. 
# `.compute()` executes the whole pipeline in parallel across partitions.
# Example 2 (Dask Delayed): Explicitly wraps each function call for each TIC with `delayed()`. 
# The dependency is explicitly passed (e.g., `d_raw = delayed(dummy_extract)(d_tpf)`). 
# A list of the final `delayed` objects (one per TIC) is created. `dask.compute()` takes 
# this list (unpacked with `*`) and executes the entire combined task graph in parallel. 
# Both achieve similar parallel task execution for this workflow. Delayed offers more 
# flexibility for complex, non-linear dependencies.
```

**Choosing between Bag and Delayed:**
*   **Dask Bag:** Simpler syntax for linear pipelines where the same function(s) are mapped across a sequence. Good for ETL-like tasks on lists or text files.
*   **Dask Delayed:** More flexible for complex workflows with arbitrary Python code, non-linear dependencies, or where different functions need to be called for different items. Provides more explicit control over the task graph construction.

Both Dask Bag and Dask Delayed provide excellent Python-native ways to parallelize task-based workflows like processing many TESS light curves. They integrate seamlessly with Dask's local and distributed schedulers, allowing easy scaling from multi-core machines to HPC clusters simply by changing the scheduler configuration or connecting a `dask.distributed.Client`.

**68.5 Handling Dependencies and Intermediate Files**

When implementing workflows, whether using simple scripts, WMS tools, or Dask, correctly handling **dependencies** between steps and managing **intermediate files** (if using file-based data passing) is crucial for correctness, efficiency, and reproducibility.

**Dependency Management:**
*   **Python Scripts (Manual):** Dependencies are managed solely by the order of function calls. The programmer must ensure Function B is called only after Function A finishes if B needs A's output. For file-based passing, this includes checking if A's output file exists before calling B (Sec 65.3). This quickly becomes complex and error-prone for non-linear workflows.
*   **WMS (Snakemake, Nextflow):** These tools excel at dependency management.
    *   **Snakemake:** Primarily uses **filename matching**. You declare the `input:` and `output:` files for each rule. Snakemake builds a DAG by inferring that if rule B requires a file produced by rule A, then A must run before B. It automatically determines the execution order and parallelizes independent rules.
    *   **Nextflow:** Uses **channel dataflow**. A process consumes items from input channels and produces items on output channels. The workflow definition connects output channels to input channels, defining the dependencies. Nextflow manages the flow of data items and triggers processes accordingly.
    *   **Parsl:** Uses **futures**. Dependencies are implicitly defined when the output future of one app call is passed as an input argument to another app call. Parsl's Data Flow Kernel tracks these dependencies to determine execution order.
WMS tools automate dependency tracking, which is a major advantage over manual scripting for complex pipelines.

**Intermediate Files:** Workflows often generate intermediate data products (e.g., calibrated images before source extraction, raw light curves before detrending). Managing these files requires consideration:
*   **Naming Conventions:** Using consistent, predictable naming schemes for intermediate files, often incorporating parameters or wildcards (like `calibrated/tic{tic}_s{sector}_cal.fits`), is essential for WMS tools (like Snakemake) to track dependencies correctly and for humans to understand the pipeline structure.
*   **Storage Location:** Intermediate files can consume significant disk space. They should ideally be written to appropriate scratch directories on HPC systems (Sec 37.2) rather than home directories. Workflows might need steps to clean up intermediate files once they are no longer needed by downstream tasks to save space, although WMSs often handle this based on rules or options. Some WMSs can optionally keep specified intermediate files for debugging.
*   **Persistence and Restart:** File-based workflows offer resilience. If the workflow fails, intermediate files from completed steps persist. WMS tools (and careful scripting with file checks) can leverage this to restart the workflow, skipping already completed steps by checking the existence and sometimes timestamps or checksums of output files relative to inputs. This "smart restart" capability is a major advantage for long-running pipelines.
*   **File Formats:** Choosing efficient formats for intermediate files (e.g., compressed FITS, HDF5, Parquet) can significantly impact I/O performance and storage usage compared to inefficient formats like plain text or uncompressed pickle files for large data.

**Dask and Intermediate Data:** Dask's approach differs slightly depending on the collection and scheduler:
*   When operating **in-memory** (datasets fit in RAM across workers), Dask often keeps intermediate results (chunks/partitions) directly in the distributed memory of the workers, passing them between tasks as needed via the network. This avoids disk I/O overhead but requires sufficient aggregate RAM and means intermediate results are lost if a worker fails (though Dask has some fault tolerance mechanisms).
*   For **larger-than-memory datasets**, Dask collections inherently work with chunks/partitions that might reside on disk (e.g., Dask Array from HDF5, Dask DataFrame from Parquet). Operations load necessary chunks, compute, and potentially write intermediate chunked results back to disk (managed by Dask) or keep them in memory if possible, before the final `.compute()` aggregates the result. This requires efficient underlying storage.
*   Users can explicitly save intermediate Dask collection results to disk using methods like `.to_zarr()`, `.to_hdf5()`, `.to_parquet()`, `.to_textfiles()`. This creates persistent intermediate files, similar to traditional file-based workflows, which can be useful for checkpointing long Dask computations or reusing intermediate results later. `dask.delayed` workflows often naturally work with filenames, similar to file-based WMS approaches.

Effectively managing dependencies (either manually, implicitly via futures, or explicitly via WMS rules/channels) and intermediate data (in memory vs. persistent files, naming, location, format) are key practical aspects of building robust, efficient, and reproducible astronomical workflows, regardless of the specific tools used for orchestration. WMS tools generally provide the most automated and robust solutions for complex, file-based dependency management and restarts. Dask excels at managing dependencies internally for array/dataframe computations, potentially keeping data in memory for performance but also allowing explicit saving of intermediate results.

**68.6 Scaling Workflow Execution (Local vs. Cluster)**

A major advantage of using explicit workflow definition tools like WMSs (Snakemake, Nextflow, Parsl) or scalable libraries like Dask is the ability to develop and test a workflow locally on a smaller scale (e.g., using a few cores on a laptop or workstation) and then **scale its execution** to much larger datasets or numbers of tasks on High-Performance Computing (HPC) clusters or cloud platforms, often with relatively minor changes to the workflow definition itself. These tools abstract away many of the details of the execution environment.

**Local Execution:** During development and testing, workflows are typically run locally:
*   **Snakemake:** `snakemake --cores N ...` executes the workflow using `N` local CPU cores, running independent rules in parallel using Python's `multiprocessing`.
*   **Nextflow:** Running `nextflow run pipeline.nf` without specific executor configuration usually defaults to running processes locally, potentially using available cores.
*   **Parsl:** Configuring Parsl with `LocalThreads` or `LocalProcesses` executors (as in App 66.B) directs execution to local threads or processes.
*   **Dask:** Using the default synchronous scheduler runs serially. Configuring the `threads` or `processes` scheduler (`dask.config.set(...)` or via `Client(processes=...)`) enables parallel execution across local cores (Sec 40.6).
Local execution is essential for debugging the workflow logic, testing individual steps on small datasets, and ensuring the basic pipeline works correctly before scaling up.

**Cluster/Cloud Execution:** To handle larger datasets or thousands of tasks requiring significant computational resources, the workflow execution needs to be scaled to an HPC cluster or cloud environment. WMSs and Dask provide mechanisms to manage this transition:
*   **Snakemake:** Supports various cluster execution modes via command-line arguments or profiles.
    *   `--cluster "sbatch [options]"`: Provides a generic command template for submitting each rule's job to a scheduler like SLURM. Snakemake substitutes resource requests (threads, memory, time, specified per rule or globally) into the template.
    *   `--profile path/to/profile`: Uses predefined profiles containing configurations for specific schedulers (SLURM, PBS, SGE, LSF) and resource specifications, simplifying submission commands.
    *   Cloud support (AWS, Google Cloud) allows execution via Kubernetes or cloud batch systems.
*   **Nextflow:** Has built-in support for numerous **executors** configured in `nextflow.config` files. Specifying `process.executor = 'slurm'` (or 'pbs', 'sge', 'awsbatch', 'google-lifesciences', 'k8s') directs Nextflow to submit process jobs to the respective platform. Resource requirements (`cpus`, `memory`, `time`) can be specified per process or globally. Profiles allow easy switching between local, cluster, and cloud configurations.
*   **Parsl:** Requires configuring the `Parsl.Config` object with appropriate **Executors** (like `HighThroughputExecutor`) and **Providers** (like `SlurmProvider`, `PBSProProvider`, `CondorProvider`, `AWSProvider`, `GoogleCloudProvider`). The Provider handles submitting pilot jobs to the cluster/cloud scheduler, which then start workers that connect back to Parsl's Interchange process to execute app tasks.
*   **Dask:** Scaling beyond a single node requires the **distributed scheduler**.
    *   **Manual Setup:** Start `dask-scheduler` and `dask-worker` processes manually across nodes.
    *   **`dask-jobqueue`:** The preferred method for HPC clusters. Use classes like `SLURMCluster`, `PBSCluster` to programmatically define worker job specifications (cores, memory, queue) and launch workers via the cluster scheduler. Connect a `dask.distributed.Client` to this cluster object (Sec 40.6, Code Example 2). Dask computations are then automatically distributed across the managed workers.
    *   Cloud integrations (e.g., `dask-cloudprovider`) exist for deploying Dask clusters on AWS, GCP, Azure.

The key advantage offered by these tools is that the **core workflow definition** (e.g., the `Snakefile`, `pipeline.nf`, Parsl Python script with app definitions, Dask computation graph) often remains **largely unchanged** when moving from local execution to cluster/cloud execution. The primary change is in the **configuration** layer, where you specify the execution backend (local vs. SLURM vs. cloud) and associated resource parameters (number of cores/nodes, memory limits, walltimes, queue names, account info).

This separation of workflow logic from execution configuration makes these tools extremely powerful for scalable science. Researchers can develop and debug workflows interactively on their local machines with small test data, and then deploy the *same* workflow definition to process large datasets on powerful HPC resources simply by changing the execution command or configuration file. This significantly lowers the barrier to leveraging HPC for complex, multi-step analysis pipelines common in astrophysics, promoting efficiency and reproducibility across different computational scales. Careful resource specification (requesting appropriate memory, cores, time) remains crucial for efficient utilization of shared cluster resources (Sec 37.6).

**(Code examples typically involve configuration files or command-line arguments specific to the WMS/Dask and the target cluster, making simple, universal illustrations difficult.)**

---
**Application 68.A: Snakemake Pipeline for Detrending Multiple TESS Sectors**

**(Paragraph 1)** **Objective:** Implement and execute the TESS light curve processing workflow (Download TPF -> Extract Raw LC -> Detrend LC) described conceptually in Application 66.A using **Snakemake** (Sec 66.2) for a set of TESS targets across multiple sectors. This application focuses on the practical `Snakefile` implementation and execution.

**(Paragraph 2)** **Astrophysical Context:** Analyzing stars observed by TESS often requires processing data from multiple sectors to build longer time-baseline light curves or search for signals that might only be present in specific sectors. Automating the download, photometry, and detrending steps for numerous target-sector combinations using a workflow manager like Snakemake ensures consistency and handles the processing efficiently.

**(Paragraph 3)** **Data Source:** A list of TESS Input Catalog IDs (`TIC_IDS`) and TESS Sector numbers (`SECTORS`). The workflow will download the required TPF files from MAST using helper scripts calling `lightkurve`. Assumes existence of helper Python scripts (`scripts/download_tpf.py`, `scripts/extract_lc.py`, `scripts/detrend_lc.py`) as described in Sec 68.2 and App 66.A.

**(Paragraph 4)** **Modules Used:** `snakemake` (command-line tool), Python (for the `Snakefile` itself and helper scripts), `lightkurve`, `numpy`, `astropy` (within helper scripts).

**(Paragraph 5)** **Technique Focus:** Practical implementation of a Snakemake workflow. Creating the `Snakefile`. Defining input lists (`TIC_IDS`, `SECTORS`). Using `expand()` to generate target output filenames (flattened light curves) for the `rule all`. Defining rules (`download_tpf`, `extract_raw_lc`, `detrend_cleaned_lc`) with `input:`, `output:`, and `shell:` sections. Using wildcards (`{tic}`, `{sector}`) effectively in input/output patterns. Ensuring dependencies are correctly inferred by Snakemake based on filename patterns. Executing the workflow locally using `snakemake --cores N`.

**(Paragraph 6)** **Processing Step 1: Create Project Structure:** Create a main directory. Inside, create subdirectories: `scripts/` (for helper Python scripts), `raw_data/` (conceptual input, though downloads go elsewhere), `tess_workflow_output/` (for outputs), and `tess_workflow_output/logs/`. Place the (hypothetical) helper scripts in `scripts/`.

**(Paragraph 7)** **Processing Step 2: Write `Snakefile`:** Create the `Snakefile` in the main directory. Define `TIC_IDS` and `SECTORS` lists. Define the `WORKDIR` variable. Use `expand` to define the list of `FINAL_LCS` (e.g., `WORKDIR + "/detrended/tic{tic}_s{sector}_flat.fits"`). Define `rule all` with `input: FINAL_LCS`. Define the three rules (`download_tpf`, `extract_raw_lc`, `detrend_cleaned_lc`) similar to the conceptual example in Sec 68.3, ensuring input/output paths match the structure and use the `WORKDIR`. The `shell:` commands should correctly call the Python scripts in the `scripts/` directory, passing `{input}` and `{output}` placeholders.

**(Paragraph 8)** **Processing Step 3: Run Snakemake (Dry Run):** Open a terminal in the main directory. Execute `snakemake --cores 1 --dryrun -p all`. Examine the output. Snakemake should print the sequence of jobs it *would* run (determined by the DAG) and the corresponding shell commands, substituting wildcards correctly. Verify the dependencies and commands look correct.

**(Paragraph 9)** **Processing Step 4: Run Snakemake (Execution):** Execute `snakemake --cores N all` (where `N` is the number of cores you want to use, e.g., `4`). Snakemake will start executing the jobs, running independent tasks (e.g., processing different TIC/Sector combinations) in parallel up to `N` cores. Monitor the progress printed to the console and check for any errors reported in the log files.

**(Paragraph 10)** **Processing Step 5: Check Final Outputs:** Once Snakemake completes, check the `tess_workflow_output/detrended/` directory. Verify that the expected flattened light curve files (`tic{tic}_s{sector}_flat.fits`) have been created for all combinations defined by `TIC_IDS` and `SECTORS`.

**Output, Testing, and Extension:** Output includes the generated intermediate and final FITS files in the specified output directories, log files for each step, and Snakemake's console output showing execution progress. **Testing:** Perform the dry run. Verify final files exist. Inspect log files for errors. Delete some intermediate files and rerun Snakemake to ensure it correctly reruns only the necessary subsequent steps. **Extensions:** (1) Add a rule to run `run_bls.py` on the flattened light curves. (2) Add a rule to generate summary plots using `plot_summary.py`. (3) Parameterize the workflow using a `config.yaml` file for TIC IDs, sectors, and script parameters. (4) Add Conda environment definitions (`environment.yaml`) and use `snakemake --use-conda ...` for enhanced reproducibility. (5) Create a Snakemake profile to submit jobs to an HPC cluster.

```python
# --- Code Example: Application 68.A ---
# This example focuses on the Snakefile content and execution command.
# Assumes helper scripts (download_tpf.py, extract_lc.py, detrend_lc.py) exist in 'scripts/'
# and will be called by the shell commands.

print("--- Snakemake Pipeline for TESS Detrending ---")

# --- Snakefile Content (Save as 'Snakefile') ---
snakefile_for_tess = """
import os

# --- Configuration ---
WORKDIR = "tess_pipeline_output" # Output base directory
TIC_IDS = ["261136679", "89025133"] # Example target TIC IDs
SECTORS = ["1", "27", "54"] # Example sectors to attempt for each TIC

# --- Helper function to generate FINAL target files ---
# We want detrended files for all existing input TPFs that were successfully downloaded.
# This makes the 'all' rule depend only on steps that should succeed.
# A simpler way is just expand over all TIC/Sector combos, but some might fail download.
# Let's use expand for simplicity here, assuming downloads might fail gracefully.
FINAL_DETRENDED_FILES = expand(WORKDIR + "/detrended/tic{tic}_s{sector}_flat.fits",
                               tic=TIC_IDS, sector=SECTORS)

# --- Rule all: Defines the final desired output ---
rule all:
    input:
        FINAL_DETRENDED_FILES
    run:
        print("Snakemake TESS pipeline finished.")
        print("Generated detrended files (if successful):")
        for f in input: print(f"  - {f}")

# --- Rule: Detrend Light Curve ---
# Input depends on output of extract_raw_lc
rule detrend_lc:
    input:
        raw_lc = WORKDIR + "/raw_lc/tic{tic}_s{sector}_rawlc.fits"
    output:
        # Use touch to create empty file upon completion for snakemake tracking
        flat_lc = touch(WORKDIR + "/detrended/tic{tic}_s{sector}_flat.fits") 
    log:
        WORKDIR + "/logs/detrend/tic{tic}_s{sector}.log"
    params:
        window=801 # Example window length for flatten
    shell:
        # Ensure output directories exist before script runs
        "mkdir -p $(dirname {output.flat_lc}); mkdir -p $(dirname {log}); "
        "echo 'Detrending {input.raw_lc}...' > {log}; "
        "python scripts/detrend_lc.py --input {input.raw_lc} --output {output.flat_lc} --window {params.window} &>> {log}"

# --- Rule: Extract Raw Light Curve ---
# Input depends on output of download_tpf
rule extract_lc:
    input:
        tpf = WORKDIR + "/tpfs/tic{tic}_s{sector}_tp.fits"
    output:
        raw_lc = touch(WORKDIR + "/raw_lc/tic{tic}_s{sector}_rawlc.fits")
    log:
        WORKDIR + "/logs/extract/tic{tic}_s{sector}.log"
    shell:
        "mkdir -p $(dirname {output.raw_lc}); mkdir -p $(dirname {log}); "
        "echo 'Extracting LC from {input.tpf}...' > {log}; "
        "python scripts/extract_lc.py --input {input.tpf} --output {output.raw_lc} &>> {log}"

# --- Rule: Download TPF ---
# This rule has only outputs defined relative to wildcards
rule download_tpf:
    output:
        tpf = touch(WORKDIR + "/tpfs/tic{tic}_s{sector}_tp.fits") # touch creates placeholder if script fails? Better check script exit code.
    log:
        WORKDIR + "/logs/download/tic{tic}_s{sector}.log"
    params: # Pass wildcards to the script
        tic = "{tic}",
        sector = "{sector}"
    shell:
        # Ensure output/log directories exist
        "mkdir -p $(dirname {output.tpf}); mkdir -p $(dirname {log}); "
        "echo 'Downloading TPF for TIC {params.tic} Sector {params.sector}...' > {log}; "
        "python scripts/download_tpf.py --tic {params.tic} --sector {params.sector} --output {output.tpf} &>> {log}"
"""
print("--- Snakefile Content ---")
print(snakefile_for_tess)

# --- Conceptual Helper Scripts (e.g., scripts/download_tpf.py) ---
# These would contain the lightkurve/argparse logic from App 65.A etc.
print("\n--- Conceptual Helper Scripts (in scripts/ directory) ---")
print("# scripts/download_tpf.py uses argparse, lightkurve.search_targetpixelfile().download()")
print("# scripts/extract_lc.py uses argparse, lk.read(tpf).to_lightcurve().to_fits()")
print("# scripts/detrend_lc.py uses argparse, lk.read(lc).flatten().to_fits()")

# --- Execution Command (in terminal) ---
print("\n--- Execution Command (run in directory with Snakefile) ---")
print("# snakemake --cores 4 all") # Run using 4 cores locally
print("# snakemake --dryrun -p all # Check execution plan first")

print("-" * 20)
```

**Application 68.B: Dask-based Parallel BLS Search on TESS Light Curves**

**(Paragraph 1)** **Objective:** Implement a parallel workflow using **Dask** (Sec 40.4, 67) to perform a Box Least Squares (BLS) transit search on a large set of pre-processed TESS light curve files. This application utilizes `dask.bag` or `dask.delayed` to distribute the computationally intensive BLS analysis for each light curve across multiple CPU cores or potentially a Dask cluster.

**(Paragraph 2)** **Astrophysical Context:** Searching for transiting exoplanets involves applying algorithms like BLS to potentially millions of light curves. BLS itself requires testing many trial periods, durations, and phases, making it computationally demanding when applied to long-baseline, high-cadence data like TESS'. Parallelizing the search across many light curves simultaneously is essential for processing large survey datasets efficiently.

**(Paragraph 3)** **Data Source:** A list of filenames pointing to pre-processed (cleaned, flattened) TESS light curve files (e.g., `detrended/tic*_flat.fits` from App 68.A). Each file contains time and normalized flux data, ready for transit searching.

**(Paragraph 4)** **Modules Used:** `dask.bag` or `dask.delayed`, `dask.diagnostics.ProgressBar`, `lightkurve` (within the worker function), `numpy`, `glob`, `os`. `dask.distributed.Client` (optional, for cluster execution).

**(Paragraph 5)** **Technique Focus:** Task parallelism using Dask. (1) Generating the list of input flattened light curve filenames using `glob.glob`. (2) Defining a worker function `run_bls_on_lc(filename)` that loads a light curve file using `lk.read()`, performs the BLS search using `lc.to_periodogram(method='bls', ...)`, extracts relevant results (best period, duration, depth, power/SNR), and returns them (e.g., as a dictionary). (3) Using either `dask.bag.from_sequence(files).map(run_bls_on_lc)` or a loop creating `dask.delayed(run_bls_on_lc)(file)` objects to build the Dask task graph. (4) Configuring a parallel Dask scheduler (e.g., `dask.config.set(scheduler='processes')`). (5) Executing the computation using `.compute()` or `dask.compute()`, potentially with a `ProgressBar`. (6) Collecting and potentially filtering the BLS results from all light curves.

**(Paragraph 6)** **Processing Step 1: Prepare File List and Worker Function:** Use `glob.glob(detrended_dir + '/*_flat.fits')` to get the list of input files. Define the function `run_bls_on_lc(filename)`: import `lightkurve`, load the FITS file, call `.to_periodogram(method='bls', ...)` specifying appropriate period range and durations, extract results like `bls.period_at_max_power`, `bls.power_at_max_power`, etc., handle potential errors during loading or BLS, and return a dictionary of results.

**(Paragraph 7)** **Processing Step 2 (Option A - Dask Bag):** Create the bag: `b = db.from_sequence(lc_files, npartitions=N)`. Map the worker: `results_bag = b.map(run_bls_on_lc)`.

**(Paragraph 8)** **Processing Step 2 (Option B - Dask Delayed):** Create an empty list `futures = []`. Loop through `lc_files`: `fut = delayed(run_bls_on_lc)(lc_file); futures.append(fut)`.

**(Paragraph 9)** **Processing Step 3: Configure Scheduler and Execute:** Set the desired scheduler, e.g., `dask.config.set(scheduler='processes', num_workers=os.cpu_count())`. Execute:
    *   For Bag: `with ProgressBar(): all_results = results_bag.compute()`.
    *   For Delayed: `with ProgressBar(): all_results = list(dask.compute(*futures))`.

**(Paragraph 10)** **Processing Step 4: Process Results:** `all_results` will be a list of dictionaries (one per light curve). Convert to a Pandas DataFrame (`results_df = pd.DataFrame(all_results)`). Filter candidates based on BLS power or other criteria (e.g., `candidates = results_df[results_df['power'] > threshold]`). Print or save the candidate list.

**Output, Testing, and Extension:** Output includes the list or DataFrame of BLS results for all processed light curves, potentially filtered to show candidates. Progress bar shown during execution. **Testing:** Verify results are obtained for all input files (check for errors in results). For a light curve known to contain a transit, check if BLS recovers the correct period and parameters. Compare execution time with serial execution on a subset of files to estimate speedup. Test with different schedulers. **Extensions:** (1) Connect a `dask.distributed.Client` to run on an HPC cluster for very large numbers of light curves. (2) Make the BLS parameters (period range, duration) configurable inputs. (3) Add a subsequent vetting step (e.g., calculating secondary eclipse depth, checking transit shape) as another mapped Dask task using `.map()` or `delayed()`. (4) Save the full BLS periodogram object for promising candidates instead of just the peak parameters.

```python
# --- Code Example: Application 68.B ---
# Note: Requires dask, lightkurve, pandas, numpy, astropy
# Assumes detrended light curve files exist (e.g., from App 68.A)
import dask.bag as db
import dask # For config
from dask.diagnostics import ProgressBar
import lightkurve as lk
import numpy as np
import os
import glob
import time
import pandas as pd
import shutil

print("Parallel BLS Transit Search using Dask Bag:")

# --- Step 1: Prepare File List and Worker Function ---
# Assume flattened LC files exist in 'dask_detrended_lcs/'
input_dir_bls = "dask_detrended_lcs"
n_lcs_bls = 32 # Number of files to create/use

# Create dummy flattened files for demonstration
os.makedirs(input_dir_bls, exist_ok=True)
print(f"\nCreating {n_lcs_bls} dummy flattened LC files in '{input_dir_bls}'...")
from astropy.table import Table
from astropy.io import fits
for i in range(n_lcs_bls):
    fname = os.path.join(input_dir_bls, f"tic123_s{i}_flat.fits")
    time_arr = np.arange(0, 27, 0.002)
    flux_arr = np.random.normal(1.0, 0.001, len(time_arr)) # Flat noise
    # Add transit? Maybe to a few?
    if i % 6 == 0: 
        period = np.random.uniform(1,10); t0 = np.random.uniform(0, period)
        phase = ((time_arr - t0 + period/2) % period) - period/2
        duration_phase = 0.05; in_transit = np.abs(phase) < duration_phase / 2
        flux_arr[in_transit] -= 0.005 # Add transit
    lc_tab = Table({'TIME': time_arr, 'FLUX': flux_arr})
    # Need primary HDU for lk.read
    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.BinTableHDU(lc_tab)
    hdulist = fits.HDUList([hdu0, hdu1])
    hdulist.writeto(fname, overwrite=True)

lightcurve_files = glob.glob(os.path.join(input_dir_bls, "*_flat.fits"))
print(f"Found {len(lightcurve_files)} input light curve files.")

# Define worker function for BLS
def run_bls_on_lc_file(filename):
    """Loads LC file, runs BLS, returns best result dictionary."""
    import lightkurve as lk
    import numpy as np
    import os
    basename = os.path.basename(filename)
    try:
        lc = lk.read(filename)
        # Ensure using TIME and FLUX columns if they exist
        if 'TIME' in lc.colnames and 'FLUX' in lc.colnames:
             lc = lk.LightCurve(time=lc['TIME'], flux=lc['FLUX'])
        else: # Fallback if columns named differently
             lc = lc # Assume first two columns are time/flux? Risky. Needs check.
             
        lc = lc.remove_nans()
        if len(lc.flux) < 50: # Need enough points for BLS
            raise ValueError("Too few points after cleaning.")
            
        # Run BLS (adjust duration/period as needed)
        bls = lc.to_periodogram(method='bls', 
                               duration=[0.05, 0.1, 0.15, 0.2]*u.day, 
                               frequency_factor=5) # Denser frequency grid
                               
        best_period = bls.period_at_max_power.value
        max_power = bls.max_power.value
        duration = bls.duration_at_max_power.value
        depth = bls.depth_at_max_power
        transit_time = bls.transit_time_at_max_power.value

        # print(f"  BLS Finished: {basename} P={best_period:.3f}") # Debug
        return {'filename': basename, 'bls_period': best_period, 
                'bls_power': max_power, 'bls_duration': duration, 
                'bls_depth': depth, 'bls_t0': transit_time, 'error': None}
                
    except Exception as e:
        # print(f"  BLS Error: {basename}: {e}") # Debug
        return {'filename': basename, 'bls_period': np.nan, 'bls_power': np.nan, 
                'bls_duration': np.nan, 'bls_depth': np.nan, 'bls_t0': np.nan, 'error': str(e)}

# --- Step 2 (Option A - Dask Bag) ---
print("\nCreating Dask Bag...")
n_cores = os.cpu_count()
# Create bag from file list, partitioning for parallelism
b = db.from_sequence(lightcurve_files, npartitions=n_cores * 2)
print(f"Dask Bag created with {b.npartitions} partitions.")

# Step 3: Apply Map Operation (Lazy)
print("Mapping BLS function (lazy)...")
results_bag = b.map(run_bls_on_lc_file)

# Step 4: Configure Scheduler and Execute
# Use processes for potentially CPU-bound BLS and lightkurve loading
dask.config.set(scheduler='processes', num_workers=n_cores) 
print(f"\nExecuting BLS computation using '{dask.config.get('scheduler')}' scheduler...")
start_time = time.time()
with ProgressBar():
    all_results_list = results_bag.compute()
end_time = time.time()
print(f"\nComputation finished. Time taken: {end_time - start_time:.3f} seconds.")

# Step 5: Process Results
if all_results_list:
    print(f"Processed {len(all_results_list)} light curves.")
    # Filter out potential errors before creating DataFrame
    valid_results = [r for r in all_results_list if r and r.get('error') is None]
    failed_results = [r for r in all_results_list if r and r.get('error') is not None]
    print(f"  ({len(valid_results)} successful, {len(failed_results)} failed)")
    
    if valid_results:
         results_df = pd.DataFrame(valid_results)
         print("\n--- Sample BLS Results (DataFrame) ---")
         print(results_df.head())
         
         # Find candidates (example: high power)
         bls_power_threshold = 15 # Example threshold
         candidates = results_df[results_df['bls_power'] > bls_power_threshold].sort_values('bls_power', ascending=False)
         print(f"\nFound {len(candidates)} potential candidates (BLS Power > {bls_power_threshold}):")
         if not candidates.empty: 
              print(candidates[['filename', 'bls_period', 'bls_power', 'bls_depth']])
    else:
         print("\nNo successful BLS results obtained.")
else:
    print("No results were computed.")

# Cleanup
finally:
     if os.path.exists(input_dir_bls): 
          shutil.rmtree(input_dir_bls)
          print(f"\nCleaned up '{input_dir_bls}'.")

print("-" * 20)
```

**Chapter 68 Summary**

This chapter focused on the practical implementation of automated and scalable astrophysical workflows, using TESS light curve analysis with the **`lightkurve`** library as a primary example. It began by recapping the common analysis steps involved in processing TESS data from Target Pixel Files (TPFs) to analysis-ready light curves: data search/download, aperture photometry, background subtraction, quality flagging/outlier removal, systematic error correction (detrending/flattening using methods like Savitzky-Golay, PLD, or CBVs), and period searching (Lomb-Scargle or BLS for transits). The importance of **encapsulating these distinct stages into modular Python functions or command-line scripts** with clear inputs and outputs was emphasized as a prerequisite for building robust workflows.

The chapter then demonstrated how to define and manage these encapsulated steps using two different approaches. First, using a dedicated **Workflow Management System (WMS)** like **Snakemake**, where the workflow is defined declaratively in a `Snakefile` using **rules** that specify input/output file patterns (often with **wildcards**) and the shell commands or scripts needed to generate outputs from inputs. Snakemake automatically infers dependencies and manages parallel execution based on these rules. Second, using the **Dask** library for **task parallelism** directly within Python. Both **Dask Bag** (using `.map()` on sequences of input items like filenames) and **Dask Delayed** (explicitly wrapping function calls and passing `Delayed` objects to define dependencies) were shown as effective methods for distributing the independent processing of many light curves across multiple cores or cluster nodes using Dask's schedulers. Practical considerations for both WMS and Dask approaches regarding **handling dependencies** (implicit vs. explicit) and managing **intermediate files** (or in-memory data flow for Dask) were discussed. Two detailed applications showed implementing a TESS download-extract-detrend pipeline using Snakemake rules and a parallel BLS transit search on multiple light curves using Dask Bag.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Lightkurve Collaboration. (2018).** Lightkurve: Kepler and TESS time series analysis in Python. *Astrophysics Source Code Library*, record ascl:1812.011. ([Link via ADS](https://ui.adsabs.harvard.edu/abs/2018ascl.soft12011L/abstract)) (See also Documentation: [https://docs.lightkurve.org/](https://docs.lightkurve.org/))
    *(Paper describing the Lightkurve package. The linked documentation is essential for understanding the functions used for TESS data download, photometry, flattening, and periodogram analysis in Sec 68.1 and the applications.)*

2.  **Köster, J., & Rahmann, S. (2012).** Snakemake—a scalable bioinformatics workflow engine. *(See reference in Chapter 66)*. (Documentation: [https://snakemake.readthedocs.io/en/stable/](https://snakemake.readthedocs.io/en/stable/))
    *(Introduces Snakemake, used in Sec 68.3 and Application 68.A.)*

3.  **Babuji, Y., et al. (2019).** Parsl: Enabling Scalable Interactive Computing in Python. *(See reference in Chapter 66)*. (Documentation: [https://parsl-project.org/](https://parsl-project.org/))
    *(Describes Parsl, mentioned as an alternative WMS approach in Sec 68.3.)*

4.  **Dask Development Team. (n.d.).** *Dask Documentation*. Dask. Retrieved January 16, 2024, from [https://docs.dask.org/en/latest/](https://docs.dask.org/en/latest/) (Specifically Bag: [https://docs.dask.org/en/latest/bag.html](https://docs.dask.org/en/latest/bag.html) and Delayed: [https://docs.dask.org/en/latest/delayed.html](https://docs.dask.org/en/latest/delayed.html))
    *(Official Dask documentation covering Dask Bag and Delayed used in Sec 68.4 and Application 68.B.)*

5.  **Ricker, G. R., Winn, J. N., Vanderspek, R., et al. (2015).** Transiting Exoplanet Survey Satellite (TESS). *Journal of Astronomical Telescopes, Instruments, and Systems*, *1*(1), 014003. [https://doi.org/10.1117/1.JATIS.1.1.014003](https://doi.org/10.1117/1.JATIS.1.1.014003)
    *(Provides background on the TESS mission, its data products (TPFs, light curves), and scientific goals relevant to the workflows discussed in this chapter.)*
