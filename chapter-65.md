**Chapter 65: Workflow Management with Python Scripting and Libraries**

While dedicated Workflow Management Systems (WMS) offer powerful features (Chapter 66), many moderately complex astronomical workflows can be effectively managed and automated directly using standard **Python scripting** combined with core libraries. This chapter explores practical techniques for structuring multi-step analysis pipelines within Python scripts, focusing on maintainability, parameterization, error handling, and basic dependency management without relying on external WMS frameworks. We begin by discussing how to **structure workflows using functions**, breaking down the analysis into modular, reusable Python functions for each logical step. Strategies for **passing data between steps** are examined, comparing in-memory transfers (via function return values) with the more common approach of saving intermediate results to files. Basic approaches to **dependency management** within scripts (ensuring steps run in the correct order, potentially checking for file existence) are outlined. We then cover methods for **parameterizing workflows**, making them flexible and configurable without hardcoding values, using **configuration files** (in formats like YAML or JSON) parsed by Python libraries and handling command-line arguments using the `argparse` module. Essential practices for **logging** workflow progress and status, and implementing robust **error handling** using `try...except` blocks, are demonstrated using Python's `logging` module. Finally, common file and directory manipulation tasks needed within workflows are showcased using the `os`, `pathlib`, and `glob` modules.

**65.1 Structuring Workflows with Functions**

The foundation of managing any non-trivial computational task in Python lies in breaking it down into smaller, well-defined **functions** (Appendix A.I.2). Applying this principle to a multi-step analysis workflow is crucial for creating code that is organized, readable, reusable, and testable. Instead of writing one long, monolithic script that performs every step sequentially, structure the workflow by encapsulating each logical stage (e.g., data download, calibration, measurement, plotting) within its own dedicated function.

Each function should ideally perform a single, well-defined task related to the workflow stage. It should have clear **inputs** (parameters passed as arguments, potentially including file paths or data loaded previously) and well-defined **outputs** (typically returned values, which could be data structures like NumPy arrays or Pandas DataFrames, or status codes indicating success/failure). Crucially, functions should also aim to be **self-contained** where possible, minimizing reliance on global variables, which makes them easier to test and reuse independently.

For example, a simple image calibration workflow might be structured with functions like:
*   `load_raw_image(filename)`: Loads a raw FITS image.
*   `create_master_bias(bias_filenames)`: Combines bias frames.
*   `create_master_flat(flat_filenames, master_bias)`: Creates a master flat.
*   `calibrate_image(raw_image_data, master_bias, master_flat)`: Performs the calibration steps.
*   `save_calibrated_image(calibrated_data, header, output_filename)`: Saves the result.
*   `run_calibration_workflow(science_filename, bias_files, flat_files, output_dir)`: The main function that orchestrates calls to the other functions in the correct sequence.

This functional decomposition offers several advantages over a single large script:
*   **Readability:** The main workflow logic (in `run_calibration_workflow`) becomes much clearer, reading like a sequence of high-level steps rather than a tangled mess of low-level operations.
*   **Modularity:** Each function handles a specific task. If you need to change how bias frames are combined, you only need to modify the `create_master_bias` function, minimizing the risk of breaking other parts of the workflow.
*   **Reusability:** Individual functions (like `load_raw_image` or `save_calibrated_image`) can potentially be reused in other analysis scripts or projects.
*   **Testability:** Each function can be tested independently (unit testing, Sec A.III.4) with known inputs and outputs, making it much easier to verify correctness and isolate bugs compared to testing a single monolithic script.

Adopting a functional approach forces you to think clearly about the inputs, outputs, and specific action of each step in your workflow. This planning process itself often leads to better designed and more robust analysis pipelines. Functions act as the fundamental building blocks for creating structured and manageable computational workflows directly within Python scripts. While dedicated WMS tools (Chapter 66) provide more advanced features, organizing your analysis into well-defined functions is the essential first step regardless of how the overall workflow execution is managed.

Good function design emphasizes clear naming conventions that reflect the function's purpose. Docstrings (Appendix A.I.2) are essential for explaining what each function does, its parameters (including expected types and units), what it returns, and any important assumptions or potential side effects. Type hinting (e.g., `def calibrate_image(raw_image_data: np.ndarray) -> np.ndarray:`) can further improve clarity and allow for static analysis checks.

When breaking down a workflow, consider the logical stages of your analysis. Calibration steps often form natural function boundaries. Data reduction tasks like source finding or photometry can be encapsulated. Analysis steps like model fitting or statistical calculations are also good candidates for separate functions. The main script then becomes an orchestrator, calling these functions in the appropriate order, potentially passing data between them (Sec 65.2).

Aim for functions that are not excessively long and perform a coherent operation. If a function becomes too complex or handles too many distinct tasks, consider breaking it down further into smaller helper functions. This hierarchical decomposition enhances clarity and maintainability.

Remember that Python modules allow you to group related functions together (Appendix A.I.3). For more complex workflows, you might organize your functions into different modules within a larger Python package (Appendix III), such as `calibration_routines.py`, `photometry_tools.py`, `analysis_models.py`, further improving organization and reusability across different projects. Structuring workflows with functions is the cornerstone of writing clean, maintainable, and testable scientific analysis code in Python.

**65.2 Passing Data Between Steps (In-Memory vs. Files)**

In a multi-step workflow structured using functions (Sec 65.1), data needs to be passed from the output of one step (function) to the input of the next. There are two primary strategies for achieving this within a Python script: passing data **in-memory** via function return values and arguments, or passing data **via files** saved to disk. The choice between these significantly impacts performance, memory usage, resilience, and the ability to inspect intermediate results.

**In-Memory Data Transfer:** This approach involves having each function return its result (e.g., a NumPy array, Pandas DataFrame, Astropy Table, or custom object) directly. The calling function (e.g., the main workflow orchestrator) captures this return value in a variable and passes this variable as an argument to the next function in the sequence. Data resides entirely in the computer's RAM as it flows between the workflow steps defined by functions.

```python
# --- Code Example: In-Memory Data Passing ---
import numpy as np
print("Conceptual In-Memory Data Passing:")

def step1_load_data(dummy_input):
    print("  Running Step 1: Load Data")
    # Simulate loading data
    data = np.random.rand(100, 100) 
    print("  Step 1 finished, returning data array.")
    return data

def step2_process_data(input_data):
    print("  Running Step 2: Process Data")
    # Simulate processing
    processed_data = np.log10(input_data + 1e-6) * 10
    print("  Step 2 finished, returning processed data.")
    return processed_data

def step3_analyze_data(processed_data):
    print("  Running Step 3: Analyze Data")
    # Simulate analysis
    mean_val = np.mean(processed_data)
    std_val = np.std(processed_data)
    print("  Step 3 finished, returning statistics.")
    return {'mean': mean_val, 'std': std_val}

# --- Workflow Orchestration ---
print("\nStarting Workflow...")
# Output of step1 becomes input of step2
intermediate_data = step1_load_data("input_specifier") 
# Output of step2 becomes input of step3
final_results_mem = step2_process_data(intermediate_data) # Renamed to avoid clash
# Final analysis based on step3 output
final_stats_mem = step3_analyze_data(final_results_mem) # Renamed

print("\nWorkflow Finished.")
print(f"Final Statistics: {final_stats_mem}")
print("-" * 20)

# Explanation: 
# - `step1_load_data` returns a NumPy array (`data`).
# - This array (`intermediate_data`) is passed directly as an argument to `step2_process_data`.
# - `step2_process_data` returns the `processed_data` array.
# - This array (`final_results_mem`) is passed directly to `step3_analyze_data`.
# - Data flows between functions via return values and arguments in memory.
```

The main advantage of in-memory transfer is **speed**. It avoids the significant overhead associated with writing data to disk and then reading it back in the next step. Disk I/O operations are typically orders of magnitude slower than accessing data already in RAM. For workflows processing small to moderately sized datasets that comfortably fit within the available system memory, this approach is often the most efficient and conceptually simplest for linear pipelines. It also keeps intermediate results directly accessible as Python objects if needed by multiple subsequent steps without reloading.

However, the major drawback is **memory limitation**. If any intermediate dataset becomes larger than the available RAM, the program will likely slow down drastically due to memory swapping or crash with an `MemoryError`. This makes the pure in-memory approach unsuitable for workflows processing very large images, data cubes, catalogs, or simulation snapshots common in modern astrophysics. Another significant disadvantage is the **lack of persistence**. If the script crashes or is interrupted during execution (e.g., due to a bug in a later step, power outage, or killed job), all intermediate results computed and held in memory are lost, forcing the entire workflow to be restarted from the beginning. This lack of resilience is problematic for long-running analyses.

**File-Based Data Transfer:** The alternative strategy is for each major workflow step (function) to write its primary output(s) to one or more files on disk. Subsequent steps then take the *filename(s)* of these intermediate files as input, read the necessary data from disk, perform their processing, and write their own output file(s). The workflow state is effectively stored on disk between steps.

```python
# --- Code Example: File-Based Data Passing (Conceptual) ---
import numpy as np
import os
import pickle # For saving/loading python objects (or use JSON, HDF5 etc)
import shutil # For cleanup

print("\nConceptual File-Based Data Passing:")
output_dir = "temp_workflow_files"
os.makedirs(output_dir, exist_ok=True)

def step1_load_save(dummy_input, output_file):
    print("  Running Step 1: Load Data and Save")
    data = np.random.rand(100, 100) 
    # Save intermediate result to file (using pickle here)
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Step 1 finished, saved data to {output_file}")
    return output_file # Return filename

def step2_process_save(input_file, output_file):
    print("  Running Step 2: Load Intermediate, Process, Save")
    # Load data from previous step's output file
    with open(input_file, 'rb') as f:
        input_data = pickle.load(f)
    # Process data
    processed_data = np.log10(input_data + 1e-6) * 10
    # Save this step's output
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    print(f"  Step 2 finished, saved processed data to {output_file}")
    return output_file

def step3_analyze_save(input_file, output_file):
    print("  Running Step 3: Load Processed, Analyze, Save")
    with open(input_file, 'rb') as f:
        processed_data = pickle.load(f)
    # Analyze
    mean_val = np.mean(processed_data)
    std_val = np.std(processed_data)
    final_stats = {'mean': mean_val, 'std': std_val}
    # Save final result
    with open(output_file, 'w') as f:
        f.write(f"Mean: {mean_val}\\nStd Dev: {std_val}\\n")
    print(f"  Step 3 finished, saved stats to {output_file}")
    return output_file

# --- Workflow Orchestration ---
print("\nStarting Workflow...")
file1 = os.path.join(output_dir, "step1_output.pkl")
file2 = os.path.join(output_dir, "step2_output.pkl")
file3 = os.path.join(output_dir, "step3_output.txt")

# Pass filenames between steps
out1_fname = step1_load_save("input_specifier", file1)
out2_fname = step2_process_save(out1_fname, file2)
out3_fname = step3_analyze_save(out2_fname, file3)

print("\nWorkflow Finished.")
print(f"Final results saved in {out3_fname}")

# Cleanup
if os.path.exists(output_dir): 
    try:
        shutil.rmtree(output_dir)
        print(f"Cleaned up {output_dir}.")
    except Exception as e_clean:
        print(f"Error during cleanup: {e_clean}")
print("-" * 20)

# Explanation:
# - Each function now takes input/output filenames as arguments.
# - `step1` loads data and saves it to `step1_output.pkl`.
# - `step2` takes `step1_output.pkl` as input, loads data, processes, saves to `step2_output.pkl`.
# - `step3` takes `step2_output.pkl` as input, loads, analyzes, saves stats to `step3_output.txt`.
# - Data flows via intermediate files on disk. Each function returns the output filename.
```

The primary advantage of file-based passing is its ability to handle **datasets much larger than RAM**, as only the data needed for the current processing step needs to be loaded. It also provides **persistence and resilience**. If the workflow crashes, the intermediate files from successfully completed steps remain on disk. With appropriate logic added to the workflow script (e.g., checking if an output file already exists before running the step that creates it), the workflow can often be **restarted**, skipping the already completed steps and resuming from the point of failure. This is crucial for long-running pipelines. Furthermore, intermediate files can be easily inspected, reused for different downstream analyses, or shared with collaborators. It also naturally facilitates workflows involving steps implemented in different programming languages or using external command-line tools.

The main disadvantage is the **performance overhead of disk I/O**. Writing and reading intermediate files, especially large ones, can significantly slow down the overall workflow execution compared to in-memory operations. This overhead can be particularly severe on slow file systems or if the workflow involves many steps generating large intermediate files. Choosing efficient file formats (like compressed FITS or HDF5 instead of plain text or pickle for large arrays) and using high-performance file systems (like cluster scratch spaces, Sec 42.2) can help mitigate this, but I/O remains a potential bottleneck. Careful management of filenames and directory structures is also required.

In practice, many scientific workflows employ a **hybrid approach**. Data might be passed in memory between closely related sub-steps within a major stage, but significant intermediate products (like calibrated images, source lists, power spectra) are written to disk before proceeding to the next major analysis stage. This balances efficiency for tightly coupled operations with resilience and scalability for the overall pipeline. Workflow Management Systems (Chapter 66) almost exclusively rely on file-based dependencies to manage task execution and restarts.

**65.3 Basic Dependency Management**

In a workflow composed of multiple functional steps (Sec 65.1), ensuring that steps are executed in the correct order based on their **dependencies** is crucial. Task B cannot start until its required input, produced by Task A, is available. While complex workflows benefit greatly from dedicated Workflow Management Systems (WMS) that automatically track file-based dependencies (Chapter 66), basic dependency management can be implemented directly within Python scripts for simpler pipelines, primarily through controlling the function call sequence and potentially adding checks for file existence.

The most straightforward way to manage dependencies in a linear workflow within a single script is simply through the **order of function calls**. If `step2_process_data` requires the output of `step1_load_data`, you ensure this by calling `step1` first, capturing its return value (if passing in memory) or output filename (if passing via files), and then calling `step2` with that result or filename as input, as illustrated in the examples in Sec 65.2. The sequential execution flow of the Python script naturally enforces the dependency order for a simple chain of tasks.

This works well for linear workflows but becomes more complex if the workflow has branches or merges (a more complex DAG structure). For example, if Task C depends on outputs from both Task A and Task B (which might run independently or in parallel), the script logic must ensure both A and B complete before C starts. This might involve running A and B (perhaps using `multiprocessing` if they are independent) and then explicitly waiting for both to finish before calling C with their combined outputs.

When passing data via intermediate files, a common technique to add resilience and allow restarting is to **check for the existence of output files** before running the task that creates them. Before calling the function for Step B, the script checks if Step B's expected output file already exists. If it does, the script can assume Step B was completed successfully in a previous run and skip executing it again, directly proceeding to Step C (which takes Step B's output file as input). This provides basic **incremental execution** and allows restarting a failed workflow without rerunning already completed parts.

```python
# --- Code Example: Checking File Existence for Basic Dependency/Restart ---
import os
import time
import numpy as np
# Assume functions step1_save, step2_save, step3_save exist 
# similar to file-based example in Sec 65.2, each saving to its output_file

print("Workflow with Basic File Existence Check:")

# Define expected output filenames
output_dir = "temp_workflow_files_dep"
os.makedirs(output_dir, exist_ok=True)
file1 = os.path.join(output_dir, "step1.npy") # Using numpy save format
file2 = os.path.join(output_dir, "step2.npy")
file3 = os.path.join(output_dir, "step3.txt")

# Dummy worker functions that save output
def step1_save(outfile): 
    print("  Running Step 1...")
    time.sleep(0.5); data=np.arange(10); np.save(outfile, data)
    print(f"  Step 1 Saved {outfile}")
def step2_save(infile, outfile): 
    print("  Running Step 2...")
    time.sleep(0.5); data=np.load(infile); res=data*2; np.save(outfile, res)
    print(f"  Step 2 Saved {outfile}")
def step3_save(infile, outfile): 
    print("  Running Step 3...")
    time.sleep(0.5); data=np.load(infile); res=np.sum(data)
    with open(outfile,'w') as f: f.write(str(res))
    print(f"  Step 3 Saved {outfile}")

# --- Workflow Execution with Checks ---
print("\nStarting Workflow Execution...")

# Step 1
if not os.path.exists(file1):
    step1_save(file1)
else:
    print(f"Skipping Step 1, output file '{file1}' already exists.")

# Step 2 (Depends on Step 1 output)
if os.path.exists(file1): # Check if input exists
    if not os.path.exists(file2):
        step2_save(file1, file2)
    else:
        print(f"Skipping Step 2, output file '{file2}' already exists.")
else:
    print("Cannot run Step 2, required input file from Step 1 missing.")

# Step 3 (Depends on Step 2 output)
if os.path.exists(file2): # Check if input exists
    if not os.path.exists(file3):
        step3_save(file2, file3)
    else:
        print(f"Skipping Step 3, output file '{file3}' already exists.")
else:
    print("Cannot run Step 3, required input file from Step 2 missing.")

print("\nWorkflow Execution Attempt Finished.")
# Verify final file exists
if os.path.exists(file3): print(f"Final output '{file3}' exists.")

# Cleanup
if os.path.exists(output_dir): shutil.rmtree(output_dir)
print(f"Cleaned up {output_dir}.")
print("-" * 20)

# Explanation:
# 1. Defines dummy functions that simulate work and save output to specified files.
# 2. Before running each step (step1_save, step2_save, step3_save), it checks if 
#    the expected *output* file for that step already exists using `os.path.exists()`.
# 3. If the output file exists, the step is skipped, assuming it completed successfully 
#    in a previous run.
# 4. Steps 2 and 3 also check if their required *input* file exists before attempting 
#    to run, preventing errors if an earlier step failed.
# Running this script once creates the files. Running it a second time will show 
# all steps being skipped because their output files exist. Deleting an intermediate 
# file (e.g., `step2.npy`) and re-running will cause only steps 2 and 3 to execute.
# This demonstrates basic dependency handling and restart capability using file checks.
```

This manual file-checking approach works for relatively simple, linear, or moderately branched workflows managed within a single script. However, it has drawbacks:
*   **Manual Implementation:** The dependency logic (checking inputs exist, skipping if outputs exist) must be explicitly coded by the user for every step, which can become complex and repetitive for large workflows.
*   **Timestamp Issues:** Relying only on file existence doesn't handle cases where an input file might have changed *after* the output file was created, necessitating a rerun. More robust checks might involve comparing file modification timestamps (using `os.path.getmtime`).
*   **Complexity for DAGs:** Manually managing the execution order and checks for complex DAGs with multiple branches and merges becomes very difficult and error-prone within a simple script.
*   **Lack of Parallelism:** This approach doesn't inherently provide automatic parallel execution of independent tasks.

These limitations are precisely what dedicated Workflow Management Systems (like Snakemake, Nextflow, Parsl) are designed to overcome. They provide specialized syntax (Rules or Process definitions) where you declare inputs and outputs for each task, and the WMS *automatically* determines the execution order, checks timestamps or checksums to decide if steps need rerunning, and manages parallel execution of independent tasks, abstracting away the complexity of manual dependency checking and scheduling logic. However, for simpler workflows run primarily as single Python scripts, careful function structuring combined with basic file existence checks can provide a workable level of dependency management and resilience.

**65.4 Parameterization: Config Files and `argparse`**

Hardcoding values like input filenames, output directories, processing parameters (e.g., filter names, detection thresholds, model parameters), or resource settings directly within a scientific analysis script severely limits its reusability and flexibility. Every time you want to run the analysis on a different dataset or with slightly different settings, you would need to manually edit the script code, which is inefficient and increases the risk of introducing errors. **Parameterization** is the practice of separating the core analysis logic (the code) from the specific configuration settings (the parameters) used for a particular run. This makes scripts much more versatile and easier to manage. Two common methods for parameterizing Python scripts are using configuration files and command-line arguments.

**Configuration Files (YAML, JSON, INI):** For workflows involving numerous parameters, storing them in an external **configuration file** is often the cleanest approach. This file contains key-value pairs defining all the parameters needed by the script. Popular formats include:
*   **YAML (`.yaml` or `.yml`):** Human-readable format using indentation to denote structure. Supports lists, dictionaries, scalars. Widely used for configuration due to its clarity. Requires the `PyYAML` library (`pip install pyyaml`).
*   **JSON (`.json`):** JavaScript Object Notation. Stricter syntax than YAML (e.g., requires double quotes for strings), less human-friendly for complex structures but easily parsed by Python's built-in `json` module.
*   **INI (`.ini` or `.cfg`):** Simple format using sections `[SectionName]` followed by `key = value` pairs. Handled by Python's built-in `configparser` module. Good for simple configurations but less flexible for nested structures or lists.

The Python script then reads this configuration file at runtime (e.g., using `yaml.safe_load()` for YAML) into a dictionary or object. Values from this configuration dictionary are then used throughout the script instead of hardcoded constants or filenames. To run the analysis with different settings, the user simply modifies the configuration file without touching the Python code.

```python
# --- Code Example 1: Using YAML Config File ---
# Requires PyYAML: pip install pyyaml
import yaml
import os

print("Using YAML Configuration File:")

# --- Create a sample config file ---
config_filename = 'analysis_config.yaml'
config_content = """
# Configuration for astrophysical analysis
input:
  catalog_file: '/data/archive/galaxy_cat_z1.hdf5'
  calibration_dir: '/data/calib_files/'
  bad_pixel_mask: 'masks/bpm_ccd1.fits'

processing:
  photometry_aperture_radius: 5.0 # pixels
  detection_threshold_sigma: 3.5
  use_psf_fitting: False
  filters_to_process: ['g', 'r', 'i']

output:
  output_catalog: 'results/processed_catalog_z1.csv'
  log_file: 'logs/analysis_run.log'
  verbosity: 1 
"""
print(f"\nWriting sample config to: {config_filename}")
with open(config_filename, 'w') as f:
    f.write(config_content)

# --- Python script reads the config ---
print(f"\nReading configuration from {config_filename}...")
config = None
try:
    with open(config_filename, 'r') as f:
        config = yaml.safe_load(f) # Load YAML into Python dictionary
    print("Config loaded successfully.")
    
    # Access parameters using dictionary keys
    catalog_path = config.get('input', {}).get('catalog_file', 'default_catalog.hdf5')
    threshold = config.get('processing', {}).get('detection_threshold_sigma', 5.0)
    filters = config.get('processing', {}).get('filters_to_process', ['g'])
    output_file = config.get('output', {}).get('output_catalog', 'default_output.csv')
    
    print("\nParameters loaded from config:")
    print(f"  Catalog: {catalog_path}")
    print(f"  Threshold: {threshold}")
    print(f"  Filters: {filters}")
    print(f"  Output: {output_file}")
    
    # ... rest of the script would use these config values ...
    
except FileNotFoundError:
     print(f"Error: Config file '{config_filename}' not found.")
except yaml.YAMLError as e:
     print(f"Error parsing YAML file '{config_filename}': {e}")
except Exception as e:
     print(f"An error occurred: {e}")

# Cleanup
if os.path.exists(config_filename): os.remove(config_filename)
print(f"\nCleaned up {config_filename}.")
print("-" * 20)

# Explanation:
# 1. Creates a sample YAML file `analysis_config.yaml` with nested structure 
#    (input, processing, output sections) storing various parameters.
# 2. Opens and reads the YAML file using `yaml.safe_load()` which parses it into 
#    a standard Python dictionary `config`.
# 3. Demonstrates accessing nested parameter values from the dictionary using 
#    `.get()` methods (providing default values is good practice).
# 4. The rest of the script would then use these variables (`catalog_path`, 
#    `threshold`, etc.) instead of hardcoded values.
```

**Command-Line Arguments (`argparse`):** For parameters that change frequently between runs (like input/output filenames, specific object IDs, or key numerical parameters like redshift or exposure time) or for controlling script behavior (like verbosity level, plotting options), using **command-line arguments** is often more convenient than editing a configuration file each time. Python's built-in `argparse` module is the standard tool for defining and parsing command-line arguments robustly.

The workflow with `argparse` involves:
1.  Import `argparse`.
2.  Create an `ArgumentParser` object, optionally providing a description.
3.  Use the `.add_argument()` method to define each expected command-line argument:
    *   Specify the argument name (e.g., `'input_file'`, or short/long flags like `'-i'`, `'--input'`).
    *   Specify `type` (e.g., `str`, `int`, `float`).
    *   Specify `help` text displayed when the user runs the script with `-h` or `--help`.
    *   Specify if required (`required=True`) or optional (providing a `default` value makes it optional).
    *   Specify actions like `'store_true'` or `'store_false'` for boolean flags.
4.  Parse the arguments provided by the user on the command line using `args = parser.parse_args()`. This returns an object (`args`) whose attributes hold the values passed by the user (e.g., `args.input_file`, `args.threshold`).
5.  Use these `args` attributes throughout the script.

```python
# --- Code Example 2: Using argparse for Command-Line Args ---
# (Save as e.g., process_image_cli.py and run from terminal)
import argparse
import time

print("Using argparse for Command-Line Arguments:")

def process_image(filename, threshold, smooth=False, output_prefix="proc"):
    """Dummy function simulating image processing."""
    print(f"\n--- Processing Image ---")
    print(f"  Input File: {filename}")
    print(f"  Threshold: {threshold}")
    print(f"  Smoothing Enabled: {smooth}")
    print(f"  Output Prefix: {output_prefix}")
    # Simulate work
    time.sleep(0.5) 
    print("Processing complete.")
    print(f"  (Results would be saved using prefix '{output_prefix}')")
    return True

# --- Main Execution Block ---
if __name__ == "__main__":
    
    # 1. Create ArgumentParser
    parser = argparse.ArgumentParser(description="Process an astronomical image file.")
    
    # 2. Add Arguments
    parser.add_argument("input_file", type=str, 
                        help="Path to the input FITS image file.")
    parser.add_argument("-t", "--threshold", type=float, required=True, 
                        help="Detection threshold value (e.g., sigma).")
    parser.add_argument("-s", "--smooth", action="store_true", default=False,
                        help="Enable Gaussian smoothing before processing.")
    parser.add_argument("-o", "--output_prefix", type=str, default="processed_image",
                        help="Prefix for output files (default: processed_image).")
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Enable verbose output.")

    # 3. Parse Arguments from command line
    args = parser.parse_args()

    # 4. Use Parsed Arguments in the script
    if args.verbose:
        print("\n*** Verbose Mode Enabled ***")
        print("Arguments received:")
        print(f"  Input File: {args.input_file}")
        print(f"  Threshold: {args.threshold}")
        print(f"  Smooth: {args.smooth}")
        print(f"  Output Prefix: {args.output_prefix}")
        print(f"  Verbose: {args.verbose}")
        
    # Call the main processing function using parsed args
    success = process_image(
        filename=args.input_file, 
        threshold=args.threshold, 
        smooth=args.smooth,
        output_prefix=args.output_prefix
    )
    
    if success: print("\nScript finished successfully.")
    else: print("\nScript encountered an error.")

# --- How to Run from Terminal ---
# python process_image_cli.py my_image.fits -t 5.0 
# python process_image_cli.py data/ngc1234.fits --threshold 3.0 --smooth -o ngc1234_proc -v
# python process_image_cli.py --help # Shows help message
```
```python
# --- Block to display conceptual argparse usage ---
print("\n--- Conceptual Argparse Script (`process_image_cli.py`) ---")
print("# (Content shown in previous code block)")
print("\n--- Conceptual Command Line Execution Examples ---")
print("# python process_image_cli.py my_image.fits -t 5.0")
print("# python process_image_cli.py data/image.fits --threshold 3.0 --smooth --output_prefix run01 --verbose")
print("# python process_image_cli.py -h")
print("-" * 20)

# Explanation:
# 1. Imports `argparse`.
# 2. Creates an `ArgumentParser`.
# 3. Uses `parser.add_argument()` to define arguments:
#    - `input_file`: A required positional argument (string).
#    - `-t` or `--threshold`: A required option (`required=True`) taking a float value.
#    - `-s` or `--smooth`: An optional flag (`action='store_true'`). If present, `args.smooth` is True.
#    - `-o` or `--output_prefix`: An optional option taking a string, with a default value.
#    - `-v` or `--verbose`: An optional flag.
# 4. `parser.parse_args()` reads arguments from `sys.argv` (command line).
# 5. The script then uses `args.input_file`, `args.threshold`, `args.smooth`, etc. 
#    to control its behavior and pass values to the `process_image` function.
# Running with `-h` automatically generates and displays help text based on the `help=` strings.
```

Often, `argparse` and configuration files are used together. `argparse` might be used to specify the *location* of the main configuration file (`--config my_run_params.yaml`), allowing easy switching between different sets of detailed parameters stored in YAML/JSON files.

Parameterizing workflows using configuration files and/or command-line arguments is essential for creating flexible, reusable, and reproducible analysis scripts. It cleanly separates the core logic from run-specific settings, making it easy to adapt the workflow to different datasets or analysis variations without modifying the underlying Python code.

**65.5 Logging and Error Handling**

Robust scientific workflows need mechanisms for tracking their progress and handling unexpected errors gracefully. Simply letting a script crash on error without informative output makes debugging difficult, and relying solely on `print()` statements for status updates can become cluttered and hard to manage, especially for complex or long-running processes. Python's built-in `logging` module provides a flexible framework for recording status messages, warnings, and errors, while `try...except` blocks are essential for structured error handling.

**Logging (`logging` module):** The `logging` module allows you to record messages about your script's execution to different destinations (like the console or a file) with varying levels of severity (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL). Compared to using `print()`, logging offers:
*   **Severity Levels:** Control which messages are displayed based on their importance. You can run normally showing only INFO and above, but switch to DEBUG level for detailed troubleshooting.
*   **Configurable Output:** Easily redirect log messages to files instead of (or in addition to) the console.
*   **Formatting:** Control the format of log messages (e.g., include timestamps, module names, severity levels).
*   **Modular Usage:** Different parts of your code (e.g., different functions or modules) can obtain their own logger instances, allowing fine-grained control over logging output.

Basic setup involves importing `logging`, configuring the basic logging behavior (level, output destination, format) usually once at the start of the script, and then using logger methods like `logging.debug()`, `logging.info()`, `logging.warning()`, `logging.error()`, `logging.critical()` throughout the code instead of `print()` for status/error reporting.

```python
# --- Code Example 1: Basic Logging Setup ---
import logging
import os
import time

# --- Basic Configuration (at script start) ---
log_filename = 'workflow.log'
# Remove old log file if it exists
if os.path.exists(log_filename): os.remove(log_filename) 

logging.basicConfig(
    level=logging.INFO, # Set minimum severity level to capture (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)-8s - %(message)s', # Include time, level, message
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename), # Log to a file
        logging.StreamHandler() # Also log to console (stderr by default)
    ]
)

logging.info("Workflow started.")

# --- Example Usage within Workflow Functions ---
def step_a(input_val):
    logging.debug(f"Starting Step A with input {input_val}") # Won't show if level=INFO
    logging.info("Performing Step A calculation...")
    time.sleep(0.1)
    if input_val < 0:
        logging.warning("Input value was negative, result might be affected.")
    result = input_val * 2
    logging.info("Step A finished.")
    return result

def step_b(input_val):
    logging.info("Performing Step B...")
    time.sleep(0.1)
    if input_val == 0:
        logging.error("Cannot process input value of zero in Step B.")
        return None # Indicate error
    result = 10 / input_val
    logging.info("Step B finished.")
    return result

# --- Run workflow ---
logging.info("Starting main processing.")
res_a = step_a(5)
res_b = step_b(res_a)
if res_b is None:
    logging.critical("Workflow failed because Step B encountered an error.")
else:
    logging.info(f"Final result from Step B: {res_b}")

res_a_neg = step_a(-3) # Should trigger warning
res_b_zero = step_b(0) # Should trigger error and critical failure message

logging.info("Workflow finished (or terminated).")
print(f"\nCheck '{log_filename}' for detailed logs.")
print("-" * 20)

# Explanation:
# 1. Imports `logging` and `os`.
# 2. Configures basic logging using `logging.basicConfig`:
#    - `level=logging.INFO`: Only messages with INFO severity or higher will be processed.
#    - `format=...`: Specifies the output format including timestamp, level name, message.
#    - `handlers=[...]`: Sets up logging to *both* a file (`workflow.log`) and the console.
# 3. The functions `step_a` and `step_b` use `logging.info()`, `logging.warning()`, 
#    `logging.error()` instead of `print()` to report status and problems. 
#    `logging.debug()` message won't appear due to the INFO level setting.
# 4. The main part logs start/end messages and checks the return from `step_b` to log 
#    a `critical` message if it failed.
# Running this script will print INFO/WARNING/ERROR/CRITICAL messages to the console 
# AND save them (along with any DEBUG messages if level was set lower) to `workflow.log`.
```
Using the `logging` module provides a much more structured and controllable way to monitor workflow execution compared to ad-hoc `print` statements.

**Error Handling (`try...except`):** Workflows can fail for many reasons: missing input files, invalid data, numerical errors (e.g., division by zero), issues calling external software, bugs in the code, etc. Letting the script crash with an unhandled exception provides little information and prevents cleanup or partial saving of results. The `try...except` block is Python's mechanism for structured error handling.

Code that might potentially raise an exception is placed inside the `try:` block. If an error occurs within the `try` block, the normal execution stops, and Python looks for a matching `except` block. If an `except` block matches the type of exception raised (or catches a general `Exception`), the code inside the `except` block is executed. This allows you to gracefully handle errors, log informative messages, perform cleanup actions (like closing files), and potentially allow the workflow to continue with the next item (if processing a batch) or terminate cleanly. A `finally:` block can be added to contain code that *always* executes, regardless of whether an error occurred in the `try` block (useful for releasing resources like file handles).

```python
# --- Code Example 2: Using try...except for Error Handling ---
import logging # Assume logging is configured as above
import os

def process_file_robustly(filename):
    """Processes a file, handling potential errors."""
    logging.info(f"Attempting to process file: {filename}")
    try:
        # Code that might fail (e.g., file access, numerical ops)
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Input file {filename} does not exist.")
            
        # Simulate reading and processing
        # data = np.loadtxt(filename) # Might raise ValueError
        # if len(data) == 0: raise ValueError("File is empty")
        # result = np.mean(data**2) # Might raise numerical warning/error
        
        # Simulate success for demonstration
        print(f"   Successfully processed {filename}")
        result = "Success"
        
    except FileNotFoundError as fnf_err:
        logging.error(f"File not found error for {filename}: {fnf_err}")
        result = "Error: File Missing"
    except ValueError as val_err:
        logging.error(f"Value error during processing {filename}: {val_err}")
        result = "Error: Bad Data"
    except Exception as e: # Catch any other unexpected exceptions
        logging.exception(f"An unexpected error occurred processing {filename}: {e}") # .exception logs traceback
        result = f"Error: Unexpected ({type(e).__name__})"
    finally:
        # Code here runs whether try succeeded or failed
        logging.debug(f"Finished attempting to process {filename}")
        # e.g., close files opened in the try block
        
    return result

# --- Example Usage ---
print("\nRobust File Processing Examples:")
# File that exists (create dummy)
dummy_file = "dummy_exists.txt"
with open(dummy_file, "w") as f: f.write("1 2 3")
process_file_robustly(dummy_file) 
os.remove(dummy_file)

# File that doesn't exist
process_file_robustly("non_existent_file.txt") 

# Simulate other errors (conceptual)
# process_file_robustly("empty_file.txt") # Assuming this file is empty
# process_file_robustly("bad_data_file.txt") # Assuming file causes ValueError

print("-" * 20)

# Explanation:
# 1. Defines `process_file_robustly` which contains the core logic within a `try` block.
# 2. Explicitly raises `FileNotFoundError` if the file doesn't exist. (Conceptual 
#    comments show where other errors like `ValueError` might occur).
# 3. `except FileNotFoundError as fnf_err:` catches only that specific error, logs it 
#    using `logging.error`, and sets an error result.
# 4. `except ValueError as val_err:` catches potential value errors (e.g., from `loadtxt` 
#    on bad data).
# 5. `except Exception as e:` acts as a catch-all for any other unexpected errors. 
#    `logging.exception` is useful here as it automatically includes the traceback 
#    in the log message.
# 6. The `finally:` block (optional) would contain cleanup code that must run regardless 
#    of success or failure within the `try` block.
# 7. The function returns a status/result, allowing the main workflow to know if 
#    processing succeeded or failed for this specific file.
```

Combining structured logging with robust `try...except` error handling is essential for building reliable automated workflows. Logging provides visibility into the execution flow and detailed diagnostics when problems occur, while error handling prevents unexpected crashes and allows the workflow to potentially recover or terminate gracefully, saving intermediate results and providing informative feedback to the user.

**65.6 Managing Files and Directories (`os`, `pathlib`, `glob`)**

Astronomical workflows invariably involve reading input data files, creating intermediate files, and writing final output products (catalogs, plots, logs). Effectively managing these files and the directory structures they reside in is a crucial aspect of workflow scripting. Python provides several built-in modules for interacting with the file system in a portable way.

The **`os` module** provides a comprehensive, low-level interface to operating system functionalities, including file system operations. Key functions relevant for workflows include:
*   `os.path.exists(path)`: Checks if a file or directory exists. Essential for dependency checking (Sec 65.3) or avoiding overwrites.
*   `os.path.join(path_part1, path_part2, ...)`: Joins path components using the correct separator for the current OS (`/` for Unix/Linux/macOS, `\` for Windows). **Crucial for creating portable paths.** Avoid manually concatenating path strings with `/` or `\`.
*   `os.makedirs(path, exist_ok=True)`: Creates a directory, including any necessary parent directories. `exist_ok=True` prevents an error if the directory already exists. Useful for setting up output directories.
*   `os.remove(path)` or `os.unlink(path)`: Deletes a file.
*   `os.rmdir(path)`: Deletes an *empty* directory.
*   `os.rename(src, dst)` or `os.replace(src, dst)`: Renames or moves a file/directory. `os.replace` atomically replaces the destination if it exists on most systems, useful for checkpointing (Sec 42.6).
*   `os.listdir(path)`: Returns a list of filenames within a directory.
*   `os.getcwd()`: Gets the current working directory.
*   `os.path.basename(path)`: Extracts the filename from a path.
*   `os.path.dirname(path)`: Extracts the directory path from a full path.
*   `os.path.isfile(path)`, `os.path.isdir(path)`: Check if a path points to a file or directory.
*   `os.environ`: A dictionary-like object providing access to environment variables (e.g., `os.getenv('HOME')`).

The **`pathlib` module** (introduced in Python 3.4) offers a more modern, object-oriented approach to file system paths. It represents paths as `Path` objects, providing intuitive methods for common operations, often replacing many functions in `os.path`.
*   `from pathlib import Path`
*   `p = Path('/path/to/file.txt')` or `p = Path('relative/path')`
*   `p.exists()`, `p.is_file()`, `p.is_dir()`
*   Joining paths: `p / 'subdir' / 'anotherfile.dat'` (uses the `/` operator intuitively).
*   Creating directories: `p.mkdir(parents=True, exist_ok=True)`
*   Reading/Writing text: `p.read_text()`, `p.write_text("content")`
*   Reading/Writing bytes: `p.read_bytes()`, `p.write_bytes(b"content")`
*   Deleting: `p.unlink()` (for files), `p.rmdir()` (for empty dirs)
*   Renaming/Moving: `p.rename(new_path)`
*   Getting parts: `p.name` (filename), `p.stem` (filename without suffix), `p.suffix` (extension), `p.parent` (directory).
`pathlib` often leads to cleaner and more readable code for path manipulations compared to using multiple `os.path` functions.

The **`glob` module** provides Unix-style pathname pattern expansion. Its primary function `glob.glob(pattern)` returns a list of file paths matching a specified pattern containing wildcards like `*` (matches anything), `?` (matches single character), `[]` (matches character set). This is extremely useful for finding all input files of a certain type in a directory or processing batches of files.

```python
# --- Code Example 1: Using os, pathlib, glob ---
import os
import pathlib # Standard name often 'Path' from pathlib import Path
from pathlib import Path # Common import style
import glob
import shutil # For removing directories containing files

print("File System Operations using os, pathlib, glob:")

# --- Setup ---
base_dir = Path("./temp_workflow_io") # Use pathlib for path object
base_dir.mkdir(exist_ok=True) # Create directory if needed
print(f"\nCreated base directory: {base_dir}")

# --- Create dummy files using pathlib ---
(base_dir / "input_data_01.txt").write_text("Data 1")
(base_dir / "input_data_02.txt").write_text("Data 2")
(base_dir / "config.yaml").write_text("param: value")
(base_dir / "results").mkdir(exist_ok=True)
print("Created dummy files and results subdirectory.")

# --- Using os module ---
print("\nUsing os module:")
config_path_os = os.path.join(str(base_dir), 'config.yaml') # Need str() for os.path
print(f"  Config path exists? {os.path.exists(config_path_os)}")
print(f"  Is it a file? {os.path.isfile(config_path_os)}")
print(f"  Directory name: {os.path.dirname(config_path_os)}")
print(f"  Base name: {os.path.basename(config_path_os)}")

# --- Using pathlib module ---
print("\nUsing pathlib module:")
config_path_pl = base_dir / 'config.yaml' # Cleaner path joining
print(f"  Config path exists? {config_path_pl.exists()}")
print(f"  Is it a file? {config_path_pl.is_file()}")
print(f"  Directory (parent): {config_path_pl.parent}")
print(f"  Filename (name): {config_path_pl.name}")
print(f"  File stem: {config_path_pl.stem}")
print(f"  File suffix: {config_path_pl.suffix}")

# --- Using glob ---
print("\nUsing glob to find input files:")
# Find all .txt files starting with 'input_data_' in the temp directory
# Need full path pattern
search_pattern = os.path.join(str(base_dir), 'input_data_*.txt') 
# Or using pathlib's glob method:
# search_pattern_pl = base_dir / 'input_data_*.txt' # Path objects have .glob
found_files_glob = glob.glob(search_pattern)
print(f"  Found files matching '{search_pattern}': {found_files_glob}")

# Using pathlib's glob method (returns generator)
print("\nUsing pathlib Path.glob():")
found_files_pl = list(base_dir.glob('input_data_*.txt'))
print(f"  Found files matching 'input_data_*.txt': {found_files_pl}")

# --- Cleanup ---
print("\nCleaning up...")
# Need shutil.rmtree to remove directory with contents
if base_dir.exists():
    shutil.rmtree(base_dir)
    print(f"Removed directory {base_dir}")

print("-" * 20)

# Explanation:
# 1. Creates a temporary directory `temp_workflow_io` using `pathlib.Path.mkdir`.
# 2. Creates dummy files inside using `Path.write_text`.
# 3. Demonstrates equivalent operations using `os.path` functions (requiring `str()` 
#    conversion from Path object) and the more object-oriented `pathlib` methods. 
#    Highlights `pathlib`'s cleaner path joining (`/` operator) and attribute access 
#    (`.name`, `.parent`, `.stem`, `.suffix`).
# 4. Shows using `glob.glob` with a pattern (`*`) to find all matching `.txt` files.
# 5. Shows the equivalent using the `.glob()` method directly on a `Path` object.
# 6. Uses `shutil.rmtree` for cleanup, as `os.rmdir` only removes empty directories.
```

Effectively managing files and directories is crucial for almost any data analysis workflow. Using `os.path.join` or `pathlib`'s `/` operator ensures cross-platform path compatibility. Functions like `os.makedirs(..., exist_ok=True)` or `Path(...).mkdir(parents=True, exist_ok=True)` robustly create necessary output directories. `glob` or `Path.glob` are essential for finding and processing batches of input files based on patterns. These standard library tools provide the necessary building blocks for handling file system interactions within your Python workflow scripts.

---

**Application 65.A: Python Script for TESS Light Curve Download and Basic Plotting**

**(Paragraph 1)** **Objective:** This application demonstrates creating a simple, parameterized Python script that uses standard libraries (`argparse`, `lightkurve`, `matplotlib`) to perform the first steps of a common astronomical workflow: downloading data for a specific target and creating a basic visualization. It reinforces concepts of structuring workflows with functions (Sec 65.1), parameterization via command-line arguments (Sec 65.4), and basic file handling/plotting (Sec 65.6).

**(Paragraph 2)** **Astrophysical Context:** Analyzing time-series data from missions like TESS often starts with retrieving the light curve for a specific star (identified by its TIC ID and observed sector) from an archive like MAST. A common initial step is to simply download the data and plot the light curve to visually inspect it for variability, transits, or data quality issues before proceeding with more complex analysis. This application automates this initial retrieval and plotting task.

**(Paragraph 3)** **Data Source:** The TESS data archive at MAST, accessed programmatically via the `lightkurve` library, which interfaces with `astroquery` behind the scenes. The input required from the user is the TESS Input Catalog (TIC) ID of the target star and optionally the sector number.

**(Paragraph 4)** **Modules Used:** `lightkurve` (for searching and downloading TESS data), `argparse` (for handling command-line arguments), `matplotlib.pyplot` (for plotting), `os` (optional, for path manipulation), `sys` (for exiting on error).

**(Paragraph 5)** **Technique Focus:** Building a command-line script using standard Python libraries. (1) Using `argparse` to define command-line arguments for the TIC ID (`--tic`), sector number (`--sector`, optional), and output plot filename (`--outfile`, optional). (2) Structuring the core logic within functions, e.g., `search_and_download_lc(tic_id, sector)` and `plot_light_curve(lc, outfile)`. (3) Using `lightkurve.search_lightcurve()` to find available light curves for the target. (4) Selecting the desired light curve (e.g., the first result or a specific sector) and downloading it using `.download()`. (5) Accessing the light curve data (time, flux) from the `LightCurve` object. (6) Using `matplotlib.pyplot` (or the light curve object's `.plot()` method) to create and save a plot. (7) Implementing basic error handling (e.g., if the target is not found or download fails).

**(Paragraph 6)** **Processing Step 1: Setup and Argument Parsing:** Import necessary libraries. Create an `ArgumentParser` instance using `argparse`. Define the required `--tic` argument (integer type) and optional arguments like `--sector` (integer) and `--outfile` (string) with appropriate help messages and default values (e.g., `default=None`). Parse the arguments using `args = parser.parse_args()`.

**(Paragraph 7)** **Processing Step 2: Define Search/Download Function:** Implement the function `search_and_download_lc(tic_id, sector=None, author="SPOC")`. Inside this function, construct the target string (`f"TIC {tic_id}"`). Call `lightkurve.search_lightcurve()` with the target, sector, and potentially author keyword (e.g., "SPOC" for primary mission data). Check if the search returned any results. If not, print an error and return `None`. If results exist, select the desired light curve (e.g., `search_result[0]`). Call the `.download()` method on the selected result, potentially specifying `download_dir` and `quality_bitmask='default'`. Include a `try...except` block around the search and download process to catch potential network or file errors, printing informative messages and returning `None` on failure. Perform basic cleaning like `lc = lc.remove_nans()`. Return the `LightCurve` object.

**(Paragraph 8)** **Processing Step 3: Define Plotting Function:** Implement the function `plot_light_curve(lc, tic_id, outfile=None)`. Check if the input `lc` object is valid (not `None`). Create a Matplotlib figure and axes (`fig, ax = plt.subplots(...)`). Use the light curve object's built-in plotting method for convenience: `lc.plot(ax=ax, marker='.', linestyle='none', label=f'TIC {tic_id} S{lc.sector}')`. Customize the plot with appropriate axis labels (using `lc.time.format` and `lc.flux.unit` if available) and a descriptive title. If an `outfile` path is provided via the arguments, use `plt.savefig(outfile)` to save the plot; otherwise, use `plt.show()` to display it interactively. Ensure the plot figure is closed (`plt.close(fig)`) after saving/showing to free memory, especially if this function might be called in a loop.

**(Paragraph 9)** **Processing Step 4: Main Script Logic (Orchestration):** In the main part of the script (typically under an `if __name__ == "__main__":` block), call the `search_and_download_lc` function, passing the parsed command-line arguments (`args.tic`, `args.sector`). Store the returned `LightCurve` object. Check if the download was successful (i.e., the returned object is not `None`). If successful, call the `plot_light_curve` function, passing the light curve object, TIC ID, and output filename from the parsed arguments. Include print statements to indicate progress and success or failure. Use `sys.exit(1)` if a critical step like downloading fails, to signal an error to any calling script.

**(Paragraph 10)** **Processing Step 5: Execution and Usage:** Save the complete code to a Python file (e.g., `get_tess_lc.py`). Make it executable if desired (`chmod +x get_tess_lc.py`). Run from the terminal, providing the necessary arguments:
`python get_tess_lc.py --tic 261136679 --sector 1` (displays plot)
`python get_tess_lc.py --tic 261136679 --sector 1 --outfile TIC_261136679_S1.png` (saves plot)
`python get_tess_lc.py --help` (shows usage information generated by `argparse`).

**Output, Testing, and Extension:** The primary output is either an interactively displayed plot or a saved PNG image file of the TESS light curve. Informative messages are printed to the console during execution. **Testing:** Run the script with known TIC IDs that have data in specific sectors and verify the output plot looks correct. Test the `--sector` option. Test providing an `--outfile` path. Test with a TIC ID that has no TESS data or an invalid sector number to ensure error handling works correctly. Check the `--help` output. **Extensions:** (1) Add an option to specify the `author` for `search_lightcurve` (e.g., 'SPOC', 'TESS-SPOC', 'QLP'). (2) Allow downloading data from multiple sectors if `--sector` is omitted or a range is given. (3) Incorporate basic light curve cleaning (`remove_outliers`, `flatten`) as optional command-line flags within the `plot_light_curve` or a separate processing function. (4) Modify the script to take a list of TIC IDs from a file and process them in a loop (serial processing). (5) Add logging using the `logging` module instead of just `print` statements (Sec 65.5).

```python
# --- Code Example: Application 65.A ---
# (Identical code to Application 65.A provided in the previous response)
# File: get_tess_lc.py
import lightkurve as lk
import matplotlib.pyplot as plt
import argparse
import sys
import os # For path check

print("TESS Light Curve Downloader and Plotter")

def search_and_download_lc(tic_id, sector=None, author="SPOC"):
    """Searches for and downloads a TESS light curve."""
    target = f"TIC {tic_id}"
    search_term = target
    search_kwargs = {'author': author}
    if sector is not None:
        search_kwargs['sector'] = sector
        search_term += f" Sector {sector}"
        
    print(f"\nSearching for {search_term} (author={author})...")
    try:
        search_result = lk.search_lightcurve(target, **search_kwargs)
        if not search_result:
            print(f"Error: No light curve found for {search_term}")
            return None
        
        # Select the first result (could add logic for multiple missions/cadences)
        lc_result = search_result[0]
        print(f"Found: {lc_result.mission} Q{lc_result.quarter} S{lc_result.sector} C{lc_result.camera} CCD{lc_result.ccd} ({lc_result.exptime.value}s cadence)")
        print("Downloading...")
        lc = lc_result.download(download_dir=".", quality_bitmask='default') 
        if lc is None:
             print("Error: Download returned None.")
             return None
        lc = lc.remove_nans() 
        print("Download and basic cleaning complete.")
        return lc
    except Exception as e:
        print(f"An error occurred during search/download: {e}")
        return None

def plot_light_curve(lc, tic_id, outfile=None):
    """Plots the light curve and optionally saves it."""
    if lc is None:
        print("No light curve object to plot.")
        return
        
    print("\nGenerating plot...")
    # Use a style that works well generally
    try: plt.style.use('seaborn-v0_8-whitegrid') 
    except: pass # Ignore if style not found
    
    fig, ax = plt.subplots(figsize=(12, 4))
    try:
        # Use lightkurve's plot method
        lc.plot(ax=ax, marker='.', linestyle='none', markersize=1, color='k', label=f'TIC {tic_id} S{lc.sector}')
        ax.set_title(f"TESS SPOC Light Curve - TIC {tic_id} Sector {lc.sector}")
    except Exception as e_plot:
         print(f"Plotting error: {e_plot}. Trying basic plot.")
         # Fallback basic plot if lc.plot fails for some reason
         try:
              ax.plot(lc.time.value, lc.flux.value, 'k.', markersize=1)
              ax.set_xlabel(f"Time [{lc.time.format.upper()}]")
              ax.set_ylabel(f"Flux")
              ax.set_title(f"TESS Light Curve - TIC {tic_id} Sector {lc.sector}")
         except Exception as e_basic:
              print(f"Basic plotting failed: {e_basic}")
              plt.close(fig)
              return

    if outfile:
        try:
            outdir = os.path.dirname(outfile)
            if outdir: os.makedirs(outdir, exist_ok=True)
            plt.savefig(outfile, dpi=150)
            print(f"Plot saved to {outfile}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    else:
        try:
             plt.show() # Display interactively if no output file given
        except Exception as e_show:
             print(f"Could not display plot interactively: {e_show}")
        
    plt.close(fig) # Close figure to free memory

# Main execution block
if __name__ == "__main__":
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Download and plot a TESS SPOC light curve.")
    parser.add_argument("--tic", required=True, type=int, help="TIC ID of the target star.")
    parser.add_argument("--sector", type=int, default=None, help="Sector number (optional, downloads first available SPOC LC if None).")
    parser.add_argument("--outfile", type=str, default=None, help="Output PNG filename for plot (optional, shows plot if None).")
    
    # Add author selection if desired
    # parser.add_argument("--author", type=str, default="SPOC", help="Pipeline author (e.g., SPOC, TESS-SPOC, QLP)")
    
    args = parser.parse_args()

    # Workflow Orchestration
    print(f"--- Processing TIC {args.tic} ---")
    lightcurve = search_and_download_lc(args.tic, args.sector, author="SPOC") # Defaulting to SPOC
    
    if lightcurve:
        plot_light_curve(lightcurve, args.tic, args.outfile)
        print(f"--- Finished TIC {args.tic} ---")
    else:
        print(f"--- Failed to process TIC {args.tic} ---")
        sys.exit(1) # Exit with error code if download failed
        
    print("\nScript finished.")
```

**Application 65.B: Parameterizing a Simple Fitting Routine**

**(Paragraph 1)** **Objective:** This application demonstrates how to make a simple scientific analysis script (performing a model fit) more flexible and reusable by parameterizing it using both command-line arguments (`argparse`, Sec 65.4) and a configuration file (e.g., YAML, Sec 65.4). This allows users to easily change input files, model parameters, initial guesses, and output settings without modifying the script's code.

**(Paragraph 2)** **Astrophysical Context:** Fitting theoretical or empirical models to observational data is a core activity in astrophysics (e.g., fitting spectra, light curves, profiles). Often, the same fitting logic needs to be applied to different datasets, using different model variations, or starting with different initial parameter guesses. Hardcoding filenames and parameters directly into the script makes it difficult to reuse and prone to errors when changes are needed. Parameterization separates the analysis logic from the specific settings for a given run.

**(Paragraph 3)** **Data Source:** An input data file (`data_to_fit.txt` or `.csv`) containing, for example, two columns (x, y) representing data points to be fitted (e.g., wavelength vs. flux, time vs. magnitude). We also need a configuration file (`fit_config.yaml`) specifying parameters like the input filename, the model type to use (e.g., 'gaussian' or 'powerlaw'), initial parameter guesses, and the output plot filename.

**(Paragraph 4)** **Modules Used:** `argparse` (to potentially specify the config file location), `yaml` (`pip install pyyaml`, for reading the config file), `numpy` (for data and model calculation), `scipy.optimize.curve_fit` (for performing the fit), `matplotlib.pyplot` (for plotting), `os` (for path handling).

**(Paragraph 5)** **Technique Focus:** Parameterization and configuration management. (1) Defining analysis parameters (input file, output file, model type, initial guesses, fitting ranges) in a structured YAML configuration file. (2) Using the `PyYAML` library to load the configuration file into a Python dictionary. (3) Using `argparse` to allow the user to optionally specify the configuration file path on the command line. (4) Structuring the fitting script to read parameters *from the configuration dictionary* instead of having hardcoded values. (5) Implementing different model functions (e.g., Gaussian, power law) and selecting which one to use based on the 'model_type' parameter read from the config. (6) Performing the fit using `curve_fit` with initial guesses read from the config. (7) Saving the results (plot, fitted parameters) to filenames specified in the config.

**(Paragraph 6)** **Processing Step 1: Create Config File:** Create `fit_config.yaml` with key-value pairs using YAML syntax (indentation matters). Include sections for inputs, model specifications, initial guesses (nested), and outputs. Ensure keys are descriptive.

**(Paragraph 7)** **Processing Step 2: Setup Argument Parser:** In the Python script, import `argparse`. Create a parser. Add an argument (e.g., `-c` or `--config`) that takes the path to the configuration file, providing a default value (e.g., `'config.yaml'`). Parse the arguments using `parser.parse_args()`.

**(Paragraph 8)** **Processing Step 3: Load Config and Data:** Get the config filename from the parsed arguments. Open and read the YAML file using `yaml.safe_load()` inside a `try...except` block to handle `FileNotFoundError` and `yaml.YAMLError`. Store the loaded config in a dictionary. Read the input data file specified in the config dictionary (using `.get()` with defaults is robust) into NumPy arrays (`x_data`, `y_data`), again using `try...except` for file errors.

**(Paragraph 9)** **Processing Step 4: Select Model and Parameters:** Define Python functions for the different possible models (`gaussian_model`, `powerlaw_model`, etc.) that accept `x` and model parameters as arguments. Retrieve the desired `model_type` string from the config dictionary. Use `if/elif/else` statements to select the correct `model_func` based on `model_type`. Retrieve the corresponding list of initial parameter guesses `p0` from the nested `initial_guesses` dictionary within the config. Handle cases where the model type is unknown or initial guesses are missing.

**(Paragraph 10)** **Processing Step 5: Perform Fit, Save, Plot:** Use `scipy.optimize.curve_fit`, passing `model_func`, `x_data`, `y_data`, and `p0`. Handle potential `RuntimeError` if the fit fails. Extract the optimal parameters (`popt`) and covariance matrix (`pcov`). Calculate parameter errors (`perr = np.sqrt(np.diag(pcov))`). Save `popt` and `perr` to the output parameter file specified in the config. Generate a plot showing the original data and the best-fit model curve (`model_func(x_plot, *popt)`), saving it to the output plot file specified in the config, ensuring output directories exist (`os.makedirs(..., exist_ok=True)`).

**Output, Testing, and Extension:** Output includes the saved plot file showing the data and fit, and a text file containing the best-fit parameters and their uncertainties. **Testing:** Verify the script runs correctly using the default `fit_config.yaml`. Create a second config file specifying a different model type ('powerlaw') or different initial guesses and run the script using `python fit_script.py --config second_config.yaml`, verifying it uses the new settings. Test error handling if the config file is missing or malformed, or if the data file cannot be read. **Extensions:** (1) Add more model types to the script and config file structure. (2) Use `astropy.modeling` instead of `curve_fit` for fitting. (3) Implement more sophisticated error handling and logging (`logging` module, Sec 65.5). (4) Integrate this parameterized fitting script as a step within a larger Snakemake or Parsl workflow. (5) Allow overriding specific config parameters directly via additional command-line arguments using `argparse`.

```python
# --- Code Example: Application 65.B ---
# File: run_fit.py
# (Identical code to Application 65.B provided in the previous response)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse
import yaml # Requires PyYAML: pip install pyyaml
import os
import sys # For sys.exit

print("Parameterized Fitting Script:")

# Define Model Functions
def gaussian_model(x, amplitude, mean, stddev, baseline):
    """ Gaussian plus baseline """
    return baseline + amplitude * np.exp(-((x - mean)**2 / (2 * stddev**2)))

def powerlaw_model(x, norm, index):
    """ Power Law: norm * x^index """
    return norm * np.power(np.maximum(x, 1e-9), index) 

# Main function
def run_fit(config_file):
    """Loads config, data, runs fit, saves results."""
    
    # --- Step 3: Load Config ---
    print(f"\nLoading configuration from: {config_file}")
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        if config is None: config = {} # Handle empty file
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found.")
        return False
    except yaml.YAMLError as e:
        print(f"Error parsing config file '{config_file}': {e}")
        return False
    print("Config loaded successfully.")
        
    # --- Load Data ---
    input_file = config.get('input_datafile', None)
    if not input_file:
        print("Error: 'input_datafile' not specified in config.")
        return False
    print(f"\nLoading data from: {input_file}")
    try:
        data = np.loadtxt(input_file)
        x_data, y_data = data[:, 0], data[:, 1]
    except FileNotFoundError:
        print(f"Error: Data file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error reading data file '{input_file}': {e}")
        return False
    print(f"Loaded {len(x_data)} data points.")

    # Optional: Apply fit range
    fit_range = config.get('fit_range', None)
    if fit_range and len(fit_range) == 2:
        mask = (x_data >= fit_range[0]) & (x_data <= fit_range[1])
        x_fit = x_data[mask]
        y_fit = y_data[mask]
        if len(x_fit) == 0:
             print(f"Error: No data points remaining within fit range {fit_range}.")
             return False
        print(f"Applied fit range: [{fit_range[0]}, {fit_range[1]}] ({len(x_fit)} points)")
    else:
        x_fit, y_fit = x_data, y_data

    # --- Step 4: Select Model and Initial Guesses ---
    model_type = config.get('model_type', '').lower()
    print(f"\nSelected model type: {model_type}")
    model_func = None
    p0 = None
    param_names = [] # To store parameter names for output
    if model_type == 'gaussian':
        model_func = gaussian_model
        p0 = config.get('initial_guesses', {}).get('gaussian', None)
        param_names = ['amplitude', 'mean', 'stddev', 'baseline']
    elif model_type == 'powerlaw':
        model_func = powerlaw_model
        p0 = config.get('initial_guesses', {}).get('powerlaw', None)
        param_names = ['norm', 'index']
    else:
        print(f"Error: Unknown or missing model_type '{model_type}' in config.")
        return False
        
    if p0: print(f"Initial parameter guesses (p0): {p0}")
    else: print("Warning: No initial guesses (p0) provided in config for this model.")

    # --- Step 5: Perform Fit ---
    print("\nPerforming fit using scipy.optimize.curve_fit...")
    try:
        # Add sigma if available in data/config? Assume no weights for now.
        popt, pcov = curve_fit(model_func, x_fit, y_fit, p0=p0)
        perr = np.sqrt(np.diag(pcov)) 
        print("Fit successful.")
        print("Best-fit parameters (popt):", popt)
        print("Parameter errors (sqrt(diag(pcov))):", perr)
    except RuntimeError as e_fit: 
        print(f"Error: curve_fit failed to converge: {e_fit}")
        return False
    except Exception as e:
        print(f"An error occurred during fitting: {e}")
        return False

    # --- Save Parameter Results ---
    output_params_file = config.get('output_params', None)
    if output_params_file:
        print(f"\nSaving parameters to: {output_params_file}")
        try:
            os.makedirs(os.path.dirname(output_params_file) or '.', exist_ok=True)
            with open(output_params_file, 'w') as f:
                f.write(f"# Fit results for model: {model_type}\n")
                # Use param_names if available, otherwise generic names
                header = "# Param_Name Value Error\n"
                if len(param_names) == len(popt):
                    header = f"# {'Param_Name':<15} {'Value':<15} {'Error':<15}\n"
                f.write(header)
                for i in range(len(popt)):
                    pname = param_names[i] if i < len(param_names) else f'param_{i}'
                    f.write(f"{pname:<15} {popt[i]:<15.6e} {perr[i]:<15.6e}\n")
            print("Parameters saved.")
        except Exception as e:
            print(f"Error saving parameters: {e}")

    # --- Generate and Save Plot ---
    output_plot_file = config.get('output_plot', None)
    if output_plot_file:
        print(f"\nGenerating plot and saving to: {output_plot_file}")
        try:
            os.makedirs(os.path.dirname(output_plot_file) or '.', exist_ok=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x_data, y_data, 'k.', markersize=4, label='Data', alpha=0.7)
            # Plot fit over finer grid within fit range
            x_smooth = np.linspace(x_fit.min(), x_fit.max(), 200)
            ax.plot(x_smooth, model_func(x_smooth, *popt), 'r-', label=f'Best Fit ({model_type})')
            ax.set_xlabel("X Data")
            ax.set_ylabel("Y Data")
            ax.set_title(f"Model Fit to {os.path.basename(input_file)}")
            ax.legend()
            ax.grid(True, alpha=0.5)
            fig.tight_layout()
            plt.savefig(output_plot_file, dpi=150)
            print("Plot saved.")
            plt.close(fig)
        except Exception as e:
            print(f"Error generating/saving plot: {e}")
            
    return True # Indicate success

# Main execution block
if __name__ == "__main__":
    # --- Step 2: Argument Parser ---
    parser = argparse.ArgumentParser(description="Run a parameterized model fit using a YAML config file.")
    parser.add_argument("-c", "--config", default="fit_config.yaml", 
                        help="Path to the YAML configuration file (default: fit_config.yaml)")
    args = parser.parse_args()

    # --- Create Dummy Config and Data if default config not found ---
    config_file_to_use = args.config
    if not os.path.exists(config_file_to_use):
        print(f"Config file '{config_file_to_use}' not found. Creating dummy files...")
        config_file_to_use = 'dummy_config.yaml' # Use dummy name
        dummy_config = {
            'input_datafile': 'dummy_data.txt',
            'output_plot': 'dummy_fit_plot.png',
            'output_params': 'dummy_fit_params.txt',
            'model_type': 'gaussian',
            'initial_guesses': {'gaussian': [0.8, 5.0, 1.0, 0.1]}, 
            'fit_range': [1, 9]
        }
        os.makedirs(os.path.dirname(config_file_to_use) or '.', exist_ok=True)
        with open(config_file_to_use, 'w') as f_cfg: yaml.dump(dummy_config, f_cfg, default_flow_style=False)
        
        x_dummy = np.linspace(0, 10, 50)
        y_dummy = gaussian_model(x_dummy, 0.8, 5.0, 1.0, 0.1) + np.random.normal(0, 0.05, 50)
        os.makedirs(os.path.dirname(dummy_config['input_datafile']) or '.', exist_ok=True)
        np.savetxt(dummy_config['input_datafile'], np.vstack((x_dummy, y_dummy)).T)
        print("Dummy files created.")
        
    # --- Run the main function ---
    success = run_fit(config_file_to_use)
    
    # --- Cleanup Demo Files ---
    if "dummy" in config_file_to_use:
         print("\nCleaning up dummy files...")
         dummy_config_loaded = {}
         try: 
              with open(config_file_to_use, 'r') as f: dummy_config_loaded = yaml.safe_load(f)
         except: pass
         if os.path.exists(dummy_config_loaded.get('input_datafile','')): os.remove(dummy_config_loaded['input_datafile'])
         if os.path.exists(dummy_config_loaded.get('output_plot','')): os.remove(dummy_config_loaded['output_plot'])
         if os.path.exists(dummy_config_loaded.get('output_params','')): os.remove(dummy_config_loaded['output_params'])
         if os.path.exists(config_file_to_use): os.remove(config_file_to_use)
         print("Cleanup done.")

    if success: print("\nScript finished successfully.")
    else: print("\nScript finished with errors."); sys.exit(1)
    
print("-" * 20)
```

**Chapter 65 Summary**

This chapter focused on managing moderately complex astronomical workflows using standard **Python scripting** and core libraries, providing techniques to enhance organization, flexibility, and robustness without resorting to dedicated Workflow Management Systems (WMS). The fundamental approach advocated was **structuring workflows using functions**, breaking down the analysis pipeline into logical, reusable, and testable functional units, each with clear inputs and outputs. Strategies for **passing data between these functional steps** were discussed, contrasting the efficiency of **in-memory** data transfer (via return values and arguments) for smaller datasets with the necessity and benefits (persistence, handling large data, interoperability) of **file-based** data transfer (saving intermediate results to disk) for larger or more complex workflows needing resilience. Basic methods for managing **dependencies** within a script, primarily through the sequential calling of functions or simple checks for the existence of required input files using the `os` module, were outlined, acknowledging their limitations compared to the automatic dependency tracking in WMS.

A significant focus was placed on **parameterizing workflows** to separate the analysis logic from specific run configurations. Techniques included using Python's built-in **`argparse`** module to define and parse command-line arguments (allowing users to specify inputs like filenames or key parameters when running the script) and employing structured **configuration files** (in human-readable formats like **YAML** or JSON, read using libraries like `PyYAML`) to manage larger sets of parameters, initial conditions, or file paths externally. Best practices for making workflows more robust included implementing **logging** using Python's `logging` module to record progress, warnings, and errors systematically, and incorporating **error handling** using `try...except` blocks to catch potential exceptions (like `FileNotFoundError` or numerical errors) and allow the script to fail gracefully or potentially continue. Finally, common file system operations essential for managing data within workflows were demonstrated using the `os`, `pathlib`, and `glob` modules for tasks like creating directories, checking file existence, constructing paths, and finding files matching patterns. Two applications demonstrated creating a parameterized TESS light curve downloader/plotter using `argparse` and `lightkurve`, and parameterizing a model fitting routine using `argparse`, YAML config files, and `scipy`.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Python Software Foundation. (n.d.).** *Python Tutorial - Chapter 4: More Control Flow Tools (Defining Functions)*. Python Documentation. Retrieved January 16, 2024, from [https://docs.python.org/3/tutorial/controlflow.html#defining-functions](https://docs.python.org/3/tutorial/controlflow.html#defining-functions)
    *(Official tutorial covering function definition, arguments, scope, relevant to Sec 65.1.)*

2.  **Python Software Foundation. (n.d.).** *argparse — Parser for command-line options, arguments and sub-commands*. Python Documentation. Retrieved January 16, 2024, from [https://docs.python.org/3/library/argparse.html](https://docs.python.org/3/library/argparse.html)
    *(Official documentation for the `argparse` module, essential for parameterization described in Sec 65.4 and App 65.A/B.)*

3.  **PyYAML Developers. (n.d.).** *PyYAML Documentation*. PyYAML. Retrieved January 16, 2024, from [https://pyyaml.org/wiki/PyYAMLDocumentation](https://pyyaml.org/wiki/PyYAMLDocumentation)
    *(Documentation for the PyYAML library, commonly used for reading YAML configuration files discussed in Sec 65.4 and App 65.B.)*

4.  **Python Software Foundation. (n.d.).** *logging — Logging facility for Python*. Python Documentation. Retrieved January 16, 2024, from [https://docs.python.org/3/library/logging.html](https://docs.python.org/3/library/logging.html) (See also Logging HOWTO: [https://docs.python.org/3/howto/logging.html](https://docs.python.org/3/howto/logging.html))
    *(Official documentation for Python's standard logging module, relevant to Sec 65.5.)*

5.  **Wilson, G., et al. (2017).** Good enough practices in scientific computing. *(See reference in Chapter 64)*.
    *(Provides practical advice on organizing projects, automating tasks with scripts, and managing data, which aligns with the goals of workflow management discussed in this chapter.)*
