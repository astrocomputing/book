**Chapter 67: Introduction to Dask for Scalable Analysis**

While Workflow Management Systems (Chapter 66) excel at orchestrating pipelines involving distinct scripts or command-line tools, often based on file dependencies, the **Dask** library offers a powerful, Python-native approach specifically designed for scaling familiar data analysis operations (like those performed with NumPy and Pandas) to datasets that exceed available RAM and distributing these computations across multiple cores or even multiple machines in a cluster. This chapter provides a focused introduction to Dask as a tool for **scalable data analysis**, revisiting and expanding on concepts introduced in Chapter 40. We will reiterate Dask's core principles of **lazy evaluation** and **task graph** generation, which allow it to optimize complex computations before execution. We delve deeper into its main parallel data collections: **Dask Array**, which mimics NumPy's `ndarray` but operates on chunked arrays potentially distributed across memory or disk; **Dask DataFrame**, which parallels Pandas DataFrames using row-wise partitioning; and **Dask Bag**, for parallel processing of generic Python objects using functional paradigms. We re-examine Dask's flexible **schedulers**, contrasting the local schedulers (synchronous, threaded, processes) suitable for single-machine parallelism with the powerful **distributed scheduler** (`dask.distributed`) enabling scaling across HPC clusters or cloud resources, including a look at its diagnostic **dashboard**. The emphasis is on how Dask allows users to write scalable parallel analysis code using familiar Python APIs with relatively minimal modification compared to techniques like MPI.

**67.1 Recap: Dask Principles (Lazy Evaluation, Task Graphs, Schedulers)**

Dask is a flexible parallel computing library for Python that distinguishes itself through its **lazy evaluation** model and explicit representation of computations as **task graphs**. Unlike standard libraries like NumPy or Pandas, which execute operations immediately (eager execution), Dask operations on its parallel collections (Arrays, DataFrames, Bags) typically do not perform computation right away. Instead, they build up a graph representing the sequence of operations (tasks) and their dependencies.

Consider adding two Dask arrays: `c = da.add(a, b)`. This operation does not immediately compute the sum. It creates a Dask array object `c` and records in its associated task graph that `c` depends on tasks that involve adding corresponding chunks of `a` and `b`. Similarly, subsequent operations like `d = c.mean()` add more nodes and edges to this graph, representing the mean calculation depending on the result of the addition tasks. This graph, often a Directed Acyclic Graph (DAG), encapsulates the entire desired computation.

This **lazy evaluation** approach offers significant advantages for parallelism and handling large datasets:
1.  **Optimization:** Before executing anything, Dask's scheduler can analyze the entire task graph, potentially fusing operations, rearranging tasks, or eliminating unnecessary intermediate computations to optimize performance.
2.  **Memory Management:** For datasets larger than RAM, Dask can arrange to load only the necessary data chunks into memory for a specific task, compute the result, and then potentially release the chunk's memory before loading the next one, enabling out-of-core computation.
3.  **Parallel Scheduling:** The scheduler can identify independent tasks (nodes in the graph with no dependency paths between them) and assign them to different worker threads, processes, or machines for parallel execution.

The actual computation is only triggered when an explicit result is requested, typically by calling the **`.compute()`** method on a Dask object (e.g., `result = d.compute()`). At this point, the appropriate Dask **scheduler** takes the task graph and executes the constituent tasks using the available computational resources.

As discussed in Sec 40.6, Dask provides several schedulers:
*   **Local Schedulers:**
    *   `'sync'`: Single-threaded, executes tasks sequentially (good for debugging).
    *   `'threads'`: Uses a thread pool (`threading`). Low overhead, good for I/O-bound tasks or code releasing the GIL (like many optimized NumPy/SciPy functions).
    *   `'processes'`: Uses a process pool (`multiprocessing`). Higher overhead but bypasses GIL for CPU-bound Python tasks.
*   **Distributed Scheduler (`dask.distributed`):** Manages a cluster of workers (potentially across multiple machines). A `Client` object connects your Python session to the cluster, and the scheduler distributes tasks intelligently across workers, handling data locality and communication. Provides scalability beyond a single machine and includes a powerful diagnostic dashboard.

This architecture—lazy evaluation building task graphs, followed by intelligent scheduling and execution—allows Dask to scale Python code involving familiar APIs like NumPy and Pandas from single cores to large clusters, often with minimal changes to the analysis logic itself. The key is to express the computation using Dask collections and let Dask manage the parallel execution via its schedulers.

**67.2 Dask Array: Scalable NumPy**

For astrophysical data analysis involving large multi-dimensional arrays (images, data cubes, simulation grids) that may not fit into the memory of a single machine, **Dask Array** (`dask.array` or `da`) provides a powerful solution by offering a **NumPy-like interface** built on the principles of lazy evaluation and chunking. A Dask array represents a single logical array that is internally composed of many smaller NumPy arrays, called **chunks**, arranged in a grid.

**Chunking** is the core concept. When creating or loading a Dask array, you specify the `chunks` tuple, which defines the maximum shape of these underlying NumPy arrays (e.g., `chunks=(1000, 1000)` for a 2D array means it's broken into blocks of size at most 1000x1000). Dask does not load all chunks into memory at once. Instead, operations on the Dask array generate tasks that operate on individual chunks or combinations of neighboring chunks.

**Creating Dask Arrays:**
*   From existing NumPy arrays (useful if already loaded but want parallel processing): `darr = da.from_array(np_arr, chunks=(...))`
*   From files supporting chunked reading (like HDF5 or Zarr): `darr = da.from_zarr(...)`, `darr = da.from_array(h5py_dataset, chunks=...)`
*   Using NumPy-like creation functions: `darr = da.ones(shape, chunks=...)`, `da.zeros(...)`, `da.random.random(...)`
*   Stacking/Concatenating other Dask arrays: `da.stack([...], axis=...)`, `da.concatenate([...], axis=...)`

```python
# --- Code Example 1: Creating Dask Arrays ---
import dask.array as da
import numpy as np
import h5py # For HDF5 example
import os

print("Creating Dask Arrays:")

# 1. From NumPy array
np_arr = np.arange(20).reshape(5, 4)
dask_arr1 = da.from_array(np_arr, chunks=(2, 2)) # Chunk into 2x2 blocks
print("\n1. Dask array from NumPy array:")
print(dask_arr1) # Shows structure, chunks, dtype
print(f"   Number of chunks: {dask_arr1.numblocks}") 

# 2. Using creation functions
shape = (10000, 5000)
chunks = (1000, 1000)
dask_arr2 = da.random.random(shape, chunks=chunks)
print("\n2. Dask array using da.random.random:")
print(dask_arr2)
print(f"   Total Size (GB, approx): {dask_arr2.nbytes / 1024**3:.2f} GB")

# 3. From HDF5 dataset (conceptual - requires existing file)
filename = 'large_data.hdf5'
datasetname = 'image_data'
# Create dummy file first (if it doesn't exist)
if not os.path.exists(filename):
     print(f"\nCreating dummy HDF5 file '{filename}'...")
     with h5py.File(filename, 'w') as f:
          # Important: Create HDF5 dataset WITH chunking for efficient dask loading
          f.create_dataset(datasetname, shape=(2048, 2048), dtype='f4', chunks=(256, 256))
     print("Dummy file created.")

print(f"\n3. Dask array from HDF5 dataset '{datasetname}':")
try:
    with h5py.File(filename, 'r') as f:
        h5_dataset = f[datasetname]
        # Let dask use HDF5's chunking, or specify different chunks
        dask_arr3 = da.from_array(h5_dataset, chunks=h5_dataset.chunks) 
        print(dask_arr3)
except Exception as e:
    print(f"  Could not load from HDF5: {e}")

# Cleanup dummy file
if os.path.exists(filename): os.remove(filename)

print("-" * 20)

# Explanation:
# 1. Creates `dask_arr1` from an existing NumPy array `np_arr`, specifying how to 
#    chunk it (`chunks=(2, 2)`).
# 2. Creates a very large Dask array `dask_arr2` (conceptually, data isn't generated 
#    until compute) using `da.random.random`, specifying shape and chunks. Note its 
#    large potential size.
# 3. Shows creating a Dask array `dask_arr3` directly from an HDF5 dataset handle 
#    (after creating a dummy chunked HDF5 file). `da.from_array` can efficiently map 
#    Dask chunks onto HDF5 chunks, enabling lazy, parallel loading from disk.
```

**Operations:** Dask Array supports a large fraction of the NumPy API. Operations like arithmetic (`+`, `-`, `*`, `/`, `**`), trigonometric/logarithmic functions (`da.sin`, `da.log`), comparisons (`>`, `<`, `==`), slicing (`darr[start:stop]`), reductions (`.sum()`, `.mean()`, `.std()`, `.min()`, `.max()`), transposing (`.T`), stacking (`da.stack`), and many linear algebra functions (`da.linalg.svd`, `da.dot`) work similarly to their NumPy counterparts but operate lazily, generating task graphs.

```python
# --- Code Example 2: Dask Array Operations (Lazy) ---
# (Continuing from previous cell, assumes dask_arr2 is defined)
import dask.array as da
import time

print("\nPerforming Operations on Dask Arrays (Lazy Evaluation):")

if 'dask_arr2' in locals():
    # Example operations
    arr_a = dask_arr2[:5000, :2500] # Lazy slicing
    arr_b = da.sin(arr_a) + 1      # Lazy element-wise ops
    arr_c = arr_b.T @ arr_b[:2500, :1000] # Lazy transpose and matrix multiply (@)
    result_lazy = arr_c.mean(axis=0) # Lazy reduction

    print(f"Defined lazy slice 'arr_a': {arr_a}")
    print(f"Defined lazy element-wise 'arr_b': {arr_b}")
    print(f"Defined lazy matmul 'arr_c': {arr_c}")
    print(f"Defined lazy final result (mean): {result_lazy}")
    
    # Nothing computed yet. Now trigger execution:
    print("\nTriggering computation with .compute()...")
    start_time = time.time()
    # Use default scheduler or configure threaded/processes/distributed
    # dask.config.set(scheduler='threads') 
    final_result_numpy = result_lazy.compute()
    end_time = time.time()
    
    print(f"\nComputation complete. Time: {end_time - start_time:.3f}s")
    print(f"Final result shape: {final_result_numpy.shape}")
    # print(f"Final result sample: {final_result_numpy[:5]}")
else:
    print("Skipping Dask Array operations as dask_arr2 not defined.")
    
print("-" * 20)

# Explanation:
# 1. Assumes `dask_arr2` (a large random Dask array) exists.
# 2. Performs several operations: slicing (`arr_a`), element-wise ufuncs (`arr_b`), 
#    transpose (`.T`) and matrix multiplication (`@`) (`arr_c`), and reduction (`.mean()`) 
#    (`result_lazy`). 
# 3. Critically, none of these operations execute immediately. They build a complex 
#    task graph stored within the `result_lazy` Dask array object. Printing the 
#    intermediate arrays shows their structure and task graph info, not computed values.
# 4. Calling `.compute()` on `result_lazy` triggers the execution of the entire graph. 
#    Dask's scheduler runs the required chunk operations (random number generation, 
#    sin, addition, transpose, slicing, matrix multiplies, partial means, final mean) 
#    in an optimized order, potentially in parallel, finally returning the computed 
#    result as a standard NumPy array (`final_result_numpy`).
```

Dask Array is invaluable for applying NumPy-style analysis to large simulation datasets (e.g., calculating power spectra from density grids), processing large observational image cubes, or performing linear algebra on matrices that don't fit in memory. By managing chunked computation and parallel execution, it allows scaling standard numerical Python techniques to much larger problems.

**67.3 Dask DataFrame: Scalable Pandas**

Just as Dask Array parallels NumPy, **Dask DataFrame** (`dask.dataframe` or `dd`) provides a parallel, larger-than-memory equivalent to the popular **Pandas** library. It allows users to perform Pandas-like operations on tabular datasets that might be spread across many files or exceed the memory capacity of a single machine. A Dask DataFrame is composed of multiple independent Pandas DataFrames, known as **partitions**, typically divided row-wise.

**Creating Dask DataFrames:** Dask DataFrames are most commonly created by reading data from storage, especially from collections of files stored in efficient columnar formats like **Parquet** or standard formats like **CSV**.
*   `dd.read_csv('data/measurements_*.csv')`: Reads multiple CSV files matching the pattern. Each file often becomes one partition. Supports many Pandas `read_csv` options.
*   `dd.read_parquet('data/output_parquet_dir/')`: Reads a directory containing multiple Parquet files (a common way to store large tabular datasets efficiently). Parquet's columnar nature and metadata support make it highly suitable for Dask.
*   `dd.from_pandas(pandas_df, npartitions=N)`: Creates a Dask DataFrame by splitting an existing Pandas DataFrame into `N` partitions. Useful for transitioning from in-memory Pandas or for testing.
*   `dd.from_dask_array(dask_array, columns=...)`: Creates from a Dask Array.
*   `dd.from_delayed([...])`: Creates from a list of Dask `delayed` objects, each returning a Pandas DataFrame (useful for custom loading/creation tasks).

```python
# --- Code Example 1: Creating Dask DataFrames ---
# Note: Requires dask[dataframe] installation: pip install "dask[dataframe]" pandas pyarrow
# pyarrow or fastparquet is needed for Parquet.
import dask.dataframe as dd
import pandas as pd
import numpy as np
import os
import shutil

print("Creating Dask DataFrames:")

# --- Create Dummy CSV Files ---
temp_csv_dir = "temp_csv_data"
os.makedirs(temp_csv_dir, exist_ok=True)
n_files = 5
n_rows_per_file = 10000
print(f"\nCreating {n_files} dummy CSV files in '{temp_csv_dir}'...")
for i in range(n_files):
    fname = os.path.join(temp_csv_dir, f"data_{i}.csv")
    df_pd = pd.DataFrame({
        'id': np.arange(i*n_rows_per_file, (i+1)*n_rows_per_file),
        'x': np.random.rand(n_rows_per_file) * 100,
        'y': np.random.normal(50, 10, n_rows_per_file),
        'category': np.random.choice(['A', 'B', 'C'], size=n_rows_per_file)
    })
    df_pd.to_csv(fname, index=False)
print("Dummy CSV files created.")

# --- Create Dask DataFrame from CSVs ---
print("\n1. Creating Dask DataFrame from CSV files:")
# Reads all CSVs matching pattern, each becomes a partition by default
ddf = dd.read_csv(os.path.join(temp_csv_dir, "data_*.csv")) 
print(ddf) # Shows structure, columns, N partitions, dtypes
print(f"  Number of partitions: {ddf.npartitions}")
# Accessing head triggers reading first part
print("\n  Sample head (triggers computation for first partition):")
print(ddf.head())

# --- Create Dask DataFrame from Pandas ---
# print("\n2. Creating Dask DataFrame from Pandas DataFrame:")
# large_pandas_df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(temp_csv_dir, "*.csv"))])
# ddf_from_pandas = dd.from_pandas(large_pandas_df, npartitions=4) # Explicitly partition
# print(ddf_from_pandas)

# Cleanup
if os.path.exists(temp_csv_dir): shutil.rmtree(temp_csv_dir)
print(f"\nCleaned up '{temp_csv_dir}'.")
print("-" * 20)

# Explanation:
# 1. Creates several dummy CSV files containing sample data.
# 2. Uses `dd.read_csv('temp_csv_data/data_*.csv')` to create a Dask DataFrame `ddf`. 
#    Dask automatically finds all matching files and creates a partition for each 
#    (or based on blocksize for single large files). Printing `ddf` shows its metadata 
#    (columns, types, number of partitions) but not the data itself.
# 3. Calling `ddf.head()` triggers Dask to actually read and compute the first few 
#    rows (typically from the first partition) for display.
# 4. Conceptually shows creating from a large Pandas DataFrame using `dd.from_pandas`, 
#    specifying the desired number of partitions.
```

**Operations:** Dask DataFrame implements a large subset of the Pandas API. Many common operations execute **embarrassingly parallel** across partitions:
*   Column selection: `ddf['x']`, `ddf[['x', 'y']]`
*   Filtering rows: `ddf[ddf.y > 50]`
*   Creating new columns based on existing ones: `ddf['z'] = ddf['x'] + ddf['y']`
*   Applying element-wise functions: `ddf['x_sin'] = np.sin(ddf['x'])` (uses underlying Pandas/NumPy)
*   Dropping rows with missing values: `ddf.dropna()`
*   Applying custom functions row-wise (`.apply(..., axis=1, meta=...)`) or element-wise (`.map_partitions(func, meta=...)`). `meta` is often needed to tell Dask the expected output structure/dtype.

Other operations require **shuffling** data between partitions, making them more computationally expensive:
*   Setting a new index (`.set_index('column_name')`): Requires sorting data across partitions. Crucial for efficient subsequent merges or time series analysis based on the index.
*   Joins/Merges (`dd.merge(ddf1, ddf2, ...)`): Efficient if joining on sorted index, otherwise requires shuffling.
*   `groupby().agg()`: Involves shuffling data to bring rows with the same key together on the same worker before performing aggregations (like `sum`, `mean`, `count`, `std`, `apply`).

Like Dask Array, all these operations are **lazy** and build a task graph. Computation only happens when `.compute()` is called or when an operation requires concrete values (like `.head()`, `.tail()`, `.unique().compute()`, saving to file).

```python
# --- Code Example 2: Dask DataFrame Operations (Lazy) ---
# (Continuing from previous cell, assumes `ddf` exists conceptually 
#  or recreated quickly for standalone example)
import dask.dataframe as dd
import pandas as pd
import numpy as np
import time
import os

print("\nPerforming Operations on Dask DataFrames (Lazy Evaluation):")

# Recreate dummy data for standalone execution if needed
temp_csv_dir = "temp_csv_data_ops"
os.makedirs(temp_csv_dir, exist_ok=True)
n_files = 5; n_rows_per_file = 1000
for i in range(n_files):
    fname = os.path.join(temp_csv_dir, f"ops_data_{i}.csv"); 
    pd.DataFrame({'id':range(i*n_rows_per_file,(i+1)*n_rows_per_file), 'x':np.random.rand(n_rows_per_file), 'category':np.random.choice(['A','B'],n_rows_per_file)}).to_csv(fname,index=False)

# Load Dask DataFrame
ddf = dd.read_csv(os.path.join(temp_csv_dir, "ops_data_*.csv"))
print(f"Loaded Dask DataFrame with {ddf.npartitions} partitions.")

# --- Define Lazy Operations ---
print("\nDefining lazy operations...")
# 1. Filter rows
filtered_ddf = ddf[ddf.x > 0.5]
print("  Defined filter: ddf[ddf.x > 0.5]")

# 2. Create a new column
new_col_ddf = filtered_ddf.assign(x_squared = filtered_ddf.x ** 2)
print("  Defined new column: .assign(x_squared = ...)")

# 3. Group by category and calculate mean x_squared
# This involves shuffling data between partitions
grouped_means_lazy = new_col_ddf.groupby('category')['x_squared'].mean()
print("  Defined groupby and mean (lazy): ...groupby('category').mean()")

# --- Trigger Computation ---
print("\nTriggering computation with .compute()...")
start_time = time.time()
# Configure scheduler if needed (e.g., dask.config.set(scheduler='processes'))
final_means_pandas = grouped_means_lazy.compute() # Result is a Pandas Series
end_time = time.time()

print(f"\nComputation finished. Time: {end_time - start_time:.3f}s")
print("Final Mean x_squared per category:")
print(final_means_pandas)

# Cleanup
if os.path.exists(temp_csv_dir): shutil.rmtree(temp_csv_dir)
print(f"\nCleaned up '{temp_csv_dir}'.")
print("-" * 20)

# Explanation:
# 1. Creates dummy CSVs and loads them into a Dask DataFrame `ddf`.
# 2. Defines several operations lazily: filtering rows where x > 0.5 (`filtered_ddf`), 
#    adding a new column `x_squared` using `.assign()`, and then grouping by 'category' 
#    and calculating the mean of `x_squared` for each group (`grouped_means_lazy`).
# 3. Calling `.compute()` on `grouped_means_lazy` executes the entire graph. Dask 
#    applies the filter and assign operations per partition in parallel. The `groupby().mean()` 
#    requires shuffling data between workers before the final aggregation. The result 
#    returned is a standard Pandas Series containing the mean `x_squared` for each category.
```

Dask DataFrame allows scaling familiar Pandas data wrangling and analysis techniques to tabular datasets that would overwhelm a single machine's memory. It is particularly powerful when reading from efficient columnar formats like Parquet and performing operations that can be parallelized across row partitions. It's a key tool for analyzing large astronomical catalogs or time-series datasets within the Python ecosystem.

**67.6 Using the Dask Distributed Scheduler and Dashboard**

While Dask's local schedulers (`threads`, `processes`) are excellent for utilizing multiple cores on a single machine, **scaling analyses to multiple nodes** in an HPC cluster or cloud environment requires the **Dask Distributed scheduler**. This scheduler provides a more sophisticated, fault-tolerant, and feature-rich environment for managing parallel task execution across a network of worker processes. A key component of `dask.distributed` is its informative **diagnostic dashboard**, usually accessed via a web browser.

**Architecture:** The distributed scheduler setup typically involves:
1.  **Scheduler Process (`dask-scheduler`):** A central process that manages the task graph, assigns tasks to workers, tracks progress, and handles communication. Usually runs on one designated node.
2.  **Worker Processes (`dask-worker`):** One or more processes that connect to the scheduler and execute assigned tasks. They typically run on the compute nodes of the cluster, utilizing the cores and memory of those nodes. Workers store intermediate results in their local memory and communicate directly with each other when necessary to transfer data dependencies (peer-to-peer data transfer).
3.  **Client (`distributed.Client`):** The interface within your Python script or interactive session. You create a `Client` object, providing the address of the running scheduler. The client submits task graphs (generated by Dask collection operations or `client.submit`/`client.map`) to the scheduler and retrieves final results.

**Setup:**
*   **Manual:** Start `dask-scheduler` on one node (it prints its address, e.g., `tcp://192.168.1.100:8786`). Start `dask-worker <scheduler-address>` on multiple compute nodes. Then, in your Python script, `client = Client('tcp://192.168.1.100:8786')`.
*   **Using `dask-jobqueue`:** For HPC clusters with SLURM, PBS, etc., this library simplifies setup. You define a cluster object (`SLURMCluster`, `PBSCluster`) specifying worker resource requirements (cores, memory, walltime, queue) in Python. Calling `.scale(N)` on the cluster object submits N batch jobs to the scheduler, each starting a `dask-worker` that automatically connects back to the scheduler (whose address is managed internally). You then create a `Client` connected to this cluster object: `client = Client(cluster)`. This automates worker deployment via the HPC job queue.

```python
# --- Code Example 1: Conceptual dask-jobqueue Setup for SLURM ---
# Note: Requires dask, distributed, dask-jobqueue. Run within Python on HPC login node.
try:
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client, progress
    dask_jq_installed = True
except ImportError:
    dask_jq_installed = False
    print("NOTE: dask[distributed] or dask-jobqueue not installed. Skipping example.")

print("Conceptual Dask Cluster Setup using dask-jobqueue (SLURM):")

if dask_jq_installed:
    # Define worker resources and cluster configuration
    print("\nDefining SLURMCluster configuration...")
    cluster = SLURMCluster(
        queue='compute',         # SLURM partition/queue name
        cores=24,                # Cores per worker job
        memory='100 GB',         # Memory per worker job
        # project='astro_project', # Your allocation account
        walltime='01:00:00',     # Max time per worker job
        # Use interface providing network address visible to workers
        # interface='ib0',         # Example: InfiniBand interface
        # Specify local directory for worker logs/files (must be accessible by nodes)
        local_directory='$SCRATCH/dask-worker-space', 
        # Optional: Load environment modules within worker jobs
        job_extra_directives=['module load python/3.9', 'source myenv/bin/activate'] 
    )
    print("SLURMCluster object created (workers not started yet).")

    # --- Scale the Cluster (Request Workers) ---
    num_workers = 4 # Request 4 worker jobs (each using 24 cores as defined above)
    print(f"\nRequesting {num_workers} worker jobs via SLURM...")
    # This submits SLURM batch jobs to start the dask-worker processes
    cluster.scale(jobs=num_workers) 
    # Or adaptively scale: cluster.adapt(minimum_jobs=1, maximum_jobs=10)

    # --- Connect Client ---
    print("Connecting Dask Client to the cluster...")
    try:
        client = Client(cluster)
        print("\nDask Client connected to SLURMCluster:")
        print(client) # Shows scheduler address and link to dashboard
        # Wait for some workers to start (optional)
        # client.wait_for_workers(n_workers=2, timeout="60s") 
        # print(f"Workers ready: {len(client.scheduler_info()['workers'])}")

        # --- Now use the client to submit work (as in Sec 40.5 Example 2) ---
        print("\nClient ready. Submit computations using client.submit, client.map,")
        print("or use Dask collections (Array, DataFrame) - they will use this client.")
        # Example: future = client.submit(my_function, arg1, arg2)
        #          results = client.map(my_function, list_of_args)
        #          dask_array.mean().compute() # Will use the client's scheduler

        # --- Remember to close the client and cluster when done ---
        # print("\nClosing client and shutting down cluster workers...")
        # client.close()
        # cluster.close() # Stops the SLURM worker jobs
        # print("Client and cluster closed.")
        
    except Exception as e_client:
        print(f"Error connecting client or running work: {e_client}")
        # Ensure cluster is closed on error if started
        # cluster.close()
        
else:
    print("Skipping dask-jobqueue example.")

print("-" * 20)

# Explanation: This code conceptually demonstrates setting up a Dask cluster on SLURM.
# 1. Imports `SLURMCluster` and `Client`.
# 2. Creates a `SLURMCluster` instance, specifying resources required *per worker job* 
#    (queue, cores, memory, walltime), account information, network interface, and 
#    potentially environment setup commands for workers.
# 3. `cluster.scale(jobs=4)` tells `dask-jobqueue` to submit 4 SLURM jobs, each requesting 
#    the specified resources. Each job starts a `dask-worker` process that connects 
#    back to the scheduler managed by the `cluster` object.
# 4. `client = Client(cluster)` connects the main Python script to this dynamically 
#    created cluster. The client object provides the scheduler address and dashboard link.
# 5. Subsequent Dask computations submitted via this client will be executed across the 
#    workers managed by the SLURM jobs.
# 6. Conceptual cleanup (`client.close()`, `cluster.close()`) stops the workers/jobs.
# This shows how `dask-jobqueue` bridges Dask with HPC schedulers.
```

**The Dask Dashboard:** A major advantage of the distributed scheduler is its integrated **web dashboard**, typically accessible via a specific port on the scheduler node (e.g., `http://scheduler_ip:8787`). This dashboard provides real-time visualization and diagnostics:
*   **Task Stream:** Shows tasks as they are scheduled, transferred, executed, and completed across different workers and threads. Helps identify bottlenecks or long-running tasks.
*   **Worker Status:** Lists connected workers, their memory usage, CPU utilization, number of tasks executed, and data stored. Helps monitor worker health and resource usage.
*   **Task Graph Visualization:** Can display the DAG for submitted computations, showing dependencies and progress.
*   **Memory Usage Plots:** Tracks total and per-worker memory usage over time, helping diagnose memory leaks or identify memory-intensive tasks.
*   **System Information:** Provides details about the cluster environment.
The dashboard is invaluable for understanding *how* Dask is executing your computation in parallel, optimizing performance, and debugging issues related to scheduling, communication, or resource limitations in a distributed setting.

Using the Dask distributed scheduler, often facilitated by tools like `dask-jobqueue` on HPC systems, allows Python users to scale their NumPy, Pandas, Scikit-learn, and custom Python analyses seamlessly from a single core to potentially thousands of cores across a cluster, managing complex task dependencies and providing rich diagnostic feedback through its dashboard. It represents a powerful and increasingly popular approach for large-scale data analysis in the Python ecosystem.

---

**Application 67.A: Processing Large FITS Images with Dask Array**

**(Paragraph 1)** **Objective:** This application demonstrates how to use `dask.array` (Sec 67.2) to perform basic image processing (e.g., calculating statistics, applying a simple filter) on a large FITS image file that might be too large to fit into memory on a single machine, leveraging Dask's chunking and parallel execution capabilities on a multi-core machine. Reinforces Sec 67.2, 40.4.

**(Paragraph 2)** **Astrophysical Context:** Modern astronomical surveys produce increasingly large images or image mosaics (e.g., from VISTA, VST, DES, HSC, Euclid, Roman). Loading an entire multi-gigabyte FITS image into memory for processing with NumPy can be infeasible on standard workstations. Dask Array allows performing NumPy-like operations on such images by loading and processing them in smaller, manageable chunks in parallel. Tasks like calculating global image statistics (mean, median, std dev), applying simple filters, or performing background estimation can be scaled using Dask.

**(Paragraph 3)** **Data Source:** A single, large FITS image file (`large_image.fits`). This should ideally be multi-gigabyte to demonstrate the benefit, but we will simulate creating a large-ish HDF5 file (as Dask integrates well with chunked HDF5) representing the image data for this example if a real large FITS isn't readily available. HDF5 allows chunked access which Dask leverages.

**(Paragraph 4)** **Modules Used:** `dask.array` as `da`, `h5py` (to create/read chunked HDF5 simulating large FITS), `numpy` as `np`, `time`. `dask.distributed.Client` or `dask.config` to manage local parallelism. `astropy.io.fits` would be used conceptually if directly reading FITS (though less efficient for chunking than HDF5/Zarr).

**(Paragraph 5)** **Technique Focus:** Using Dask Array for out-of-core and parallel array processing. (1) Creating a Dask Array representing the large image, either directly from a chunked HDF5 dataset using `da.from_array(h5_dataset, chunks=...)` or conceptually from a large FITS file (which might involve more complex chunking setup). (2) Performing standard NumPy-like operations lazily on the Dask Array (e.g., `dask_image.mean()`, `dask_image.std()`, `dask_image * 2 + 100`). (3) Applying a simple filter operation, like a uniform filter, potentially using `dask_image.map_blocks()` or `dask_image.map_overlap()` which apply a function to each chunk (with overlap for filters needing neighbors). (4) Triggering computation using `.compute()` with a parallel scheduler (`threads` or `processes`). (5) Comparing execution time and memory usage (conceptually) to attempting the same operation with NumPy on the full dataset (which might fail).

**(Paragraph 6)** **Processing Step 1: Prepare Large Data File:** Create a large HDF5 file containing a 2D dataset with chunking enabled. This simulates having a large image file accessible in a chunkable format.

```python
# --- Setup: Create Large Dummy HDF5 File ---
import h5py
import numpy as np
import os
import dask.array as da # Need da here too

large_filename = 'large_image_dask.hdf5'
dset_name = 'image'
N_large = 8192 # e.g., 8k x 8k image
chunk_size = 512 # Chunk size

if not os.path.exists(large_filename):
     print(f"Creating large dummy HDF5 file '{large_filename}' ({N_large}x{N_large})...")
     with h5py.File(large_filename, 'w') as f:
          dset = f.create_dataset(dset_name, shape=(N_large, N_large), dtype='f4', 
                                  chunks=(chunk_size, chunk_size))
          # Write data chunk by chunk to simulate large file creation
          for r in range(0, N_large, chunk_size):
               for c in range(0, N_large, chunk_size):
                    dset[r:r+chunk_size, c:c+chunk_size] = np.random.rand(chunk_size, chunk_size)
     print("Dummy file created.")
else:
     print(f"Using existing file: '{large_filename}'")
# ---------------------------------------------
```

**(Paragraph 7)** **Processing Step 2: Create Dask Array from File:** Open the HDF5 file using `h5py`. Get the dataset object. Create the Dask Array using `da.from_array(h5_dataset, chunks=h5_dataset.chunks)` which tells Dask to align its chunks with the HDF5 file's chunks for efficient lazy loading.

**(Paragraph 8)** **Processing Step 3: Define Lazy Computations:** Define operations on the Dask Array `dask_image`:
    *   Calculate global mean: `global_mean = dask_image.mean()`
    *   Apply a simple operation: `scaled_image = (dask_image - global_mean) / dask_image.std()` (Note: `.std()` also triggers computation graph)
    *   Conceptual simple smoothing using block processing (more complex filters need `map_overlap`): `smoothed_image = dask_image.map_blocks(lambda block: block - np.mean(block), dtype=dask_image.dtype)` (Example: Subtract local mean per block).

**(Paragraph 9)** **Processing Step 4: Execute with Parallel Scheduler:** Use `dask.config.set(scheduler='processes')` (or `'threads'`) or set up a `dask.distributed.Client` for local parallel execution. Trigger computations using `.compute()`. Use `ProgressBar` for feedback. Time the execution.

**(Paragraph 10)** **Processing Step 5: Analyze Results/Timing:** Print the calculated global mean. Compare the Dask execution time with the time it would hypothetically take (or error out) loading and processing the entire array with NumPy. Discuss how Dask handled the large size by operating chunk-wise in parallel.

**Output, Testing, and Extension:** Output includes timing results for Dask computations and the calculated statistics (e.g., global mean). **Testing:** Verify the results computed by Dask match those computed by NumPy on a smaller version of the data that fits in memory. Check memory usage during Dask computation (should remain relatively low compared to loading the whole array). Test different chunk sizes and local schedulers. **Extensions:** (1) Implement a more realistic filter (e.g., Gaussian blur) using `dask_image.map_overlap` to handle boundary data between chunks correctly. (2) Read data directly from a large multi-extension FITS file using Dask Array (might require custom chunking logic). (3) Perform more complex analysis chains involving filtering, thresholding, and statistical calculation using Dask Array operations. (4) Use `dask_image.to_zarr()` or `dask_image.to_hdf5()` to save processed Dask arrays back to disk efficiently. (5) Connect to a distributed Dask cluster and run the analysis on multiple nodes.

```python
# --- Code Example: Application 67.A ---
# Note: Requires dask, h5py, numpy. Assumes dummy file created above.
import dask.array as da
import dask # For config
from dask.diagnostics import ProgressBar
import h5py
import numpy as np
import time
import os

print("Processing Large Image with Dask Array:")

large_filename = 'large_image_dask.hdf5'
dset_name = 'image'

if not os.path.exists(large_filename):
    print(f"Error: Data file '{large_filename}' not found. Please run setup step first.")
    dask_array_available = False
else:
    dask_array_available = True

if dask_array_available:
    # --- Step 2: Create Dask Array from HDF5 ---
    print(f"\nCreating Dask Array from '{large_filename}'...")
    try:
        h5f = h5py.File(large_filename, 'r')
        h5_dataset = h5f[dset_name]
        # Use chunks defined in the HDF5 file
        dask_image = da.from_array(h5_dataset, chunks=h5_dataset.chunks)
        print("Dask Array created:")
        print(dask_image)
    except Exception as e:
        print(f"Error creating Dask array: {e}")
        dask_image = None

    if dask_image is not None:
        # --- Step 3: Define Lazy Computations ---
        print("\nDefining computations...")
        # 1. Global Mean (reduction across all chunks)
        global_mean = dask_image.mean()
        print(f"  Defined global mean: {global_mean}")
        
        # 2. Simple filter: Subtract local mean per block
        # map_blocks applies a function independently to each chunk/block
        # Note: This doesn't handle block boundaries correctly for proper filters!
        smoothed_image = dask_image.map_blocks(lambda block: block - np.mean(block), 
                                               dtype=dask_image.dtype)
        print(f"  Defined smoothed image (local mean subtraction): {smoothed_image}")
        # Calculate std dev of smoothed image
        smoothed_std = smoothed_image.std()
        print(f"  Defined smoothed image std dev: {smoothed_std}")

        # --- Step 4: Execute with Parallel Scheduler ---
        # Configure local scheduler (e.g., processes)
        # n_workers = os.cpu_count()
        # dask.config.set(scheduler='processes', num_workers=n_workers)
        # print(f"\nExecuting using '{dask.config.get('scheduler')}' scheduler ({n_workers} workers)...")
        print(f"\nExecuting using default scheduler (often 'sync' or 'threads')...")
        
        print("Computing global mean...")
        start_mean = time.time()
        with ProgressBar():
            mean_val = global_mean.compute()
        end_mean = time.time()
        print(f"  Global Mean = {mean_val:.4f} (Time: {end_mean-start_mean:.3f}s)")
        
        print("\nComputing std dev of smoothed image...")
        start_std = time.time()
        with ProgressBar():
             std_val = smoothed_std.compute()
        end_std = time.time()
        print(f"  Smoothed Std Dev = {std_val:.4f} (Time: {end_std-start_std:.3f}s)")

        # Optional: Try loading full array with NumPy for comparison (will likely fail/be slow)
        # print("\nAttempting NumPy load (may fail or be very slow)...")
        # start_np = time.time()
        # try:
        #     np_image = h5_dataset[:] # Read whole dataset
        #     np_mean = np.mean(np_image)
        #     end_np = time.time()
        #     print(f"  NumPy mean = {np_mean:.4f} (Time: {end_np - start_np:.3f}s)")
        # except MemoryError:
        #     print("  MemoryError: NumPy could not load the full array.")
        # except Exception as e_np:
        #     print(f"  NumPy failed: {e_np}")

    # Close HDF5 file handle
    if 'h5f' in locals() and h5f: h5f.close()

# --- Cleanup ---
if os.path.exists(large_filename): 
    os.remove(large_filename)
    print(f"\nCleaned up {large_filename}.")

print("-" * 20)
```

**Application 67.B: Parallel Group-By Operations on TESS Catalogs with Dask DataFrame**

**(Paragraph 1)** **Objective:** This application demonstrates using `dask.dataframe` (Sec 67.3) to perform a common catalog analysis task – a `groupby()` aggregation – in parallel on a large dataset that potentially doesn't fit into memory. We will simulate reading a large catalog of TESS Object of Interest (TOI) properties distributed across multiple files and use Dask DataFrame to calculate the median transit duration per stellar effective temperature bin across the entire dataset.

**(Paragraph 2)** **Astrophysical Context:** Analyzing large catalogs from missions like TESS or Kepler often involves grouping objects based on certain properties (like host star type, temperature, metallicity, or galactic population) and then calculating aggregate statistics (mean, median, counts, standard deviation) for other properties within those groups. For example, studying how exoplanet properties (like radius, period, duration) vary with host star properties requires grouping by stellar parameters (T<0xE1><0xB5><0x8A><0xE1><0xB5><0x93><0xE1><0xB5><0x93>) and aggregating planet parameters. Dask DataFrame allows performing these `groupby().agg()` operations efficiently on catalogs too large for Pandas on a single machine.

**(Paragraph 3)** **Data Source:** Multiple CSV or Parquet files containing a simulated catalog of planet candidates (e.g., TOIs). Each file represents a partition of the full catalog. The catalog should contain columns like `tic_id`, `planet_number`, `transit_period`, `transit_duration`, `stellar_Teff`, `stellar_logg`, etc. We will simulate creating these partitioned files.

**(Paragraph 4)** **Modules Used:** `dask.dataframe` as `dd`, `pandas` as `pd` (for creating dummy data and for the final result type), `numpy` as `np`, `os`, `shutil` (for cleanup), `dask.diagnostics.ProgressBar`.

**(Paragraph 5)** **Technique Focus:** Using Dask DataFrame for scalable tabular analysis. (1) Creating partitioned data on disk (e.g., multiple CSV or preferably Parquet files). (2) Creating a Dask DataFrame pointing to these files using `dd.read_csv()` or `dd.read_parquet()`. (3) Performing standard Pandas-like operations lazily: selecting columns, potentially binning a continuous variable (like `stellar_Teff`) into categories using `dd.to_numeric` / `pd.cut` via `map_partitions`. (4) Performing a `groupby()` operation on a categorical column (e.g., Teff bins). (5) Calculating an aggregation (e.g., `.median()`) on another column (`transit_duration`) within each group. (6) Triggering the computation using `.compute()`, which involves shuffling data between partitions for the groupby. (7) Analyzing the resulting Pandas Series/DataFrame containing the aggregated statistics per group.

**(Paragraph 6)** **Processing Step 1: Prepare Partitioned Data:** Create a temporary directory. Generate several Pandas DataFrames containing simulated TOI data (TIC ID, Teff, Duration, etc.) and save each as a separate CSV or Parquet file within the directory. Ensure Teff and Duration cover a reasonable range. Parquet format (`df.to_parquet(...)`, requires `pyarrow` or `fastparquet`) is generally much more efficient for Dask DataFrames than CSV.

**(Paragraph 7)** **Processing Step 2: Create Dask DataFrame:** Use `ddf = dd.read_parquet(temp_dir + '/')` or `ddf = dd.read_csv(temp_dir + '/toi_*.csv')` to create the Dask DataFrame representing the entire distributed catalog. Print `ddf` and `ddf.npartitions`.

**(Paragraph 8)** **Processing Step 3: Define Lazy Computation (Groupby-Aggregation):**
    *   Define temperature bins: `temp_bins = [3000, 4000, 5000, 6000, 7000]` and labels `temp_labels = ['<4k', '4-5k', '5-6k', '6-7k']`.
    *   Create a new column 'Teff_bin' by applying `pd.cut` to the `stellar_Teff` column. Since `cut` needs bin edges, it might need to be applied per partition using `ddf.map_partitions(lambda df: df.assign(Teff_bin=pd.cut(df['stellar_Teff'], bins=temp_bins, labels=temp_labels)), meta=...)`. Define the `meta` (expected output structure/dtypes) carefully.
    *   Perform the groupby and aggregation: `median_duration_by_temp = ddf.groupby('Teff_bin')['transit_duration'].median(split_every=...)`. The `median` is an approximate calculation in Dask by default; `split_every` can control intermediate aggregation steps for large groups.

**(Paragraph 9)** **Processing Step 4: Execute Computation:** Use `with ProgressBar(): result_series = median_duration_by_temp.compute()`. This triggers the reading of partitions, the `map_partitions` for binning, the expensive shuffle operation for the `groupby`, the calculation of median within each group on workers, and final aggregation. The result `result_series` will be a Pandas Series indexed by the Teff bins.

**(Paragraph 10)** **Processing Step 5: Display Results:** Print the resulting Pandas Series showing the median transit duration for each stellar temperature bin. Interpret the result qualitatively (e.g., do hotter stars show different median durations in this simulated data?).

**Output, Testing, and Extension:** Output includes the final Pandas Series with median durations per Teff bin, along with progress messages from Dask. **Testing:** Verify the calculation completes. Check the results are plausible (correct bins, reasonable median values). Manually calculate the median duration for a small subset of the data corresponding to one bin and compare with the Dask result. Test performance by varying the number of partitions or running with different Dask schedulers. **Extensions:** (1) Use `dd.read_parquet` for better performance if data saved in Parquet. (2) Calculate multiple aggregations simultaneously using `.agg(['mean', 'median', 'std', 'count'])`. (3) Perform a more complex analysis, like grouping by Teff bin and Planet Period bin, calculating median duration in each 2D bin. (4) Connect to a distributed Dask cluster using `dask.distributed.Client` and run the groupby on a truly large dataset spread across multiple nodes.

```python
# --- Code Example: Application 67.B ---
# Note: Requires dask[dataframe], pandas, numpy, pyarrow or fastparquet
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import pandas as pd
import numpy as np
import os
import shutil
import time

print("Parallel GroupBy on TESS Catalog with Dask DataFrame:")

# Step 1: Prepare Partitioned Data (Parquet is preferred)
temp_parquet_dir = "temp_toi_parquet"
n_partitions = 8
n_rows_total = 500000
n_rows_per_partition = n_rows_total // n_partitions

print(f"\nCreating {n_partitions} dummy Parquet files in '{temp_parquet_dir}'...")
os.makedirs(temp_parquet_dir, exist_ok=True)
# Ensure pyarrow or fastparquet is installed: pip install pyarrow
for i in range(n_partitions):
    fname = os.path.join(temp_parquet_dir, f"toi_part_{i}.parquet")
    df_part = pd.DataFrame({
        'tic_id': np.random.randint(1e8, 1e9, size=n_rows_per_partition),
        'transit_duration_hours': np.random.uniform(1, 8, size=n_rows_per_partition),
        'stellar_Teff': np.random.normal(5500, 1000, size=n_rows_per_partition).clip(3000, 8000)
    })
    df_part.to_parquet(fname)
print("Dummy Parquet files created.")

# Step 2: Create Dask DataFrame
print("\nCreating Dask DataFrame from Parquet directory...")
try:
    # read_parquet automatically handles partitioning from directory structure
    ddf = dd.read_parquet(temp_parquet_dir)
    print(f"Dask DataFrame created with {ddf.npartitions} partitions.")
    print(" Columns:", list(ddf.columns))
    # print(" DTypes:\n", ddf.dtypes) # Compute dtypes requires reading sample
except Exception as e:
    print(f"Error reading parquet: {e}. Ensure pyarrow or fastparquet is installed.")
    ddf = None

if ddf is not None:
    # Step 3: Define Lazy Computation (Bin Teff, Groupby, Median)
    print("\nDefining lazy groupby aggregation...")
    # Define bins and labels for Teff
    temp_bins = [3000, 4000, 5000, 6000, 7000, 8000]
    temp_labels = ['3-4k K', '4-5k K', '5-6k K', '6-7k K', '7-8k K']
    
    # Add the 'Teff_bin' column using map_partitions
    # Need meta to define output structure/dtypes
    meta_with_bin = pd.DataFrame({
        'tic_id': pd.Series([], dtype='int64'),
        'transit_duration_hours': pd.Series([], dtype='float64'),
        'stellar_Teff': pd.Series([], dtype='float64'),
        'Teff_bin': pd.Series([], dtype='category') # Specify category dtype
    })
    ddf_binned = ddf.map_partitions(
        lambda df: df.assign(
            Teff_bin=pd.cut(df['stellar_Teff'], bins=temp_bins, labels=temp_labels, right=False)
        ), 
        meta=meta_with_bin
    )
    print("  Added 'Teff_bin' column definition (lazy).")

    # Define the groupby aggregation
    # Median is approximate in Dask by default
    median_duration_lazy = ddf_binned.groupby('Teff_bin')['transit_duration_hours'].median(split_out=4) 
    # split_out controls intermediate partitions during shuffle for median
    print("  Defined groupby('Teff_bin')['transit_duration_hours'].median() (lazy).")

    # Step 4: Execute Computation
    print("\nTriggering computation with .compute()...")
    # Configure scheduler if desired (e.g., multiple processes)
    # import dask
    # dask.config.set(scheduler='processes') 
    start_time = time.time()
    with ProgressBar():
        result_pandas_series = median_duration_lazy.compute()
    end_time = time.time()
    print(f"\nComputation finished. Time taken: {end_time - start_time:.3f}s")

    # Step 5: Display Results
    print("\n--- Median Transit Duration per Stellar Teff Bin ---")
    print(result_pandas_series)

# Cleanup
finally: # Ensure cleanup happens
    if os.path.exists(temp_parquet_dir): 
        shutil.rmtree(temp_parquet_dir)
        print(f"\nCleaned up '{temp_parquet_dir}'.")

print("-" * 20)
```

**Chapter 67 Summary**

This chapter provided a focused overview of the **Dask** library as a powerful Python-native tool for **scalable data analysis**, building upon concepts introduced earlier. It revisited Dask's fundamental principles of **lazy evaluation** – where operations build a **task graph** rather than executing immediately – and intelligent **task scheduling**, which allows for optimization and parallel execution of the graph. The core parallel data collections were explored in more detail: **Dask Array** (`dask.array`) for scaling NumPy-like operations on chunked, potentially larger-than-memory arrays; **Dask DataFrame** (`dask.dataframe`) for scaling Pandas-like operations on partitioned tabular data, including efficient reading from formats like Parquet; and **Dask Bag** (`dask.bag`) for parallel processing of generic Python items using functional programming paradigms (`map`, `filter`, `fold`, `groupby`), suitable for task parallelism on semi-structured data.

The different **Dask schedulers** controlling execution were re-examined: the local synchronous scheduler (`'sync'`) for debugging; the local threaded scheduler (`'threads'`) effective for I/O-bound tasks or code releasing the Python Global Interpreter Lock (GIL); the local multiprocessing scheduler (`'processes'`) which bypasses the GIL for CPU-bound tasks on a single machine using separate processes; and the highly scalable **distributed scheduler** (`dask.distributed`). The architecture of the distributed scheduler (central scheduler process, multiple worker processes across nodes, client interface) and its associated diagnostic **web dashboard** were described, along with helper libraries like `dask-jobqueue` for simplifying cluster deployment on HPC systems with schedulers like SLURM or PBS. Two applications illustrated these concepts: using Dask Array for chunked processing (calculating statistics) of a simulated large FITS/HDF5 image, demonstrating out-of-core and parallel computation on a single node; and using Dask DataFrame to perform a parallel `groupby().median()` aggregation on a large, partitioned catalog of simulated TESS planet candidates, showcasing scalable tabular analysis.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Dask Development Team. (n.d.).** *Dask Documentation*. Dask. Retrieved January 16, 2024, from [https://docs.dask.org/en/latest/](https://docs.dask.org/en/latest/)
    *(The essential reference, covering Dask concepts, data collections (Array, DataFrame, Bag), schedulers, distributed computing, APIs, and examples.)*

2.  **Rocklin, M. (2015).** Dask: Parallel Computation with Blocked algorithms and Task Scheduling. In *Proceedings of the 14th Python in Science Conference (SciPy 2015)* (pp. 130–136). [https://doi.org/10.25080/Majora-7b98e3ed-013](https://doi.org/10.25080/Majora-7b98e3ed-013)
    *(The original conference paper introducing Dask.)*

3.  **Dask Development Team. (n.d.).** *Dask Examples*. Dask Examples. Retrieved January 16, 2024, from [https://examples.dask.org/](https://examples.dask.org/)
    *(A collection of practical examples and tutorials demonstrating various Dask use cases, including array analysis, dataframe manipulation, machine learning, and distributed computing setups.)*

4.  **Dask Development Team. (n.d.).** *Dask-Jobqueue Documentation*. Dask Jobqueue. Retrieved January 16, 2024, from [https://jobqueue.dask.org/en/latest/](https://jobqueue.dask.org/en/latest/)
    *(Documentation for the library that simplifies deploying Dask workers on HPC clusters using job schedulers like SLURM, PBS, SGE, LSF.)*

5.  **VanderPlas, J. (2016).** *Python Data Science Handbook*. O'Reilly Media. ([Chapter 5 on Machine Learning](https://jakevdp.github.io/PythonDataScienceHandbook/05.00-machine-learning.html))
    *(While focused on Scikit-learn, this book provides excellent context on the types of data manipulation (using Pandas, covered in Ch 2) and analysis tasks that libraries like Dask aim to scale.)* (Note: Finding references specifically comparing Dask performance *in astrophysics* might require searching recent conference proceedings or specialized papers.)
