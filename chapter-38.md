**Chapter 38: Parallel Programming Fundamentals**

Having introduced High-Performance Computing (HPC) environments in the previous chapter, we now delve into the fundamental concepts and basic techniques of **parallel programming** required to effectively utilize these resources. The goal of parallel programming is to decompose a computational task into smaller parts that can be executed concurrently on multiple processors or cores, thereby reducing the overall execution time. This chapter lays the conceptual groundwork for the more advanced parallel techniques discussed later in Part VII. We begin by defining key concepts like **parallelism versus concurrency**, measures of parallel performance (**speedup, efficiency**), and the theoretical limitation imposed by **Amdahl's Law**. We differentiate between **task parallelism** (executing independent tasks simultaneously) and **data parallelism** (operating on different subsets of data simultaneously), the latter being dominant in large scientific simulations. We then introduce Python's built-in capabilities for achieving parallelism on a single, multi-core machine: **process-based parallelism** using the **`multiprocessing`** module (including the convenient `Pool` object), suitable for CPU-bound tasks, and **thread-based parallelism** using the **`threading`** module, often better suited for I/O-bound tasks due to Python's Global Interpreter Lock (GIL). Finally, we provide conceptual introductions to the dominant paradigms used for large-scale distributed memory parallelism on HPC clusters: the **Message Passing Interface (MPI)** standard for inter-node communication and **OpenMP** directives for shared-memory parallelism within a node (primarily for compiled languages).

**38.1 Concepts: Parallelism vs. Concurrency, Speedup, Efficiency, Amdahl's Law**

Before diving into specific programming techniques, it's essential to clarify some fundamental concepts related to executing multiple tasks or operations seemingly at the same time. A key distinction is between **parallelism** and **concurrency**. **Parallelism** refers to the *simultaneous* execution of multiple computations, typically achieved by distributing tasks across multiple physical processing units (CPU cores, GPUs, compute nodes). The primary goal of parallelism is to *speed up* computation by doing more work in the same amount of time. **Concurrency**, on the other hand, deals with managing multiple tasks that are *making progress* over overlapping time periods, even if they aren't executing simultaneously at every instant (e.g., on a single-core CPU switching rapidly between tasks). Concurrency focuses on structuring a program to handle multiple tasks logically, often important for responsiveness in interactive applications or managing I/O operations, but it doesn't necessarily imply faster execution of a single computational problem. Our main focus in HPC is achieving true parallelism for computational speedup.

To quantify the benefit of parallel execution, several metrics are used. The most basic is **Speedup (S<0xE1><0xB5><0x96>)**. It measures how much faster a program runs on `p` processors compared to a single processor. It's defined as the ratio of the serial execution time (T₁) to the parallel execution time (T<0xE1><0xB5><0x96>):
S<0xE1><0xB5><0x96> = T₁ / T<0xE1><0xB5><0x96>
Ideally, we hope for **linear speedup**, where S<0xE1><0xB5><0x96> = p. This would mean doubling the number of processors halves the execution time. However, perfect linear speedup is rarely achieved in practice due to various overheads.

**Parallel Efficiency (E<0xE1><0xB5><0x96>)** measures how effectively the processors are utilized. It's defined as the speedup per processor:
E<0xE1><0xB5><0x96> = S<0xE1><0xB5><0x96> / p = T₁ / (p * T<0xE1><0xB5><0x96>)
Efficiency ranges from 0 to 1 (or 0% to 100%). An efficiency of 1 corresponds to perfect linear speedup. Lower efficiency indicates that adding more processors provides diminishing returns, possibly due to communication overhead, load imbalance, or sequential parts of the algorithm. Analyzing how speedup and efficiency change as `p` increases (scalability analysis) is crucial for understanding a parallel program's performance.

A fundamental theoretical limitation on speedup is described by **Amdahl's Law**. It considers a program where a fraction `f` of the total execution time (on a single processor) is spent on inherently **sequential** parts of the code that cannot be parallelized, while the remaining fraction `(1-f)` can be perfectly parallelized across `p` processors. Amdahl's Law states that the maximum possible speedup S<0xE1><0xB5><0x96> is limited by:
S<0xE1><0xB5><0x96> ≤ 1 / [ f + (1-f)/p ]
As the number of processors `p` becomes very large, the term `(1-f)/p` approaches zero, and the speedup approaches `1/f`. This means that the **sequential fraction `f` ultimately limits the achievable speedup**. For example, if even 10% of a program is inherently sequential (f = 0.1), the maximum possible speedup, even with an infinite number of processors, is only 1 / 0.1 = 10x. This highlights the critical importance of minimizing the sequential portions and maximizing the parallelizable fraction of any algorithm intended for large-scale parallel execution.

Factors preventing perfect linear speedup and reducing efficiency include:
*   **Sequential Bottlenecks:** Parts of the code that inherently cannot be parallelized (Amdahl's Law).
*   **Communication Overhead:** Time spent sending data between processors (e.g., via MPI messages) or accessing shared memory with necessary synchronization (locks, barriers). This overhead often increases with the number of processors.
*   **Load Imbalance:** Some processors might have significantly more work to do than others, leading to idle time while waiting for the slowest processor to finish its task in a synchronized step.
*   **Synchronization Costs:** Waiting for all processors to reach a certain point before proceeding (e.g., barriers in shared memory, collective operations in MPI).
*   **Redundant Computation:** Sometimes parallel algorithms require processors to perform extra computations compared to the serial version.

Understanding these concepts – parallelism vs. concurrency, speedup, efficiency, Amdahl's Law, and the sources of parallel overhead – provides the necessary foundation for designing effective parallel algorithms, interpreting performance measurements, and appreciating the challenges involved in scaling applications to massively parallel HPC systems. The goal is always to maximize the parallelizable fraction, minimize communication and synchronization, and ensure good load balance to approach linear speedup as closely as possible within the constraints of the problem and the hardware.

```python
# --- Code Example: Illustrating Amdahl's Law Concept ---

import numpy as np
import matplotlib.pyplot as plt

print("Illustrating Amdahl's Law:")

# Define sequential fraction (f)
# Example 1: 5% sequential part
f1 = 0.05 
# Example 2: 20% sequential part
f2 = 0.20 

# Number of processors (p) to consider
processors = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])

# Calculate theoretical maximum speedup using Amdahl's Law formula
# Speedup = 1 / (f + (1-f)/p)
speedup_f1 = 1.0 / (f1 + (1.0 - f1) / processors)
speedup_f2 = 1.0 / (f2 + (1.0 - f2) / processors)

# Ideal linear speedup for comparison
ideal_speedup = processors

print(f"\nSequential fraction f1 = {f1*100:.0f}% => Max Speedup Limit = {1.0/f1:.1f}x")
print(f"Sequential fraction f2 = {f2*100:.0f}% => Max Speedup Limit = {1.0/f2:.1f}x")

# Plotting
print("\nGenerating Amdahl's Law plot...")
plt.figure(figsize=(8, 6))
plt.plot(processors, ideal_speedup, 'k--', label='Ideal Linear Speedup')
plt.plot(processors, speedup_f1, 'bo-', label=f'Amdahl Speedup (f={f1:.2f})')
plt.plot(processors, speedup_f2, 'rs-', label=f'Amdahl Speedup (f={f2:.2f})')

# Plot maximum speedup limits
plt.axhline(1.0/f1, color='blue', linestyle=':', label=f'Max Speedup (f={f1:.2f}) = {1.0/f1:.1f}x')
plt.axhline(1.0/f2, color='red', linestyle=':', label=f'Max Speedup (f={f2:.2f}) = {1.0/f2:.1f}x')

plt.xscale('log', base=2) # Use log scale for processors
plt.yscale('log', base=2) # Use log scale for speedup
plt.xlabel("Number of Processors (p) [log₂ scale]")
plt.ylabel("Theoretical Maximum Speedup (S<0xE1><0xB5><0x96>) [log₂ scale]")
plt.title("Amdahl's Law: Limit on Parallel Speedup")
plt.legend()
plt.grid(True, which='both', alpha=0.4)
# Set ticks for powers of 2
proc_ticks = [2**i for i in range(11)]
plt.xticks(proc_ticks, labels=[str(p) for p in proc_ticks])
plt.yticks(proc_ticks, labels=[str(p) for p in proc_ticks])
plt.ylim(bottom=1)

plt.tight_layout()
# plt.show()
print("Plot generated.")
plt.close()
print("-" * 20)

# Explanation: This code calculates and plots the theoretical maximum speedup 
# predicted by Amdahl's Law for different sequential fractions (f=0.05 and f=0.20).
# 1. It defines the sequential fractions `f1`, `f2`.
# 2. It defines an array `processors` representing the number of processing units `p`.
# 3. It calculates the speedup `S = 1 / (f + (1-f)/p)` for both `f1` and `f2` for each `p`.
# 4. It plots the ideal linear speedup (S=p) as a dashed line.
# 5. It plots the calculated Amdahl speedups for f=0.05 and f=0.20.
# 6. It plots horizontal lines showing the theoretical maximum speedup limit (1/f) for each case.
# The plot clearly shows that even with a small sequential fraction (f=0.05), the 
# speedup quickly deviates from ideal linear scaling and saturates at 1/f = 20x. 
# With a larger sequential fraction (f=0.20), the maximum speedup is limited to only 5x. 
# This visually demonstrates how the non-parallelizable part of a task fundamentally 
# limits the gains from adding more processors.
```

**38.2 Task Parallelism vs. Data Parallelism**

When designing a parallel algorithm, a fundamental consideration is *how* to divide the work among the available processors or cores. Two primary strategies exist: **task parallelism** and **data parallelism**. The choice between them depends largely on the structure of the problem being solved.

**Task Parallelism:** In this paradigm, the problem is decomposed into a collection of largely **independent tasks** that can be executed concurrently, potentially operating on the same or different data. The parallelism arises from executing different *functions* or *code sections* simultaneously. A classic example is a graphical user interface (GUI) where one task handles user input events while another task performs background calculations or updates the display. In scientific computing, task parallelism is common in workflows involving multiple independent steps or processing numerous independent data items.

Examples suitable for task parallelism include:
*   Processing a large batch of independent files (e.g., analyzing thousands of individual light curves or FITS images using the same analysis script, as conceptually shown in App 38.A). Each file processing constitutes an independent task.
*   Running a parameter sweep where the same simulation or analysis code is executed many times with different input parameters. Each run is an independent task.
*   Performing certain types of Monte Carlo simulations where many independent random trials need to be computed.
*   Web servers handling multiple client requests concurrently.

Task parallelism is often implemented using **process pools** or **thread pools**. A fixed number of worker processes or threads are created, and tasks (e.g., filenames, parameter sets) are distributed among them from a queue. As a worker finishes a task, it takes the next one from the queue. Python's `multiprocessing.Pool` (Sec 38.3) is excellent for task parallelism across CPU cores on a single machine. Frameworks like Dask (Chapter 40) extend this concept to distributed environments. The main challenge in task parallelism often lies in efficiently managing the task queue and handling dependencies if tasks are not entirely independent.

**Data Parallelism:** This is the dominant paradigm for large-scale scientific simulations and array processing. Here, the parallelism arises from performing essentially the *same* operation or computation concurrently on different **subsets of a large dataset**. The data itself (e.g., a large array, a grid, a set of particles) is partitioned or distributed across the available processors, and each processor executes the same code instructions on its assigned portion of the data.

Examples suitable for data parallelism include:
*   Applying a filter or mathematical operation element-wise to a very large image or array. The array can be split into chunks, and each core processes one chunk simultaneously.
*   Updating the state of grid cells in a finite difference or finite volume hydrodynamics simulation. The grid is decomposed into subdomains (domain decomposition), and each processor updates the cells in its subdomain (often requiring communication with neighbors for boundary information).
*   Calculating forces on particles in an N-body simulation. The particle list can be distributed, and each processor calculates forces for its assigned particles (requiring communication to get positions of other particles).
*   Performing large matrix operations (multiplication, factorization) common in linear algebra or machine learning. Rows or columns of the matrices are distributed across processors.

Data parallelism is typically implemented using techniques like **domain decomposition** combined with communication libraries like **MPI** (for distributed memory systems, Chapter 39) or using compiler directives like **OpenMP** or parallel loop constructs (`prange` in Numba) for shared memory systems (Sec 38.6, Chapter 41). GPU programming heavily relies on fine-grained data parallelism (SIMD/SIMT execution, Chapter 41). Libraries like NumPy implicitly use data parallelism in their vectorized operations when linked against multi-threaded libraries (like MKL or OpenBLAS), utilizing multiple cores on a single node transparently for array operations. Dask DataFrames and Arrays (Chapter 40) explicitly implement data parallelism for large datasets across distributed systems.

The main challenges in data parallelism often involve efficiently partitioning the data, minimizing communication overhead required between processors working on different data chunks (especially for operations requiring neighbor information), and ensuring good **load balance** so that all processors have roughly equal amounts of work to do.

In many complex applications, particularly large simulations, **hybrid approaches** combining task and data parallelism are used. For example, different physics modules might run as separate tasks, while the core hydro or gravity calculation within each module is parallelized using data parallelism (domain decomposition with MPI/OpenMP).

Understanding whether your problem structure lends itself more naturally to task parallelism (many independent operations) or data parallelism (same operation on many data pieces) is crucial for choosing the appropriate parallelization strategy and programming tools (e.g., `multiprocessing`/Dask for tasks vs. MPI/OpenMP/NumPy vectorization/Dask arrays for data). Many astrophysical analysis workflows involve task parallelism (processing many files), while core simulation codes primarily use data parallelism.

**(Code examples are better suited for sections discussing specific libraries like `multiprocessing` or MPI.)**

**38.3 Process-based Parallelism: `multiprocessing`**

For leveraging the multiple CPU cores available on a typical modern workstation or a single HPC compute node (shared memory system, Sec 37.3), Python's built-in **`multiprocessing`** module provides a standard and effective way to achieve **process-based parallelism**. Unlike threading (Sec 38.4), which runs multiple threads within the *same* process sharing the same memory space (and often limited by Python's Global Interpreter Lock for CPU-bound tasks), `multiprocessing` creates *separate operating system processes*. Each process has its own independent memory space and its own Python interpreter, allowing CPU-bound tasks to run truly in parallel on different cores, bypassing the GIL limitations.

The `multiprocessing` module offers several ways to create and manage parallel processes. A common and convenient approach, particularly well-suited for **task parallelism** (Sec 38.2), is using the **`Pool`** object. A `Pool` object manages a fixed pool of worker processes. You submit tasks to the pool, and the worker processes execute them concurrently.

The typical workflow using `multiprocessing.Pool` is:
1.  **Define the Task Function:** Create a Python function that performs the work to be parallelized on a single input item (e.g., process one file, run one simulation parameter set). This function must be defined at the top level of a module (or be pickleable) so it can be sent to worker processes.
2.  **Prepare Input Data:** Create an iterable (like a list) containing the input arguments for each task (e.g., a list of filenames, a list of parameter tuples).
3.  **Create Pool:** Instantiate a `Pool` object, optionally specifying the number of worker processes (`processes=N`). If omitted, it often defaults to the number of available CPU cores (`os.cpu_count()`). Using a `with` statement is recommended for automatic pool cleanup: `with Pool(processes=N) as pool:`.
4.  **Map Tasks to Processes:** Use one of the pool's mapping methods to distribute the input data across the worker processes and execute the task function:
    *   `pool.map(func, iterable)`: Applies the function `func` to each item in the `iterable`. It blocks until *all* tasks are completed and returns a list containing the results in the same order as the input iterable. Conceptually simple but less flexible if tasks have very different run times.
    *   `pool.imap(func, iterable)`: Similar to `map` but returns an iterator. Results are still returned in order, but computations might proceed more lazily.
    *   `pool.imap_unordered(func, iterable)`: Returns an iterator that yields results as soon as they become available, *regardless* of the input order. This can be more efficient if tasks have highly variable execution times, as faster tasks don't have to wait for slower ones to finish before their results are returned.
    *   `pool.apply_async(func, args)`: Submits a single task asynchronously. Returns an `AsyncResult` object immediately, which can be used later to check status (`.ready()`) and retrieve the result (`.get()`). Useful for more complex task management.
5.  **Collect Results:** Retrieve the results returned by the mapping functions (e.g., the list from `pool.map` or by iterating through the result of `imap_unordered`).
6.  **Cleanup:** The `with` statement automatically ensures the pool is properly closed and worker processes are terminated (`pool.close()`, `pool.join()`).

```python
# --- Code Example 1: Using multiprocessing.Pool for Task Parallelism ---
import multiprocessing
import time
import os
import numpy as np # For dummy calculation

print("Parallel execution using multiprocessing.Pool:")

# Step 1: Define Task Function (must be defined at top level or importable)
def process_item(item_id):
    """Simulates a task that takes some time."""
    # print(f"  Process {os.getpid()} starting item {item_id}") # Uncomment for debug
    # Simulate work (e.g., complex calculation)
    result = 0
    for i in range(10**6 + item_id * 1000): # Vary work slightly per item
        result += np.sin(i * 0.1)
    time.sleep(0.1 * np.random.rand()) # Add small random sleep
    # print(f"  Process {os.getpid()} finished item {item_id}")
    return item_id, result # Return tuple (original id, calculated result)

# Step 2: Prepare Input Data
n_items = 20
input_items = list(range(n_items)) # Simple list [0, 1, ..., 19]
print(f"\nInput items to process: {input_items}")

# --- Serial Execution (for comparison) ---
print("\nRunning tasks serially...")
start_serial = time.time()
serial_results = {}
for item in input_items:
    item_id, item_result = process_item(item)
    serial_results[item_id] = item_result
end_serial = time.time()
time_serial = end_serial - start_serial
print(f"Serial execution finished. Time: {time_serial:.2f}s")

# --- Parallel Execution using Pool.map ---
print("\nRunning tasks in parallel using Pool.map...")
# Determine number of cores to use
n_cores = os.cpu_count()
print(f"Using {n_cores} worker processes (detected cores).")
start_parallel = time.time()
parallel_results_map = {}

# Need if __name__ == '__main__': guard on Windows/macOS with spawn start method
if __name__ == '__main__': # Essential for cross-platform compatibility
    # Step 3: Create Pool (using 'with' statement)
    with multiprocessing.Pool(processes=n_cores) as pool:
        # Step 4: Map tasks to processes
        # pool.map blocks until all results are ready, returns list of results
        # The function process_item is applied to each element in input_items
        results_list = pool.map(process_item, input_items) 
    
    # Step 5: Collect Results (results_list contains (id, result) tuples)
    for item_id, item_result in results_list:
        parallel_results_map[item_id] = item_result
        
    end_parallel = time.time()
    time_parallel = end_parallel - start_parallel
    print(f"Parallel execution (map) finished. Time: {time_parallel:.2f}s")

    # Calculate Speedup
    if time_parallel > 0:
        speedup = time_serial / time_parallel
        print(f"Speedup: {speedup:.2f}x (Ideal max: {n_cores}x)")
        
    # Verify results match (optional)
    # print(f"Serial results match parallel? {serial_results == parallel_results_map}")

print("-" * 20)

# Explanation: This code demonstrates task parallelism using `multiprocessing.Pool`.
# 1. It defines a function `process_item` that simulates some work taking variable time.
# 2. It creates a list `input_items` to be processed.
# 3. It runs the tasks serially using a standard loop and times it.
# 4. It determines the number of CPU cores (`os.cpu_count()`).
# 5. **Crucially**, it uses `if __name__ == '__main__':` which is necessary on systems 
#    that use 'spawn' instead of 'fork' to create processes (like Windows, sometimes macOS) 
#    to prevent issues with re-importing the main script in child processes.
# 6. Inside the `if`, it creates a `Pool` of worker processes within a `with` statement.
# 7. `pool.map(process_item, input_items)` distributes the items from `input_items` 
#    to the worker processes, each worker executes `process_item` on its assigned items. 
#    `pool.map` waits for all tasks to complete and returns the results as a list 
#    in the original order.
# 8. The parallel execution time is measured and speedup calculated. The speedup should 
#    be significant (approaching `n_cores` if the task is purely CPU-bound and overhead is low).
```

Because `multiprocessing` creates separate processes, each with its own memory space, sharing large amounts of data between the main process and worker processes can involve significant overhead due to **inter-process communication (IPC)**. Data passed as arguments to pool methods (like the `iterable` in `map`) and results returned need to be **serialized** (pickled), sent to the worker, and deserialized, which can be slow for large objects. For large NumPy arrays, `multiprocessing` can sometimes use more efficient shared memory mechanisms behind the scenes if configured correctly or if arrays are created in shared memory explicitly, but care is needed.

`multiprocessing` is highly effective for **CPU-bound tasks** where each task involves significant computation and relatively little communication or shared state between tasks (often called "embarrassingly parallel" problems). Examples include processing independent files, running independent simulations in a parameter sweep, or performing certain types of Monte Carlo calculations. It provides a relatively straightforward way to achieve true parallelism and utilize multiple cores on a single machine, overcoming the limitations of the Global Interpreter Lock for CPU-bound work. However, for tasks involving extensive communication or fine-grained sharing of large datasets between parallel units, message passing with MPI (Chapter 39) or potentially shared-memory threading with compiled code (OpenMP, Cython releasing GIL) might be more appropriate, especially when scaling beyond a single node.

**38.4 Thread-based Parallelism (`threading`) and the GIL**

An alternative approach to achieving concurrency (and sometimes limited parallelism) within a single Python program is using **threads**, available through the built-in `threading` module. Unlike processes (`multiprocessing`), threads exist *within* the same process and share the same memory space. This makes creating threads generally faster and less resource-intensive than creating processes, and sharing data between threads appears simpler as they can directly access the same objects in memory.

The `threading` module provides tools to create `Thread` objects, start their execution (`thread.start()`), manage them (e.g., wait for completion using `thread.join()`), and synchronize access to shared resources using primitives like `Lock`, `RLock`, `Semaphore`, `Event`, and `Condition`. One might create multiple threads, assign each a specific function to execute, start them, and then wait for them all to finish.

However, for achieving computational speedup on **CPU-bound tasks** in standard CPython (the most common Python implementation), threading has a major limitation: the **Global Interpreter Lock (GIL)**. The GIL is a mutex (a lock) that protects access to Python objects, preventing multiple threads from executing Python bytecode *simultaneously* within the same process, even on multi-core hardware. Only one thread can hold the GIL and execute Python bytecode at any given time. Other threads waiting to execute Python code must wait for the GIL to be released.

The consequence of the GIL is that for tasks that spend most of their time executing pure Python code or standard Python library functions that are CPU-intensive (e.g., complex numerical calculations, string processing), using multiple threads will **not** result in true parallelism on multi-core processors. The threads will run concurrently (switching between them), but not simultaneously. In fact, the overhead associated with thread creation, context switching, and GIL contention can sometimes make multi-threaded CPU-bound Python code run *slower* than the equivalent serial code.

So, when is `threading` useful? It excels primarily for **I/O-bound tasks**. These are tasks where the program spends significant time waiting for external operations to complete, such as reading/writing files from disk, making network requests (like downloading files or querying web APIs), or waiting for user input. While one thread is blocked waiting for I/O (during which the GIL is typically released by the I/O function), other threads *can* acquire the GIL and execute, allowing the program to make progress on other tasks or respond to other events concurrently.

For example, if you need to download multiple data files from different web servers, launching each download in a separate thread using `threading` can significantly speed up the overall process compared to downloading them sequentially. While one thread waits for data to arrive from a slow server, other threads can initiate or continue downloads from other servers. Similarly, threads can be used in GUI applications to keep the user interface responsive while background tasks are performed.

```python
# --- Code Example 1: Using threading for Concurrent I/O (Conceptual Download) ---
import threading
import time
import random
import os # For dummy file creation/deletion

print("Using threading for concurrent (simulated) downloads:")

# List of dummy URLs to "download"
urls = [f"http://example.com/datafile_{i}.dat" for i in range(10)]

# Function executed by each thread
def download_file_task(url, thread_id):
    """Simulates downloading a file with random delay."""
    filename = os.path.basename(url)
    print(f"  Thread-{thread_id}: Starting download for {filename}")
    # Simulate download time (I/O wait)
    download_time = random.uniform(0.5, 2.0) 
    time.sleep(download_time) 
    # Simulate saving file
    # with open(filename, 'w') as f: f.write("dummy content")
    print(f"  Thread-{thread_id}: Finished download for {filename} (took {download_time:.2f}s)")
    return filename # Return filename when done

# --- Serial Execution ---
print("\nRunning downloads serially...")
start_serial = time.time()
serial_files = []
for i, url in enumerate(urls):
    serial_files.append(download_file_task(url, thread_id=0)) # Run task directly
end_serial = time.time()
time_serial = end_serial - start_serial
print(f"Serial downloads finished. Time: {time_serial:.2f}s")

# --- Threaded Execution ---
print("\nRunning downloads concurrently using threading...")
start_threaded = time.time()
threads = []
# Cannot easily get return values directly like Pool.map
# Need queues or shared list with locks if return values are critical
downloaded_files_threaded = [] # Simple list (careful with race conditions if modified by threads)

for i, url in enumerate(urls):
    # Create a thread targeting the function with arguments
    thread = threading.Thread(target=download_file_task, args=(url, i+1))
    threads.append(thread)
    thread.start() # Start thread execution

# Wait for all threads to complete
print("Waiting for threads to finish...")
for thread in threads:
    thread.join() # Blocks until this thread terminates

end_threaded = time.time()
time_threaded = end_threaded - start_serial # Bug: should be end_threaded - start_threaded
time_threaded = end_threaded - start_threaded # Corrected
print(f"Threaded downloads finished. Time: {time_threaded:.2f}s")

# Calculate Speedup (expected > 1 for I/O bound task)
if time_threaded > 0:
    speedup = time_serial / time_threaded
    print(f"Speedup: {speedup:.2f}x")

# Note: Getting return values from threads is more complex than multiprocessing.Pool
# Often involves using thread-safe queues (queue.Queue) or shared mutable state with Locks.

print("-" * 20)

# Explanation: This code demonstrates using `threading` for concurrent I/O-like tasks.
# 1. It defines a function `download_file_task` that simulates work dominated by 
#    waiting (`time.sleep`), representing an I/O operation.
# 2. It runs the tasks serially and measures the total time (which is roughly the sum 
#    of individual sleep times).
# 3. It then creates and starts a separate `threading.Thread` for each URL. `thread.start()` 
#    begins execution concurrently.
# 4. `thread.join()` is called in a separate loop to wait for all threads to finish 
#    before proceeding.
# 5. The total time for the threaded execution is measured. Because the tasks are I/O bound 
#    (simulated by `sleep`), the GIL is released during the wait, allowing other threads 
#    to run. Therefore, the total threaded time is expected to be significantly *less* 
#    than the serial time (closer to the time of the *longest* individual task plus some 
#    overhead), demonstrating speedup for I/O concurrency. 
# NOTE: This example doesn't explicitly retrieve return values from threads, which 
# requires more complex synchronization mechanisms like Queues.
```

Because threads within the same process share memory, coordinating access to shared mutable data structures requires careful use of **synchronization primitives** provided by the `threading` module:
*   `Lock`: A simple mutual exclusion lock. Only one thread can acquire the lock at a time. Used to protect critical sections of code that modify shared data.
*   `RLock`: A re-entrant lock that can be acquired multiple times by the same thread.
*   `Semaphore`: Allows a fixed number of threads to acquire it simultaneously.
*   `Event`: A simple mechanism for one thread to signal an event to other threads.
*   `Condition`: Allows threads to wait for a condition to become true, often used with a Lock.
Incorrect use of shared data without proper locking can lead to race conditions and corrupted data, making multi-threaded programming notoriously difficult to debug.

While `threading` is suitable for I/O-bound concurrency, for achieving true CPU parallelism on multi-core machines within Python, `multiprocessing` (Sec 38.3) is generally the preferred approach due to the GIL. Alternatively, for CPU-bound code, one can use libraries like `Numba` (Chapter 41) which can release the GIL for compiled functions, or write performance-critical sections in languages like C or Cython and release the GIL manually, allowing multi-threading to achieve true speedup. Libraries like Dask (Chapter 40) also provide higher-level abstractions for managing both threaded and processed-based parallelism.

In summary, Python's `threading` module allows for concurrent execution by running tasks in separate threads within the same process, sharing memory. It is highly effective for improving performance in I/O-bound applications by overlapping waiting times. However, due to the Global Interpreter Lock (GIL) in CPython, it does **not** provide true parallelism for CPU-bound Python code on multi-core processors. For CPU-bound parallelism within Python on a single machine, `multiprocessing` is generally the better choice.

**38.5 Introduction to Message Passing Interface (MPI)**

When computations need to scale beyond the resources of a single shared-memory node – requiring the combined power and memory of multiple independent compute nodes connected by a network in an HPC cluster (distributed memory architecture, Sec 37.3) – a different parallel programming paradigm is needed. The dominant standard for programming distributed memory systems is the **Message Passing Interface (MPI)**. MPI is not a language itself, but rather a **specification** for a library of functions that allow separate processes (typically running on different nodes) to communicate and coordinate by explicitly **sending** and **receiving messages** containing data.

MPI provides a portable interface, meaning code written using standard MPI calls can often be compiled and run on different HPC systems using vendor-specific or open-source MPI implementations (like OpenMPI, MPICH, Intel MPI) that adhere to the standard. The core concepts of MPI include:

*   **Communicator:** Defines a group of processes that can communicate with each other. The most common communicator is `MPI_COMM_WORLD`, which includes all processes launched as part of the MPI job.
*   **Rank:** Each process within a communicator is assigned a unique integer identifier called its **rank**, starting from 0 up to `size - 1`. Processes use ranks to identify senders and receivers of messages.
*   **Size:** The total number of processes within the communicator.
*   **Messages:** Data is exchanged between processes in the form of messages. A message typically consists of the data buffer itself, the data type, the number of elements, the rank of the sending/receiving process, a message tag (an integer for distinguishing different types of messages), and the communicator.

MPI defines a rich set of communication routines, broadly categorized as:
*   **Point-to-Point Communication:** Involves data transfer between *two* specific processes.
    *   `MPI_Send`: Sends data from the calling process to a specified destination rank.
    *   `MPI_Recv`: Receives data sent from a specified source rank (or any source).
    These can be **blocking** (function call doesn't return until the communication is complete, e.g., message sent or received) or **non-blocking** (`MPI_Isend`, `MPI_Irecv`), which initiate the communication and return immediately, allowing computation to overlap with communication (requires later calls like `MPI_Wait` to check for completion). Blocking communication is simpler to reason about but can lead to deadlock if send/receive pairs are not carefully matched. Non-blocking communication offers higher potential performance but requires more complex management.
*   **Collective Communication:** Involves *all* processes within a communicator participating in a coordinated communication pattern. Common collective operations include:
    *   `MPI_Bcast`: Broadcasts data from one designated root process to all other processes in the communicator.
    *   `MPI_Scatter`: Distributes different chunks of an array from a root process to all processes (including the root).
    *   `MPI_Gather`: Collects data chunks from all processes onto a root process, forming a larger array.
    *   `MPI_Allgather`: Gathers data from all processes, and distributes the *entire* gathered array back to all processes.
    *   `MPI_Reduce`: Combines data from all processes using a specified operation (e.g., SUM, MAX, MIN, PROD, logical operators) and delivers the result to a root process.
    *   `MPI_Allreduce`: Performs a reduction, and distributes the final result back to all processes.
    *   `MPI_Barrier`: A synchronization point; processes wait here until all processes in the communicator have reached the barrier.
Collective operations often have highly optimized implementations tailored to the cluster's network topology and are crucial for tasks involving data distribution, result aggregation, and synchronization in parallel algorithms like N-body or grid-based simulations.

MPI programs are typically written using the **Single Program, Multiple Data (SPMD)** model. The same program code is executed by all MPI processes, but each process uses its unique rank (obtained via `MPI_Comm_rank`) and the total size (`MPI_Comm_size`) to determine which part of the data it should work on and with which other processes it needs to communicate. Conditional statements (`if rank == 0: ... else: ...`) are common for tasks performed only by the root process (like reading input, printing final results) or for managing communication patterns.

While MPI is traditionally used with compiled languages like C, C++, and Fortran, the **`mpi4py`** library (`pip install mpi4py`) provides standard Python bindings for MPI. `mpi4py` allows Python programmers to leverage the power of MPI for distributed memory parallelism directly within Python scripts. It provides Python objects and methods that closely mirror the standard MPI functions (`comm.send`, `comm.recv`, `comm.bcast`, `comm.scatter`, `comm.gather`, `comm.reduce`, `comm.Barrier`, etc.). It handles the serialization (pickling) of standard Python objects for sending/receiving, and provides optimized communication for NumPy arrays, minimizing overhead. Using `mpi4py` (covered in Chapter 39) enables scaling Python-based analysis tasks or even prototyping parallel algorithms across multiple nodes of an HPC cluster.

```python
# --- Code Example: Conceptual MPI 'Hello World' using mpi4py ---
# Note: Requires mpi4py installation and an MPI implementation (e.g., OpenMPI)
# Run using mpirun: mpirun -np 4 python your_script_name.py 
# (where 4 is the number of processes)

try:
    from mpi4py import MPI
    mpi4py_installed = True
except ImportError:
    mpi4py_installed = False
    print("NOTE: mpi4py not installed or MPI environment not found. Skipping MPI example.")

print("Conceptual MPI 'Hello World' using mpi4py:")

if mpi4py_installed:
    # --- Initialize MPI Environment ---
    comm = MPI.COMM_WORLD # Get the default communicator
    rank = comm.Get_rank() # Get rank of this specific process
    size = comm.Get_size() # Get total number of processes launched
    
    # --- Each process executes this code ---
    hostname = MPI.Get_processor_name() # Get hostname where process is running
    
    message = f"Hello from rank {rank}/{size} on host {hostname}!"
    
    # --- Demonstrate Communication (Gather all messages to rank 0) ---
    # Each process has its own 'message' string
    # Use comm.gather (lowercase 'g' for Python objects needing pickle)
    # This collects the 'message' from each rank into a list on rank 0
    all_messages = comm.gather(message, root=0)
    
    # --- Rank 0 Prints Results ---
    if rank == 0:
        print(f"\nRank 0 received messages from all {size} processes:")
        for i, msg in enumerate(all_messages):
            print(f"  Message from rank {i}: '{msg}'")
            
    # --- Demonstrate Broadcast ---
    # Rank 0 creates data, broadcasts it to all others
    if rank == 0:
        data_to_broadcast = {'value': 123, 'source': 'Rank 0'}
    else:
        data_to_broadcast = None # Other ranks initialize to None
        
    # Use comm.bcast (lowercase 'b' for Python objects)
    # Broadcasts object from root (rank 0) to all others
    received_data = comm.bcast(data_to_broadcast, root=0)
    
    print(f"\nRank {rank} received broadcast data: {received_data}")
    
    # --- Synchronization Barrier ---
    # print(f"Rank {rank} reaching barrier...")
    # comm.Barrier() # All processes wait here until everyone arrives
    # print(f"Rank {rank} passed barrier.")

else:
    print("Skipping MPI execution.")

print("-" * 20)

# How to Run:
# 1. Save this code as e.g., `mpi_hello.py`.
# 2. Ensure you have an MPI implementation (like OpenMPI) and mpi4py installed.
# 3. Run from terminal using mpirun: 
#    mpirun -np 4 python mpi_hello.py 
#    (This launches 4 independent Python processes running this script)

# Explanation: This code demonstrates basic MPI concepts using `mpi4py`.
# 1. It initializes MPI, getting the communicator (`comm`), rank (`rank`), and size (`size`).
# 2. Each process constructs a unique message including its rank and hostname.
# 3. `comm.gather(message, root=0)` collects the `message` string from every process 
#    into a list `all_messages` available *only* on rank 0. Rank 0 then prints this list.
# 4. Rank 0 creates some data (`data_to_broadcast`). `comm.bcast(..., root=0)` sends 
#    this data from rank 0 to all other processes. Each process receives the data 
#    into `received_data` and prints it.
# 5. (Conceptual) `comm.Barrier()` would force all processes to wait at that point.
# Running this with `mpirun -np 4` will show output from 4 distinct processes, 
# illustrating parallel execution and basic collective communication (gather, broadcast).
```

MPI provides the fundamental communication infrastructure required for large-scale parallel simulations (N-body, hydro, MHD) and data analysis tasks distributed across multiple compute nodes in HPC environments. While direct MPI programming can be complex, libraries like `mpi4py` make its power accessible within the Python ecosystem, enabling Python-based tools to scale beyond single-node limitations.

**38.6 Introduction to OpenMP**

While MPI is the standard for communication between processes on *different* nodes (distributed memory), **OpenMP (Open Multi-Processing)** provides a complementary approach for parallelism *within* a single node that has multiple CPU cores sharing the same memory (shared memory architecture, Sec 37.3). OpenMP is primarily an **API specification** based on **compiler directives** (pragmas in C/C++, comments in Fortran) that allows programmers to easily parallelize sections of their code, most commonly loops, across multiple **threads** running on the available cores of a single machine.

The core idea of OpenMP is **incremental parallelism**. A programmer identifies computationally intensive loops or code sections that can be executed independently for different iterations or tasks. They then insert specific OpenMP pragmas or directives before these sections. A compiler that supports OpenMP (like modern GCC, Clang, Intel compilers) recognizes these directives and automatically generates multi-threaded code to execute that section in parallel using a pool of threads managed by the OpenMP runtime library.

Key OpenMP concepts include:
*   **Parallel Regions:** Defined using `#pragma omp parallel` (C/C++) or `!$OMP PARALLEL` (Fortran). The code block within this region is executed by multiple threads concurrently.
*   **Worksharing Constructs:** Directives placed within a parallel region to distribute work among the threads. The most common is the **loop construct** (`#pragma omp for` or `!$OMP DO`), which automatically divides the iterations of the following `for` or `DO` loop among the available threads. Each thread executes a subset of the loop iterations.
*   **Data Environment Clauses:** Clauses added to directives to control how variables are shared or private among threads. Variables declared outside the parallel region are typically **shared** by default (all threads access the same memory location), while loop iteration variables are usually **private** (each thread gets its own copy). Clauses like `private(var_list)`, `shared(var_list)`, `reduction(operator:var_list)` (for safely combining results like sums or products across threads) allow explicit control over data scoping.
*   **Synchronization:** Directives like `#pragma omp critical` (only one thread executes the block at a time), `#pragma omp atomic` (for safe updates to single memory locations), and `#pragma omp barrier` (all threads wait until all arrive) provide mechanisms for coordinating threads when necessary, although minimizing synchronization is key for performance.
*   **Runtime Library:** OpenMP includes a runtime library providing functions to control the number of threads (`omp_set_num_threads()`), get the current thread ID (`omp_get_thread_num()`), etc. The number of threads used is often controlled by the `OMP_NUM_THREADS` environment variable.

The primary advantage of OpenMP is its **relative ease of use** for parallelizing existing serial code, especially loop-heavy scientific codes written in C, C++, or Fortran. By adding just a few compiler directives, significant parallelism can often be achieved on multi-core processors without drastically restructuring the code or dealing with explicit thread creation and management. The compiler and runtime handle the details of thread creation, scheduling, and synchronization for common worksharing constructs.

However, OpenMP is limited to **shared memory parallelism**. It cannot be used directly for communication between different compute nodes in a distributed memory cluster; MPI is required for that. Effective OpenMP parallelization still requires careful consideration of data dependencies, race conditions (when multiple threads access shared data unsafely), load balancing within parallel loops, and minimizing synchronization overhead. Performance gains depend heavily on the fraction of code parallelized (Amdahl's Law) and the efficiency of memory access by multiple cores.

**OpenMP in Python:** Directly using OpenMP pragmas is not possible in standard Python code because Python is interpreted, not compiled in the same way as C/C++/Fortran. However, OpenMP parallelism *can* be leveraged by Python code in several ways:
*   **NumPy/SciPy with Threaded Libraries:** Many core NumPy and SciPy functions performing array operations or linear algebra rely on underlying compiled libraries (like BLAS, LAPACK) which are often themselves parallelized using OpenMP or similar threading techniques (e.g., Intel MKL, OpenBLAS). By installing appropriately configured versions of these libraries, basic NumPy operations might automatically utilize multiple cores on your machine without explicit directives in your Python code (though GIL limitations might still apply for operations involving significant Python overhead).
*   **Cython with OpenMP:** Cython allows writing C-extensions for Python. Within Cython code (`.pyx` files), you can release the GIL for computationally intensive sections and use OpenMP pragmas (enabled via compiler flags during compilation) to parallelize loops using threads directly, calling this compiled extension from Python.
*   **Numba with `prange`:** The Numba JIT compiler (Chapter 41) provides the `numba.prange` function. When used within a `@numba.njit(parallel=True)` decorated function, `prange` attempts to automatically parallelize the loop across multiple threads using mechanisms similar in spirit to OpenMP's loop parallelism, often releasing the GIL for the compiled code.

```python
# --- Code Example 1: Conceptual OpenMP in C ---
# This is C code, not Python, illustrating the pragma usage.

c_code_omp = """
#include <stdio.h>
#include <stdlib.h>
#include <omp.h> // Include OpenMP header

int main(int argc, char *argv[]) {
    int n = 10000000; // Size of arrays
    double *a, *b, *c;
    double sum = 0.0;
    int i;

    a = (double*)malloc(n * sizeof(double));
    b = (double*)malloc(n * sizeof(double));
    c = (double*)malloc(n * sizeof(double));

    // Initialize arrays (can be parallelized too)
    #pragma omp parallel for // Example: Parallelize initialization loop
    for (i = 0; i < n; i++) {
        a[i] = (double)i * 0.5;
        b[i] = (double)i * 1.5;
    }
    printf("Arrays initialized.\\n");

    // Set number of threads (e.g., from environment variable)
    // omp_set_num_threads(4); 

    double start_time = omp_get_wtime();

    // Parallelize the main computation loop using OpenMP 'for' directive
    // The 'reduction(+:sum)' clause safely calculates the sum across threads
    #pragma omp parallel for reduction(+:sum) private(i) shared(a, b, c, n)
    for (i = 0; i < n; i++) {
        c[i] = a[i] + b[i] * 2.0; // Example computation
        sum += c[i]; // Reduction handles summing 'sum' safely
    }

    double end_time = omp_get_wtime();

    printf("Calculation finished.\\n");
    printf("Sum = %f\\n", sum);
    printf("Time taken: %f seconds\\n", end_time - start_time);
    
    free(a); free(b); free(c);
    return 0;
}
"""
print("--- Conceptual OpenMP Usage in C Code ---")
print(c_code_omp)
# To compile (using GCC): gcc -fopenmp my_omp_code.c -o my_omp_code
# To run (e.g., using 4 threads): export OMP_NUM_THREADS=4; ./my_omp_code
print("\n--- Compilation and Execution (Conceptual Shell Commands) ---")
print("# gcc -fopenmp my_omp_code.c -o my_omp_code")
print("# export OMP_NUM_THREADS=4")
print("# ./my_omp_code")
print("-" * 20)

# Explanation: This shows C code using OpenMP directives.
# - `#include <omp.h>` includes the OpenMP library header.
# - `#pragma omp parallel for`: This directive tells the compiler to parallelize the 
#   following `for` loop across multiple threads. The OpenMP runtime automatically 
#   divides the loop iterations (0 to n-1) among the available threads.
# - `reduction(+:sum)`: This clause ensures that the updates to the `sum` variable 
#   from different threads are combined correctly (summed) at the end of the parallel 
#   loop, avoiding race conditions.
# - `private(i)` makes the loop counter `i` private to each thread. `shared(...)` 
#   confirms which variables are shared (default for most).
# - `omp_get_wtime()` is used for timing.
# This illustrates how adding pragmas allows relatively easy parallelization of loops 
# in compiled languages for shared memory execution.
```

```python
# --- Code Example 2: Conceptual Parallel Loop with Numba ---
# Note: Requires numba installation: pip install numba
import numpy as np
import numba
import time

print("Conceptual parallel loop using Numba @njit(parallel=True) and prange:")

# Function to parallelize
@numba.njit(parallel=True, fastmath=True) # Enable parallel execution
def parallel_calculation(x, y):
    """Performs an element-wise calculation potentially in parallel."""
    result = np.empty_like(x)
    # Use numba.prange instead of range for parallel loop execution
    for i in numba.prange(len(x)): 
        # Example computation within the loop
        result[i] = np.sin(x[i])**2 + np.cos(y[i])**2 
    return result

# Generate data
n_points = 10**7
x_in = np.random.rand(n_points)
y_in = np.random.rand(n_points)
print(f"\nGenerated input data with {n_points} points.")

# Run the Numba function (compiles on first call)
print("Running Numba parallel function...")
start_time_numba = time.time()
result_numba = parallel_calculation(x_in, y_in)
end_time_numba = time.time()
print(f"Calculation finished. Time taken: {end_time_numba - start_time_numba:.3f}s")
# Check result shape
print(f"Result array shape: {result_numba.shape}")

# Note: Actual speedup depends on number of cores, overhead, and if the loop 
# body is computationally intensive enough to benefit from parallelization.
# Numba's parallelism works best on loops with independent iterations.

print("-" * 20)

# Explanation: This code demonstrates using Numba for shared-memory parallelism in Python.
# 1. It defines a function `parallel_calculation` decorated with `@numba.njit(parallel=True)`. 
#    `njit` compiles the function to fast machine code, and `parallel=True` enables 
#    Numba's automatic parallelization capabilities.
# 2. Inside the function, the loop uses `numba.prange` instead of Python's `range`. 
#    Numba attempts to distribute the iterations of this `prange` loop across multiple 
#    CPU cores using threads, similar to OpenMP's `#pragma omp for`.
# 3. The code generates input data and calls the Numba-compiled function, measuring its 
#    execution time. For computationally intensive loops with independent iterations, 
#    this Numba approach can achieve significant speedups on multi-core machines 
#    directly from Python code, effectively leveraging shared memory parallelism.
```

In summary, OpenMP provides a directive-based standard primarily for shared-memory parallelism in compiled languages (C/C++/Fortran), allowing easy parallelization of loops and code blocks across threads on multi-core nodes. While not directly usable in standard Python bytecode due to the GIL, its parallel execution benefits can be accessed via optimized underlying libraries (NumPy/SciPy linked with threaded BLAS/LAPACK), by writing extensions in Cython with OpenMP pragmas, or more conveniently through JIT compilers like Numba that support automatic parallelization (`prange`) for suitable loops within compiled functions. It serves as a key tool for intra-node parallelism, often used in conjunction with MPI for inter-node parallelism in hybrid HPC applications.

**Application 38.A: Parallel Processing of FITS Files using `multiprocessing`**

**(Paragraph 1)** **Objective:** This application demonstrates a practical use case for task parallelism (Sec 38.2) on a single multi-core machine using Python's `multiprocessing` module (Sec 38.3). The goal is to accelerate the processing of a large number of independent FITS files by distributing the analysis of each file across multiple CPU cores. We will compare the execution time of serial processing versus parallel processing using `multiprocessing.Pool`.

**(Paragraph 2)** **Astrophysical Context:** Many astronomical analysis workflows involve applying the same processing steps or analysis function to a large collection of individual data files. Examples include calculating statistics for thousands of light curves from TESS or Kepler, detecting sources in hundreds of survey image cutouts, extracting spectra from numerous multi-extension FITS files, or running a photometric pipeline on individual CCD images from an observing run. Since the processing of each file is typically independent of the others, this type of task is "embarrassingly parallel" and ideally suited for task parallelism using multiple CPU cores to reduce the total wall-clock time significantly.

**(Paragraph 3)** **Data Source:** A directory containing a moderate number (e.g., 20-100 for demonstration, could be thousands in reality) of independent FITS image files (`image_*.fits`). Each file contains image data. We also need a Python function `analyze_image(filename)` that performs some representative analysis on a single FITS file (e.g., loads the image, calculates basic statistics like mean, median, std dev, maybe finds bright pixels) and takes a noticeable amount of CPU time (e.g., a fraction of a second to several seconds per file). We will simulate the files and the analysis function.

**(Paragraph 4)** **Modules Used:** `multiprocessing` (specifically `Pool`, `cpu_count`), `os` (for file/path handling), `numpy` (for dummy analysis), `astropy.io.fits` (within the analysis function), `time` (for timing).

**(Paragraph 5)** **Technique Focus:** Implementing task parallelism using `multiprocessing.Pool.map`. (1) Defining a worker function `analyze_image` that takes a filename as input and returns some result. (2) Creating a list of input filenames. (3) Timing the execution of the worker function applied serially to all files using a standard `for` loop. (4) Creating a `multiprocessing.Pool` with a number of processes equal to or related to the number of available CPU cores (`os.cpu_count()`). (5) Using `pool.map(analyze_image, filename_list)` to distribute the filenames across the worker processes and execute the analysis concurrently. (6) Timing the parallel execution. (7) Calculating and comparing the speedup achieved. Emphasizing the `if __name__ == '__main__':` guard for cross-platform compatibility.

**(Paragraph 6)** **Processing Step 1: Setup Data and Worker Function:**
    *   Create a temporary directory. Simulate creating multiple dummy FITS files inside it (e.g., simple NumPy arrays saved as FITS).
    *   Define the `analyze_image(filename)` function. This function should open the FITS file, access the data, perform some non-trivial calculation (e.g., `np.std`, `np.percentile`, maybe a loop over pixels for simulation), and return a result (e.g., a dictionary of statistics or a single value). Ensure this function is defined at the top level or imported.
    *   Use `glob.glob` or `os.listdir` combined with `os.path.join` to create a list `fits_files` containing the full paths to all the dummy FITS files.

**(Paragraph 7)** **Processing Step 2: Serial Execution:** Record the start time (`time.time()`). Use a standard `for` loop to iterate through `fits_files`. Inside the loop, call `analyze_image(filename)`. Store the results if needed. Record the end time and calculate the total serial execution time (`time_serial`).

**(Paragraph 8)** **Processing Step 3: Parallel Execution:**
    *   Determine the number of worker processes `n_workers = os.cpu_count()`.
    *   Use the `if __name__ == '__main__':` guard.
    *   Inside the guard, record the start time.
    *   Create the `Pool` using `with multiprocessing.Pool(processes=n_workers) as pool:`.
    *   Call `parallel_results = pool.map(analyze_image, fits_files)`. The `map` function blocks until all processes complete. `parallel_results` will be a list containing the return value of `analyze_image` for each input file, in the original order.
    *   Record the end time and calculate the total parallel execution time (`time_parallel`).

**(Paragraph 9)** **Processing Step 4: Compare and Report:** Calculate the speedup `S = time_serial / time_parallel`. Print the serial time, parallel time, number of workers used, and the calculated speedup. Optionally, verify that the `parallel_results` list contains the expected results and has the same length as the input file list.

**(Paragraph 10)** **Processing Step 5: Cleanup:** Remove the temporary directory and the dummy FITS files created during setup.

**Output, Testing, and Extension:** Output includes printouts showing progress, the measured serial and parallel execution times, the number of cores used, and the calculated speedup factor. **Testing:** Verify the speedup is greater than 1 and ideally approaches the number of cores used for CPU-bound tasks (minus overhead for process creation and communication). Check that the results obtained from the parallel run match those from the serial run (if results were stored). Ensure the `if __name__ == '__main__':` guard is used. **Extensions:** (1) Vary the number of processes used in the `Pool` (from 1 up to `os.cpu_count()`) and plot speedup vs. number of processes to see how performance scales. (2) Replace `pool.map` with `pool.imap_unordered` and see if performance changes, especially if the `analyze_image` function has variable runtime per file. (3) Modify `analyze_image` to be more I/O-bound (e.g., reading/writing larger files) and compare the speedup achieved with `multiprocessing` versus using `threading` (Sec 38.4), expecting threading might be competitive or better for pure I/O tasks. (4) Implement a progress bar for the parallel execution (can be tricky with `map`, sometimes easier with `apply_async` or libraries like `tqdm` integrated carefully).

```python
# --- Code Example: Application 38.A ---
# Note: Uses multiprocessing, may behave differently on different OS (fork vs spawn)
# The 'if __name__ == "__main__":' guard is crucial.

import multiprocessing
import time
import os
import glob
import numpy as np
from astropy.io import fits
import shutil # For cleanup

print("Parallel FITS File Processing using multiprocessing.Pool:")

# Step 1: Setup Data and Worker Function
temp_dir = "temp_fits_files"
n_files = 32 # Number of files to create (should be >= num cores for good test)
n_pix = 128 # Size of dummy images

def create_dummy_fits(filename):
    """Creates a simple FITS file with random data."""
    hdu = fits.PrimaryHDU(np.random.rand(n_pix, n_pix) * 1000)
    hdu.header['FILENAME'] = os.path.basename(filename)
    hdu.writeto(filename, overwrite=True)

def analyze_image(filename):
    """Worker function: Opens FITS, performs dummy analysis, returns stats."""
    try:
        with fits.open(filename) as hdul:
            data = hdul[0].data.astype(np.float64) # Ensure float for stats
            # Simulate some CPU work
            mean_val = np.mean(data)
            std_val = np.std(data)
            # Add a slightly heavier calculation
            log_data = np.log10(np.maximum(data, 1e-6)) # Avoid log(0)
            median_log = np.median(log_data)
            # Simulate result
            result = {'mean': mean_val, 'std': std_val, 'median_log': median_log}
            # print(f"  Processed {os.path.basename(filename)} on {os.getpid()}") # Debug
            return os.path.basename(filename), result
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return os.path.basename(filename), None

# Create dummy files
print(f"\nCreating {n_files} dummy FITS files in '{temp_dir}'...")
os.makedirs(temp_dir, exist_ok=True)
for i in range(n_files):
    fname = os.path.join(temp_dir, f"image_{i:03d}.fits")
    create_dummy_fits(fname)
fits_files = glob.glob(os.path.join(temp_dir, "*.fits"))
print(f"Found {len(fits_files)} files.")

# Step 2: Serial Execution
print("\nRunning analysis serially...")
start_serial = time.time()
serial_results_dict = {}
for f in fits_files:
    fname, result_data = analyze_image(f)
    if result_data: serial_results_dict[fname] = result_data
end_serial = time.time()
time_serial = end_serial - start_serial
print(f"Serial execution time: {time_serial:.3f} seconds")

# Step 3 & 4: Parallel Execution
# IMPORTANT: Pool creation and map call must be inside this block
if __name__ == "__main__": # Essential guard!
    n_workers = min(os.cpu_count(), n_files) # Use available cores, up to num files
    print(f"\nRunning analysis in parallel using {n_workers} workers...")
    start_parallel = time.time()
    
    results_list_parallel = []
    try:
        # Create pool within context manager
        with multiprocessing.Pool(processes=n_workers) as pool:
            # Map the analyze_image function to the list of filenames
            # Returns list of results [(fname1, res1), (fname2, res2), ...]
            results_list_parallel = pool.map(analyze_image, fits_files) 
            
        end_parallel = time.time()
        time_parallel = end_parallel - start_parallel
        print(f"Parallel execution time: {time_parallel:.3f} seconds")

        # Step 5: Compare and Report
        if time_serial > 0 and time_parallel > 0:
            speedup = time_serial / time_parallel
            efficiency = speedup / n_workers
            print(f"\nSpeedup factor: {speedup:.2f}x (Ideal max: {n_workers}x)")
            print(f"Parallel Efficiency: {efficiency:.2f}")
        
        # Optional: Check results consistency (can be tricky with floats)
        # parallel_results_dict = {fname: res for fname, res in results_list_parallel if res}
        # print(f"Result lengths match? {len(serial_results_dict) == len(parallel_results_dict)}")
        
    except Exception as e_pool:
         print(f"An error occurred during parallel processing: {e_pool}")

# Step 6: Cleanup
finally:
     if os.path.exists(temp_dir):
         print(f"\nCleaning up '{temp_dir}'...")
         shutil.rmtree(temp_dir)
         print("Cleanup complete.")

print("-" * 20)
```

*Caveat:* These code examples are simplified to illustrate the parallelization *structure* and data handling concepts. They are **not** optimized implementations for calculating correlation functions efficiently. Real-world calculations for large N would use highly optimized libraries like `Corrfunc` or compiled code.

---

**Application 38.B: Conceptual Parallel Pair Counting (with Code Illustrations)**

**(Paragraph 1)** **Objective:** This application conceptually explores how a computationally intensive analysis task common in cosmology and galaxy studies – calculating pair counts for correlation functions – could be parallelized using both shared memory (`multiprocessing`) and distributed memory (MPI) approaches. We provide simple Python code snippets using `multiprocessing` and `mpi4py` to illustrate the *logic* of task distribution, data handling, and communication, reinforcing concepts of data vs. task parallelism (Sec 38.2) and architectural differences (Sec 37.3), while acknowledging these are not production-optimized codes.

**(Paragraph 2)** **Astrophysical Context:** The two-point correlation function, ξ(r), measures the excess probability of finding pairs of galaxies (or dark matter halos) separated by a distance `r` compared to a random distribution. Calculating ξ(r) typically involves counting pairs of objects within specific radial separation bins (`DD(r)` counts), comparing this to counts from a random catalog (`RR(r)`), and potentially cross-counts (`DR(r)`). The pair counting step, especially `DD` on large datasets, is computationally dominated by calculating distances between potentially N*(N-1)/2 pairs, making parallelization essential for large N.

**(Paragraph 3)** **Data Source:** A catalog of 3D object positions (e.g., `x`, `y`, `z` coordinates for N galaxies or halos). We will simulate this as a NumPy array.

**(Paragraph 4)** **Modules Used (for illustration):** `numpy` (for arrays, distances), `multiprocessing` (for shared memory parallelism), `mpi4py` (for distributed memory parallelism), `scipy.spatial.KDTree` (optional, for slightly more efficient neighbor finding within examples), `matplotlib.pyplot` (for plotting results).

**(Paragraph 5)** **Technique Focus:** Illustrating parallelization strategies for a data-intensive task.
    *   **Shared Memory:** Using `multiprocessing.Pool` to parallelize the outer loop of pair counting. Each process works on a subset of particles `i` but needs access to the *full* dataset to find pairs `j`. Requires data duplication or shared memory mechanisms for efficiency with large N.
    *   **Distributed Memory:** Using `mpi4py` conceptually with **domain decomposition**. Each process holds only a *fraction* of the data. Requires explicit **communication** between processes to count pairs across subdomain boundaries, followed by a global reduction to combine results.

**(Paragraph 6)** **Serial Approach (Baseline):** First, define a function to calculate pair counts serially. For simplicity, we can use `KDTree.query_pairs` or just count pairs within fixed distance bins using `np.histogram` on calculated distances (less efficient but illustrates binning).

```python
# --- Serial Pair Counting (Baseline / Helper) ---
import numpy as np
from scipy.spatial import KDTree # Using KDTree for slightly better serial performance

def calculate_serial_pair_counts(positions, r_bins):
    """Calculates pair counts in radial bins using KDTree (serial)."""
    print(f"  Serial: Building KDTree for {len(positions)} points...")
    kdtree = KDTree(positions)
    # Find pairs within the largest radius
    max_radius = r_bins[-1]
    pairs = kdtree.query_pairs(r=max_radius, output_type='ndarray')
    print(f"  Serial: Found {len(pairs)} pairs within {max_radius:.2f}")
    
    if len(pairs) == 0:
        return np.zeros(len(r_bins) - 1), r_bins
        
    # Calculate distances for found pairs
    pos1 = positions[pairs[:, 0]]
    pos2 = positions[pairs[:, 1]]
    distances = np.sqrt(np.sum((pos1 - pos2)**2, axis=1))
    
    # Histogram distances into bins
    counts, _ = np.histogram(distances, bins=r_bins)
    return counts, r_bins

# --- Simulate Data ---
np.random.seed(42)
N_points = 2000 # Keep moderately small for serial/multiprocessing demo
positions = np.random.rand(N_points, 3) * 100.0 # Points in a 100x100x100 box
r_bins = np.linspace(0, 10, 21) # 20 bins from 0 to 10
bin_centers = 0.5*(r_bins[:-1] + r_bins[1:])

print(f"Generated {N_points} points. Calculating serial pair counts...")
# serial_counts, _ = calculate_serial_pair_counts(positions, r_bins)
# print(f"Serial Counts: {serial_counts}") 
# (Skipping execution here to focus on parallel structure below)
print("  (Serial calculation conceptually defined)")
```

**(Paragraph 7)** **Shared Memory Parallelism (`multiprocessing`):** We parallelize the outer loop. Each worker process gets a *chunk of indices* `i` to process, but needs access to the *full* position array to find pairs `j`. The worker calculates a partial histogram, and results are summed at the end.

```python
# --- Code Example: Shared Memory Parallel Pair Counting (Conceptual) ---
import numpy as np
import multiprocessing
from scipy.spatial import KDTree 
import time
import os 

# Worker function for multiprocessing
# Must be defined at top level or importable
def worker_pair_counts_shared(args):
    """Calculates partial pair counts for a chunk of indices i."""
    chunk_indices, all_positions, r_bins_local = args
    n_all = len(all_positions)
    partial_counts = np.zeros(len(r_bins_local) - 1, dtype=np.int64)
    
    # Build KDTree *once* if efficient (or pass it if feasible)
    # For simplicity, assume rebuilding or using simpler method here.
    # Naive loop for illustration (inefficient):
    for i in chunk_indices:
        # Compare point i to points j > i in the *full* dataset
        for j in range(i + 1, n_all): 
            dist = np.sqrt(np.sum((all_positions[i] - all_positions[j])**2))
            if dist < r_bins_local[-1]:
                # Find bin index
                bin_idx = np.searchsorted(r_bins_local, dist, side='right') - 1
                if bin_idx >= 0 and bin_idx < len(partial_counts):
                    partial_counts[bin_idx] += 1
                    
    # Using KDTree within worker (more realistic but requires passing all_positions)
    # tree_local = KDTree(all_positions)
    # pairs_in_chunk = tree_local.query_pairs(r=r_bins_local[-1], output_type='ndarray')
    # # Filter pairs to only count those involving chunk_indices correctly (avoid double count)
    # mask = np.isin(pairs_in_chunk[:,0], chunk_indices) # Only count pairs starting from my chunk
    # relevant_pairs = pairs_in_chunk[mask]
    # if len(relevant_pairs) > 0:
    #    pos1 = all_positions[relevant_pairs[:, 0]]
    #    pos2 = all_positions[relevant_pairs[:, 1]]
    #    distances = np.sqrt(np.sum((pos1 - pos2)**2, axis=1))
    #    partial_counts, _ = np.histogram(distances, bins=r_bins_local)
        
    # print(f"  Worker {os.getpid()} finished chunk.") # Debug
    return partial_counts

# --- Main Execution (Shared Memory) ---
print("\n--- Shared Memory Parallel Pair Counting (multiprocessing) ---")
# Use data generated previously: positions, r_bins
n_cores = os.cpu_count()
print(f"Using {n_cores} cores.")

# Divide indices into chunks for workers
indices = np.arange(len(positions))
index_chunks = np.array_split(indices, n_cores)

# Prepare arguments for workers
# Each worker needs its indices, the FULL position array, and the bins
# WARNING: Sending full 'positions' array to each worker involves overhead!
# For large N, use explicit multiprocessing shared memory arrays.
worker_args = [(chunk, positions, r_bins) for chunk in index_chunks]

start_time_mp = time.time()
parallel_counts_shared = np.zeros(len(r_bins) - 1, dtype=np.int64)

if __name__ == "__main__": # Crucial guard
    try:
        with multiprocessing.Pool(processes=n_cores) as pool:
            # Map the worker function to the arguments
            partial_results = pool.map(worker_pair_counts_shared, worker_args)
        
        # Sum the partial histograms from all workers
        for partial_hist in partial_results:
            parallel_counts_shared += partial_hist
            
        end_time_mp = time.time()
        print(f"Parallel calculation time: {end_time_mp - start_time_mp:.3f} seconds")
        print(f"Parallel Counts: {parallel_counts_shared}")
        # Compare with serial_counts if calculated earlier

    except Exception as e:
        print(f"Multiprocessing failed: {e}")

print("--- End Shared Memory Example ---")
print("-" * 20)

# Explanation:
# 1. Defines a worker function `worker_pair_counts_shared` that takes a chunk of particle 
#    indices, the *full* position array, and the bins.
# 2. It calculates pairs involving indices in its chunk against the full dataset 
#    (naive O(N_chunk * N_all) loop shown for simplicity; KDTree usage commented).
# 3. It returns a *partial* histogram of pair counts for its subset of work.
# 4. The main part splits the particle indices into chunks.
# 5. It creates a `multiprocessing.Pool`. 
# 6. `pool.map` sends each task (chunk indices, full positions, bins) to a worker. 
#    This highlights the need for all workers to access the full `positions` array, 
#    which implies either data duplication or use of shared memory for large N.
# 7. The partial histograms returned by `pool.map` are summed to get the final counts.
```

**(Paragraph 8)** **Distributed Memory Parallelism (MPI with `mpi4py`):** Here, the data is physically distributed. Each MPI process `k` only stores the positions `local_positions_k` for particles in its subdomain. It calculates local pairs (`DD_local`) and then must **communicate** to count cross-boundary pairs (`DD_cross`).

```python
# --- Code Example: Distributed Memory Parallel Pair Counting (Conceptual MPI) ---
# Note: Requires mpi4py and MPI environment. Run with e.g., mpirun -np 4 python script.py
try:
    from mpi4py import MPI
    mpi4py_installed = True
except ImportError:
    mpi4py_installed = False
    print("NOTE: mpi4py not installed or MPI environment not found. Skipping MPI example.")

print("\n--- Distributed Memory Parallel Pair Counting (Conceptual MPI) ---")

if mpi4py_installed:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # --- Simulate Data Distribution ---
    # Each rank generates/loads only its portion of the data
    N_local = N_points // size # Assume perfect division for simplicity
    if rank == size - 1: N_local += N_points % size # Last rank gets remainder
        
    # Generate local positions (ensure consistency if loading from file)
    np.random.seed(42 + rank) # Different seed per rank
    local_positions = np.random.rand(N_local, 3) * 100.0 
    # Add rank offset to simulate spatial domain decomposition
    local_positions[:, 0] += rank * 100.0 / size 
    
    print(f"Rank {rank}: Has {N_local} local particles.")

    # Define bins (same on all ranks)
    r_bins_mpi = np.linspace(0, 10, 21) 
    local_counts = np.zeros(len(r_bins_mpi) - 1, dtype=np.int64)

    # --- 1. Calculate Local Pairs ---
    print(f"Rank {rank}: Calculating local pairs...")
    if N_local > 1:
        # Use serial function on local data
        local_counts, _ = calculate_serial_pair_counts(local_positions, r_bins_mpi)
    print(f"Rank {rank}: Found {np.sum(local_counts)} local pairs.")

    # --- 2. Communicate Boundary Data (HIGHLY CONCEPTUAL) ---
    # This is the complex part requiring knowledge of domain decomposition and neighbors
    print(f"Rank {rank}: Conceptual boundary data exchange...")
    # a. Identify particles near boundaries shared with neighbors.
    # b. Determine neighbor ranks (e.g., based on domain decomposition grid).
    # c. Send boundary particle positions to neighbors (`comm.Send` or `comm.Isend`).
    # d. Receive boundary particle positions from neighbors (`comm.Recv` or `comm.Irecv`).
    # (Requires careful buffer management and handling non-blocking communication/waits)
    print(f"  (Skipping actual MPI Send/Recv implementation for boundary exchange)")
    # Store received neighbor data in `neighbor_positions`.
    neighbor_positions = np.empty((0, 3)) # Placeholder for received data

    # --- 3. Calculate Cross-Boundary Pairs ---
    print(f"Rank {rank}: Calculating cross-boundary pairs...")
    cross_counts = np.zeros_like(local_counts)
    if N_local > 0 and len(neighbor_positions) > 0:
        # Build KDTree for neighbors? Or loop.
        # Compare local particles to received neighbor particles
        # Calculate distances, histogram into cross_counts
        # Need careful logic to avoid double counting pairs across boundaries
        print(f"  (Skipping actual cross-pair calculation)")
        pass # Placeholder for cross-pair calculation logic
    print(f"Rank {rank}: Found {np.sum(cross_counts)} cross pairs (conceptual).")

    # Combine local and cross counts for this rank
    total_partial_counts = local_counts + cross_counts

    # --- 4. Global Reduction ---
    print(f"Rank {rank}: Performing global reduction (Sum)...")
    # Allocate array on rank 0 to receive sum, or use None on others for Reduce
    final_counts = np.zeros_like(local_counts) if rank == 0 else None
    
    # Sum the 'total_partial_counts' from all ranks onto rank 0
    comm.Reduce(
        [total_partial_counts, MPI.INT64_T], # Send buffer and datatype
        [final_counts, MPI.INT64_T],       # Receive buffer and datatype (only relevant on root)
        op=MPI.SUM,                        # Operation to perform
        root=0                             # Rank receiving the result
    )

    # --- Rank 0 Prints Final Result ---
    if rank == 0:
        print("\n--- Final Pair Counts (Rank 0) ---")
        print(f"Total Counts per bin: {final_counts}")
        # Plotting or further analysis happens here
        
    # Wait for all processes before exiting (good practice)
    comm.Barrier() 

else:
    print("Skipping MPI execution.")

print("-" * 20)

# How to Run:
# 1. Save as e.g., `mpi_pairs.py`.
# 2. Ensure mpi4py and MPI environment exist.
# 3. Run from terminal: mpirun -np 4 python mpi_pairs.py (e.g., with 4 processes)

# Explanation:
# 1. Simulates Data Distribution: Each MPI rank generates/holds only its `local_positions`.
# 2. Local Pair Counting: Each rank calls the serial pair counting function (`calculate_serial_pair_counts`) 
#    on its *local* data only, obtaining `local_counts`.
# 3. Boundary Communication (Conceptual): **Critically**, it highlights where explicit 
#    MPI communication (`Send`/`Recv`) *would* be needed to exchange particle data near 
#    subdomain boundaries with neighboring ranks. This step is complex and *not* implemented here.
# 4. Cross-Pair Counting (Conceptual): Shows where pairs between local data and received 
#    neighbor data would be calculated.
# 5. Global Reduction: Uses `comm.Reduce` with `op=MPI.SUM` to sum the partial pair counts 
#    (local + conceptual cross-boundary) from all ranks onto rank 0.
# 6. Final Output: Rank 0 holds the final combined histogram `final_counts`.
# This illustrates the distributed data, local computation, explicit communication, and 
# global reduction pattern typical of MPI data parallelism for such problems.
```

**(Paragraph 9)** **Comparison Revisited:** The code snippets illustrate the fundamental difference. The `multiprocessing` version parallelizes the *loop over particles `i`*, but each process needs access to *all* particle positions `j`. This works well if the entire dataset fits comfortably in shared RAM but doesn't scale beyond a single node or to datasets larger than RAM. The `mpi4py` version involves **distributing the data** itself. Each process handles a smaller amount of data locally, enabling scaling to massive datasets across many nodes. However, this necessitates complex **explicit communication** steps to handle interactions across the boundaries of the distributed data chunks, which adds overhead and implementation complexity absent in the shared-memory approach.

*Caveat:* These code examples are simplified to illustrate the parallelization *structure* and data handling concepts. They are **not** optimized implementations for calculating correlation functions efficiently. Real-world calculations for large N would use highly optimized libraries like `Corrfunc` or compiled code.


**(Paragraph 10)** **Summary:** Parallelizing pair counting for correlation functions highlights the trade-offs between shared-memory and distributed-memory approaches. Shared memory (`multiprocessing`) is conceptually simpler for tasks fitting on one node but requires global data access. Distributed memory (MPI/`mpi4py`) scales to larger problems by partitioning data but demands explicit communication logic for boundary interactions and global reductions. For performance-critical correlation function calculations on large astrophysical datasets, highly optimized, compiled libraries using MPI (and often OpenMP/GPU hybrids) like `Corrfunc` are typically employed, but understanding the underlying parallelization strategies illustrated here provides valuable conceptual insight.

**Chapter 38 Summary**

This chapter provided a conceptual introduction to parallel programming, laying the groundwork for understanding how computationally intensive astrophysical tasks are tackled on multi-core processors and distributed HPC clusters. It began by defining core concepts: **parallelism** (simultaneous execution for speedup) versus **concurrency** (managing overlapping tasks), performance metrics like **speedup** (T₁/T<0xE1><0xB5><0x96>) and **efficiency** (Speedup/p), and the fundamental limitation on speedup due to irreducible sequential code sections described by **Amdahl's Law**. Two major parallelization strategies were contrasted: **task parallelism**, where independent tasks are executed concurrently (suited for processing batches of files or parameter sweeps), and **data parallelism**, where the same operation is applied concurrently to different subsets of a large dataset (dominant in large simulations and array processing), often involving domain decomposition.

The chapter then explored Python's built-in mechanisms for achieving parallelism primarily on a single, shared-memory machine. **Process-based parallelism** using the `multiprocessing` module, particularly the `Pool` object (`Pool.map`), was presented as an effective way to leverage multiple CPU cores for CPU-bound tasks, as separate processes bypass the Global Interpreter Lock (GIL). **Thread-based parallelism** using the `threading` module was discussed, highlighting its lower overhead and suitability for I/O-bound concurrency but its inability to provide true speedup for CPU-bound Python code due to the GIL. Finally, conceptual introductions were provided for the dominant paradigms used in large-scale HPC: the **Message Passing Interface (MPI)** standard for communication between processes on distributed memory systems (nodes in a cluster), enabling data parallelism via explicit message exchange (point-to-point and collective operations, accessible in Python via `mpi4py`), and **OpenMP**, a directive-based API for easier shared-memory parallelism within a node using threads, primarily used in compiled languages but accessible indirectly via optimized libraries, Cython, or Numba's `prange`.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Downey, A. B. (2016).** *Think HPC: Programming and Problem Solving on High Performance Computing Clusters*. O'Reilly Media (Often available via Green Tea Press online: [https://greenteapress.com/wp/think-hpc/](https://greenteapress.com/wp/think-hpc/)).
    *(Provides accessible introductions to parallel concepts like speedup, Amdahl's Law, task/data parallelism, and basic ideas behind MPI and OpenMP.)*

2.  **Eijkhout, V. (2022).** *Introduction to High Performance Scientific Computing*. (Online textbook). [http://pages.tacc.utexas.edu/~eijkhout/istc/istc.html](http://pages.tacc.utexas.edu/~eijkhout/istc/istc.html)
    *(Covers parallel concepts, shared vs. distributed memory, Amdahl's Law, MPI, and OpenMP in detail.)*

3.  **Python Software Foundation. (n.d.).** *multiprocessing — Process-based parallelism*. Python Documentation. Retrieved January 16, 2024, from [https://docs.python.org/3/library/multiprocessing.html](https://docs.python.org/3/library/multiprocessing.html)
    *(Official documentation for Python's `multiprocessing` module, including `Pool`, relevant to Sec 38.3 and Application 38.A.)*

4.  **Python Software Foundation. (n.d.).** *threading — Thread-based parallelism*. Python Documentation. Retrieved January 16, 2024, from [https://docs.python.org/3/library/threading.html](https://docs.python.org/3/library/threading.html)
    *(Official documentation for Python's `threading` module, explaining thread creation, synchronization, and the GIL, relevant to Sec 38.4.)*

5.  **Quinn, M. J. (2003).** *Parallel Programming in C with MPI and OpenMP*. McGraw-Hill Education.
    *(A classic textbook providing a practical introduction to both MPI and OpenMP programming, primarily using C examples, but explaining the core concepts relevant to Sec 38.5 and 38.6.)*
