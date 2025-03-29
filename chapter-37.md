**Chapter 37: Introduction to HPC Environments**

As astrophysical datasets and simulation requirements continue to grow exponentially (Chapter 7, Chapter 31), many research tasks quickly exceed the capabilities of standard desktop or laptop computers. Analyzing petabyte-scale survey data, running large N-body or hydrodynamical simulations with billions of elements, or performing complex statistical inference like MCMC on high-dimensional models often demands significantly more computational power, memory, and storage than locally available. This chapter serves as the entry point into **Part VII: High-Performance Computing (HPC) for Astrophysics**, providing a necessary introduction to the concepts, architectures, and basic usage patterns of the powerful cluster computing environments that enable these large-scale computational endeavors. We begin by motivating *why* HPC is essential for modern astrophysics. We then dissect the typical **anatomy of an HPC cluster**, explaining the roles of different types of nodes, the interconnect network, storage systems, and the crucial job scheduler. We differentiate between **shared and distributed memory architectures**, the fundamental paradigms influencing parallel programming strategies. Practical aspects of **accessing HPC resources** are covered, including secure shell (SSH) login, navigating remote file systems, and managing software environments using module systems. We introduce the concept of **job schedulers** (like SLURM or PBS/Torque) used for submitting and managing computationally intensive tasks as batch jobs. Finally, we briefly discuss **resource allocation and quotas**, common constraints users encounter when working on shared HPC systems.

**37.1 Why HPC? Problems Exceeding Desktop Capabilities**

While personal computers have become incredibly powerful, the demands of cutting-edge astrophysical research frequently push beyond the limits of even high-end workstations. Problems involving massive datasets or computationally intensive calculations often necessitate the use of **High-Performance Computing (HPC)** resources. Understanding *why* and *when* HPC becomes necessary is the first step towards leveraging these powerful systems effectively.

The most obvious driver is **computational speed**. Many astrophysical simulations, particularly those involving N-body gravity with large particle counts (Sec 33) or complex hydrodynamics/MHD with high resolution (Sec 34), require trillions upon trillions of floating-point operations (teraflops or petaflops). Running such simulations on a single CPU core could take years or even centuries. HPC clusters provide access to hundreds or thousands of interconnected processor cores working in parallel, allowing these calculations to be completed in days, weeks, or months – a feasible timescale for research projects. Similarly, analyzing massive observational datasets, performing complex model fitting (like large MCMC runs, Sec 17), or running computationally expensive algorithms (like certain machine learning training processes, Sec 24) can be drastically accelerated by distributing the workload across many cores.

Another major limitation of local machines is **memory (RAM)**. Loading and processing large simulation snapshots, high-resolution images or data cubes, or extensive catalogs can require tens or hundreds of gigabytes, or even terabytes, of RAM. Standard desktops or laptops typically have 8GB, 16GB, or perhaps 32GB/64GB of RAM, which is often insufficient. HPC nodes are frequently equipped with much larger amounts of memory (e.g., 128GB, 256GB, 512GB, or more per node), and distributed memory clusters allow the total memory footprint of a problem to be spread across many nodes, enabling the analysis of datasets that simply cannot fit into the memory of a single workstation.

**Data storage and I/O bandwidth** also become critical factors. Petabyte-scale survey archives (Sec 7.1) cannot be stored locally. Even downloading multi-terabyte subsets requires substantial local disk space, often beyond personal capacity. HPC centers provide large-scale storage systems (often petabytes) designed for scientific data, typically using high-performance parallel file systems (Sec 42.2). Furthermore, accessing data quickly is crucial. Reading or writing terabytes of data from local disks can be extremely slow. HPC file systems are designed for high bandwidth and concurrent access from many compute nodes, mitigating the I/O bottleneck that can cripple large-scale data analysis on local storage.

Many astrophysical problems are **inherently parallel**, meaning they can naturally be broken down into smaller pieces that can be computed simultaneously. Examples include: calculating gravitational forces between N particles (each particle's force calculation is largely independent), updating grid cells in a hydro simulation (updates often depend only on neighboring cells), processing independent regions of a large sky survey image, or running the same analysis pipeline on thousands of different light curves or spectra (task parallelism). HPC environments provide the hardware (many cores/nodes) and software infrastructure (MPI, parallel libraries) needed to exploit this inherent parallelism effectively (Sec 32.6, Part VII).

Access to specialized **hardware accelerators** like Graphics Processing Units (GPUs) or Tensor Processing Units (TPUs) is another reason to use HPC. These devices offer massive parallelism for specific types of computations (like matrix operations common in deep learning or some numerical algorithms), providing significant speedups over traditional CPUs for suitable workloads (Sec 41). HPC centers often deploy nodes equipped with powerful GPUs specifically for these tasks.

Beyond raw performance, HPC centers provide a **stable, managed environment**. Hardware is maintained, system software is kept up-to-date, backups are often performed, and support staff are available. This offloads significant system administration burdens from individual researchers. They also provide access to a wide range of pre-installed scientific software packages and libraries, often managed via module systems (Sec 37.4), ensuring a consistent and reproducible software environment.

Collaboration is also facilitated by shared HPC resources. Research groups can access the same large datasets stored centrally and utilize common software environments and analysis pipelines running on the cluster, simplifying data sharing and collaborative development compared to managing disparate local setups.

Of course, using HPC resources also introduces complexities: users need to learn how to connect remotely, navigate different file systems, use job schedulers, potentially write parallel code, and manage resource allocations. However, for a growing number of scientifically ambitious projects in astrophysics involving large simulations or datasets, the computational power, memory capacity, storage, and specialized hardware offered by HPC environments are no longer just advantageous – they are absolutely essential for making progress. Recognizing when your problem's scale exceeds local capabilities is the trigger to explore utilizing these powerful shared resources.

**37.2 Anatomy of an HPC Cluster**

High-Performance Computing (HPC) clusters, the workhorses of large-scale scientific computation, are complex systems composed of several interconnected hardware and software components working in concert. Understanding the basic **anatomy of a typical cluster** helps users navigate the environment, request resources appropriately, and understand how their jobs are executed. While specific architectures vary, most clusters share common elements.

The core computational power resides in the **compute nodes**. These are essentially individual servers, each typically containing multiple processors (CPUs), with each CPU having multiple cores (e.g., a node might have two 32-core CPUs for a total of 64 cores). Each compute node has its own local memory (RAM, e.g., 128GB or 256GB) directly accessible by the cores on that node. Large simulations or analyses often run across many compute nodes simultaneously, requiring communication between them. Some nodes might be specialized, e.g., containing GPUs (**GPU nodes**) for accelerated computing or having significantly larger amounts of RAM (**high-memory nodes**) for memory-intensive tasks. Users typically do *not* log in directly to compute nodes; access is managed by the scheduler.

Users usually interact with the cluster via one or more **login nodes** (also called head nodes or front-end nodes). These are servers accessible directly from the outside network (e.g., via SSH). Login nodes are where users compile their code, edit files, manage data, submit jobs to the scheduler, and monitor their progress. They typically have access to the same file systems as the compute nodes. Importantly, login nodes are *shared* resources for *all* users and are **not** intended for running computationally intensive tasks directly. Running heavy computations on login nodes can severely impact performance for everyone else and is usually prohibited by usage policies. Their purpose is primarily interaction, compilation, and job management.

Connecting all the nodes (compute, login, storage) is a high-speed **interconnect network**. While standard Gigabit Ethernet might be used for basic management, performance-critical communication between compute nodes (e.g., for MPI message passing, Sec 39) typically relies on much faster, lower-latency interconnects like **InfiniBand** or specialized high-speed Ethernet variants (e.g., 100GbE, 200GbE). The topology and bandwidth of this interconnect are crucial for the scalability of parallel applications that require significant inter-process communication.

Data storage is managed through one or more **file systems**, accessible from both login and compute nodes. Common types include:
*   **Home Directories (`/home/username`):** Usually relatively small storage quotas, backed up regularly, intended for storing source code, configuration files, important scripts, and small essential data. Not intended for large datasets or intensive I/O during computation.
*   **Scratch Space (`/scratch`, `/tmp`, `/lustre/scratch`):** Large, high-performance storage (often based on parallel file systems like Lustre or GPFS) designed for temporary storage of large input/output files during job execution. Typically *not* backed up and often subject to automatic purging policies (files deleted after a certain period, e.g., 30-90 days). Ideal for simulation outputs or intermediate analysis files while jobs are running.
*   **Project Space (`/project`, `/work`, `/data`):** Often larger quotas than home directories, potentially backed up (check local policy), intended for longer-term storage of shared group data, important datasets, and final results. Performance might be intermediate between home and scratch.
Understanding the purpose, quotas, performance characteristics, and backup/purge policies of the different available file systems is essential for effective data management on an HPC cluster.

The allocation of compute nodes to user jobs and the management of job execution order are handled by a **job scheduler** or **batch system**. This crucial piece of software acts as a gatekeeper, receiving job requests (submitted by users as scripts), managing a queue of pending jobs, allocating available compute nodes based on requested resources (cores, memory, time) and user priorities (or fair-share policies), launching the job on the allocated nodes, monitoring its execution, and cleaning up after completion or failure. Common schedulers include **SLURM** (Simple Linux Utility for Resource Management), **PBS** (Portable Batch System) and its variants (Torque, OpenPBS), and historically LSF or SGE. Users interact with the scheduler primarily through command-line tools to submit jobs (`sbatch`, `qsub`), check job status (`squeue`, `qstat`), and delete jobs (`scancel`, `qdel`). (Schedulers are discussed further in Sec 37.5).

Finally, the cluster runs a **software stack** typically based on a Linux operating system distribution. Access to compilers (GCC, Intel, PGI/Nvidia), parallel libraries (MPI implementations like OpenMPI, MVAPICH, Intel MPI), scientific libraries (GSL, FFTW, HDF5, NetCDF), performance analysis tools, debuggers, and numerous scientific applications and Python environments is usually managed through an **environment module system** (like Lmod or environment-modules). Users load specific modules (`module load python/3.9`, `module load gcc openmpi`) to configure their shell environment with the necessary paths and settings before compiling or running software (Sec 37.4).

Understanding this basic anatomy – login nodes for interaction, compute nodes for execution (CPU/GPU/high-memory), high-speed interconnect for communication, tiered storage systems (home/scratch/project), a job scheduler for resource management, and a module system for software access – provides the necessary context for effectively utilizing HPC resources for demanding astrophysical computations.

**37.3 Shared vs. Distributed Memory Architectures**

When discussing parallel computing on HPC systems, a fundamental architectural distinction is made between **shared memory** and **distributed memory** systems. This distinction relates to how different processing units (CPU cores) access the system's main memory (RAM) and dictates the primary programming models used to parallelize applications. Many modern HPC clusters incorporate aspects of both.

A **shared memory architecture** is one where multiple processing units (cores) have direct access to the *same* physical memory space. A typical multi-core desktop or workstation, or a single compute node within an HPC cluster (often called a Symmetric Multiprocessing or SMP node), represents a shared memory system. All cores within that node can read from and write to any location in the node's main RAM using standard memory load/store operations. This allows for relatively easy sharing of data between tasks running concurrently on different cores within the same node, as they can all access the same data structures in memory directly.

The primary programming model for shared memory parallelism involves **threads**. Multiple threads of execution can run concurrently within a single process, sharing the process's memory space. Communication and synchronization between threads often occur implicitly through shared variables, although careful use of synchronization primitives (like mutexes, locks, semaphores) is required to prevent race conditions where multiple threads try to modify the same memory location simultaneously, leading to corrupted data or incorrect results. Common threading libraries include **pthreads** (POSIX Threads, used in C/C++/Fortran) and Python's `threading` module (Sec 38.4). Another popular approach for scientific computing is using compiler directives like **OpenMP** (Open Multi-Processing), where pragmas inserted into Fortran or C/C++ code instruct the compiler to automatically parallelize loops or code sections across multiple threads (Sec 38.6). `Numba` (Chapter 41) can also generate multi-threaded code for CPUs. Shared memory programming is generally considered conceptually simpler for certain types of parallelism (like loop parallelization) as explicit data movement between processors is not required.

A **distributed memory architecture**, on the other hand, consists of multiple independent nodes connected by a network. Each node has its own processor(s), its own local memory, and typically runs its own copy of the operating system. Crucially, a processor on one node **cannot** directly access the local memory of another node. Data sharing and coordination between tasks running on different nodes *must* occur through explicit **message passing** over the network interconnect. An HPC cluster composed of many compute nodes is the canonical example of a distributed memory system.

The dominant programming model for distributed memory systems is the **Message Passing Interface (MPI)** (Sec 38.5, Chapter 39). MPI is a standardized library specification (with implementations like OpenMPI, MPICH, Intel MPI) that provides functions for processes running on different nodes to send and receive messages containing data. Processes are identified by a unique rank within a communicator (like `MPI.COMM_WORLD`). Communication can be **point-to-point** (one process sends data directly to another using `MPI_Send`/`MPI_Recv`) or **collective** (involving all processes in the communicator simultaneously, e.g., `MPI_Bcast` to broadcast data from one process to all others, `MPI_Reduce` to combine data from all processes onto one, `MPI_Scatter` to distribute parts of an array, `MPI_Gather` to collect parts back). MPI programming requires explicitly managing data distribution and communication between processes, making it generally more complex than shared memory threading or OpenMP, but it is essential for scaling applications beyond a single node to potentially thousands of cores across an entire cluster.

Modern HPC clusters typically exhibit a **hybrid architecture**. Each individual compute node is a shared memory system (multiple cores sharing RAM). Multiple such nodes are connected via a network, forming a distributed memory system overall. Therefore, high-performance applications often use a **hybrid programming model**: **MPI** is used for communication *between* nodes, while **OpenMP** or **threading** (or GPU programming, Chapter 41) is used for parallelism *within* each node, utilizing the multiple cores sharing memory on that node. For example, one might run one MPI process per node, and each MPI process launches multiple OpenMP threads to utilize all the cores on its node. This hybrid approach aims to leverage both levels of parallelism effectively.

Understanding the distinction between shared and distributed memory is crucial for choosing appropriate parallelization strategies and tools. For tasks that can run entirely within the memory limits of a single multi-core node, shared memory parallelism using `multiprocessing` (which simulates separate processes but can use shared memory mechanisms on one node), `threading` (for I/O bound tasks primarily, due to Python's GIL), `Numba` parallel loops, or OpenMP (if using compiled code) might be sufficient and simpler. For problems requiring more memory or computational power than available on a single node, or for utilizing hundreds/thousands of cores, distributed memory parallelism using MPI (often via `mpi4py` in Python, Chapter 39) becomes necessary, requiring explicit management of data distribution and communication across the network connecting the cluster nodes.

**37.4 Accessing HPC Resources: SSH, Modules, File Systems**

Gaining access to and effectively utilizing an HPC cluster involves interacting with its specific environment, which typically differs from a standard desktop setup. Key practical skills include connecting securely, navigating the file systems, and managing the available software stack using environment modules.

**Connecting via SSH:** Access to HPC clusters is almost universally provided through the **Secure Shell (SSH)** protocol. SSH allows users to establish a secure, encrypted connection from their local machine (the client) to the cluster's login node(s) (the server) over the network. From a terminal on a Linux or macOS system, the command is typically:
`ssh username@cluster_hostname`
where `username` is your assigned user account on the cluster and `cluster_hostname` is the address of the login node (e.g., `hpc.astro.university.edu`). Windows users can use SSH clients like PuTTY, MobaXterm, or the built-in SSH client available in newer Windows versions (via Command Prompt or PowerShell). Authentication usually involves entering your cluster password, although setting up **SSH keys** (a pair of public and private cryptographic keys) is highly recommended for passwordless and more secure login, especially for scripting or frequent access. Once connected, you are presented with a command-line shell prompt on the login node.

**Navigating File Systems:** Once logged in, you interact with the cluster using standard Unix/Linux command-line tools. Key commands for navigation include:
*   `pwd`: Print Working Directory (shows your current location).
*   `ls`: List directory contents (`ls -l` for detailed view, `ls -a` to show hidden files).
*   `cd directory_name`: Change Directory (move into a subdirectory).
*   `cd ..`: Move up one directory level.
*   `cd ~` or `cd`: Go to your home directory.
*   `mkdir directory_name`: Create a new directory.
*   `rm filename`: Remove a file (use with caution!).
*   `rmdir directory_name`: Remove an empty directory.
*   `mv source destination`: Move or rename a file/directory.
*   `cp source destination`: Copy a file/directory.
As discussed in Sec 37.2, HPC systems typically have multiple file systems (`/home`, `/scratch`, `/project`, etc.) with different purposes, quotas, and performance. It's crucial to understand where your data and code should reside. Use commands like `df -h` (disk free) or specific quota commands provided by the system administrators to check available space and usage limits in different file systems. Most intensive computation and large file I/O should occur in designated scratch directories, not your home directory.

**Managing Software with Environment Modules:** Scientific software on HPC clusters (compilers, libraries like MPI/HDF5, applications like GADGET/Enzo, Python environments with specific packages like Astropy/SciPy/yt) is rarely installed system-wide in standard locations. Instead, multiple versions of software packages are often managed using an **environment module system** (like Lmod or environment-modules). This system allows users to dynamically modify their shell environment (specifically environment variables like `PATH`, `LD_LIBRARY_PATH`, `PYTHONPATH`) to access specific versions of software packages without conflicts.

Common `module` commands include:
*   `module avail`: List all available software modules on the system.
*   `module list`: Show the modules currently loaded in your environment.
*   `module load module_name[/version]`: Add a specific module (and its required dependencies) to your environment (e.g., `module load gcc/9.3.0`, `module load python/3.9.6`, `module load astropy/5.1-python3.9`).
*   `module unload module_name`: Remove a module from your environment.
*   `module swap module1 module2`: Unload module1 and load module2 (useful for switching versions).
*   `module purge`: Unload all currently loaded modules.
*   `module help module_name`: Display help information for a specific module.
Before compiling code or running analysis scripts, you typically need to load the necessary modules (compiler, MPI library, Python version, required Python packages) to make the software accessible and ensure compatibility. The specific modules needed are often specified in documentation or job submission scripts (Sec 37.5). Checking `module list` after loading confirms the environment is correctly configured.

```python
# --- Code Example: Interacting via Shell Commands (Conceptual Sequence) ---
# This represents commands typed in a terminal after SSHing into a cluster.
# It's not Python code, but illustrates the interaction flow.

# 1. Connect via SSH (from local machine terminal)
# ssh your_username@hpc.cluster.address 
# (Enter password or use SSH key)

# --- Now on the cluster login node ---

# 2. Check current location
# pwd 

# 3. Navigate to project directory (assuming it exists)
# cd /project/your_group/your_project

# 4. List contents
# ls -l

# 5. Check available disk space
# df -h .

# 6. Check available Python modules
# module avail python

# 7. Load a specific Python environment (e.g., Anaconda or specific version)
# module load anaconda3/2023.09  # Or: module load python/3.9.12 gcc/10.2.0 openmpi/4.1.1
print("\n--- Conceptual Shell Commands ---")
print("# ssh your_username@hpc.cluster.address")
print("# pwd")
print("# cd /project/your_group/your_project")
print("# ls -l")
print("# df -h .")
print("# module avail python")
print("# module load python/3.10.4  # Load desired Python")
print("# module load astropy/5.3-python3.10 # Load specific package module")

# 8. Check loaded modules
# module list
print("# module list")

# 9. Check Python version and package availability
# python --version
# python -c "import astropy; print(astropy.__version__)"
print('# python --version')
print('# python -c "import astropy; print(astropy.__version__)"')

# 10. Transfer a file from local machine (run this in a *local* terminal)
# scp my_local_script.py your_username@hpc.cluster.address:/project/your_group/your_project/
print("\n# On LOCAL machine:")
print("# scp data.fits your_username@hpc.cluster.address:/project/your_group/your_project/data/")

# 11. Verify file transfer on cluster
# ls -lh data/data.fits
print("\n# Back on CLUSTER:")
print("# ls -lh data/data.fits")
print("--- End Conceptual Shell Commands ---")
print("-" * 20)

# Explanation: This block lists a sequence of typical shell commands used when 
# interacting with an HPC cluster. 
# - `ssh` establishes the connection.
# - `pwd`, `cd`, `ls` are standard file system navigation.
# - `df` checks disk space.
# - `module avail` discovers software, `module load` activates specific versions 
#   (e.g., Python 3.10, Astropy 5.3 built for that Python), `module list` confirms loading.
# - Basic `python` commands verify the loaded environment.
# - `scp` (Secure Copy) is shown (run from the local machine) to transfer a file 
#   (`data.fits`) to a specific directory on the cluster.
# - `ls` on the cluster verifies the transfer.
# This illustrates the essential command-line interactions for accessing the cluster, 
# managing software via modules, and transferring files.
```

Mastering these basic interactions – secure connection via SSH, navigating the different file systems (`/home`, `/scratch`, `/project`), and managing your software environment using the `module` system – is the essential first step for any researcher starting to use HPC resources. Always consult the specific documentation provided by your HPC center for details on hostnames, file system paths, available modules, and usage policies.

**37.5 Job Schedulers (SLURM, PBS/Torque)**

Computationally intensive tasks, such as running large simulations or complex data analysis pipelines, should **never** be executed directly on the cluster's login nodes. Login nodes are shared resources designed for interactive tasks like editing, compiling, managing files, and submitting jobs. Running heavy computations there degrades performance for all other users. Instead, resource-intensive work must be submitted as **batch jobs** to the cluster's **job scheduler** (also known as a batch queuing system). The scheduler manages access to the cluster's compute nodes, allocating resources fairly and efficiently among many users and jobs.

The core concept is **batch processing**. Users encapsulate the commands needed to run their task (e.g., loading modules, setting environment variables, executing the simulation code or analysis script) within a **submission script**. This script also includes directives specifying the computational resources required (number of nodes, cores per node, memory per node/core, estimated runtime/walltime) and potentially other job characteristics (job name, output/error file locations, project account to charge). The user submits this script to the scheduler using a command like `sbatch` (for SLURM) or `qsub` (for PBS/Torque).

The scheduler places the submitted job into a **queue**. It then monitors the available compute nodes and the pending jobs. Based on factors like requested resources, job priority (which might depend on user quotas, fair-share policies, or job size/duration), and queue policies set by administrators, the scheduler selects jobs from the queue and allocates the required compute nodes to them. Once nodes are allocated, the scheduler initiates the execution of the user's submission script on the primary allocated node. The script runs non-interactively in the background on the compute node(s).

Common job schedulers found on HPC clusters include:
*   **SLURM (Simple Linux Utility for Resource Management):** A widely used, powerful, and scalable open-source scheduler. Submission scripts use `#SBATCH` directives. Key commands: `sbatch script.slurm` (submit), `squeue` (view queue), `scancel jobid` (delete job), `sinfo` (view node status), `sacct` (view accounting info).
*   **PBS (Portable Batch System) and its variants (Torque, OpenPBS):** Another popular family of schedulers. Submission scripts use `#PBS` directives. Key commands: `qsub script.pbs` (submit), `qstat` (view queue), `qdel jobid` (delete job), `pbsnodes` (view node status).
While the specific commands and directive syntax differ, the underlying concepts are similar across different schedulers. Always consult the local HPC documentation for the specific scheduler used and its command/directive syntax.

A typical submission script (e.g., for SLURM) contains:
1.  **Shebang:** `#!/bin/bash` (or other shell).
2.  **Scheduler Directives:** Lines starting with `#SBATCH` (or `#PBS`) specifying resources:
    *   `--nodes=N`: Number of compute nodes requested.
    *   `--ntasks-per-node=C`: Number of tasks (often MPI processes) to run per node.
    *   `--cpus-per-task=T`: Number of CPU cores allocated per task (for multi-threaded tasks using OpenMP or similar). Total cores = N * C * T (or similar depending on specific flags like `--ntasks`).
    *   `--mem=XG`: Total memory requested per node (e.g., `120G`). Or `--mem-per-cpu=XG`.
    *   `--time=DD-HH:MM:SS`: Maximum wall clock time requested for the job. Exceeding this limit will cause the scheduler to terminate the job.
    *   `--job-name=MyJobName`: A descriptive name for the job.
    *   `--output=job_%j.out`: File to redirect standard output ( `%j` is replaced by job ID).
    *   `--error=job_%j.err`: File to redirect standard error.
    *   `--partition=queue_name`: Specify which queue or partition to submit to (e.g., 'compute', 'gpu', 'highmem').
    *   `--account=project_id`: Specify the project/account to charge resources against.
3.  **Environment Setup:** `module load ...` commands to set up the necessary software environment.
4.  **Execution Commands:** The actual command(s) to run the simulation or analysis script (e.g., `mpirun -np $SLURM_NTASKS ./my_mpi_code input.param` or `python my_analysis.py data/`). It's often good practice to `cd` into a specific working directory (e.g., on scratch space) before execution.

```python
# --- Code Example 1: Example SLURM Submission Script ---
# This is a SHELL script (e.g., save as submit_my_job.slurm), not Python.

slurm_script_content = """#!/bin/bash

#SBATCH --job-name=AstroSim      # Job name
#SBATCH --output=astro_sim_%j.out # Standard output log (%j expands to job ID)
#SBATCH --error=astro_sim_%j.err  # Standard error log
#SBATCH --partition=compute       # Partition (queue) name
#SBATCH --nodes=4                 # Request 4 nodes
#SBATCH --ntasks-per-node=32      # Request 32 tasks (MPI processes) per node
#SBATCH --cpus-per-task=1         # 1 CPU core per MPI task (no OpenMP here)
#SBATCH --mem=120G                # Memory per node (use G for Gigabytes)
#SBATCH --time=02-00:00:00        # Max wall time (2 days)
#SBATCH --account=astro_project   # Account to charge

# --- Environment Setup ---
echo "Loading modules..."
module purge # Start with a clean environment
module load gcc/10.2.0 openmpi/4.1.1 hdf5/1.10.7-parallel # Load compiler, MPI, parallel HDF5
module load python/3.9.6 # Load Python if needed by setup/analysis
echo "Modules loaded."
module list # Print loaded modules to output log

# --- Execution ---
# Get total number of tasks allocated by SLURM
TOTAL_TASKS=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
echo "Running on $SLURM_NNODES nodes with $TOTAL_TASKS total MPI tasks."

# Navigate to scratch directory (recommended)
SCRATCH_DIR="/scratch/${USER}/sim_run_${SLURM_JOB_ID}"
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR
echo "Working directory: $SCRATCH_DIR"

# Copy necessary input files (assuming they are in submission directory)
# cp $SLURM_SUBMIT_DIR/input.param .
# cp $SLURM_SUBMIT_DIR/my_sim_code . 
echo "Input files copied (conceptual)."

# Run the parallel simulation code using mpirun/srun
echo "Starting simulation executable..."
# Use srun (SLURM's runner) or mpirun (check local HPC docs)
srun --mpi=pmix -n $TOTAL_TASKS ./my_sim_code input.param # Example execution
# Or: mpirun -np $TOTAL_TASKS ./my_sim_code input.param

SIM_EXIT_CODE=$? # Get exit code of the simulation
echo "Simulation finished with exit code: $SIM_EXIT_CODE"

# Optional: Copy important results back to project space
# echo "Copying results..."
# cp output_*.hdf5 $SLURM_SUBMIT_DIR/results/
# echo "Results copied."

# Optional: Cleanup scratch space if needed and successful
# if [ $SIM_EXIT_CODE -eq 0 ]; then
#   echo "Cleaning up scratch directory..."
#   rm -rf $SCRATCH_DIR
# fi

echo "Job finished."
"""

print("--- Example SLURM Submission Script (submit_my_job.slurm) ---")
print(slurm_script_content)

# --- Commands to Submit and Monitor (Run in Shell on Login Node) ---
shell_commands = """
# Submit the job
sbatch submit_my_job.slurm 
# (Output: Submitted batch job 12345)

# Check job status (replace 12345 with actual job ID)
squeue -u $USER 
# (Output shows job 12345, partition, name, user, status (PD/R), time, nodes)

# Check status of specific job
scontrol show job 12345 
# (Detailed info including requested/allocated resources, run time)

# Cancel the job (if needed)
scancel 12345

# Check accounting information after job finishes
sacct -j 12345 --format=JobID,JobName,Partition,AllocCPUS,State,ExitCode,MaxRSS,Elapsed
"""
print("\n--- Example Shell Commands for SLURM ---")
print(shell_commands)
print("-" * 20)

# Explanation:
# 1. Script Content: Shows a typical SLURM script (`.slurm`). 
#    - `#SBATCH` directives request resources: 4 nodes, 32 tasks/node, 1 cpu/task (total 128 tasks), 
#      120GB RAM/node, 2 days max time, specific partition and account. Output/error files 
#      are named using the job ID (`%j`).
#    - Environment Setup: Uses `module purge` and `module load` to set up required software.
#    - Execution: Calculates total tasks, creates and changes to a job-specific scratch directory, 
#      conceptually copies inputs, runs the parallel code (`./my_sim_code`) using `srun` 
#      (SLURM's parallel launcher) with the allocated number of tasks. It captures the exit code.
#    - Post-processing (Conceptual): Shows how results might be copied back and scratch cleaned up.
# 2. Shell Commands: Shows how to submit (`sbatch`), monitor (`squeue`, `scontrol`), cancel (`scancel`), 
#    and check accounting (`sacct`) for the submitted job from the login node terminal.
```

After submitting a job, it enters the queue and its state changes (e.g., `PENDING` (PD), `RUNNING` (R), `COMPLETING` (CG), `COMPLETED` (CD), `FAILED` (F), `TIMEOUT` (TO)). Users use commands like `squeue` or `qstat` to monitor the status of their jobs. When the job runs, standard output and standard error are redirected to the files specified in the script (or default locations). Examining these output/error files is crucial for checking progress and diagnosing problems. Accounting tools (`sacct`, `qacct`) allow checking resource usage after the job completes.

Effectively using an HPC cluster requires understanding how to write submission scripts that accurately request the necessary resources (requesting too much might increase queue wait times, too little might cause the job to fail or run slowly) and how to use the scheduler commands to submit, monitor, and manage batch jobs. Consulting the specific documentation for the cluster's scheduler and local policies is essential.

**37.6 Resource Allocation and Quotas**

Access to High-Performance Computing (HPC) resources is typically managed through **allocations** and constrained by **quotas**. Since HPC clusters are expensive shared facilities, mechanisms are needed to ensure fair access among users and research groups, track resource consumption, and prevent any single user or group from monopolizing the system. Understanding the allocation model and storage quotas of your HPC environment is crucial for planning projects and managing your computations effectively.

**Compute Allocations:** Access to computing time on the cluster's nodes is usually granted in units of **core-hours** (or CPU-hours, node-hours, or GPU-hours). A research group or project is typically awarded a specific allocation (e.g., 500,000 core-hours per year) by the institution or funding agency operating the facility. When a user submits a batch job (Sec 37.5) requesting specific resources (e.g., 128 cores for 48 hours), the scheduler checks if the job's potential consumption (128 cores * 48 hours = 6144 core-hours) fits within the remaining balance of the project's allocation. The core-hours actually consumed by completed jobs are then deducted from the allocation balance. Running out of allocated time can prevent users from submitting new jobs until the allocation is replenished or a new allocation period begins.

**Job Priority and Fair Share:** Schedulers often implement **priority schemes** to decide which job runs next when resources become available. Priority might be influenced by factors like:
*   Job size (smaller jobs sometimes get higher priority to improve turnaround).
*   Time waiting in the queue (priority often increases over time).
*   **Fair Share:** A mechanism that dynamically adjusts user or group priority based on their recent resource usage relative to their allocated share. Users who have recently consumed a large fraction of their fair share might have their job priority temporarily lowered compared to users who have used less, ensuring that resources are distributed equitably over time, even among users within the same allocation grant. Understanding the local fair-share policy can help predict queue wait times.

**Storage Quotas:** Similar to compute time, storage space on the cluster's file systems (`/home`, `/project`, `/scratch`) is usually subject to **quotas**. These quotas typically limit both the total **disk space** (in Gigabytes or Terabytes) and sometimes the total **number of files** (inodes) a user or group can occupy within a specific file system. Exceeding a quota will usually prevent the user from writing new files to that file system, which can cause running jobs to fail if they try to write output or checkpoint files. Users need to monitor their disk usage regularly (using commands like `du -sh ~` for home directory size, `quota -s`, or specific local commands) and actively manage their files (deleting unnecessary intermediate data, compressing outputs, moving data between scratch and project space, archiving old data) to stay within limits.

**Tracking Usage:** HPC centers provide tools for users to track their resource consumption. Commands like `sacct` (SLURM) or `qacct` (PBS/SGE variants) can often show historical job information, including the actual walltime used and resources consumed. Specific commands or web portals might be available to check the current balance of compute allocations (core-hours remaining) and storage quotas. Regularly monitoring usage helps in planning future computations and avoiding unexpected interruptions due to exceeding limits.

**Requesting Resources:** Obtaining allocations usually involves submitting proposals to allocation committees or institutional resource managers, justifying the computational needs based on scientific goals. Renewing allocations typically requires demonstrating productive use of previous allocations, often evidenced by publications, presentations, or progress reports acknowledging the facility. Understanding the allocation process and cycles at your institution is important for securing the necessary resources for large computational projects.

**Consequences of Exceeding Limits:** Running jobs that significantly exceed requested time limits might be automatically killed by the scheduler. Exceeding storage quotas prevents writing new data. Consistently exceeding fair-share usage might lead to lower job priority for subsequent submissions. Being a responsible user involves accurately estimating resource needs, requesting appropriate allocations, monitoring usage, and actively managing data storage to operate within the established limits and policies of the shared facility.

In summary, HPC resources are typically allocated based on compute time (core-hours) and storage space, subject to quotas and fair-share policies managed by the job scheduler and system administrators. Understanding the specific allocation model, monitoring usage, respecting quotas, and accurately requesting resources in job submission scripts are essential practical aspects of effectively and responsibly utilizing shared HPC environments for large-scale astrophysical simulations and data analysis. Consulting local HPC documentation and support staff is crucial for navigating specific policies and procedures.

**Application 37.A: Writing and Submitting a Basic SLURM Job Script**

**(Paragraph 1)** **Objective:** This application provides a hands-on, practical example of creating a basic job submission script for the widely used **SLURM** scheduler (Sec 37.5) and submitting it to run a simple task (like a short Python script) on a compute node within an HPC environment. It reinforces understanding of scheduler directives, resource requests, module loading, and job submission/monitoring commands.

**(Paragraph 2)** **Astrophysical Context:** Many astrophysical analysis tasks, even if not requiring massive parallelism, might still benefit from being run as batch jobs on an HPC cluster. This could be because the task takes a moderately long time (e.g., several hours), requires more memory than available locally, needs specific software versions managed by modules, or is part of a larger automated workflow. Learning to submit even simple serial or single-node parallel jobs via the scheduler is a fundamental HPC skill. We'll simulate running a Python script that performs a hypothetical calculation.

**(Paragraph 3)** **Data Source:** Not applicable for data input. The "input" is the Python script to be executed (`my_python_script.py`) and the SLURM submission script (`submit.slurm`) itself. The Python script should be simple but perhaps take a minute or two to run (e.g., perform a loop with some calculations) and produce some output to a file.

**(Paragraph 4)** **Modules Used:**
*   **Shell:** Standard Linux shell (`bash`) for the script itself and commands like `sbatch`, `squeue`, `scancel`, `cat`, `ls`.
*   **Text Editor:** For creating the Python script and the SLURM script (e.g., `nano`, `vim`, `emacs` used via SSH on the login node).
*   **Python:** The script `my_python_script.py` will use basic Python, potentially `numpy` or `time`.
*   **Environment Modules:** The `module` command within the SLURM script.

**(Paragraph 5)** **Technique Focus:** Creating a SLURM submission script (`.slurm` file). Writing `#SBATCH` directives to request specific resources (nodes, tasks, CPUs, memory, time). Using `module load` within the script to set up the required software environment (e.g., a specific Python version). Writing the command(s) to execute the Python script. Submitting the script using `sbatch`. Monitoring the job's status using `squeue`. Checking the output files generated by the job.

**(Paragraph 6)** **Processing Step 1: Create Python Script:** Create a simple Python script named `my_python_script.py`. It should perform some noticeable work and write output to a file.
```python
# my_python_script.py
import time
import numpy as np
import socket # To show hostname

start_time = time.time()
hostname = socket.gethostname()
print(f"Python script running on node: {hostname}")
print("Performing a dummy calculation...")
result = 0
for i in range(10**7): # Loop to take some time
    result += np.sin(i * 0.01) 
    
end_time = time.time()
duration = end_time - start_time
print(f"Calculation finished. Result = {result:.5f}")
print(f"Duration: {duration:.2f} seconds.")

# Write result to an output file
output_filename = "script_output.txt"
with open(output_filename, "w") as f:
    f.write(f"Hostname: {hostname}\n")
    f.write(f"Result: {result}\n")
    f.write(f"Duration: {duration}\n")
print(f"Output written to {output_filename}")
```

**(Paragraph 7)** **Processing Step 2: Create SLURM Script:** Create a file named `submit.slurm` using a text editor on the cluster's login node. Populate it with `#SBATCH` directives and the execution commands.
```bash
#!/bin/bash

#SBATCH --job-name=SimplePyJob   # Job name
#SBATCH --output=pyjob_%j.out    # Standard output log (%j = job ID)
#SBATCH --error=pyjob_%j.err     # Standard error log
#SBATCH --partition=debug        # Use a debug/short queue if available
#SBATCH --nodes=1                # Request 1 node
#SBATCH --ntasks=1               # Request 1 task (process)
#SBATCH --cpus-per-task=1        # Request 1 CPU core for the task
#SBATCH --mem=1G                 # Request 1 GB of memory
#SBATCH --time=00:05:00          # Request max 5 minutes walltime

echo "SLURM JOB: ${SLURM_JOB_ID}"
echo "Running on host: $(hostname)"
echo "Loading Python module..."

# Load the required Python module (adjust version as needed)
module purge
module load python/3.9.6 # Or Anaconda, etc.
echo "Python module loaded. Version: $(python --version)"

# Define the script to run
PYTHON_SCRIPT="my_python_script.py"
echo "Running Python script: ${PYTHON_SCRIPT}"

# Execute the Python script
python $PYTHON_SCRIPT

EXIT_CODE=$? # Get exit code of the python script
echo "Python script finished with exit code: $EXIT_CODE"

echo "SLURM job finished."
```

**(Paragraph 8)** **Processing Step 3: Submit Job:** Ensure both `my_python_script.py` and `submit.slurm` are in your current directory on the login node. Submit the job using the command: `sbatch submit.slurm`. The scheduler will respond with a job ID (e.g., `Submitted batch job 12345`).

**(Paragraph 9)** **Processing Step 4: Monitor Job:** Check the job's status using `squeue -u $USER` (replace `$USER` with your username if needed). The status will likely be `PENDING` (PD) initially, then change to `RUNNING` (R) once resources are allocated and the job starts on a compute node. It might run very quickly if submitted to a debug queue. You can also use `scontrol show job <jobid>` for detailed status.

**(Paragraph 10)** **Processing Step 5: Check Output:** Once the `squeue` command shows the job is no longer running (it disappears or shows state `COMPLETED` (CD)), check the contents of the output files created in the submission directory:
*   `ls pyjob_*`: Should show `pyjob_<jobid>.out` and `pyjob_<jobid>.err`.
*   `cat pyjob_<jobid>.out`: Should contain the `echo` messages from the script and the standard output printed by the Python script (e.g., "Python script running on node: compute-node-X...", "Calculation finished...", "Output written...").
*   `cat pyjob_<jobid>.err`: Should ideally be empty if the job ran without errors.
*   `cat script_output.txt`: Should contain the results written by the Python script itself.

**Output, Testing, and Extension:** The outputs are the files generated by the SLURM job: `pyjob_<jobid>.out`, `pyjob_<jobid>.err`, and `script_output.txt`. **Testing:** Verify the job runs and completes successfully (check exit code in `.out` file, should be 0). Check the contents of the `.out` file show the expected messages, including the hostname of a compute node (different from the login node). Verify `script_output.txt` contains the correct calculated results. **Extensions:** (1) Modify the Python script to accept a command-line argument (using `sys.argv` or `argparse`) and pass an argument to it from the SLURM script (e.g., `python $PYTHON_SCRIPT 100`). (2) Modify the `#SBATCH` directives to request more resources (e.g., more time, more memory) and observe the effect on queue time or execution. (3) Make the Python script use multiple cores (e.g., using `multiprocessing`) and request `--cpus-per-task=N` in the SLURM script accordingly. (4) Create a job array (`#SBATCH --array=1-5`) to run the same script multiple times with different inputs controlled by the `$SLURM_ARRAY_TASK_ID` environment variable.

```python
# --- Code Example: Application 37.A ---
# This combines the Python script and SLURM script content for clarity.
# In practice, these would be two separate files.

# --- 1. Python Script (my_python_script.py) ---
python_script_content = """
import time
import numpy as np
import socket # To show hostname
import sys # To potentially get arguments

print(f"--- Start Python Script ---")
start_time = time.time()
hostname = socket.gethostname()
job_id = os.getenv('SLURM_JOB_ID', 'N/A')
task_id = os.getenv('SLURM_ARRAY_TASK_ID', 'N/A') # For job arrays

print(f"Running on host: {hostname}")
print(f"SLURM Job ID: {job_id}")
if task_id != 'N/A': print(f"SLURM Array Task ID: {task_id}")

print("Performing a dummy calculation...")
# Use argument if provided, otherwise default loop size
loop_size = int(float(sys.argv[1])) if len(sys.argv) > 1 else 10**7
print(f"Loop size: {loop_size}")

result = 0
for i in range(loop_size): # Loop to take some time
    result += np.sin(i * 0.01) 
    
end_time = time.time()
duration = end_time - start_time
print(f"Calculation finished. Result = {result:.5f}")
print(f"Duration: {duration:.2f} seconds.")

# Write result to an output file named uniquely if in an array job
output_filename = f"script_output_{job_id}_{task_id}.txt" if task_id != 'N/A' else f"script_output_{job_id}.txt"
try:
    with open(output_filename, "w") as f:
        f.write(f"Hostname: {hostname}\\n")
        f.write(f"Job ID: {job_id}\\n")
        if task_id != 'N/A': f.write(f"Array Task ID: {task_id}\\n")
        f.write(f"Loop Size: {loop_size}\\n")
        f.write(f"Result: {result}\\n")
        f.write(f"Duration: {duration}\\n")
    print(f"Output written to {output_filename}")
except Exception as e:
     print(f"Error writing output file: {e}")

print(f"--- End Python Script ---")
"""
print("--- Content for: my_python_script.py ---")
print(python_script_content)
# Note: Need to save this exact content to a file named 'my_python_script.py'

# --- 2. SLURM Submission Script (submit.slurm) ---
slurm_script_content = """#!/bin/bash

#SBATCH --job-name=SimplePyJob   # Job name
#SBATCH --output=pyjob_%j.out    # Standard output log (%j = job ID)
#SBATCH --error=pyjob_%j.err     # Standard error log
#SBATCH --partition=debug        # Use a debug/short queue if available
#SBATCH --nodes=1                # Request 1 node
#SBATCH --ntasks=1               # Request 1 task (process)
#SBATCH --cpus-per-task=1        # Request 1 CPU core for the task
#SBATCH --mem=1G                 # Request 1 GB of memory
#SBATCH --time=00:05:00          # Request max 5 minutes walltime
#SBATCH --account=your_account   # IMPORTANT: Replace with your actual account/allocation name

echo "======================================================"
echo "SLURM JOB: ${SLURM_JOB_ID}"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Job submitted from: ${SLURM_SUBMIT_DIR}"
echo "======================================================"

echo "Loading modules..."
# Load the required Python module (adjust version as needed for your cluster)
module purge
module load python/3.9 # Or anaconda module etc.
echo "Python module loaded. Version: $(python --version)"
echo "------------------------------------------------------"

# Define the script to run (assuming it's in the same directory as submit.slurm)
PYTHON_SCRIPT="my_python_script.py"

# Check if script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "ERROR: Python script '$PYTHON_SCRIPT' not found!"
  exit 1
fi

echo "Running Python script: ${PYTHON_SCRIPT}"
# Execute the Python script (no arguments needed for default loop size)
python $PYTHON_SCRIPT 
# To pass an argument: python $PYTHON_SCRIPT 5000000 

EXIT_CODE=$? # Get exit code of the python script
echo "------------------------------------------------------"
echo "Python script finished with exit code: $EXIT_CODE"
echo "======================================================"
echo "SLURM job finished."
"""
print("\n--- Content for: submit.slurm ---")
print(slurm_script_content)
# Note: Need to save this content to 'submit.slurm' and replace '--account'

# --- 3. Commands (To run in SHELL on HPC login node) ---
shell_commands_run = """
# Ensure both files exist in the current directory
# ls -l my_python_script.py submit.slurm 

# Submit the job (replace --account in script first!)
# sbatch submit.slurm
# (Note the Job ID, e.g., 12345)

# Monitor the job
# squeue -j 12345 
# (Wait until it disappears or shows COMPLETED)

# Check the output files after completion
# cat pyjob_12345.out
# cat pyjob_12345.err  (Should be empty ideally)
# cat script_output_12345_N/A.txt 
"""
print("\n--- Example SHELL Commands ---")
print(shell_commands_run)
print("-" * 20)

# Explanation: 
# 1. Provides the content for the Python script `my_python_script.py`. This script 
#    prints hostname/job info, runs a calculation loop, prints duration, and writes 
#    results to a unique file including the job ID. It can optionally take a command-line 
#    argument to change the loop size.
# 2. Provides the content for the SLURM script `submit.slurm`. This includes 
#    `#SBATCH` directives for resources, prints job information, loads a Python module, 
#    checks for the script's existence, executes the Python script, and reports the exit code. 
#    Crucially, the user MUST replace `--account=your_account` with their valid allocation.
# 3. Shows the shell commands needed on the HPC login node to submit the job (`sbatch`), 
#    monitor it (`squeue`), and check the resulting output files (`cat`).
# This provides a complete, runnable example (after saving files and setting account).
```

**Application 37.B: Basic Cluster Interaction: Login, Modules, File Transfer**

**(Paragraph 1)** **Objective:** This application focuses on the fundamental steps required to begin working on an HPC cluster: establishing a secure connection, navigating the file system, checking and loading necessary software modules, and transferring a data file from a local machine to the cluster. Reinforces Sec 37.4.

**(Paragraph 2)** **Astrophysical Context:** Before running simulations or analysis jobs, researchers need to set up their environment on the HPC cluster. This involves logging in, creating project directories, ensuring required software (like specific Python versions, compilers, or astronomical libraries like Astropy) is accessible, and transferring necessary input data files (e.g., initial conditions, observational data files like FITS images or catalogs) from their local machine or other sources to the cluster's file system. Mastering these basic interactive steps is a prerequisite for any HPC work.

**(Paragraph 3)** **Data Source:** A small-to-medium sized data file residing on the user's **local** machine (e.g., `my_data.fits` or `input_params.txt`). Access credentials (username, cluster hostname, potentially password or SSH key setup) for a target HPC cluster.

**(Paragraph 4)** **Modules Used:**
*   **Local Machine:** SSH client (command-line `ssh` or GUI like PuTTY), Secure Copy client (`scp` command-line utility or GUI equivalent like WinSCP).
*   **HPC Cluster (via SSH):** Standard Linux shell commands (`pwd`, `ls`, `cd`, `mkdir`, `df`), Environment Module command (`module`).

**(Paragraph 5)** **Technique Focus:** Demonstrating the sequence of standard command-line operations for basic cluster interaction: (1) Using `ssh` to connect securely from local machine to cluster login node. (2) Using `pwd`, `ls`, `cd`, `mkdir` to navigate the remote file system and create directories. (3) Using `df -h` or `quota` to check storage space. (4) Using `module avail` to see available software and `module load` to activate specific modules (e.g., Python, Astropy). (5) Using `scp` (run on the local machine) to securely transfer a file from local storage to a specific directory on the remote cluster. (6) Verifying the transfer using `ls` on the cluster.

**(Paragraph 6)** **Processing Step 1: SSH Connection (Local Terminal):** Open a terminal on your local machine. Use the `ssh` command with your username and the cluster's hostname: `ssh your_username@cluster.hpc.institution.edu`. Enter your password when prompted, or ensure your SSH key is set up for passwordless login. You should see a welcome message and a command prompt indicating you are now on the cluster's login node.

**(Paragraph 7)** **Processing Step 2: Navigate and Check Environment (Remote Terminal):** Once logged in:
    *   Check your location: `pwd` (likely `/home/your_username`).
    *   Navigate to your project or scratch space: `cd /scratch/your_username/my_astro_project` (create directories using `mkdir -p /scratch/your_username/my_astro_project` if they don't exist). Use appropriate paths for your cluster.
    *   Check disk space: `df -h .` (shows space for the current file system). Check quotas if necessary (`quota -s`).
    *   List directory contents: `ls -l`.

**(Paragraph 8)** **Processing Step 3: Manage Software Modules (Remote Terminal):**
    *   See what software is available: `module avail` (might show many entries, can filter e.g., `module avail python`).
    *   Load needed modules: `module load python/3.10` then `module load astropy/5.3-python3.10` (module names and versions are specific to each cluster). Use `module purge` first for a clean slate if needed.
    *   Verify loading: `module list`. Check software versions: `python --version`, `python -c "import astropy; print(astropy.__version__)"`.

**(Paragraph 9)** **Processing Step 4: Transfer File (Local Terminal):** Open *another* terminal window on your **local** machine (or use a separate SCP client application). Use the `scp` command to copy the file:
`scp /path/to/local/my_data.fits your_username@cluster.hpc.institution.edu:/scratch/your_username/my_astro_project/`
Replace paths and hostname appropriately. This command copies `my_data.fits` from your local machine to the specified directory on the cluster. Enter your password if prompted (or use SSH keys). For larger files or resuming transfers, `rsync -avzP source destination` is often preferred over `scp`.

**(Paragraph 10)** **Processing Step 5: Verify Transfer (Remote Terminal):** Go back to your SSH session on the cluster. Navigate to the target directory (`cd /scratch/your_username/my_astro_project/`). Use `ls -lh` to list files with sizes. Verify that `my_data.fits` is now present and has the expected file size. You can now use this file in job scripts running on the cluster. Log out using `exit`.

**Output, Testing, and Extension:** The "output" is the successful execution of these commands and the presence of the transferred file on the cluster. **Testing:** Confirm successful login. Verify `module list` shows the loaded modules. Ensure `scp` completes without errors and `ls -lh` on the cluster confirms the file's arrival and size. Test transferring a file *from* the cluster *to* your local machine (reverse source and destination in `scp`). **Extensions:** (1) Set up SSH keys for passwordless login. (2) Use `rsync` instead of `scp` and explore its options (e.g., `--progress`, `-z` for compression). (3) Practice editing a file on the cluster using a terminal editor like `nano` or `vim`. (4) Create a simple script on the cluster that uses the loaded Python/Astropy modules to open the transferred FITS file. (5) Explore tools like `sshfs` (on Linux/macOS) to mount the remote cluster file system locally for easier graphical file management (use with caution for performance).

```python
# --- Code Example: Application 37.B ---
# This shows the sequence of SHELL commands executed on LOCAL and REMOTE terminals.

# --- Assumed Setup ---
# LOCAL Machine: File 'local_data.fits' exists in current directory.
# REMOTE HPC: Account 'user123', hostname 'hpc.example.edu', 
#             home dir '/home/user123', scratch '/scratch/user123'.
#             Modules 'python/3.9.12' and 'astropy/5.1-python3.9' exist.

print("--- Basic Cluster Interaction Workflow (Shell Commands) ---")

# === Step 1: Connect (Run on LOCAL terminal) ===
prompt_ssh = "# ssh user123@hpc.example.edu"
print(f"\n1. Connect via SSH (on LOCAL terminal):\n   {prompt_ssh}")
print("   (Enter password or use SSH key)")

# === Step 2 & 3: Navigate and Check Modules (Run on REMOTE terminal after login) ===
remote_commands_1 = """
# pwd
# module purge
# module avail python/3.9
# module load python/3.9.12
# module load astropy/5.1-python3.9 # Assumes this depends on python/3.9.12
# module list
# python --version
# python -c "import astropy; print(f'Astropy version: {astropy.__version__}')"
# mkdir -p /scratch/user123/my_project/data # Create directories
# cd /scratch/user123/my_project/data
# pwd
# df -h .
"""
print(f"\n2 & 3. Navigate & Setup Modules (on REMOTE terminal):\n   (Commands represented by following printout)")
for cmd in remote_commands_1.strip().split('\n'): print(f"   remote$ {cmd}")

# === Step 4: Transfer File (Run on LOCAL terminal) ===
local_file = 'local_data.fits' # Assume this exists locally
remote_path = 'user123@hpc.example.edu:/scratch/user123/my_project/data/'
# Create a dummy local file for demo completeness
with open(local_file, 'w') as f: f.write("Dummy FITS data simulation") 
print(f"\n4. Transfer File (on LOCAL terminal):")
print(f"   (Ensured '{local_file}' exists locally)")
prompt_scp = f"# scp {local_file} {remote_path}"
print(f"   {prompt_scp}")
print("   (Enter password or use SSH key again)")

# === Step 5: Verify Transfer (Run on REMOTE terminal) ===
remote_commands_2 = f"""
# ls -lh 
# cat {local_file}  # Verify content (optional for small files)
"""
print(f"\n5. Verify Transfer (on REMOTE terminal):")
for cmd in remote_commands_2.strip().split('\n'): print(f"   remote$ {cmd}")

# Clean up dummy local file
if os.path.exists(local_file): os.remove(local_file)

print("\n--- Workflow Complete ---")
print("-" * 20)

# Explanation:
# This block simulates the interactive shell commands for basic HPC interaction.
# 1. Shows the `ssh` command run locally.
# 2. Shows commands run remotely after login: checking path (`pwd`), cleaning/loading 
#    specific `module` versions for Python and Astropy, verifying versions, creating 
#    a project directory in scratch space, moving into it (`cd`), and checking disk space (`df`).
# 3. Shows the `scp` command run locally to copy `local_data.fits` to the specific 
#    directory created on the cluster. A dummy local file is created first.
# 4. Shows commands run remotely (`ls -lh`) to verify the file arrived successfully.
# 5. Cleans up the local dummy file.
# This sequence demonstrates the essential steps for connecting, setting up the 
# environment, and transferring data to an HPC cluster.
```

**Chapter 37 Summary**

This chapter served as an essential introduction to High-Performance Computing (HPC) environments, motivating their necessity for tackling the large-scale computational challenges prevalent in modern astrophysics, such as complex simulations and the analysis of petabyte-scale datasets that exceed the capabilities of local workstations in terms of processing power, memory, and storage I/O. It dissected the typical **anatomy of an HPC cluster**, describing the roles of login nodes (user interaction, job submission), compute nodes (CPU, GPU, high-memory for executing jobs), the high-speed interconnect network crucial for parallel communication, tiered storage systems (home, project, high-performance scratch), the central **job scheduler** (like SLURM or PBS/Torque) managing resource allocation and batch job execution, and the **environment module system** used for managing different versions of software and libraries. The fundamental difference between **shared memory architectures** (within a node, parallelized using threads/OpenMP) and **distributed memory architectures** (across nodes, parallelized using MPI) was explained, noting that most clusters employ a hybrid model.

Practical aspects of **accessing and interacting** with HPC resources were covered, emphasizing secure connection via **SSH**, essential Linux command-line navigation (`cd`, `ls`, `pwd`, `mkdir`), awareness of different file systems and their usage policies, and the critical use of the `module` command (`module load`, `module avail`, `module list`) to configure the software environment required for specific tasks (e.g., loading correct compiler, MPI library, or Python/Astropy versions). The concept of **batch processing** via job schedulers was detailed, explaining how users write submission scripts containing resource requests (`#SBATCH` or `#PBS` directives specifying nodes, cores, memory, time) and the commands to be executed non-interactively on compute nodes. Submitting (`sbatch`, `qsub`), monitoring (`squeue`, `qstat`), and checking the output of these jobs were outlined. Finally, the chapter briefly discussed the concepts of **compute resource allocations** (e.g., core-hours) and storage **quotas**, highlighting the need for users to monitor their usage and manage resources responsibly within the shared HPC environment.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Your Local HPC Center's Documentation. (n.d.).** *(e.g., University IT Research Computing, National Supercomputing Center website)*.
    *(This is arguably the *most* important reference. Specific details about login hostnames, file system paths/quotas, available modules, the exact scheduler used (SLURM/PBS), queue names, resource limits, and usage policies are cluster-specific. Always consult your local documentation first.)*

2.  **SLURM Workload Manager. (n.d.).** *SLURM Documentation*. SchedMD. Retrieved January 16, 2024, from [https://slurm.schedmd.com/](https://slurm.schedmd.com/)
    *(Official documentation for the widely used SLURM scheduler, covering commands (`sbatch`, `squeue`, `scancel`, `sinfo`, `sacct`), submission script directives (`#SBATCH`), and configuration options, relevant to Sec 37.5.)*

3.  **Downey, A. B. (2016).** *Think HPC: Programming and Problem Solving on High Performance Computing Clusters*. O'Reilly Media (Often available via Green Tea Press online: [https://greenteapress.com/wp/think-hpc/](https://greenteapress.com/wp/think-hpc/)).
    *(A practical, accessible introduction to using HPC clusters, covering concepts like SSH, shells, modules, schedulers, parallel architectures, and basic parallel programming ideas, suitable background for this entire Part.)*

4.  **Eijkhout, V. (2022).** *Introduction to High Performance Scientific Computing*. (Online textbook). [http://pages.tacc.utexas.edu/~eijkhout/istc/istc.html](http://pages.tacc.utexas.edu/~eijkhout/istc/istc.html)
    *(A detailed online textbook covering HPC architectures, parallel programming models (MPI, OpenMP), performance analysis, and numerical algorithms, providing deeper background for concepts introduced here and expanded in later chapters.)*

5.  **Tanenbaum, A. S., & Austin, T. (2013).** *Structured Computer Organization* (6th ed.). Pearson.
    *(While a general computer architecture textbook, chapters on processor architecture, memory hierarchy, and parallel computer architectures provide fundamental background for understanding shared vs. distributed memory systems discussed in Sec 37.3.)*
