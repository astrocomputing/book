**Chapter 39: Distributed Computing with MPI and `mpi4py`**

Building upon the fundamental concepts of parallel computing introduced previously, this chapter focuses specifically on the dominant paradigm for large-scale parallel processing on distributed memory High-Performance Computing (HPC) clusters: the **Message Passing Interface (MPI)**. When problems exceed the memory or computational capacity of a single shared-memory node, MPI provides the standard mechanism for multiple independent processes, often running on different physical nodes, to coordinate and communicate by explicitly sending and receiving messages over the network. We will delve into the core concepts of the MPI model, including communicators (`MPI.COMM_WORLD`), process ranks and sizes, and the different modes of **point-to-point communication** (`send`/`recv`, blocking vs. non-blocking). We will then explore essential **collective communication** operations (`bcast`, `scatter`, `gather`, `reduce`) that involve coordinated data movement or computation among all processes in a communicator. Our practical focus will be on using MPI from Python through the **`mpi4py`** library, demonstrating how to initialize MPI, determine rank/size, and send/receive both standard Python objects (requiring pickling) and, crucially, NumPy arrays efficiently. We illustrate how to parallelize simple tasks and loops by distributing work based on rank, touching upon basic load balancing ideas and common domain decomposition strategies used in simulations. Finally, we cover the practicalities of launching `mpi4py` scripts on HPC clusters using commands like `mpirun` or `srun`.

**39.1 MPI Model: Communicator, Rank, Size, Point-to-Point Communication**

The Message Passing Interface (MPI) is a specification for a library of functions that enables parallel programming on distributed memory systems. It standardizes how independent processes coordinate their work and exchange data by sending and receiving explicit messages. Understanding the core MPI execution model and basic communication mechanisms is fundamental to writing parallel programs that can scale across multiple compute nodes.

An MPI program typically follows the **Single Program, Multiple Data (SPMD)** model. The same executable code is launched simultaneously on multiple processors (cores), often across different nodes. Each instance of the running program is called an **MPI process**. Although running the same code, each process operates largely independently, having its own memory space and execution flow. Crucially, each process can determine its unique identity and its relationship to other processes within the defined communication group.

The primary concept defining a group of communicating processes is the **Communicator**. A communicator encapsulates a set of processes that can interact with each other. The most fundamental communicator is `MPI.COMM_WORLD` (in `mpi4py`), which automatically includes *all* processes launched when the MPI program starts. While more advanced applications might create sub-communicators involving subsets of processes, most basic MPI tasks operate within `MPI.COMM_WORLD`.

Within a communicator, each process is assigned a unique, non-negative integer identifier called its **rank**. Ranks typically range from 0 to `size - 1`, where `size` is the total number of processes in the communicator. The process with rank 0 often plays a special role, such as reading initial input, distributing work, gathering final results, or performing I/O. Every process can query the communicator to find out its own rank (`comm.Get_rank()` or `comm.rank` in `mpi4py`) and the total number of processes (`comm.Get_size()` or `comm.size`). This rank and size information is essential for partitioning work and coordinating communication patterns within the SPMD model (e.g., `if rank == 0: ... else: ...`).

Communication between processes occurs by sending and receiving **messages**. The fundamental type is **point-to-point communication**, where one specific process sends a message directly to another specific process. The two basic operations are **send** and **receive**.
*   **Send:** A process calls a send function (like `MPI_Send` or `comm.send`/`comm.Send` in `mpi4py`), specifying the data buffer to send, the number of elements, the data type, the rank of the **destination** process, a message **tag** (an integer used by the receiver to distinguish between different types of messages), and the communicator.
*   **Receive:** A process calls a receive function (like `MPI_Recv` or `comm.recv`/`comm.Recv` in `mpi4py`), specifying a buffer to store the incoming data, the expected number of elements, the data type, the rank of the **source** process (or `MPI.ANY_SOURCE` to receive from any process), the expected message tag (or `MPI.ANY_TAG`), and the communicator. The receive operation typically also returns a status object containing information about the received message (actual source rank, tag, number of elements received).

A crucial aspect of point-to-point communication is its **synchronization behavior**:
*   **Blocking Communication:** Standard `MPI_Send` and `MPI_Recv` are typically *blocking*. `MPI_Send` might not return control to the calling process until the message data has been safely copied out of the send buffer (either into internal MPI buffers or potentially until the receiver has started receiving). `MPI_Recv` *always* blocks until a matching message has actually arrived and been received into the specified buffer. This blocking behavior ensures data integrity but requires careful coordination between sender and receiver to avoid **deadlock**, where two or more processes are blocked waiting for each other to send/receive in a circular fashion (e.g., Process 0 tries to receive from 1 before sending to 1, while Process 1 tries to receive from 0 before sending to 0).
*   **Non-blocking Communication:** Functions like `MPI_Isend` (immediate send) and `MPI_Irecv` (immediate receive) initiate the communication operation but return control *immediately* to the calling process, allowing it to perform other computations while the message transfer occurs in the background (handled by the MPI library and network hardware). Non-blocking operations return a request object. The programmer must later explicitly check for the completion of the non-blocking operation using functions like `MPI_Wait` or `MPI_Test` on the request object before it is safe to reuse the send buffer (for `Isend`) or access the data in the receive buffer (for `Irecv`). While offering potential performance gains by overlapping communication and computation, non-blocking communication requires more complex program logic to manage request objects and ensure completion.

In `mpi4py`, lowercase methods like `comm.send(data, dest=...)` and `data = comm.recv(source=...)` handle arbitrary Python objects. They automatically **pickle** the object on the sending side and unpickle it on the receiving side, making it very convenient but potentially slow for large objects due to serialization overhead. Uppercase methods like `comm.Send(np_array, dest=...)` and `comm.Recv(np_array, source=...)` are designed for efficient communication of **NumPy arrays** (or other objects supporting the Python buffer protocol). They typically transfer the raw data bytes directly without pickling overhead, offering much higher performance for numerical data. When using `Send`/`Recv`, the receive buffer (`np_array` in `comm.Recv`) must already be allocated with the correct size and data type.

```python
# --- Code Example 1: Basic Point-to-Point Communication with mpi4py ---
# Note: Requires mpi4py and MPI environment. Run with e.g., mpirun -np 2 python script.py

try:
    from mpi4py import MPI
    import numpy as np
    mpi4py_installed = True
except ImportError:
    mpi4py_installed = False
    print("NOTE: mpi4py not installed or MPI environment not found. Skipping example.")

print("MPI Point-to-Point Communication Example (using mpi4py):")

if mpi4py_installed:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Ensure we run this example with exactly 2 processes
    if size != 2:
        if rank == 0:
            print("Error: This example requires exactly 2 MPI processes.")
        # All ranks exit if not size 2
        MPI.Finalize() # Cleanly exit MPI
        exit()

    # --- Using lowercase send/recv (Python objects via pickle) ---
    print(f"\nRank {rank}: Testing lowercase send/recv (pickled)...")
    if rank == 0:
        data_to_send = {'a': 1, 'b': [2.0, 3.0], 'c': 'hello'}
        print(f"Rank 0: Sending object: {data_to_send}")
        # Blocking send to rank 1, tag=0
        comm.send(data_to_send, dest=1, tag=0) 
        print("Rank 0: Object sent.")
        # Now receive response from rank 1, tag=1
        received_reply = comm.recv(source=1, tag=1)
        print(f"Rank 0: Received reply: {received_reply}")
    
    elif rank == 1:
        # Blocking receive from rank 0, tag=0
        received_object = comm.recv(source=0, tag=0)
        print(f"Rank 1: Received object: {received_object}")
        # Send a reply back
        reply_message = f"Rank 1 received object type {type(received_object)}"
        print(f"Rank 1: Sending reply: '{reply_message}'")
        comm.send(reply_message, dest=0, tag=1)
        print("Rank 1: Reply sent.")

    comm.Barrier() # Wait for both processes before next part

    # --- Using uppercase Send/Recv (NumPy arrays, faster) ---
    print(f"\nRank {rank}: Testing uppercase Send/Recv (NumPy array)...")
    # Rank 0 sends a NumPy array to Rank 1
    if rank == 0:
        send_array = np.arange(10, dtype=np.float64) * 1.1
        print(f"Rank 0: Sending NumPy array: {send_array}")
        # Blocking Send (capital S) requires buffer-like object
        comm.Send([send_array, MPI.DOUBLE], dest=1, tag=10) # Pass buffer info [data, MPI_TYPE]
        print("Rank 0: Array sent.")
    
    elif rank == 1:
        # Allocate receive buffer *before* receiving
        recv_array = np.empty(10, dtype=np.float64) 
        print("Rank 1: Receiving NumPy array...")
        # Blocking Recv (capital R) fills the pre-allocated buffer
        comm.Recv([recv_array, MPI.DOUBLE], source=0, tag=10) # Pass buffer info
        print(f"Rank 1: Received NumPy array: {recv_array}")

else:
    print("Skipping MPI execution.")

print("-" * 20)

# How to Run:
# mpirun -np 2 python your_script_name.py
# Explanation: This code demonstrates basic point-to-point communication between two MPI processes.
# 1. It first checks if exactly 2 processes were launched.
# 2. Lowercase `send`/`recv`: Rank 0 sends a Python dictionary `data_to_send` to rank 1 
#    using `comm.send()`. Rank 1 receives it using `comm.recv()`. This involves pickling/unpickling. 
#    Rank 1 then sends a string reply back to rank 0. This demonstrates the ease of sending 
#    general Python objects but highlights the blocking nature (rank 1 waits for rank 0's send).
# 3. Uppercase `Send`/`Recv`: Rank 0 creates a NumPy array `send_array`. It sends this using 
#    `comm.Send()`, providing the buffer (`send_array`) and its MPI datatype (`MPI.DOUBLE`). 
#    Rank 1 pre-allocates a NumPy array `recv_array` of the correct size and type. It then 
#    calls `comm.Recv()`, providing the buffer to be filled. This method avoids pickling 
#    and is much more efficient for large numerical arrays.
# `comm.Barrier()` is used to synchronize before the second part.
```

Point-to-point communication forms the basis for many parallel algorithms where processes need to exchange specific data with specific partners, such as exchanging boundary data in domain decomposition schemes used in simulations or implementing distributed algorithms involving direct peer-to-peer interaction. Understanding blocking vs. non-blocking semantics and choosing efficient methods for transferring data types (pickle vs. buffer protocol/NumPy) are key practical aspects of using MPI effectively via `mpi4py`.

**39.2 Collective Communication (`bcast`, `scatter`, `gather`, `reduce`)**

While point-to-point communication handles data exchange between specific pairs of MPI processes, many parallel algorithms require operations involving *all* processes within a communicator simultaneously. **Collective communication** routines in MPI provide optimized implementations for these common group-wide data movement and computation patterns. Using collective operations is often simpler to code and potentially much more efficient than implementing the same pattern using multiple point-to-point messages, as MPI libraries can leverage knowledge of the underlying network topology for optimization.

Several fundamental collective operations exist:
*   **`MPI_Bcast` / `comm.bcast` / `comm.Bcast`:** **Broadcast.** Sends data from a single designated **root** process (specified by rank) to *all* other processes (including the root itself) within the communicator. Useful for distributing initial parameters, configuration data, or common arrays read by one process to all workers. `comm.bcast` (lowercase) handles pickleable Python objects, while `comm.Bcast` (uppercase) efficiently broadcasts buffer-like objects like NumPy arrays (the receive buffer must be pre-allocated on non-root ranks).

*   **`MPI_Scatter` / `comm.scatter` / `comm.Scatter`:** **Scatter.** Takes an array (or list for lowercase `scatter`) existing on the **root** process, splits it into equal chunks, and sends a *different* chunk to each process in the communicator (including the root). Useful for distributing distinct parts of a dataset or workload across processes at the beginning of a parallel computation (e.g., distributing rows of a matrix or chunks of a particle list). Again, lowercase `scatter` handles general objects (via pickle), uppercase `Scatter` handles buffer-like objects (like NumPy arrays) efficiently, requiring pre-allocated receive buffers on all ranks.

*   **`MPI_Gather` / `comm.gather` / `comm.Gather`:** **Gather.** The inverse of Scatter. Each process sends its chunk of data to a designated **root** process. The root process collects these chunks and assembles them into a larger array (or list for lowercase `gather`) in rank order. Useful for collecting partial results computed independently by each process back onto a single process for final processing or output. Lowercase `gather` returns a list of objects on the root, uppercase `Gather` requires a pre-allocated receive buffer on the root to store the combined buffer-like data.

*   **`MPI_Allgather` / `comm.allgather` / `comm.Allgather`:** **Allgather.** Similar to Gather, but the assembled result (containing data from all processes) is delivered to *all* processes in the communicator, not just the root. Useful when all processes need access to the complete, combined dataset after a distributed computation phase.

*   **`MPI_Reduce` / `comm.reduce` / `comm.Reduce`:** **Reduce.** Combines data from all processes using a specified **reduction operation** (like sum, product, maximum, minimum, logical AND/OR) and delivers the single final result to a designated **root** process. Essential for computing global sums, finding global maxima/minima, or performing other aggregate calculations across distributed data. MPI defines standard operations (`MPI.SUM`, `MPI.MAX`, `MPI.MIN`, `MPI.PROD`, `MPI.LAND`, `MPI.LOR`, etc.). Lowercase `reduce` works with common Python operators on single items, uppercase `Reduce` operates element-wise on NumPy arrays using MPI operations, requiring pre-allocated send and receive buffers.

*   **`MPI_Allreduce` / `comm.allreduce` / `comm.Allreduce`:** **Allreduce.** Performs a reduction operation but delivers the final combined result to *all* processes in the communicator, not just the root. Useful when all processes need the result of a global calculation (e.g., a global sum needed for normalization).

*   **`MPI_Barrier` / `comm.Barrier`:** **Barrier Synchronization.** A simple synchronization point. When a process calls Barrier, it blocks until *all* other processes in the communicator have also called Barrier. Useful for ensuring all processes have completed a certain phase before proceeding, often used for timing or debugging, but should be used sparingly as it can introduce idle time.

These collective operations are fundamental building blocks for structuring SPMD (Single Program, Multiple Data) parallel algorithms. They provide efficient, standardized ways to handle common patterns of data distribution (Bcast, Scatter), data collection (Gather, Allgather), global computation (Reduce, Allreduce), and synchronization (Barrier). `mpi4py` provides Pythonic interfaces to these operations, distinguishing between methods for general Python objects (lowercase, using pickle) and optimized methods for NumPy arrays (uppercase, using buffer protocol).

```python
# --- Code Example 1: Using MPI Collectives with mpi4py ---
# Note: Requires mpi4py and MPI environment. Run with e.g., mpirun -np 4 python script.py

try:
    from mpi4py import MPI
    import numpy as np
    mpi4py_installed = True
except ImportError:
    mpi4py_installed = False
    print("NOTE: mpi4py not installed or MPI environment not found. Skipping example.")

print("MPI Collective Communication Example (using mpi4py):")

if mpi4py_installed:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0 # Define rank 0 as the root process

    # --- Broadcast Example ---
    print(f"\nRank {rank}: Broadcast Example...")
    if rank == root:
        data_to_bcast = {'config': 'value', 'param': 12.3} # Data only exists on root initially
    else:
        data_to_bcast = None
    # Broadcast Python object from root to all
    data_received_bcast = comm.bcast(data_to_bcast, root=root)
    print(f"Rank {rank}: Received broadcast data: {data_received_bcast}")
    comm.Barrier() # Synchronize before next step

    # --- Scatter Example (NumPy array) ---
    print(f"\nRank {rank}: Scatter Example...")
    send_buffer_scatter = None
    if rank == root:
        # Create data array on root, size must be multiple of 'size' for simple Scatter
        total_elements = size * 3 # e.g., 3 elements per process
        send_buffer_scatter = np.arange(total_elements, dtype=np.float64) * (rank + 1.1)
        print(f"Rank {root}: Sending Scatter buffer: {send_buffer_scatter}")
    # Create receive buffer on *all* ranks
    recv_buffer_scatter = np.empty(3, dtype=np.float64) # Each rank receives 3 elements
    # Scatter NumPy array data from root to all ranks
    comm.Scatter(send_buffer_scatter, recv_buffer_scatter, root=root)
    print(f"Rank {rank}: Received Scatter data: {recv_buffer_scatter}")
    comm.Barrier()

    # --- Gather Example (NumPy array) ---
    print(f"\nRank {rank}: Gather Example...")
    # Each rank has some data to send (e.g., its received scattered data squared)
    send_buffer_gather = recv_buffer_scatter**2 
    print(f"Rank {rank}: Sending Gather data: {send_buffer_gather}")
    recv_buffer_gather = None
    if rank == root:
        # Allocate receive buffer *only on root*, large enough for all data
        recv_buffer_gather = np.empty(size * 3, dtype=np.float64) 
    # Gather NumPy array data from all ranks onto root
    comm.Gather(send_buffer_gather, recv_buffer_gather, root=root)
    if rank == root:
        print(f"Rank {root}: Received Gather data: {recv_buffer_gather}")
    comm.Barrier()

    # --- Reduce Example (Sum) ---
    print(f"\nRank {rank}: Reduce Example...")
    local_value = float(rank * 10 + np.random.randint(5)) # Each rank has a local value
    print(f"Rank {rank}: Local value = {local_value}")
    # Calculate global sum across all ranks, result only on root
    global_sum = comm.reduce(local_value, op=MPI.SUM, root=root)
    if rank == root:
        print(f"Rank {root}: Global sum (via reduce) = {global_sum}")
    comm.Barrier()
        
    # --- Allreduce Example (Max) ---
    print(f"\nRank {rank}: Allreduce Example...")
    local_max = float(rank + np.random.rand() * 10)
    print(f"Rank {rank}: Local max = {local_max:.2f}")
    # Calculate global maximum, result available on *all* ranks
    global_max = comm.allreduce(local_max, op=MPI.MAX)
    print(f"Rank {rank}: Global maximum (via allreduce) = {global_max:.2f}")

else:
    print("Skipping MPI execution.")

print("-" * 20)

# How to Run: mpirun -np 4 python your_script_name.py
# Explanation: This code demonstrates several MPI collective operations using mpi4py.
# 1. Broadcast: Rank 0 creates a dictionary `data_to_bcast`. `comm.bcast` sends this 
#    object to all other ranks. Each rank prints the received data.
# 2. Scatter: Rank 0 creates a NumPy array `send_buffer_scatter`. `comm.Scatter` (uppercase 
#    for NumPy) splits this array and sends a unique chunk to each rank, which receives 
#    it into its pre-allocated `recv_buffer_scatter`.
# 3. Gather: Each rank squares its received chunk (`send_buffer_gather`). `comm.Gather` 
#    collects these chunks from all ranks onto rank 0's pre-allocated `recv_buffer_gather`. 
#    Only rank 0 prints the full gathered array.
# 4. Reduce: Each rank has a `local_value`. `comm.reduce` calculates the sum of these 
#    values across all ranks using `op=MPI.SUM`, delivering the final `global_sum` 
#    only to rank 0.
# 5. Allreduce: Each rank has a `local_max`. `comm.allreduce` finds the maximum value 
#    across all ranks using `op=MPI.MAX`, and delivers the `global_max` to *all* ranks.
# Barriers are used conceptually to separate output from different sections.
```

Using collective operations effectively is key to writing efficient and scalable MPI programs. They often encapsulate complex communication patterns that are optimized by the MPI library for the specific hardware interconnect, leading to better performance than implementing the same logic with numerous point-to-point messages. Understanding when to use broadcast/scatter for data distribution and gather/reduce for result collection is fundamental to the SPMD parallel programming model used with MPI.

**39.3 Using `mpi4py`**

While MPI itself is a specification usually implemented in C or Fortran, the **`mpi4py`** package provides standard, efficient Python bindings, making the power of MPI accessible directly within Python scripts. It allows Python programmers to write parallel applications that can run across multiple nodes of an HPC cluster, leveraging distributed memory parallelism without needing to write code in lower-level languages for the communication aspects. `mpi4py` aims to provide an interface that closely mirrors the standard MPI C/C++/Fortran bindings but with a Pythonic feel.

**Installation:** `mpi4py` typically needs to be installed (`pip install mpi4py`) in an environment where an underlying MPI implementation (like OpenMPI, MPICH, Intel MPI) is already installed and configured correctly. The `pip` installation attempts to automatically detect and link against the available MPI library. On HPC clusters, loading the appropriate `mpi` module (`module load openmpi`) *before* installing or running `mpi4py` is usually necessary.

**Initialization and Basic Information:** To use `mpi4py`, you first import the MPI module: `from mpi4py import MPI`. The MPI environment is usually initialized automatically upon import. The primary object is the communicator, typically `comm = MPI.COMM_WORLD`, representing all processes launched. You can then query this communicator for essential information:
*   `rank = comm.Get_rank()` (or `comm.rank`): Gets the integer rank (0 to size-1) of the current process.
*   `size = comm.Get_size()` (or `comm.size`): Gets the total number of processes in the communicator.
These are fundamental for controlling program flow in the SPMD model. You can also get the processor/hostname the process is running on using `hostname = MPI.Get_processor_name()`.

**Point-to-Point Communication:** `mpi4py` provides methods on the communicator object for point-to-point messages:
*   **General Python Objects (using pickle):**
    *   `comm.send(obj, dest=destination_rank, tag=0)`: Sends any pickleable Python object `obj`.
    *   `received_obj = comm.recv(source=source_rank, tag=0)`: Receives a pickled object. Returns the unpickled object. `source` and `tag` can use `MPI.ANY_SOURCE` and `MPI.ANY_TAG`. These are convenient but incur serialization overhead.
*   **Buffer-like Objects (NumPy arrays, efficient):**
    *   `comm.Send(numpy_array_buffer, dest=...)`: Sends data from a buffer-like object (e.g., NumPy array). The argument is typically `[data_buffer, MPI_datatype]` or just `data_buffer` if type can be inferred. E.g., `comm.Send([my_np_array, MPI.DOUBLE], dest=1)`.
    *   `comm.Recv(numpy_array_buffer, source=...)`: Receives data directly into a pre-allocated buffer-like object. E.g., `recv_buf = np.empty(10, dtype=np.float64); comm.Recv([recv_buf, MPI.DOUBLE], source=0)`. This avoids pickling and is much faster for large arrays. Requires specifying the MPI datatype (e.g., `MPI.INT`, `MPI.FLOAT`, `MPI.DOUBLE`). Non-blocking versions `comm.isend`/`comm.Irecv` and `comm.Isend`/`comm.Irecv` also exist, returning request objects.

**Collective Communication:** `mpi4py` provides methods for standard collective operations, again with lowercase versions for general Python objects (using pickle) and uppercase versions for efficient buffer communication (typically NumPy arrays):
*   `comm.bcast(obj, root=0)` / `comm.Bcast(np_array_buffer, root=0)`
*   `comm.scatter(list_of_objs, root=0)` / `comm.Scatter(send_np_buffer, recv_np_buffer, root=0)`
*   `comm.gather(obj_to_send, root=0)` / `comm.Gather(send_np_buffer, recv_np_buffer, root=0)`
*   `comm.allgather(obj_to_send)` / `comm.Allgather(send_np_buffer, recv_np_buffer)`
*   `comm.reduce(value_to_send, op=MPI.SUM, root=0)` / `comm.Reduce(send_np_buffer, recv_np_buffer, op=MPI.SUM, root=0)`
*   `comm.allreduce(value_to_send, op=MPI.SUM)` / `comm.Allreduce(send_np_buffer, recv_np_buffer, op=MPI.SUM)`
*   `comm.Barrier()`
The lowercase versions are convenient for simple data structures or single values, while the uppercase versions are essential for performance when distributing or collecting large NumPy arrays. Remember that receive buffers for uppercase Scatter/Gather/Reduce operations must be pre-allocated with the correct size and type on the receiving rank(s).

```python
# --- Code Example: Using Different mpi4py Communication Methods ---
# Note: Requires mpi4py and MPI environment. Run with e.g., mpirun -np 4 python script.py

try:
    from mpi4py import MPI
    import numpy as np
    mpi4py_installed = True
except ImportError:
    mpi4py_installed = False
    print("NOTE: mpi4py not installed or MPI environment not found. Skipping example.")

if mpi4py_installed:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0

    # --- Point-to-Point: Lowercase (pickle) vs Uppercase (buffer) ---
    if rank == 0:
        # Send dictionary via pickle
        py_obj = {'rank': rank, 'data': list(range(5))}
        comm.send(py_obj, dest=1 % size, tag=1) # Send to next rank (if size > 1)
        # Send numpy array via buffer
        np_array = np.ones(5, dtype='i') * rank
        comm.Send([np_array, MPI.INT], dest=1 % size, tag=2)
    
    if rank == 1 % size: # Receiving rank
        received_obj = comm.recv(source=0, tag=1)
        print(f"Rank {rank}: Received pickled object: {received_obj}")
        
        recv_buf = np.empty(5, dtype='i')
        comm.Recv([recv_buf, MPI.INT], source=0, tag=2)
        print(f"Rank {rank}: Received numpy array via buffer: {recv_buf}")
    comm.Barrier()

    # --- Collective: Bcast (pickle) vs Bcast (buffer) ---
    bcast_data = None
    if rank == root:
        bcast_data = {'message': 'Broadcast This!', 'value': 3.14}
        bcast_np_array = np.arange(size*2, dtype='f') # Array size depends on 'size'
    else:
        bcast_np_array = np.empty(size*2, dtype='f') # Allocate buffer on non-roots
        
    # Broadcast Python object
    bcast_data = comm.bcast(bcast_data, root=root) 
    # Broadcast NumPy array
    comm.Bcast(bcast_np_array, root=root)
    
    print(f"Rank {rank}: Received bcast object: {bcast_data}")
    print(f"Rank {rank}: Received Bcast array (first 4): {bcast_np_array[:4]}")
    comm.Barrier()
    
    # --- Collective: Reduce (pickle) vs Reduce (buffer) ---
    local_val = float(rank)
    local_array = np.ones(3, dtype='i') * rank

    # Reduce single value (sum)
    sum_val = comm.reduce(local_val, op=MPI.SUM, root=root)
    
    # Reduce NumPy array (element-wise sum)
    sum_array_recv = None
    if rank == root: sum_array_recv = np.empty(3, dtype='i') # Allocate on root
    comm.Reduce([local_array, MPI.INT], [sum_array_recv, MPI.INT], op=MPI.SUM, root=root)
    
    if rank == root:
        print(f"\nRank {root}: Global sum (reduce): {sum_val}")
        print(f"Rank {root}: Global summed array (Reduce): {sum_array_recv}")
        # Expected sum_val = 0+1+2+...+(size-1)
        # Expected sum_array = [0*N, 1*N, 2*N] where N=size if array was [0,1,2]*rank
        # Here it's [0*1 + 1*1 + ..., 0*1 + 1*1 + ..., 0*1 + 1*1 + ...] -> [size-1]*3? No, [0+1+..]*[1,1,1]
        # Example: np.ones(3)*rank -> [0,0,0], [1,1,1], [2,2,2], [3,3,3] if size=4
        # Sum = [0+1+2+3, 0+1+2+3, 0+1+2+3] = [6, 6, 6]
else:
    print("Skipping MPI execution.")

print("-" * 20)

# Explanation: This code contrasts lowercase (pickle) and uppercase (buffer) methods.
# 1. Point-to-Point: Rank 0 sends a dictionary using `comm.send` and a NumPy array 
#    using `comm.Send` to Rank 1. Rank 1 uses corresponding `recv`/`Recv` calls.
# 2. Broadcast: Rank 0 broadcasts a dictionary using `comm.bcast` and a NumPy array 
#    using `comm.Bcast`. All ranks receive both. Note non-root ranks must pre-allocate 
#    the buffer `bcast_np_array` for `Bcast`.
# 3. Reduce: Each rank has a local scalar `local_val` and array `local_array`. 
#    `comm.reduce` sums the scalar values onto rank 0. `comm.Reduce` performs an 
#    element-wise sum of the `local_array`s onto rank 0's `sum_array_recv` buffer 
#    (which must be allocated).
# This highlights the different method calls and buffer requirements for general 
# Python objects versus efficient NumPy array communication in mpi4py.
```

Error Handling in `mpi4py`: By default, MPI errors are fatal and terminate the entire MPI job. You can change this behavior using `comm.Set_errhandler(MPI.ERRORS_RETURN)` which causes MPI errors to raise Python exceptions (like `mpi4py.MPI.Exception`) that can potentially be caught with `try...except`, although handling errors robustly in complex parallel scenarios remains challenging.

`mpi4py` provides a powerful and relatively user-friendly way to write parallel distributed memory programs in Python. It allows leveraging the full capabilities of the MPI standard for point-to-point and collective communication, crucially offering optimized routines for NumPy arrays essential for scientific computing performance. It enables Python users to develop applications that scale beyond single nodes onto large HPC clusters.

**39.4 Parallelizing Simple Loops and Tasks**

One common application of parallel programming is accelerating computations that involve processing many independent items or iterating through large loops where each iteration is largely independent. Both `multiprocessing` (for shared memory, Sec 38.3) and `mpi4py` (for distributed memory, Sec 39.3) can be used for this, employing different strategies for distributing the work.

**Parallelizing with `multiprocessing.Pool` (Shared Memory):** As shown in Application 38.A, `Pool.map` or `Pool.imap_unordered` are ideal for task parallelism where a worker function needs to be applied to each item in a list (e.g., filenames, parameters). The pool automatically distributes the *items* from the input iterable among the worker processes. Each worker executes the function on its assigned items. This works well when:
*   The task performed by the worker function on each item is independent of other items.
*   The overhead of sending the input item and receiving the result back is small compared to the computation time of the worker function itself.
*   All necessary data (if large) can fit into the shared memory of the single node where the pool processes are running, or efficient shared memory mechanisms are used.

**Parallelizing with MPI / `mpi4py` (Distributed Memory):** When the total data is too large for one node, or when using multiple nodes, MPI is needed. Parallelizing a loop (e.g., `for i in range(N_total): do_work(i)`) or processing a list of tasks using MPI typically involves **manually dividing the workload** based on process rank and size.

A common strategy is **static load balancing**: divide the `N_total` iterations or tasks as evenly as possible among the `size` available MPI processes. Each process `rank` calculates the range of indices `i` it is responsible for. For example, rank `r` might handle indices from `r * (N_total / size)` to `(r+1) * (N_total / size) - 1` (adjusting for integer division and remainders). Each process then executes the `do_work(i)` loop *only* for its assigned range of `i`.

```python
# --- Code Example 1: Distributing Loop Iterations with mpi4py ---
# Note: Requires mpi4py and MPI environment. Run with mpirun.

try:
    from mpi4py import MPI
    import numpy as np
    import time
    mpi4py_installed = True
except ImportError:
    mpi4py_installed = False
    print("NOTE: mpi4py not installed or MPI environment not found. Skipping example.")

print("Distributing loop iterations using mpi4py:")

if mpi4py_installed:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    N_total_iterations = 100 # Total work items

    # --- Static Load Balancing: Divide iterations among ranks ---
    items_per_rank = N_total_iterations // size
    remainder = N_total_iterations % size
    
    start_index = rank * items_per_rank + min(rank, remainder)
    end_index = start_index + items_per_rank + (1 if rank < remainder else 0)
    
    my_indices = list(range(start_index, end_index))
    
    print(f"Rank {rank}/{size}: Responsible for indices {my_indices}")

    # --- Each rank performs work on its assigned indices ---
    local_results = []
    print(f"Rank {rank}: Starting work...")
    start_work = time.time()
    for index in my_indices:
        # Simulate work for this index
        result = np.sin(index * 0.1) ** 2 + np.cos(index * 0.05)
        local_results.append(result)
        time.sleep(0.01) # Simulate small work per item
    end_work = time.time()
    print(f"Rank {rank}: Finished work. Time: {end_work - start_work:.2f}s")

    # --- Gather results (if needed) ---
    # Use lowercase gather for list of floats (or results objects)
    print(f"Rank {rank}: Gathering results...")
    all_results_list = comm.gather(local_results, root=0)

    # Rank 0 processes the combined results
    if rank == 0:
        print("\n--- Rank 0: Received All Results ---")
        # all_results_list is a list of lists, one sublist per rank
        # Flatten the list if needed
        final_results = [item for sublist in all_results_list for item in sublist]
        print(f"  Total results gathered: {len(final_results)}")
        # print(f"  Sample final results: {final_results[:10]}")
    
    comm.Barrier() # Ensure all processes finish before ending

else:
    print("Skipping MPI execution.")

print("-" * 20)

# How to Run: mpirun -np 4 python your_script_name.py
# Explanation: This code demonstrates distributing loop iterations using MPI.
# 1. It calculates the range of indices (`start_index` to `end_index`) each rank 
#    is responsible for, distributing `N_total_iterations` as evenly as possible, 
#    handling remainders.
# 2. Each rank executes the `for` loop *only* over `my_indices`. Inside the loop, 
#    it performs some work (simulated here) and stores its `local_results`.
# 3. After the loop, `comm.gather(local_results, root=0)` collects the list of 
#    results computed by each rank into `all_results_list` on rank 0.
# 4. Rank 0 then processes the gathered list (here, just printing the total count).
# This shows the common pattern of dividing iterations based on rank and size, 
# performing local computation, and then using a collective (gather or reduce) 
# to combine results if necessary.
```

This static load balancing works well if the amount of work per iteration (`do_work(i)`) is roughly constant. However, if the work per item varies significantly, some processes might finish much earlier than others, leading to **load imbalance** and reduced efficiency.

For tasks with highly variable work times per item, **dynamic load balancing** strategies might be needed, often using a **manager-worker** pattern (though less common in pure MPI than static decomposition). Rank 0 acts as the manager, holding a queue of tasks. Worker ranks (1 to size-1) repeatedly request a task from the manager, perform the work, send the result back, and request another task, until the queue is empty. This ensures faster workers process more tasks, but involves more point-to-point communication overhead. Libraries like `Dask.distributed` (Chapter 40) provide higher-level abstractions for dynamic task scheduling.

When parallelizing loops involving large **data arrays** using MPI, the data itself usually needs to be distributed first using `comm.Scatter` or `comm.Bcast` (if all need read access) or read in chunks by each process from parallel files (Chapter 42). Each process operates on its local chunk of the array within the loop. If the calculation requires data from neighboring chunks (e.g., finite difference stencils), explicit point-to-point communication (`Send`/`Recv` or non-blocking variants) is needed near the boundaries, often involving "ghost cells" or halo regions. Results are then typically combined using `comm.Gather` or `comm.Reduce`. This data decomposition and boundary communication pattern is fundamental to large-scale simulations using domain decomposition.

Choosing the right strategy – `multiprocessing.Pool` for single-node task parallelism, static iteration division with MPI for balanced distributed tasks, dynamic load balancing for variable tasks, or data decomposition with boundary exchange for array processing – depends heavily on the nature of the task, the amount of data involved, the required communication patterns, and the target parallel architecture. `mpi4py` provides the tools for implementing these distributed memory parallelization patterns in Python.

**39.5 Domain Decomposition Strategies**

As mentioned in the previous section and Sec 32.6, **domain decomposition** is the fundamental strategy for achieving data parallelism in large-scale simulations (N-body, hydro, MHD) running on distributed memory HPC clusters using MPI. The core idea is to spatially divide the overall physical simulation domain (e.g., the cosmological box or the region being modeled) into smaller, non-overlapping subdomains and assign each subdomain (along with the particles or grid cells it contains) to a specific MPI process. Each process is then primarily responsible for evolving the state within its own subdomain.

The way the domain is decomposed significantly impacts performance, particularly **load balance** and **communication overhead**. Common strategies include:
1.  **Slab Decomposition:** The simplest approach, often used in 1D or sometimes 2D/3D. The domain is divided into slices or "slabs" along one dimension (e.g., dividing a 3D box along the x-axis). Process `i` gets the slab covering x-coordinates from `xᵢ` to `xᵢ₊₁`. Communication only occurs between processes holding adjacent slabs (across the y-z plane boundaries). Simple but can lead to poor load balance if the density or computational work is highly inhomogeneous along the slicing dimension.

2.  **Pencil Decomposition:** Extends slabs to 2D. The domain is divided along two dimensions (e.g., x and y), creating long "pencils" along the third dimension (z). Process `(i, j)` handles the region `[xᵢ, xᵢ₊₁)`, `[yⱼ, yⱼ₊₁)`, `[z_min, z_max)`. Communication occurs with neighbors in both x and y directions. Offers better load balancing potential than slabs for isotropic problems but still might struggle with highly clustered distributions.

3.  **Block (or Cubic) Decomposition:** Divides the domain along all three dimensions, assigning a roughly cubic subdomain to each process. Process `(i, j, k)` handles `[xᵢ, xᵢ₊₁)`, `[yⱼ, yⱼ₊₁)`, `[z<0xE2><0x82><0x97>, z<0xE2><0x82><0x97>₊₁)`. This generally minimizes the surface area-to-volume ratio for each subdomain, which often minimizes the amount of boundary data that needs to be communicated relative to the amount of local computation, potentially leading to better scalability for many physics calculations (like short-range forces or hydro fluxes). Communication occurs with neighbors in all three dimensions (up to 26 neighbors for interior blocks). Block decomposition is very common in large N-body and grid codes.

4.  **Hierarchical/Tree-Based Decomposition:** Used implicitly by Tree codes (Sec 32.5) like GADGET or AREPO for gravity calculations. The spatial domain is recursively divided by an octree structure. Nodes of the tree representing different spatial regions (and the particles they contain) can be assigned to different processes. This naturally adapts to inhomogeneous particle distributions. Load balancing often involves distributing tree nodes or sections of the sorted particle list (based on a space-filling curve like Peano-Hilbert) across processes to ensure roughly equal computational cost (number of particle interactions or active cells) per process. Communication involves exchanging tree structure information and particle data needed for force calculations between processes handling interacting nodes.

5.  **Adaptive Mesh Refinement (AMR) Decomposition:** AMR grid codes (Sec 34.2) use a hierarchical grid structure. Parallelization often involves distributing the coarse grid patches (blocks) among processes. When a patch is refined, the finer sub-patches might remain on the same process or be distributed to other processes for load balancing. Managing the distribution of patches across refinement levels and handling communication for boundary conditions (ghost cells) and data transfer between levels on different processes is complex but essential for AMR scalability.

Regardless of the strategy, communication is required at the boundaries between subdomains assigned to different processes. For calculations involving nearest neighbors (like finite difference stencils in grid codes, short-range forces in N-body, or SPH neighbor finding), each process needs data from **ghost cells** or **halo regions** – copies of data from the adjacent edges of neighboring subdomains. Processes exchange this boundary data using MPI point-to-point messages (`Send`/`Recv` or non-blocking variants) before performing local computations. The size of this communication overhead relative to the computation within the subdomain is a key factor determining parallel efficiency. Load balancing strategies aim to ensure that the computation time per step is roughly equal across all processes, preventing some from sitting idle while waiting for others. Techniques might involve dynamically adjusting subdomain boundaries or migrating data/tasks between processes during the simulation run.

While the implementation details of domain decomposition and load balancing reside within the complex simulation codes (usually C++/Fortran with MPI), understanding these concepts is helpful for simulation users. It explains why simulations are run with specific numbers of MPI tasks (often related to how the domain is best divided), how performance scales with the number of nodes/cores, and why communication bottlenecks can arise. When analyzing simulation output that might be distributed across multiple files (one per MPI rank), knowledge of the domain decomposition used can sometimes be relevant for reconstructing the global state or understanding file structures.

**(No simple Python code can adequately illustrate the complexity of implementing robust domain decomposition and communication patterns used in large-scale simulation codes.)**

**Application 39.A: Parallel Calculation of Global Average Property**

**(Paragraph 1)** **Objective:** This application provides a concrete example of using `mpi4py` (Sec 39.3) for a common data-parallel task: calculating a global average property (like the mean temperature or mean particle mass) from a large dataset distributed across multiple MPI processes. It demonstrates the standard pattern of local calculation followed by **collective communication** using `comm.Reduce` (or `comm.Allreduce`) (Sec 39.2) to combine results.

**(Paragraph 2)** **Astrophysical Context:** Analyzing large simulation snapshots or observational catalogs often requires calculating global summary statistics across the entire dataset, which might be too large to fit in one node's memory. Examples include finding the average gas temperature in a cosmological simulation box, the mean metallicity of stars in a large survey catalog, or the total number of particles/objects meeting certain criteria. Parallel computation allows distributing the data and calculating partial results locally, followed by a global reduction to get the final answer.

**(Paragraph 3)** **Data Source:** A large dataset (conceptually) distributed across MPI processes. For this example, each process `rank` will *generate* its own local chunk of data (e.g., a NumPy array of particle temperatures or masses). In a real scenario, this local data might be read from separate files corresponding to a domain decomposition or scattered from a root process.

**(Paragraph 4)** **Modules Used:** `mpi4py.MPI` (for communication), `numpy` (for local data arrays and calculations).

**(Paragraph 5)** **Technique Focus:** Implementing the "map-reduce" pattern using MPI. (1) **Data Distribution:** Each rank holds/generates its local portion of the data. (2) **Local Calculation:** Each rank calculates the necessary partial sums needed for the global average independently on its local data (e.g., local sum of values `local_sum`, and local count of values `local_n`). (3) **Global Reduction:** Using `comm.Reduce` with `op=MPI.SUM` to sum the `local_sum` values from all processes onto the root process (rank 0), and similarly summing the `local_n` values. (4) **Final Calculation (on root):** Rank 0 calculates the global average by dividing the `total_sum` by the `total_n`. Optionally, use `comm.Allreduce` if all processes need the final average. Using uppercase `comm.Reduce` for NumPy arrays if calculating averages of multiple quantities simultaneously.

**(Paragraph 6)** **Processing Step 1: Initialize MPI and Generate Local Data:** Import libraries. Initialize `comm`, `rank`, `size`. Each rank generates a NumPy array `local_data` (e.g., random temperatures).

**(Paragraph 7)** **Processing Step 2: Local Calculation:** Each rank calculates the sum of its local data `local_sum = np.sum(local_data)` and the number of elements in its local data `local_n = len(local_data)`. Use appropriate data types (e.g., `float64` for sums to avoid overflow).

**(Paragraph 8)** **Processing Step 3: Global Reduction:**
    *   Use `comm.Reduce(sendbuf=[local_sum, MPI.DOUBLE], recvbuf=[total_sum_recv, MPI.DOUBLE], op=MPI.SUM, root=0)`. `total_sum_recv` needs to be allocated on rank 0 (e.g., `np.zeros(1, dtype=np.float64)`).
    *   Use `comm.Reduce(sendbuf=[np.array(local_n, dtype='i'), MPI.INT], recvbuf=[total_n_recv, MPI.INT], op=MPI.SUM, root=0)`. `total_n_recv` needs to be allocated on rank 0 (e.g., `np.zeros(1, dtype='i')`).
    *(Note: `mpi4py` might allow reducing scalars directly with lowercase `comm.reduce` without explicit buffers, which is simpler: `total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)`)*

**(Paragraph 9)** **Processing Step 4: Final Calculation and Output:** If `rank == 0`, calculate `global_average = total_sum / total_n` (using the received values). Print the global average from rank 0. Add checks for `total_n > 0`.

**(Paragraph 10)** **Processing Step 5: Synchronization (Optional):** Include `comm.Barrier()` at the end if necessary to ensure all processes finish cleanly before the program exits.

**Output, Testing, and Extension:** The primary output is the global average value printed by the root process (rank 0). **Testing:** Run the script with different numbers of processes (`mpirun -np P ...`). Verify that the calculated global average remains consistent regardless of the number of processes used. Compare the parallel result with a serial calculation performed on the equivalent total dataset (if feasible to generate/load serially) to confirm correctness. **Extensions:** (1) Use `comm.Allreduce` instead of `comm.Reduce` so that *all* processes receive the final global average. (2) Modify the code to calculate the global standard deviation in parallel (requires reducing both sum and sum-of-squares). (3) Implement the data distribution step explicitly using `comm.Scatter` from rank 0 instead of generating data locally on each rank. (4) Adapt the code to read local data chunks from separate files named according to rank (simulating analysis of domain-decomposed simulation output).

```python
# --- Code Example: Application 39.A ---
# Note: Requires mpi4py and MPI environment. Run with e.g., mpirun -np 4 python script.py

try:
    from mpi4py import MPI
    import numpy as np
    import time
    mpi4py_installed = True
except ImportError:
    mpi4py_installed = False
    print("NOTE: mpi4py not installed or MPI environment not found. Skipping example.")

print("Parallel Calculation of Global Average using mpi4py:")

if mpi4py_installed:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0

    # Step 1: Generate Local Data (e.g., Temperatures)
    # Each rank generates some data
    np.random.seed(rank) # Ensure different data per rank
    local_n = np.random.randint(5000, 10000) # Variable number of points per rank
    local_data = np.random.normal(loc=100.0 + rank*10, scale=20.0, size=local_n)
    print(f"Rank {rank}: Generated {local_n} local data points.")

    # Step 2: Local Calculation
    local_sum = np.sum(local_data, dtype=np.float64) # Use float64 for sum
    local_count = np.array(local_n, dtype='i') # Use integer type for count

    print(f"Rank {rank}: Local Sum = {local_sum:.2f}, Local Count = {local_count}")

    # Step 3: Global Reduction
    # Use lowercase reduce for scalars (simpler if sum/count fit in standard types)
    print(f"Rank {rank}: Performing Reduce...")
    
    # Reduce sum
    # Need to handle potential None return on non-root ranks for reduce
    total_sum = comm.reduce(local_sum, op=MPI.SUM, root=root)
    
    # Reduce count
    total_count = comm.reduce(local_count, op=MPI.SUM, root=root)

    # --- Alternative using Uppercase Reduce (more robust for large numbers/arrays) ---
    # Allocate receive buffers on root ONLY
    # total_sum_buf = np.zeros(1, dtype=np.float64) if rank == root else None
    # total_count_buf = np.zeros(1, dtype='i') if rank == root else None
    # comm.Reduce([np.array(local_sum), MPI.DOUBLE], total_sum_buf, op=MPI.SUM, root=root)
    # comm.Reduce([local_count, MPI.INT], total_count_buf, op=MPI.SUM, root=root)
    # if rank == root:
    #     total_sum = total_sum_buf[0]
    #     total_count = total_count_buf[0]
    # -------------------------------------------------------------------------------

    # Step 4: Final Calculation and Output (on Root)
    if rank == root:
        print("\n--- Rank 0: Reduction Results ---")
        print(f"  Total Sum = {total_sum:.2f}")
        print(f"  Total Count = {total_count}")
        if total_count > 0:
            global_average = total_sum / total_count
            print(f"  Global Average = {global_average:.3f}")
        else:
            print("  Cannot calculate average (total count is zero).")
            
    # Step 5: Synchronization
    comm.Barrier()
    if rank == 0: print("\nAll processes finished.")

else:
    print("Skipping MPI execution.")

print("-" * 20)
# How to Run: mpirun -np 4 python your_script_name.py
```

**Application 39.B: Parallel Image Filtering using Domain Decomposition**

**(Paragraph 1)** **Objective:** This application illustrates how a task typically involving local neighborhood operations – image filtering (e.g., median filter or convolution) – can be parallelized across multiple MPI processes using a **domain decomposition** strategy. It highlights the need for communication between processes to handle data dependency at the boundaries of the subdomains (**ghost cells** or **halo exchange**). Reinforces Sec 39.1, 39.2, 39.5.

**(Paragraph 2)** **Astrophysical Context:** Processing large astronomical images (e.g., from wide-field surveys or large mosaics) often involves applying filters to smooth noise, detect sources, or enhance features. Standard filtering operations (like median filters, Gaussian blurs, convolutions) require access to neighboring pixel values. When distributing a large image across multiple processors for parallel filtering, each processor needs access not only to its assigned image section but also to a small border region ("ghost cells" or "halo") from adjacent sections handled by neighboring processes to correctly compute filter outputs near the boundaries.

**(Paragraph 3)** **Data Source:** A large 2D NumPy array representing an astronomical image. For this application, we assume the image is initially loaded or generated by the root process (rank 0) and then scattered to other processes.

**(Paragraph 4)** **Modules Used:** `mpi4py.MPI`, `numpy`, `scipy.ndimage` (for the filter operation, e.g., `median_filter`).

**(Paragraph 5)** **Technique Focus:** Implementing parallel image filtering using domain decomposition and halo exchange. (1) **Data Distribution:** The root process scatters horizontal strips (or 2D blocks) of the image to all processes using `comm.Scatter` (uppercase for NumPy array). (2) **Halo Exchange:** Each process identifies its top and bottom neighboring ranks (handling edge cases for rank 0 and size-1). Using point-to-point communication (`comm.Send` and `comm.Recv`, potentially non-blocking `Isend`/`Irecv`), each process sends its topmost row(s) needed by the neighbor below and its bottommost row(s) needed by the neighbor above. It simultaneously receives corresponding boundary rows from its neighbors into pre-allocated "ghost cell" arrays. (3) **Local Computation:** Each process applies the filter (e.g., `scipy.ndimage.median_filter`) to its local data strip *plus* the received ghost cell rows, ensuring correct calculations near the boundaries. (4) **Result Gathering:** The root process gathers the filtered strips (excluding the ghost cell regions) back from all processes using `comm.Gather` into a final combined image.

**(Paragraph 6)** **Processing Step 1: Setup and Data Distribution:** Initialize MPI (`comm`, `rank`, `size`). Rank 0 creates/loads the full image `global_image`. Determine the number of rows per process `local_rows` and scatter strips of the image using `comm.Scatter` into `local_strip` on each rank.

**(Paragraph 7)** **Processing Step 2: Halo Exchange:**
    *   Determine `rank_up` and `rank_down` neighbors (handle boundaries, e.g., using `MPI.PROC_NULL` for no neighbor or periodic conditions).
    *   Allocate receive buffers (`ghost_row_above`, `ghost_row_below`) for the boundary rows needed from neighbors.
    *   Implement communication:
        *   Send bottom row `local_strip[-1, :]` to `rank_down`.
        *   Send top row `local_strip[0, :]` to `rank_up`.
        *   Receive from `rank_up` into `ghost_row_above`.
        *   Receive from `rank_down` into `ghost_row_below`.
        *   Using non-blocking `Isend`/`Irecv` with `Waitall` is generally more efficient and avoids deadlocks compared to naive blocking Send/Recv pairs. (Need multiple rows if filter size > 3).

**(Paragraph 8)** **Processing Step 3: Local Filtering with Halo:** Create an extended local array by stacking `ghost_row_above`, `local_strip`, and `ghost_row_below`. Apply the filter (e.g., `median_filter(extended_strip, size=3)`) to this extended array. Extract the filtered result corresponding *only* to the original `local_strip` dimensions (discarding results based on ghost cells). Store this in `filtered_local_strip`.

**(Paragraph 9)** **Processing Step 4: Gather Results:** Use `comm.Gather` to collect the `filtered_local_strip` arrays from all processes onto rank 0's pre-allocated `final_filtered_image` buffer.

**(Paragraph 10)** **Processing Step 5: Finalize:** Rank 0 now has the complete filtered image. It can save it or perform further processing. Include `comm.Barrier()` if needed.

**Output, Testing, and Extension:** The output is the final filtered image assembled on rank 0. **Testing:** Compare the parallel filtered image pixel-by-pixel with the result obtained by applying the same filter serially to the original global image using `scipy.ndimage`. Pay close attention to the boundary regions between strips to ensure the halo exchange worked correctly. Test with different numbers of processes. **Extensions:** (1) Implement 2D block decomposition instead of 1D strips (requires more complex 8-neighbor communication). (2) Use non-blocking communication (`Isend`/`Irecv`/`Waitall`) for the halo exchange to overlap communication and computation. (3) Implement a different filter (e.g., Gaussian blur using `scipy.ndimage.gaussian_filter` or convolution using `scipy.signal.convolve2d`). (4) Package the parallel filter into a reusable function.

```python
# --- Code Example: Application 39.B ---
# Note: Requires mpi4py, numpy, scipy. Run with mpirun.
# Implements 1D slab decomposition and basic halo exchange.

try:
    from mpi4py import MPI
    import numpy as np
    from scipy.ndimage import median_filter
    import time
    mpi4py_installed = True
except ImportError:
    mpi4py_installed = False
    print("NOTE: mpi4py, numpy, scipy not installed/found. Skipping example.")

print("Parallel Image Filtering using MPI Domain Decomposition:")

if mpi4py_installed:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0

    # --- Step 1: Setup and Data Distribution ---
    global_image = None
    N_rows_global, N_cols = 0, 0 # Define dimensions

    if rank == root:
        N_rows_global = 128 * size # Ensure divisible by size for simple scatter
        N_cols = 256
        print(f"\nRoot generating global image ({N_rows_global} x {N_cols})...")
        # Add a pattern for visual verification
        global_image = np.random.rand(N_rows_global, N_cols) * 50
        global_image[N_rows_global//4 : 3*N_rows_global//4, N_cols//4 : 3*N_cols//4] += 100
        print("Global image generated.")
    
    # Broadcast dimensions to all ranks
    dims = comm.bcast((N_rows_global, N_cols) if rank == root else None, root=root)
    if rank != root: N_rows_global, N_cols = dims

    # Calculate local strip dimensions (assuming N_rows_global % size == 0)
    local_rows = N_rows_global // size
    local_shape = (local_rows, N_cols)
    
    # Allocate local strip buffer on all ranks
    local_strip = np.empty(local_shape, dtype=global_image.dtype if rank == root else 'f8')

    # Scatter image strips from root to all ranks
    print(f"Rank {rank}: Scattering/Receiving image strip ({local_rows} rows)...")
    # Uppercase Scatter for NumPy array. Need buffer info [data, MPI_TYPE].
    # Send buffer is contiguous global_image on root, None elsewhere
    # Recv buffer is local_strip on all ranks
    sendbuf = [global_image, MPI.DOUBLE] if rank == root else None
    recvbuf = [local_strip, MPI.DOUBLE]
    comm.Scatter(sendbuf, recvbuf, root=root)
    print(f"Rank {rank}: Received strip.")

    # --- Step 2: Halo Exchange (for filter size 3x3, need 1 ghost row) ---
    print(f"Rank {rank}: Performing Halo Exchange...")
    # Determine neighbors (simple non-periodic boundaries)
    rank_up = rank - 1 if rank > 0 else MPI.PROC_NULL
    rank_down = rank + 1 if rank < size - 1 else MPI.PROC_NULL
    
    # Allocate ghost rows
    ghost_row_above = np.empty(N_cols, dtype=local_strip.dtype)
    ghost_row_below = np.empty(N_cols, dtype=local_strip.dtype)
    
    # Non-blocking communication is generally better here, but using blocking for simplicity
    # Send bottom row down, receive ghost row from above
    comm.Sendrecv(local_strip[-1, :], dest=rank_down, sendtag=0,
                  recvbuf=ghost_row_above, source=rank_up, recvtag=0)
                  
    # Send top row up, receive ghost row from below
    comm.Sendrecv(local_strip[0, :], dest=rank_up, sendtag=1,
                  recvbuf=ghost_row_below, source=rank_down, recvtag=1)
                  
    # Handle edges where neighbors are MPI.PROC_NULL (optional: fill ghosts with boundary value)
    if rank_up == MPI.PROC_NULL: ghost_row_above[:] = local_strip[0, :] # Example: Reflect boundary
    if rank_down == MPI.PROC_NULL: ghost_row_below[:] = local_strip[-1, :] # Example: Reflect boundary
    print(f"Rank {rank}: Halo exchange complete.")

    # --- Step 3: Local Filtering with Halo ---
    print(f"Rank {rank}: Applying median filter locally...")
    # Create extended strip including ghost rows
    # Need to handle shape carefully if local_rows < 1
    if local_rows > 0:
         extended_strip = np.vstack((ghost_row_above, local_strip, ghost_row_below))
         # Apply filter to extended strip
         filtered_extended = median_filter(extended_strip, size=3)
         # Extract the relevant part corresponding to the original local_strip
         filtered_local_strip = filtered_extended[1:-1, :] # Exclude ghost row results
    else: # Handle case where a process gets no rows
         filtered_local_strip = np.empty((0, N_cols), dtype=local_strip.dtype)

    print(f"Rank {rank}: Filtering complete.")

    # --- Step 4: Gather Results ---
    print(f"Rank {rank}: Gathering filtered strips...")
    # Allocate receive buffer on root only
    final_filtered_image = None
    if rank == root:
        final_filtered_image = np.empty((N_rows_global, N_cols), dtype=local_strip.dtype)
    
    # Gather filtered local strips onto root process
    sendbuf_gather = [filtered_local_strip, MPI.DOUBLE]
    recvbuf_gather = [final_filtered_image, MPI.DOUBLE] if rank == root else None
    comm.Gather(sendbuf_gather, recvbuf_gather, root=root)
    
    # --- Step 5: Finalize ---
    if rank == root:
        print("\n--- Rank 0: Received Final Filtered Image ---")
        print(f"  Final image shape: {final_filtered_image.shape}")
        # Optional: Save or display the image
        # plt.figure(); plt.imshow(final_filtered_image); plt.title("Parallel Filtered Image")
        # plt.show()

    comm.Barrier()
    if rank == 0: print("\nAll processes finished.")

else:
    print("Skipping MPI execution.")

print("-" * 20)

# How to Run: mpirun -np 4 python your_script_name.py
```

**Chapter 39 Summary**

This chapter provided a practical introduction to distributed memory parallel programming using the **Message Passing Interface (MPI)** standard, focusing on its implementation within Python via the **`mpi4py`** library. It began by outlining the core MPI execution model (SPMD), explaining fundamental concepts like **communicators** (defining groups of processes, e.g., `MPI.COMM_WORLD`), unique process **ranks** (0 to size-1), and communicator **size**. The basic mechanism of **point-to-point communication** was detailed, covering blocking `send`/`recv` operations (and the risk of deadlock) and non-blocking `isend`/`irecv` operations (allowing computation/communication overlap but requiring careful management). The crucial distinction in `mpi4py` between convenient lowercase methods (`send`, `recv`) for general Python objects (using slower pickle serialization) and efficient uppercase methods (`Send`, `Recv`) for buffer-like objects like NumPy arrays (avoiding pickle overhead) was emphasized.

The chapter then explored essential **collective communication** routines that involve coordinated operations among all processes in a communicator. These included **`bcast`** (distributing data from one root to all), **`scatter`** (distributing different chunks from root to all), **`gather`** (collecting chunks from all onto root), **`allgather`** (gathering data onto all), **`reduce`** (combining data from all onto root using operations like `MPI.SUM`, `MPI.MAX`), **`allreduce`** (combining data and distributing result to all), and **`Barrier`** (synchronization). The usage of both lowercase (object) and uppercase (buffer/NumPy) versions of these collectives in `mpi4py` was demonstrated. Techniques for **parallelizing simple loops or tasks** using MPI were illustrated, primarily through static load balancing (dividing iterations based on rank and size) and the gather/reduce pattern for collecting results. The concept of **domain decomposition** as the standard strategy for data parallelism in large simulations was discussed, outlining different strategies (slab, pencil, block, tree-based) and highlighting the necessity of **halo exchange** (communication of boundary data) using point-to-point messages when local computations depend on neighboring subdomains. Finally, the chapter covered the practicalities of **launching `mpi4py` scripts** on HPC clusters using commands like `mpirun` or `srun`.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Dalcin, L., Paz, R., Storti, M., & D'Elia, J. (2011).** MPI for Python: performance improvements and MPI-2 extensions. *Journal of Parallel and Distributed Computing*, *71*(5), 655–668. [https://doi.org/10.1016/j.jpdc.2011.01.005](https://doi.org/10.1016/j.jpdc.2011.01.005) (See also `mpi4py` documentation: [https://mpi4py.readthedocs.io/en/stable/](https://mpi4py.readthedocs.io/en/stable/))
    *(A key paper describing `mpi4py` and its features. The linked documentation is the essential practical reference for using the library, covering all functions discussed in the chapter.)*

2.  **Gropp, W., Lusk, E., & Skjellum, A. (1999).** *Using MPI: Portable Parallel Programming with the Message-Passing Interface* (2nd ed.). MIT Press. (Also relevant: Gropp, W., Lusk, E., & Thakur, R. (1999). *Using MPI-2: Advanced Features of the Message-Passing Interface*.)
    *(Classic textbooks providing comprehensive coverage of the MPI-1 and MPI-2 standards, explaining the concepts behind point-to-point and collective communication routines.)*

3.  **Pacheco, P. S. (2011).** *An Introduction to Parallel Programming*. Morgan Kaufmann.
    *(A textbook covering various parallel programming paradigms, including detailed chapters on MPI programming with examples, providing good foundational knowledge.)*

4.  **Eijkhout, V. (2022).** *Introduction to High Performance Scientific Computing*. (Online textbook). [http://pages.tacc.utexas.edu/~eijkhout/istc/istc.html](http://pages.tacc.utexas.edu/~eijkhout/istc/istc.html)
    *(Covers MPI programming concepts, including point-to-point, collectives, communicators, and domain decomposition strategies within the broader context of HPC.)*

5.  **MPI Forum. (n.d.).** *MPI Documents*. MPI Forum. Retrieved January 16, 2024, from [https://www.mpi-forum.org/docs/](https://www.mpi-forum.org/docs/)
    *(The official MPI standard documents (MPI 3.1, MPI 4.0). While highly technical, they are the ultimate reference for the precise definition and behavior of all MPI functions and concepts.)*
