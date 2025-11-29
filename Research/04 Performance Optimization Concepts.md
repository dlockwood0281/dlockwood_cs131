
- [[#**SIMT Execution:**|**SIMT Execution:**]]
- [[#**Shared Memory:**|**Shared Memory:**]]
	- [[#**Shared Memory:**#Global Memory Access|Global Memory Access]]
	- [[#**Shared Memory:**#Shared Memory Access|Shared Memory Access]]
	- [[#**Shared Memory:**#When to use what|When to use what]]
- [[#**Occupancy:**|**Occupancy:**]]

# Performance Optimization
## **SIMT Execution:** 
Single Instruction, Multiple Thread execution model.

Key Characteristics of SIMT:

- **Warp/Wavefront Execution:** 
    
    Threads are grouped into units called "warps" (NVIDIA CUDA) or "wavefronts" (AMD GCN). All threads within a warp execute in a lock-step fashion, meaning they all execute the same instruction at the same time.
    
- **Thread Independence:** 
    
    While executing the same instruction, each thread has its own program counter, registers, and memory space, allowing it to process different data elements.
    
- **Handling Divergence:** 
    
    If threads within a warp encounter conditional branches (e.g., `if-else` statements) and follow different execution paths, the warp handles this "divergence" by serially executing each divergent path. Threads not on the active path are temporarily disabled until the paths reconverge. This can impact performance if divergence is frequent.
    
- **Scalability:** 
    
    SIMT allows for massive parallelism by enabling a single instruction to be applied across thousands or even millions of data points concurrently, making it highly efficient for data-parallel workloads common in graphics rendering, scientific computing, and machine learning.

- **Memory Access Patterns:** Optimizing global and shared memory access for coalescing and bank conflicts.

## **Shared Memory:** 
Efficiently utilizing shared memory for data reuse and inter-thread communication within a block.

To optimize memory access, aim for **coalescing** in global memory and avoid **bank conflicts** in shared memory by ensuring threads in a warp access distinct memory banks. Coalescing is achieved when threads access contiguous global memory locations, allowing multiple reads/writes to be combined into a single operation. For shared memory, ensure that when a warp accesses it, each thread is not accessing a different location in the same bank, which forces serialization. 

### Global Memory Access

- **Coalescing:** Global memory access is most efficient when threads in a warp access consecutive memory addresses.
    - **Why it works:** The GPU groups these accesses into a single, larger transaction to retrieve a large chunk of memory, even if individual threads only need a small piece of the data.
    - **How to achieve it:**
        - Structure your data and access patterns so that threads with adjacent thread indices access adjacent memory locations.
        - For row-major storage, access rows within a block rather than columns across the entire matrix to ensure coalesced access.
        - The goal is to have each thread in a warp read or write to a contiguous chunk of memory. 

### Shared Memory Access

- **Bank Conflicts:** Shared memory is divided into 32 banks, and multiple threads in a warp accessing different locations in the **same bank** simultaneously will cause a bank conflict.
    - **Why it hurts performance:** The hardware must serialize these conflicting accesses, which creates a performance bottleneck as the instruction is replayed for each conflicting thread.
    - **How to avoid them:**
        - **Strive for disjoint banks:** Arrange your data and access patterns so that threads in a warp access different memory banks.
        - **Adjacent access:** If threads in a warp access addresses that are separated by a multiple of the number of banks (32), they will land in the same bank. Avoid this.
        - **Padding:** If you are accessing 64-bit (8-byte) data, you can pad it to 96 bits (12 bytes) to spread it across three consecutive banks, avoiding conflicts, as noted in the [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/requesting-clarification-for-shared-memory-bank-conflicts-and-shared-memory-access/268574).
        - **Data layout:** For a matrix, accessing elements in a way that spreads them across banks can be beneficial. For example, a 1D array can be accessed with a stride of 1. For a 2D array, a stride of 1 can still lead to conflicts if the stride is a multiple of the number of banks.
        - **Use shared memory for staging:** Use shared memory to stage data from global memory in a coalesced manner to then allow for fast, non-conflicting accesses for computation. 

### When to use what

- Use global memory for data that is not reused or does not have locality, and use shared memory for data that will be accessed multiple times by threads within a block.
- To maximize performance, stage data from global memory into shared memory first, ensuring the initial global memory read is coalesced. Then, access the shared memory with a pattern that avoids bank conflicts for subsequent computations.

- **Synchronization:** `__syncthreads()` for coordinating threads within a block.

`__syncthreads()` is a synchronization function in CUDA that acts as a barrier, making all threads within a thread block wait until every thread in that same block has reached the barrier. It is crucial for coordinating threads when they need to share data via shared memory to prevent race conditions and ensure a correct, consistent state before proceeding. 

How it works

- **Block-level scope:** `__syncthreads()` only synchronizes threads within the same block. It does not synchronize threads in other blocks or the entire grid.
- **Synchronization barrier:** When a thread encounters `__syncthreads()`, it stops and waits. Once all other threads in the same block have also reached that point, all threads are released to continue execution together.
- **Use with shared memory:** This function is essential when threads write to shared memory and then other threads need to read that data. `__syncthreads()` ensures the data is fully written before any thread attempts to read it. 

Important considerations

- **Conditional execution:** Placing `__syncthreads()` inside an `if` statement can lead to undefined behavior or a hang if not all threads in the block take the same path.
- **Divergent code:** If threads take different execution paths and not all of them hit the `__syncthreads()` call, the program will likely deadlock because the threads waiting at the barrier will never be joined by the threads that are not waiting.
- **Usage:** `__syncthreads()` can be called from a `__global__` kernel or a `__device__` function.

## **Occupancy:** 
Maximizing the number of active warps on the GPU.

Maximizing the number of active warps on a GPU, also known as occupancy, is a key strategy for improving performance by keeping the GPU's execution units busy and hiding memory latency. It is achieved by increasing the number of active warps to the maximum allowed per multiprocessor (SM). This is done by carefully tuning hardware-dependent factors like block size, register usage, and shared memory, and can be analyzed using tools like the NVIDIA Occupancy Calculator. 

How it works

- **Latency hiding**: When a warp stalls, such as when waiting for data from global memory, the warp scheduler can immediately switch to another ready warp. By having many active warps, the GPU can switch between them, effectively hiding the latency and keeping the execution units busy.
- **Resource allocation**: Each streaming multiprocessor (SM) has a limited number of resources, such as registers and shared memory. The number of active warps is limited by the hardware's ability to support the resources each thread in a warp requires.
- **The goal**: The goal is to reach an occupancy level where there are enough active warps to ensure the warp scheduler can always find a ready warp to execute, maximizing throughput. 

Factors that influence occupancy

- **Block size**: The number of threads per block is a crucial factor. For example, a kernel with a small block size will have fewer warps per block and may limit occupancy if the maximum number of blocks per SM is reached.
- **Register usage**: Each thread in a warp uses a certain number of registers. If a kernel uses a large number of registers per thread, fewer thread blocks (and thus fewer warps) can be active on an SM at one time.
- **Shared memory**: The amount of shared memory allocated per thread block also affects how many can be active on an SM at once.
- **Synchronization**: Overuse of synchronization primitives can lead to low occupancy because warps may have to wait for each other to complete certain tasks. 

How to maximize occupancy

- **Use the Occupancy Calculator**: Use tools like the CUDA Occupancy Calculator or spreadsheets provided by NVIDIA to calculate the theoretical occupancy for your kernel based on resource constraints.
- **Tune block size**: Experiment with different block sizes to find a balance that keeps the GPU busy without hitting resource limits.
- **Minimize resource usage**: Reduce register usage and shared memory per thread block where possible, while ensuring performance is not negatively impacted.
- **Analyze your kernel**: Use tools like Nsight Compute to analyze your kernel's performance, identify bottlenecks like low eligible warps, and see how resource usage affects occupancy.