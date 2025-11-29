
- [[#**Host and Device:**|**Host and Device:**]]
	- [[#**Host and Device:**#CPU (host) role|CPU (host) role]]
	- [[#**Host and Device:**#GPU (device) role|GPU (device) role]]
	- [[#**Host and Device:**#Interaction and communication|Interaction and communication]]
- [[#**Kernels:**|**Kernels:**]]
	- [[#**Kernels:**#Key characteristics of kernels include:|Key characteristics of kernels include:]]
		- [[#Key characteristics of kernels include:#**Parallel Execution:**|**Parallel Execution:**]]
		- [[#Key characteristics of kernels include:#**Device Execution:**|**Device Execution:**]]
		- [[#Key characteristics of kernels include:#**Data Parallelism:**|**Data Parallelism:**]]
		- [[#Key characteristics of kernels include:#**Thread Organization:**|**Thread Organization:**]]
		- [[#Key characteristics of kernels include:#**Special Syntax:**|**Special Syntax:**]]
		- [[#Key characteristics of kernels include:#**Thread Hierarchy:**|**Thread Hierarchy:**]]

# CUDA Programming Model

## **Host and Device:**
Understanding the interaction between the CPU (host) and GPU (device).

The CPU (host) and GPU (device) interact through a process of data transfer and command execution, where the CPU acts as the orchestrator, managing workflow, allocating memory, and initiating tasks on the GPU. Data must be copied from the CPU's main memory to the GPU's separate memory, processed in parallel by the GPU's many cores, and then copied back to the CPU for the results to be used. This communication typically happens over a PCI-e bus.
### CPU (host) role

- **Task scheduling:** The CPU is the primary processor that schedules and controls the overall application flow.
- **Memory management:** It allocates memory on both the host and device and handles the data transfers between them.
- **Command submission:** The CPU sends commands to the GPU to execute specific tasks, such as running parallel "kernels". 

### GPU (device) role

- **Parallel processing:** The GPU has thousands of cores to execute highly parallelizable tasks, like complex computations or graphics rendering, much faster than the CPU.
- **Data processing:** It performs the computations on the data it receives from the host.
- **Returning results:** Once the computation is complete, the GPU sends the results back to the CPU. 

### Interaction and communication

- **Data transfer:** Data must be moved from the host's main memory to the device's memory before the GPU can work on it. This transfer overhead is a key performance consideration.
- **Command execution:** The CPU issues commands to the GPU. These commands can be asynchronous, meaning the CPU can continue with other work while the GPU performs the requested task.
- **Synchronization:** In many cases, the CPU needs to wait for the GPU to finish a task before proceeding. Synchronization methods, such as those available in APIs like CUDA, are used to manage this dependency.
- **[PCI-e bus](https://www.google.com/search?q=PCI-e+bus&sca_esv=de22d866bfd6854f&rlz=1C1CHBF_enUS860US860&sxsrf=AE3TifMdeqXRkc7sFMs6_ytgugX6-e1faw%3A1760123321093&ei=uVnpaLC5Be2fkPIP4p2z6A8&ved=2ahUKEwiJ8oa8y5qQAxXzKEQIHXYsIOEQgK4QegQICBAE&uact=5&oq=-+**Host+and+Device%3A**%C2%A0Understanding+the+interaction+between+the+CPU+%28host%29+and+GPU+%28device%29.&gs_lp=Egxnd3Mtd2l6LXNlcnAiXi0gKipIb3N0IGFuZCBEZXZpY2U6KirCoFVuZGVyc3RhbmRpbmcgdGhlIGludGVyYWN0aW9uIGJldHdlZW4gdGhlIENQVSAoaG9zdCkgYW5kIEdQVSAoZGV2aWNlKS5IAFAAWABwAHgBkAEAmAEAoAEAqgEAuAEDyAEA-AEC-AEBmAIAoAIAmAMAkgcAoAcAsgcAuAcAwgcAyAcA&sclient=gws-wiz-serp&mstk=AUtExfDeQxhjoxkqrCzpZDaSog9S2CTVIchDXuL8dzORqqpaJ_vW8b3z0RwLmIRMMt9ixY6vBB27pduhO7I-qgWevLneL0Jr7IirqeHq8sGlpBxuD-X02naN9Km5i5UMxBoUbECXij2YWrY1FOQAp6rg5wR5a2HkySypBOARvPmHHVFgh1M&csui=3):** The physical and logical connection for communication between the CPU and GPU is typically the PCI Express (PCIe) bus.


## **Kernels:** 
Functions executed on the GPU by multiple threads.

A kernel in the context of General Purpose Graphics Processing Unit (GPGPU) programming, such as with NVIDIA's CUDA, refers to a function or routine designed to be executed in parallel on the GPU by a multitude of threads.

### Key characteristics of kernels include:

#### **Parallel Execution:**
  Unlike CPU functions that execute sequentially or with a limited number of threads, kernels are designed for massive parallelism. A single kernel call initiates its execution by thousands or even millions of threads simultaneously on the GPU.
    
#### **Device Execution:**
   Kernels are launched from the host (CPU) but are executed entirely on the device (GPU).
    
#### **Data Parallelism:**
   Kernels are particularly suited for tasks exhibiting data parallelism, where the same operation is applied to different data elements concurrently. Each thread typically processes a subset of the data.
    
#### **Thread Organization:**
   Threads executing a kernel are organized into a hierarchical structure, typically involving thread blocks and grids. This structure allows for efficient management and synchronization of threads.
    
#### **Special Syntax:**
In CUDA, kernels are identified by a `__global__` specifier and are launched using a special "execution configuration" syntax (e.g., `<<<numBlocks, threadsPerBlock>>>`) to specify the number of threads and blocks.

#### **Thread Hierarchy:** 
**Grids, Blocks, and Threads** – how they are organized and launched.

In the CUDA thread hierarchy, **grids** are the highest level, consisting of one or more **blocks**, which are themselves composed of multiple **threads**. Threads are the smallest unit of execution, performing a single kernel task. Threads within a block can communicate and synchronize, while blocks within a grid are independent and execute on different Streaming Multiprocessors (SMs) as scheduled by the runtime. 

Organization

- **Threads:** The most basic level. Each thread executes a stream of instructions independently.
- **Blocks:** A collection of threads that can cooperate.
    - They share a common, fast memory called [SRAM](https://www.google.com/search?q=SRAM&sca_esv=de22d866bfd6854f&rlz=1C1CHBF_enUS860US860&sxsrf=AE3TifMiRZGnLJZk2yHoVb2bNCJH1sq2TQ%3A1760132261294&ei=pXzpaL_gEdqQvMcP9cvQ-Qg&ved=2ahUKEwjLkPDhy5qQAxUcDEQIHVjzDGwQgK4QegQIAxAD&uact=5&oq=-+**Thread+Hierarchy%3A**%C2%A0Grids%2C+Blocks%2C+and+Threads+%E2%80%93+how+they+are+organized+and+launched.&gs_lp=Egxnd3Mtd2l6LXNlcnAiXC0gKipUaHJlYWQgSGllcmFyY2h5OioqwqBHcmlkcywgQmxvY2tzLCBhbmQgVGhyZWFkcyDigJMgaG93IHRoZXkgYXJlIG9yZ2FuaXplZCBhbmQgbGF1bmNoZWQuSOgSUOgJWOgJcAN4AJABAJgBAKABAKoBALgBA8gBAPgBAvgBAZgCAKACAJgDAIgGAZIHAKAHALIHALgHAMIHAMgHAA&sclient=gws-wiz-serp&mstk=AUtExfAlB_G394slkuzwHZ9rUXLAntgyrorLYZDbk7ytMsWpDyggAhm7mfIiEUHnw1i5ErEJtBO88Nq5yV5zWxCELjZxap-YyJC3fOh_3j6LOWWeO91jOWgDvNDjpKT0v2ZvWiZABo1doOXPeRgaIU9XYtLVgSDOFqOX7fUsPJ25VV-c-ny6-NoCS_mf0wC_5pAprouLVjsN0zeAI1a1LEVBnLWA-iSO9zpD-Sq1IwB8RZ4BHIqPeqspEYMsFHhweJHe990vDuZIvuyfWZ_-TPRpwgR3&csui=3) and can synchronize using barriers.
    - Blocks are executed on a single SM.
- **Grids:** A collection of thread blocks launched by a single kernel invocation.
    - Blocks within the same grid do not share SRAM and are scheduled by the runtime to run on different SMs.
    - Grids can be organized in one, two, or three dimensions for flexible mapping to the problem. 

Launching

- **Kernel Launch:** When a kernel is launched, a grid is created.
- **Grid and Block Dimensions:** The programmer specifies the dimensions of the grid (number of blocks) and the dimensions of each block (number of threads per block) during the launch.
- **Execution:** The CUDA runtime automatically schedules the blocks for execution on the available SMs.
- **Mapping:** Each thread is assigned a unique index within its block and grid, allowing it to access its specific part of the data. The formula `$threadIdx.x + blockIdx.x * blockDim.x$` is a common way to calculate a 1D index.
- **Example:** A 2D grid of 2 blocks, with each block containing 4 threads, would be launched with a configuration like `<<<2, 4>>>`. A 2D mapping would use both x and y dimensions.


- **Memory Model:** Global memory, shared memory, constant memory, and local memory – their characteristics and uses.

Global memory is large, accessible by all threads, and slower, while shared memory is fast, limited in size, and accessible only by threads within a block. Constant memory is read-only, cached, and fast for data accessed by all threads, and local memory is thread-specific, residing in global memory and carrying the same high latency. 

|Memory Type|Characteristics|Uses|
|---|---|---|
|**Global Memory**|* **Size:** Largest space on the GPU.  <br>* **Accessibility:** Visible to all threads across all blocks in an application.  <br>* **Speed:** Slowest; high latency (hundreds of clock cycles).|* Main storage for large datasets, such as images or matrices.  <br>* Data transfer between the host (CPU) and the device (GPU).|
|**Shared Memory**|* **Size:** Small, on-chip memory.  <br>* **Accessibility:** Visible to all threads within a single thread block.  <br>* **Speed:** Very fast, comparable to register speed, but can suffer from bank conflicts.|* Facilitating fast data sharing and communication between threads within a block.  <br>* Used to reduce global memory access by caching frequently used data that is shared among threads.|
|**Constant Memory**|* **Size:** Small, limited amount (e.g., 64KB).  <br>* **Accessibility:** Visible to all threads in the application.  <br>* **Speed:** Very fast because it is cached, but only for read-only data that remains constant during a kernel's execution.|* Storing values that do not change during a kernel run, such as a constant scaling factor.|
|**Local Memory**|* **Size:** Part of the global memory space.  <br>* **Accessibility:** Accessible only by the thread that owns it.  <br>* **Speed:** Slow, same as global memory.|* Used when a thread needs to store data that is too large for registers or whose size isn't known at compile time.  <br>* Stores local variables when all registers are in use (known as "register spilling").|