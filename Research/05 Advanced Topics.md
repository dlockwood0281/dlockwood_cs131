- [[#**CUDA Libraries:**|**CUDA Libraries:**]]
	- [[#**CUDA Libraries:**#cuBLAS|cuBLAS]]
	- [[#**CUDA Libraries:**#cuFFT|cuFFT]]
	- [[#**CUDA Libraries:**#cuDNN|cuDNN]]
	- [[#**CUDA Libraries:**#cuSOLVER|cuSOLVER]]
	- [[#**CUDA Libraries:**#cuSPARSE|cuSPARSE]]
	- [[#**CUDA Libraries:**#Other notable libraries|Other notable libraries]]
- [[#Development tools|Development tools]]
	- [[#Development tools#Nsight|Nsight]]
		- [[#Nsight#Nsight Systems|Nsight Systems]]
		- [[#Nsight#Nsight Compute|Nsight Compute]]
	- [[#Development tools#**Error Handling:**|**Error Handling:**]]

# Advanced Topics

## **CUDA Libraries:** 
Introduction to cuBLAS, cuFFT, cuDNN, etc.

The CUDA toolkit provides libraries that accelerate common computational tasks by harnessing the power of NVIDIA GPUs. Rather than rewriting complex parallel algorithms from scratch, developers can integrate these optimized libraries into their C, C++, and Python applications to achieve significant performance gains. 

Core CUDA libraries

These are some of the most fundamental and widely used libraries for high-performance computing and AI. 

### cuBLAS

- **Full name:** CUDA Basic Linear Algebra Subprograms.
- **Function:** This library provides GPU-accelerated implementations of standard BLAS routines for vector-vector, matrix-vector, and matrix-matrix operations.
- **Key use cases:**
    - **Deep Learning:** Accelerating foundational operations like matrix multiplication, which are central to training neural networks.
    - **Scientific Computing:** Powering physics simulations, computational fluid dynamics, and other areas that rely on dense linear algebra.
- **Key features:**
    - Supports single-, double-, and half-precision floating-point arithmetic.
    - Includes specialized routines that leverage Tensor Cores on modern NVIDIA GPUs for higher performance. 

### cuFFT

- **Full name:** CUDA Fast Fourier Transform.
- **Function:** Provides a highly optimized library for computing Fast Fourier Transforms (FFTs) on NVIDIA GPUs.
- **Key use cases:**
    - **Signal Processing:** Analyzing and manipulating signals in fields like radar, acoustics, and telecommunications.
    - **Computational Physics:** Solving partial differential equations and other problems in quantum mechanics and fluid dynamics.
- **Key features:**
    - Supports 1D, 2D, and 3D transforms for complex and real data.
    - Offers APIs similar to the popular CPU-based FFTW library to simplify porting. 

### cuDNN

- **Full name:** CUDA Deep Neural Network library.
- **Function:** Provides a GPU-accelerated library of primitives designed specifically for deep neural networks.
- **Key use cases:**
    - **Deep Learning:** Accelerating the core building blocks of deep neural networks, such as convolutional layers, normalization, pooling, and attention mechanisms.
- **Key features:**
    - Highly optimized kernels for common deep learning operations.
    - Integrates seamlessly with popular frameworks like TensorFlow and PyTorch. 

### cuSOLVER

- **Full name:** CUDA Solver.
- **Function:** A high-level library built on cuBLAS and cuSPARSE that offers LAPACK-like functionality, such as common matrix factorizations and linear system solvers.
- **Key use cases:**
    - **Numerical Simulation:** Providing dense and sparse solvers for complex engineering and scientific problems.
- **Key features:**
    - Includes dense solvers for LU, QR, and SVD factorizations.
    - Provides specialized solvers for sparse matrices. 

### cuSPARSE

- **Full name:** CUDA Sparse.
- **Function:** Provides a library of optimized routines for performing basic linear algebra subroutines on sparse matrices.
- **Key use cases:**
    - **Machine Learning and AI:** Used in applications where data is sparse, such as text analysis or recommender systems.
    - **Computational Fluid Dynamics and Seismic Exploration:** Handling large matrices where most elements are zero.
- **Key features:**
    - Supports common sparse matrix formats like COO, CSR, and CSC.
    - Provides specialized operations that leverage Tensor Cores for performance. 

### Other notable libraries

NVIDIA provides a vast ecosystem of additional libraries for specific application domains. 

- **cuRAND:** A library for generating high-quality pseudorandom and quasirandom numbers on the GPU.
- **cuTENSOR:** A linear algebra library for high-performance tensor contractions, reductions, and element-wise operations.
- **NPP (NVIDIA Performance Primitives):** A collection of functions for GPU-accelerated image, video, and signal processing.
- **NCCL (NVIDIA Collective Communications Library):** A library for fast, multi-GPU and multi-node communication. 

## Development tools

### Nsight

For developers who need to debug and profile applications that use these libraries, NVIDIA offers the **Nsight** suite of tools. 

#### Nsight Systems

- **Function:** A system-wide performance analysis tool that provides a holistic view of your application's behavior across both the CPU and GPU.
- **Key use cases:**
    - **Bottleneck Detection:** Visualizing how different system resources are being used on a unified timeline to identify performance bottlenecks.
    - **CPU-GPU Interaction:** Tracing and visualizing CUDA API calls and library functions to optimize the interplay between the host and device. 

#### Nsight Compute

- **Function:** An interactive kernel profiler for fine-grained analysis of individual CUDA kernels, providing detailed performance metrics.
- **Key use cases:**
    - **Code Optimization:** Helping developers understand exactly how their kernels are performing to optimize code at a low level.

- **Streams:** Asynchronous execution and overlapping computation with data transfer.

Streams are queues of operations for GPUs that allow for asynchronous execution and the overlapping of computation with data transfer. By placing independent operations, like memory copies and kernel launches, into different streams, they can run concurrently instead of one after another. This overlap is crucial for performance, as it keeps the GPU busy by having it compute on one set of data while another set is being transferred. For asynchronous data transfers, operations must be issued to non-default streams, and pinned host memory is required. 

How streams achieve overlap

- **Asynchronous operations:** By default, GPU operations are synchronous, meaning the CPU waits for each one to finish before starting the next. Asynchronous functions, like `cudaMemcpyAsync()`, allow a program to initiate a data transfer and move on to the next task without waiting for the transfer to complete.
- **Multiple streams:** Creating and using multiple streams allows you to manage concurrent operations. For example, you can put a data transfer in Stream 1 and a kernel launch in Stream 2. If the GPU supports it and the data is ready, these can run at the same time.
- **Pipelining:** This technique creates a pipeline where the output of one operation can become the input for the next, without waiting for the first to fully complete. For example, a data transfer can start for the next batch of data while a kernel is finishing the current batch.
- **Pinned memory:** To enable asynchronous and overlapping operations between the host (CPU) and the device (GPU), the host memory must be "pinned" or "page-locked". This prevents the operating system from moving the memory to a different location, ensuring it's always accessible for the asynchronous transfer. 

Benefits

- **Increased GPU utilization:** Overlapping keeps the GPU busy, minimizing idle time and maximizing throughput.
- **Improved performance:** By not having to wait for data transfers to complete, applications can finish tasks much faster.
- **Efficient data handling:** It's a key technique for applications that process large datasets, as it can significantly reduce the time it takes to move data between the host and the GPU.

### **Error Handling:** 
Best practices for handling CUDA errors.

Best practices for handling CUDA errors involve a combination of diligent checking, robust logging, and graceful recovery mechanisms.

- **Check CUDA API Return Values:** 
    
    Always verify the `cudaError_t` return value of every CUDA API call. This includes memory allocations (`cudaMalloc`), data transfers (`cudaMemcpy`), and device synchronization calls (`cudaDeviceSynchronize`). 
    
- **Synchronize for Kernel Errors:** 
    
    Kernel launches are asynchronous, meaning errors may not be immediately apparent. Call `cudaDeviceSynchronize()` after kernel launches to force error detection and retrieve any asynchronous errors using `cudaGetLastError()`. 
    
- **Implement Comprehensive Logging:** 
    
    When an error occurs, log detailed information using `cudaGetErrorString()` to translate the error code into a human-readable message. Include contextual information like the function where the error occurred, timestamps, and relevant device or memory states.
    
- **Create Error-Checking Macros/Functions:** 
    
    Define custom macros or wrapper functions to encapsulate CUDA API calls and automatically check for errors, reducing boilerplate code and promoting consistency.
    

C++

```c++
    #define CHECK_CUDA_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)    
{       
    if (code != cudaSuccess)       
    {          
	    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);          
		    if (abort) exit(code);       
	}    
}
```

- **Handle Sticky Errors:** 
    
    In cases of unrecoverable "sticky" errors (e.g., `CUDA_ERROR_LAUNCH_FAILED`), the CUDA context may become unusable. The host process initiating the error might need to be terminated and potentially restarted with a new context.
    
- **Implement Graceful Fallbacks:** 
    
    Design applications to handle critical CUDA errors gracefully. This might involve switching to CPU-based alternatives, using a different GPU in a multi-GPU environment, or implementing cleanup routines to prevent resource leaks.
    
- **Utilize CUDA Debugging Tools:** 
    
    Leverage NVIDIA's debugging tools like CUDA-MEMCHECK for memory access violations and Nsight Compute/Nsight Systems for deeper analysis of performance and correctness issues.
    
- **Validate Memory Operations:** 
    
    Pay close attention to memory-related operations. Ensure proper allocation, deallocation, and access within bounds to prevent common memory errors.

- **Debugging and Profiling:** Tools like Nsight for analyzing and optimizing CUDA code.

NVIDIA Nsight tools are a comprehensive suite designed for debugging, profiling, and optimizing software that leverages NVIDIA GPUs, particularly for CUDA and graphics-intensive applications. These tools offer insights into various aspects of application performance and behavior.

Key Nsight Tools for Debugging and Profiling:

- **NVIDIA Nsight Systems:** 
    
    This is a system-wide performance analysis tool providing a unified timeline view of CPU and GPU activities, including kernel execution, memory transfers, and API calls. It helps identify bottlenecks, optimize resource utilization, and understand the interactions between different system components.
    
- **NVIDIA Nsight Compute:** 
    
    An interactive kernel profiler specifically for CUDA applications. It provides detailed performance metrics and API debugging capabilities for individual CUDA kernels, aiding in the identification of performance issues and offering guidance for optimization.
    
- **NVIDIA Nsight Graphics:** 
    
    A standalone developer tool with ray-tracing support, enabling debugging, profiling, and frame export for applications built with various graphics APIs like Direct3D, Vulkan, OpenGL, and OpenVR. 
    
- **NVIDIA Nsight Visual Studio Edition/Code Edition/Eclipse Edition:** 
    
    These are integrations into popular Integrated Development Environments (IDEs) like Visual Studio, Visual Studio Code, and Eclipse. They provide features such as IntelliSense, debugger views, and productivity enhancements specifically for CUDA development, enabling seamless debugging and profiling within the familiar IDE environment.
    
- **CUDA-GDB:** 
    
    An extension to the GNU Project Debugger (GDB), providing a command-line mechanism for debugging CUDA applications running on actual hardware, including support for CPU and GPU debugging, breakpoints, stepping, and variable inspection.
    
- **Compute Sanitizer:** 
    
    A functional correctness checking suite that helps detect various errors in CUDA applications, such as memory access errors, shared memory data access hazards, uninitialized accesses, and invalid synchronization primitive usage. 
    

These tools collectively provide a powerful ecosystem for developers to analyze, debug, and optimize their CUDA and GPU-accelerated applications, ensuring maximum performance and correctness.
