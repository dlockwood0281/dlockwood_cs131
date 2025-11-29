
- [[#**CUDA C/C++ Syntax:**|**CUDA C/C++ Syntax:**]]
	- [[#**CUDA C/C++ Syntax:**#Kernel launch syntax|Kernel launch syntax]]
		- [[#Kernel launch syntax#`cudaMalloc`:|`cudaMalloc`:]]
		- [[#Kernel launch syntax#`cudaMemcpy`:|`cudaMemcpy`:]]
		- [[#Kernel launch syntax#`cudaFree`:|`cudaFree`:]]
	- [[#**CUDA C/C++ Syntax:**#Example Workflow:|Example Workflow:]]

# CUDA Programming

## **CUDA C/C++ Syntax:** 
Extending C/C++ with CUDA-specific keywords and APIs.

CUDA extends C/C++ with new keywords, functions, and a specific syntax for managing and launching parallel code on NVIDIA GPUs. A typical CUDA application consists of both host code (running on the CPU) and device code (running on the GPU). 

Function execution qualifiers

These qualifiers determine where a function is executed and from where it can be called. 

|Qualifier|Execution Location|Callable From|Purpose|
|---|---|---|---|
|`__global__`|GPU (device)|CPU (host) or GPU|Defines a _kernel_, the primary function for parallel execution on the GPU. Must have `void` return type and asynchronous behavior.|
|`__device__`|GPU (device)|GPU (device) only|Defines a helper function that can be called by kernels or other `__device__` functions.|
|`__host__`|CPU (host)|CPU (host) only|Defines a standard CPU function. It is often combined with `__device__` for a function to be compiled for and runnable on both the CPU and GPU.|

### Kernel launch syntax

To execute a `__global__` kernel function, you use the special triple-angle-bracket syntax `<<<...>>>`.

- **Memory Management:** `cudaMalloc`, `cudaMemcpy`, `cudaFree` for allocating and transferring data between host and device.

CUDA provides functions for managing memory on the GPU device and transferring data between the host (CPU) and the device. These functions are crucial for any CUDA application.

#### `cudaMalloc`:

- **Purpose:** Allocates a specified amount of linear memory on the device (GPU).
- **Syntax:**

C++

```c++
    cudaError_t cudaMalloc(void** devPtr, size_t size);
```

- `devPtr`: A pointer to a pointer that will store the address of the newly allocated device memory.
- `size`: The size in bytes of the memory to allocate.
- **Usage:** Similar to `malloc` in C, but allocates memory on the GPU, making it accessible to CUDA kernels.

#### `cudaMemcpy`:

- **Purpose:** Copies data between host memory and device memory, or between different locations within device memory.
- **Syntax:**

C++

```c++
    cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
```

- `dst`: Pointer to the destination memory.
- `src`: Pointer to the source memory.
- `count`: The number of bytes to copy.
- `kind`: Specifies the direction of the copy (e.g., `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, `cudaMemcpyDeviceToDevice`).
- **Usage:** Essential for transferring input data from the CPU to the GPU before kernel execution and transferring results back from the GPU to the CPU after kernel completion.

#### `cudaFree`:

- **Purpose:** Frees memory previously allocated on the device using `cudaMalloc`.
- **Syntax:**

C++

```c++
    cudaError_t cudaFree(void* devPtr);
```

- `devPtr`: A pointer to the device memory to be freed.
- **Usage:** Important for preventing memory leaks and releasing GPU resources when they are no longer needed. Similar to `free` in C.

### Example Workflow:

- **Allocate host memory:** Use `malloc` or `new` on the CPU for input data.
- **Allocate device memory:** Use `cudaMalloc` to allocate corresponding memory on the GPU.
- **Copy data from host to device:** Use `cudaMemcpy` with `cudaMemcpyHostToDevice` to transfer input data.
- **Launch CUDA kernel:** Execute the GPU computation.
- **Copy data from device to host:** Use `cudaMemcpy` with `cudaMemcpyDeviceToHost` to retrieve results.
- **Free device memory:** Use `cudaFree` to release the allocated GPU memory.
- **Free host memory:** Use `free` or `delete` on the CPU.

- **Kernel Launch:** Specifying the grid and block dimensions for kernel execution.

Kernel launch in parallel computing frameworks like CUDA involves defining the execution configuration for a `__global__` function (kernel). This configuration primarily specifies the grid dimensions and block dimensions, which dictate how threads are organized and scheduled on the GPU.

- **Grid Dimensions:**
    
    - The grid represents the overall collection of thread blocks that will execute the kernel.
    - It is typically defined by `gridSize`, a `dim3` structure (or similar in other frameworks) that specifies the number of blocks in the X, Y, and Z dimensions.
    - For example, `dim3 gridSize(8, 4, 1)` would launch a grid with 8 blocks in the X-dimension and 4 blocks in the Y-dimension, for a total of 32 blocks.
    
- **Block Dimensions:**
    
    - Each block within the grid is a group of threads that can communicate and synchronize efficiently.
    - It is defined by `blockSize`, another `dim3` structure, specifying the number of threads within each block in the X, Y, and Z dimensions.
    - For example, `dim3 blockSize(32, 4, 1)` would define a block containing 32 threads in the X-dimension and 4 threads in the Y-dimension, for a total of 128 threads per block.
    

Execution Configuration Syntax (CUDA example):

C++

```
myKernel<<<gridSize, blockSize>>>(arg1, arg2, ...);
```

This syntax launches `myKernel` with the specified `gridSize` and `blockSize`. Within the kernel, variables like `gridDim`, `blockDim`, `blockIdx`, and `threadIdx` can be used to determine the unique position of each thread within the grid and its respective block.

Considerations for Choosing Dimensions:

- **Hardware Limits:** 
    
    GPUs have limitations on the maximum number of threads per block and the dimensions of blocks and grids.
    
- **Performance:** 
    
    Optimal performance often involves choosing block sizes that are multiples of the warp size (typically 32 threads) and ensuring sufficient occupancy of the GPU's streaming multiprocessors (SMs).
    
- **Problem Structure:** 
    
    The dimensions should align with the data structure and computational patterns of the problem being solved.

- **Simple Examples:** Demonstrating basic CUDA programs like vector addition.

```c++
//
// Demonstration using a single 1D grid and 1D block size
//
/*
 * Example of vector addition :
 * Array of floats x is added to array of floats y and the 
 * result is placed back in y
 *
 */
#include <math.h>   // ceil function
#include <stdio.h>  // printf
#include <iostream> // alternative cout print for illustration

#include <cuda.h>

void initialize(float *x, float *y, int N);
void verifyCorrect(float *y, int N);
void getArguments(int argc, char **argv, int *blockSize);

///////
// error checking macro taken from Oakridge Nat'l lab training code:
// https://github.com/olcf/cuda-training-series
////////
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

// Kernel function based on 1D grid of 1D blocks of threads
// In this version, thread number is:
//  its block number in the grid (blockIdx.x) times 
// the threads per block plus which thread it is in that block.
//
// This thread id is then the index into the 1D array of floats.
// This represents the simplest type of mapping:
// Each thread takes care of one element of the result
__global__ void vecAdd(float *x, float *y, int n)
{
    // Get our global thread ID designed to be an array index
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
 
    // Make sure we do not go out of bounds;
    // Threads allocated could be larger than array length
    if (id < n)
        y[id] = x[id] + y[id];
}

////////////////////                   main
int main(int argc, char **argv)
{
  printf("Vector addition by managing memory ourselves.\n");
  // Set up size of arrays for vectors 
  int N = 32*1048576;
  // TODO: try changng the size of the arrays by doubling or
  //       halving (32 becomes 64 or 16). Note how the grid size changes.
  printf("size (N) of 1D arrays are: %d\n\n", N);

  // host vectors, which are arrays of length N
  float *x, *y;

  // Size, in bytes, of each vector
  size_t bytes = N*sizeof(float);

  // 1.1 Allocate memory for each vector on host
  x = (float*)malloc(bytes);
  y = (float*)malloc(bytes);

  // 1.2 initialize x and y arrays on the host
  initialize(x, y, N);  // set values in each vector

   // device array storage
  float *d_x;
  float *d_y;

  printf("allocate vectors and copy to device\n");

  // 2. Allocate memory for each vector on GPU device
  cudaMalloc(&d_x, bytes);
  cudaMalloc(&d_y, bytes);
  cudaCheckErrors("allocate device memory");

  // 3. Copy host vectors to device
  cudaMemcpy( d_x, x, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy( d_y, y, bytes, cudaMemcpyHostToDevice);
  cudaCheckErrors("mem copy to device");

  // Default number of threads in each thread block
  int blockSize = 256;
  getArguments(argc, argv, &blockSize); //update blocksize from cmd line
 
  // Number of thread blocks in grid needs to be based on array size
  // and block size
  int gridSize = (int)ceil((float)N/blockSize);
 
  printf("add vectors on device using grid with ");
  printf("%d blocks of %d threads each.\n", gridSize, blockSize);
  // 4. Execute the kernel
  vecAdd<<<gridSize, blockSize>>>(d_x, d_y, N);
  cudaCheckErrors("vecAdd kernel call");

  // 5. Ensure that device is finished
  cudaDeviceSynchronize();
  cudaCheckErrors("Failure to synchronize device");

  // 6. Copy array back to host (use original y for this)
  // Note that essentially the device gets synchronized
  // before this is performed.
  cudaMemcpy( y, d_y, bytes, cudaMemcpyDeviceToHost);
  cudaCheckErrors("mem copy device to host");

  // 7. Check that the computation ran correctly
  verifyCorrect(y, N); 

  printf("execution complete\n");

  // 8.1 Free device memory
  cudaFree(d_x);
  cudaFree(d_y);
  cudaCheckErrors("free cuda memory");

  // 8.2 Release host memory
  free(x);
  free(y);

  return 0;
}
///////////////////////// end main

///////////////////////////////// helper functions

// To initialize or reset the arrays for each trial
void initialize(float *x, float *y, int N) {
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
}

// check whether the kernel functions worked as expected
void verifyCorrect(float *y, int N) {
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmaxf(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
}

// simple argument gather for this simple example program
void getArguments(int argc, char **argv, int *blockSize) {

  if (argc == 2) {
    *blockSize = atoi(argv[1]);
  }
```