
CUDA "hello world" example that does vector addition—adding two arrays of 1 million elements in parallel on the GPU. This is more useful than just printing text because it shows the complete CUDA workflow:

**Key concepts in this example:**

- **`__global__`** keyword marks a function as a GPU kernel
- **Memory management**: `cudaMalloc()` for GPU, `malloc()` for CPU
- **Data transfer**: `cudaMemcpy()` moves data between CPU and GPU
- **Kernel launch**: `<<<blocks, threads>>>` syntax
- **Thread ID calculation**: Each thread figures out which array element it handles
- **Synchronization**: `cudaDeviceSynchronize()` waits for GPU to finish

To compile and run:

bash

```bash
!nvcc -arch=sm_75 -o hello_cuda hello_cuda.cu
!./hello_cuda
```

This demonstrates the basic pattern you'll use in almost all CUDA programs: allocate memory, transfer data to GPU, launch kernel, get results back, clean up.


```c++
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel - runs on GPU
// __global__ means this function is called from CPU but runs on GPU
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we don't go out of bounds
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int n = 1000000;  // 1 million elements
    size_t bytes = n * sizeof(float);
    
    // Allocate memory on host (CPU)
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    // Initialize arrays on host
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // Allocate memory on device (GPU)
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    
    // Time: Copy data from host to device
    printf("\n=== Timing Breakdown ===\n");
    cudaEventRecord(start);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Host to Device transfer: %.3f ms\n", milliseconds);
    
    // Setup kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("\nLaunching kernel with %d blocks of %d threads\n", 
           blocksPerGrid, threadsPerBlock);
    
    // Time: Kernel execution
    cudaEventRecord(start);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution: %.3f ms\n", milliseconds);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Make sure kernel is complete
    cudaDeviceSynchronize();
    
    // Time: Copy result back to host
    cudaEventRecord(start);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Device to Host transfer: %.3f ms\n", milliseconds);
    
    // Verify result
    bool success = true;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != 3.0f) {
            printf("Error at index %d: expected 3.0, got %f\n", i, h_c[i]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("\nSuccess! Added %d elements on GPU\n", n);
    }
    
    printf("======================\n");
    
    // Free CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}

/* 
 * To compile and run:
 * 
 * LOCAL:
 * nvcc -o hello_cuda hello_cuda.cu
 * ./hello_cuda
 * 
 * GOOGLE COLAB:
 * !nvcc -arch=sm_75 -o hello_cuda hello_cuda.cu
 * !./hello_cuda
 * 
 * (Use -arch=sm_75 for Tesla T4, -arch=sm_37 for K80, -arch=sm_60 for P100)
 * 
 * Key concepts demonstrated:
 * 1. __global__ keyword for kernel functions
 * 2. Memory allocation with cudaMalloc()
 * 3. Data transfer with cudaMemcpy()
 * 4. Kernel launch syntax: kernel<<<blocks, threads>>>()
 * 5. Thread ID calculation: blockIdx.x * blockDim.x + threadIdx.x
 * 6. Synchronization with cudaDeviceSynchronize()
 * 7. Memory cleanup with cudaFree() and free()
 * 8. Timing with CUDA events
 * 9. Error checking with cudaGetLastError()
 */
```