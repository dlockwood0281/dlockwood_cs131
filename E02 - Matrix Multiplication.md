
```c++
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 2048  // Matrix size (N x N)

// CPU implementation - sequential matrix multiplication
void matrixMulCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// GPU kernel - parallel matrix multiplication
__global__ void matrixMulGPU(float *a, float *b, float *c, int n) {
    // Calculate row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

// Utility function to initialize matrix with random values
void initMatrix(float *mat, int n) {
    for (int i = 0; i < n * n; i++) {
        mat[i] = (float)(rand() % 10);
    }
}

// Utility function to verify results match
bool verifyResults(float *cpu, float *gpu, int n, float tolerance = 0.01f) {
    for (int i = 0; i < n * n; i++) {
        if (abs(cpu[i] - gpu[i]) > tolerance) {
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, cpu[i], gpu[i]);
            return false;
        }
    }
    return true;
}

int main() {
    printf("Matrix Multiplication: CPU vs GPU\n");
    printf("Matrix size: %d x %d\n\n", N, N);
    
    size_t bytes = N * N * sizeof(float);
    
    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c_cpu = (float*)malloc(bytes);
    float *h_c_gpu = (float*)malloc(bytes);
    
    // Initialize matrices
    srand(time(NULL));
    initMatrix(h_a, N);
    initMatrix(h_b, N);
    
    // ============================================
    // CPU COMPUTATION
    // ============================================
    printf("Running on CPU...\n");
    clock_t start_cpu = clock();
    
    matrixMulCPU(h_a, h_b, h_c_cpu, N);
    
    clock_t end_cpu = clock();
    double time_cpu = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    printf("CPU Time: %.3f seconds\n\n", time_cpu);
    
    // ============================================
    // GPU COMPUTATION
    // ============================================
    printf("Running on GPU...\n");
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Setup kernel launch parameters
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    
    printf("Grid: %d x %d blocks\n", blocksPerGrid.x, blocksPerGrid.y);
    printf("Block: %d x %d threads\n", threadsPerBlock.x, threadsPerBlock.y);
    printf("Total threads: %d\n\n", 
           blocksPerGrid.x * blocksPerGrid.y * threadsPerBlock.x * threadsPerBlock.y);
    
    // Create CUDA events for timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    
    // Record start time
    cudaEventRecord(start_gpu);
    
    // Launch kernel
    matrixMulGPU<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    // Record end time
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    
    // Calculate elapsed time
    float time_gpu_ms;
    cudaEventElapsedTime(&time_gpu_ms, start_gpu, stop_gpu);
    double time_gpu = time_gpu_ms / 1000.0;
    
    printf("GPU Time: %.3f seconds\n\n", time_gpu);
    
    // Make sure kernel is complete before copying
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // Check for any CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    // ============================================
    // RESULTS
    // ============================================
    printf("========================================\n");
    printf("PERFORMANCE COMPARISON\n");
    printf("========================================\n");
    printf("CPU Time:     %.3f seconds\n", time_cpu);
    printf("GPU Time:     %.3f seconds\n", time_gpu);
    printf("Speedup:      %.2fx faster\n", time_cpu / time_gpu);
    printf("========================================\n\n");
    
    // Verify results match
    printf("Verifying results...\n");
    if (verifyResults(h_c_cpu, h_c_gpu, N)) {
        printf("✓ Results match! GPU computation is correct.\n");
    } else {
        printf("✗ Results don't match. Something went wrong.\n");
    }
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    
    return 0;
}
```
