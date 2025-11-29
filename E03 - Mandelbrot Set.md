
**Features:**

- Renders 1920x1080 HD images (over 2 million pixels)
- Creates a zoom sequence from 1x to 1000x magnification
- Implements both CPU (sequential) and GPU (parallel) versions
- Uses beautiful rainbow color gradients
- Times both implementations accurately
- Saves output as PPM image files

**What you'll see:**

- **CPU**: Several seconds per frame (~0.5-3 FPS)
- **GPU**: Milliseconds per frame (~30-300+ FPS)
- **Speedup**: Typically 50-500x faster on GPU

**To compile and run:**

```bash
nvcc -o mandelbrot mandelbrot.cu
./mandelbrot
```

**To view the images:** The program generates PPM files. To convert to PNG:

```bash
convert mandelbrot_gpu_frame1.ppm mandelbrot_gpu_frame1.png
```

Or open directly with GIMP, Photoshop, or most image viewers.

**For your presentation:** You can increase `numFrames` to generate more frames and create a video showing smooth real-time zooming:

```bash
ffmpeg -framerate 30 -i mandelbrot_gpu_frame%d.ppm -c:v libx264 mandelbrot_zoom.mp4
```

## Strategy: Create a "Race" Video

The most impactful approach is to show **both CPU and GPU rendering the same zoom sequence side-by-side in real-time**. The GPU side zooms smoothly while the CPU side stutters along, creating a visceral understanding of the performance gap.

Here's how to do it:

### Step 1: Modify the code to generate more frames

Change these lines in the code:

```cpp
int numFrames = 100;  // Generate 100 frames for a longer zoom
double zoomEnd = 10000.0;  // Deeper zoom
```

### Step 2: Generate all frames

The program will output frames for both CPU and GPU with timing information.

### Step 3: Create videos at their actual FPS

After running the program, note the actual FPS from the output (e.g., CPU: 2 FPS, GPU: 60 FPS).

**Create CPU video at its actual FPS:**

```bash
ffmpeg -framerate 2 -i mandelbrot_cpu_frame%d.ppm -c:v libx264 -pix_fmt yuv420p cpu_realtime.mp4
```

**Create GPU video at its actual FPS:**

```bash
ffmpeg -framerate 60 -i mandelbrot_gpu_frame%d.ppm -c:v libx264 -pix_fmt yuv420p gpu_realtime.mp4
```

### Step 4: Create side-by-side comparison (Most Dramatic!)

```bash
ffmpeg -i cpu_realtime.mp4 -i gpu_realtime.mp4 -filter_complex \
"[0:v]scale=960:1080,drawtext=text='CPU':fontsize=48:fontcolor=white:x=(w-text_w)/2:y=30[cpu]; \
[1:v]scale=960:1080,drawtext=text='GPU (CUDA)':fontsize=48:fontcolor=white:x=(w-text_w)/2:y=30[gpu]; \
[cpu][gpu]hstack=inputs=2[v]" \
-map "[v]" -c:v libx264 -pix_fmt yuv420p comparison.mp4
```

This creates a split-screen where:

- **Left side (CPU)**: Stutters through frames slowly
- **Right side (GPU)**: Zooms smoothly in real-time
- The GPU side will complete the zoom while the CPU is still rendering early frames!

### Alternative: "Time-lapse" Approach

If you want to show the full sequence without waiting forever for the CPU:

**Option A - Speed up CPU to match duration:**

```bash
# Speed up CPU video to match GPU video duration
ffmpeg -i cpu_realtime.mp4 -filter:v "setpts=0.033*PTS" cpu_timelapse.mp4
```

Then create side-by-side with both sped up equally to show "what if they finished at the same time":

```bash
ffmpeg -i cpu_timelapse.mp4 -i gpu_realtime.mp4 -filter_complex \
"[0:v]drawtext=text='CPU (30x speed-up)':fontsize=36:fontcolor=white:x=10:y=30[cpu]; \
[1:v]drawtext=text='GPU (Real-time)':fontsize=36:fontcolor=white:x=10:y=30[gpu]; \
[cpu][gpu]hstack=inputs=2[v]" \
-map "[v]" -c:v libx264 -pix_fmt yuv420p comparison_normalized.mp4
```

### My Recommendation: The "Progress Bar Race"

Create a video showing both rendering simultaneously with a progress bar:

```bash
ffmpeg -i cpu_realtime.mp4 -i gpu_realtime.mp4 -filter_complex \
"[0:v]scale=960:1080,drawtext=text='CPU: %{frame_num} frames':fontsize=36:fontcolor=white:x=10:y=30:box=1:boxcolor=black@0.5[cpu]; \
[1:v]scale=960:1080,drawtext=text='GPU: %{frame_num} frames':fontsize=36:fontcolor=white:x=10:y=30:box=1:boxcolor=black@0.5[gpu]; \
[cpu][gpu]hstack=inputs=2[v]" \
-map "[v]" -t 10 -c:v libx264 -pix_fmt yuv420p race_10sec.mp4
```

In a 10-second clip:

- **CPU** renders maybe 20 frames
- **GPU** renders 600 frames

The audience literally **watches the GPU complete the entire zoom** while the CPU barely starts!

### For Your Presentation

I'd recommend:

1. Show the **side-by-side race** for 10-15 seconds
2. Pause to let the dramatic difference sink in
3. Then show the final frames side-by-side to prove they're computing the same thing

Would you like me to create a modified version of the code that automatically generates the frames and provides the exact ffmpeg commands based on your measured FPS?


```c++
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define WIDTH 1920
#define HEIGHT 1080
#define MAX_ITER 256

// Color palette for beautiful visualization
struct Color {
    unsigned char r, g, b;
};

// Generate smooth color gradient
__host__ __device__ Color getColor(int iter, int max_iter) {
    if (iter == max_iter) {
        return {0, 0, 0};  // Black for points in the set
    }
    
    // Smooth coloring using continuous iteration count
    float t = (float)iter / max_iter;
    
    Color c;
    // Create a rainbow gradient
    c.r = (unsigned char)(9 * (1 - t) * t * t * t * 255);
    c.g = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * 255);
    c.b = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
    
    return c;
}

// CPU implementation - sequential computation
void mandelbrotCPU(Color *image, int width, int height, 
                   double centerX, double centerY, double zoom, int maxIter) {
    double scale = 4.0 / (zoom * width);
    
    for (int py = 0; py < height; py++) {
        for (int px = 0; px < width; px++) {
            // Map pixel to complex plane
            double x0 = centerX + (px - width / 2.0) * scale;
            double y0 = centerY + (py - height / 2.0) * scale;
            
            double x = 0.0;
            double y = 0.0;
            int iter = 0;
            
            // Iterate: z = z^2 + c
            while (x*x + y*y <= 4.0 && iter < maxIter) {
                double xtemp = x*x - y*y + x0;
                y = 2*x*y + y0;
                x = xtemp;
                iter++;
            }
            
            image[py * width + px] = getColor(iter, maxIter);
        }
    }
}

// GPU kernel - parallel computation
__global__ void mandelbrotGPU(Color *image, int width, int height,
                               double centerX, double centerY, double zoom, int maxIter) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px >= width || py >= height) return;
    
    double scale = 4.0 / (zoom * width);
    
    // Map pixel to complex plane
    double x0 = centerX + (px - width / 2.0) * scale;
    double y0 = centerY + (py - height / 2.0) * scale;
    
    double x = 0.0;
    double y = 0.0;
    int iter = 0;
    
    // Iterate: z = z^2 + c
    while (x*x + y*y <= 4.0 && iter < maxIter) {
        double xtemp = x*x - y*y + x0;
        y = 2*x*y + y0;
        x = xtemp;
        iter++;
    }
    
    image[py * width + px] = getColor(iter, maxIter);
}

// Save image as PPM file (can be opened with most image viewers)
void savePPM(const char *filename, Color *image, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    fwrite(image, sizeof(Color), width * height, fp);
    fclose(fp);
    printf("Saved: %s\n", filename);
}

int main() {
    printf("===========================================\n");
    printf("MANDELBROT SET: CPU vs GPU Performance\n");
    printf("Resolution: %dx%d (%d megapixels)\n", WIDTH, HEIGHT, (WIDTH*HEIGHT)/1000000);
    printf("===========================================\n\n");
    
    size_t imageSize = WIDTH * HEIGHT * sizeof(Color);
    
    // Allocate host memory
    Color *h_image_cpu = (Color*)malloc(imageSize);
    Color *h_image_gpu = (Color*)malloc(imageSize);
    
    // Zoom sequence parameters
    double centerX = -0.7;      // Interesting location in the set
    double centerY = 0.0;
    double zoomStart = 1.0;
    double zoomEnd = 1000.0;
    int numFrames = 5;
    
    printf("Rendering zoom sequence from %fx to %fx zoom...\n\n", zoomStart, zoomEnd);
    
    // ============================================
    // CPU RENDERING
    // ============================================
    printf("CPU Rendering:\n");
    printf("------------------------------------------\n");
    
    clock_t totalCPUTime = 0;
    
    for (int frame = 0; frame < numFrames; frame++) {
        double zoom = zoomStart * pow(zoomEnd / zoomStart, (double)frame / (numFrames - 1));
        
        clock_t start = clock();
        mandelbrotCPU(h_image_cpu, WIDTH, HEIGHT, centerX, centerY, zoom, MAX_ITER);
        clock_t end = clock();
        
        double frameTime = ((double)(end - start)) / CLOCKS_PER_SEC;
        totalCPUTime += (end - start);
        
        printf("Frame %d (zoom: %.1fx): %.3f seconds\n", frame + 1, zoom, frameTime);
        
        // Save first and last frame
        if (frame == 0 || frame == numFrames - 1) {
            char filename[100];
            sprintf(filename, "mandelbrot_cpu_frame%d.ppm", frame + 1);
            savePPM(filename, h_image_cpu, WIDTH, HEIGHT);
        }
    }
    
    double avgCPUTime = ((double)totalCPUTime / CLOCKS_PER_SEC) / numFrames;
    printf("\nAverage CPU time per frame: %.3f seconds\n", avgCPUTime);
    printf("CPU FPS: %.2f\n\n", 1.0 / avgCPUTime);
    
    // ============================================
    // GPU RENDERING
    // ============================================
    printf("GPU Rendering:\n");
    printf("------------------------------------------\n");
    
    // Allocate device memory
    Color *d_image;
    cudaMalloc(&d_image, imageSize);
    
    // Setup kernel launch parameters
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((WIDTH + 15) / 16, (HEIGHT + 15) / 16);
    
    printf("Grid: %dx%d blocks, Block: %dx%d threads\n", 
           blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);
    printf("Total GPU threads: %d\n\n", 
           blocksPerGrid.x * blocksPerGrid.y * threadsPerBlock.x * threadsPerBlock.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float totalGPUTime = 0.0f;
    
    for (int frame = 0; frame < numFrames; frame++) {
        double zoom = zoomStart * pow(zoomEnd / zoomStart, (double)frame / (numFrames - 1));
        
        cudaEventRecord(start);
        
        mandelbrotGPU<<<blocksPerGrid, threadsPerBlock>>>(
            d_image, WIDTH, HEIGHT, centerX, centerY, zoom, MAX_ITER);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float frameTime;
        cudaEventElapsedTime(&frameTime, start, stop);
        totalGPUTime += frameTime;
        
        printf("Frame %d (zoom: %.1fx): %.3f seconds\n", frame + 1, zoom, frameTime / 1000.0);
        
        // Copy result back and save first and last frame
        if (frame == 0 || frame == numFrames - 1) {
            cudaMemcpy(h_image_gpu, d_image, imageSize, cudaMemcpyDeviceToHost);
            char filename[100];
            sprintf(filename, "mandelbrot_gpu_frame%d.ppm", frame + 1);
            savePPM(filename, h_image_gpu, WIDTH, HEIGHT);
        }
    }
    
    float avgGPUTime = (totalGPUTime / 1000.0) / numFrames;
    printf("\nAverage GPU time per frame: %.3f seconds\n", avgGPUTime);
    printf("GPU FPS: %.2f\n\n", 1.0 / avgGPUTime);
    
    // ============================================
    // PERFORMANCE COMPARISON
    // ============================================
    printf("===========================================\n");
    printf("PERFORMANCE SUMMARY\n");
    printf("===========================================\n");
    printf("Average CPU time: %.3f seconds (%.2f FPS)\n", avgCPUTime, 1.0 / avgCPUTime);
    printf("Average GPU time: %.3f seconds (%.2f FPS)\n", avgGPUTime, 1.0 / avgGPUTime);
    printf("Speedup:          %.2fx faster on GPU\n", avgCPUTime / avgGPUTime);
    printf("===========================================\n\n");
    
    if (avgGPUTime < 0.033) {
        printf("✓ GPU can render at 30+ FPS! Real-time zoom is possible!\n");
    } else if (avgGPUTime < 0.016) {
        printf("✓ GPU can render at 60+ FPS! Buttery smooth real-time zoom!\n");
    }
    
    printf("\nOutput files:\n");
    printf("- mandelbrot_cpu_frame1.ppm & mandelbrot_cpu_frame%d.ppm (CPU)\n", numFrames);
    printf("- mandelbrot_gpu_frame1.ppm & mandelbrot_gpu_frame%d.ppm (GPU)\n", numFrames);
    printf("\nTo view: convert frame1.ppm frame1.png (using ImageMagick)\n");
    printf("Or open directly with GIMP, Photoshop, or most image viewers\n");
    
    // Cleanup
    cudaFree(d_image);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_image_cpu);
    free(h_image_gpu);
    
    return 0;
}

/*
 * To compile and run:
 * nvcc -o mandelbrot mandelbrot.cu
 * ./mandelbrot
 * 
 * This will generate PPM image files showing the Mandelbrot zoom.
 * 
 * To convert PPM to PNG (for easier viewing):
 * convert mandelbrot_gpu_frame1.ppm mandelbrot_gpu_frame1.png
 * 
 * Expected performance:
 * - CPU: Several seconds per frame (0.5-3 FPS)
 * - GPU: Milliseconds per frame (30-300+ FPS)
 * - Speedup: 50-500x depending on hardware
 * 
 * Why this demonstrates GPU power:
 * - 2+ million pixels computed independently
 * - Same iterative calculation for each pixel
 * - Perfect data parallelism
 * - CPU does them one at a time, GPU does them all at once
 * - Visually stunning output shows the computational power
 * 
 * For presentation:
 * You can generate a full zoom sequence by increasing numFrames
 * and create a video from the frames to show smooth real-time
 * zooming that would be impossible on CPU alone.
 */
```