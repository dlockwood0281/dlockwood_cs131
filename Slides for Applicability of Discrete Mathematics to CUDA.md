
- [[#Slide 1. **Graph Algorithms**|Slide 1. Graph Algorithms]]
- [[#Example 01|Example 01]]
- [[#Slide 2. **Combinatorial Optimization**|Slide 2. Combinatorial Optimization]]
- [[#Slide 3. **Algorithm Complexity Analysis**|Slide 3. Algorithm Complexity Analysis]]
- [[#Example 02|Example 02]]
- [[#Slide 4. **Boolean Algebra and Bit Operations**|Slide 4. Boolean Algebra and Bit Operations]]
- [[#Slide 5. **Number Theory and Cryptography**|Slide 5. Number Theory and Cryptography]]
- [[#Slide 6. **Discrete Probability and Monte Carlo Methods**|Slide 6. Discrete Probability and Monte Carlo Methods]]
- [[#Slide 7. **Recurrence Relations**|Slide 7. Recurrence Relations]]
- [[#Slide 8. **Set Operations**|Slide 8. Set Operations]]
- [[#Slide 9. Real-World Example:|Slide 9. Real-World Example:]]
- [[#Example 03|Example 03]]
- [[#Slide 10. The Bottom Line:|Slide 10. The Bottom Line:]]


## Slide 1. **Graph Algorithms**

Graph processing is one of the biggest uses of GPU computing:

- **Breadth-First Search (BFS)** - Each node's neighbors can be explored in parallel
- **Shortest path algorithms** - Dijkstra's, Bellman-Ford parallelized on GPUs
- **PageRank** - Google's algorithm runs efficiently on GPUs
- **Graph coloring, matching, clustering** - All benefit from parallel computation
- Social network analysis, recommendation systems, route finding

Graph problems are naturally parallel because you can process many nodes/edges simultaneously. This is exactly what GPUs excel at.

## Example 01
CUDA "hello world" example that does vector addition—adding two arrays of 1 million elements in parallel on the GPU. This is more useful than just printing text because it shows the complete CUDA workflow:

**Key concepts in this example:**

- **`__global__`** keyword marks a function as a GPU kernel
- **Memory management**: `cudaMalloc()` for GPU, `malloc()` for CPU
- **Data transfer**: `cudaMemcpy()` moves data between CPU and GPU
- **Kernel launch**: `<<<blocks, threads>>>` syntax
- **Thread ID calculation**: Each thread figures out which array element it handles
- **Synchronization**: `cudaDeviceSynchronize()` waits for GPU to finish

=== Timing Breakdown ===
Host to Device transfer: 2.284 ms
Launching kernel with 3907 blocks of 256 threads
Kernel execution: 23.714 ms
Device to Host transfer: 0.994 ms
Success! Added 1000000 elements on GPU

## Slide 2. **Combinatorial Optimization**

Many discrete optimization problems parallelize well:

- **Traveling Salesman Problem** - Parallel branch and bound
- **Knapsack problems** - Exploring solution space in parallel
- **SAT solvers** - Boolean satisfiability solved faster on GPUs
- **Constraint satisfaction** - Parallel search through possibilities

## Slide 3. **Algorithm Complexity Analysis**

Understanding Big-O notation helps you identify CUDA-suitable problems:

- **O(n²) or O(n³) algorithms** → Great GPU candidates (matrix operations, all-pairs problems)
- **O(log n) with dependencies** → Poor GPU candidates (binary search with sequential dependencies)
- **O(n) embarrassingly parallel** → Perfect for GPU (element-wise operations)

Discrete math teaches you to analyze whether a problem has **inherent parallelism** vs **forced sequencing**.

## Example 02

Multiply two 2048x2048 matrices (over 8 million operations)
- Same computation on both CPU and GPU
- Time both implementations accurately
- Verify the results match
- Show the speedup factor

**CPU approach:**
- Triple nested loop executing sequentially ($\huge O(n^3)$)
- One element computed at a time
- Takes seconds to complete

**GPU approach:**
- Thousands of threads running in parallel 
- Each thread computes one output element simultaneously
- Uses 2D grid and block organization
- Takes milliseconds to complete

### **Results:**
Matrix Multiplication: CPU vs GPU 
Matrix size: 2048 x 2048

Running on CPU... 
CPU Time: 95.640 seconds 

Running on GPU... 
Grid: 128 x 128 blocks 
Block: 16 x 16 threads 
Total threads: 4194304 
GPU Time: 0.075 seconds 

`======================================== `
PERFORMANCE COMPARISON 
`======================================== `
CPU Time: 95.640 seconds 
GPU Time: 0.075 seconds 
Speedup: 1275.58x faster 
`======================================== `
Verifying results... ✓ Results match! GPU computation is correct.

## Slide 4. **Boolean Algebra and Bit Operations**

- **Bitwise parallel operations** - GPUs can do billions of bit operations simultaneously
- **Boolean circuits** - Simulating logic gates in parallel
- **Bit manipulation algorithms** - Population count, finding set bits, etc.

## Slide 5. **Number Theory and Cryptography**

- **Prime number generation** - Parallel sieving
- **Modular arithmetic** - RSA encryption operations on GPUs
- **Hashing algorithms** - Parallel hash computations (blockchain mining)
- **Random number generation** - Parallel RNG for simulations

## Slide 6. **Discrete Probability and Monte Carlo Methods**

- **Simulations** - Run millions of trials in parallel
- **Random walks** - Parallel exploration of state spaces
- **Probabilistic algorithms** - Each trial independent and parallel

## Slide 7. **Recurrence Relations**

Understanding which recurrences parallelize:

- **Independent recurrences** → Good for GPU (Fibonacci across many starting points)
- **Dependent recurrences** → Bad for GPU (computing single Fibonacci sequence)

## Slide 8. **Set Operations**

Working with large sets on GPUs:

- **Union, intersection, difference** - Parallel set operations on millions of elements
- **Cardinality counting** - Parallel reduction operations
- **Set membership testing** - Bloom filters on GPU

## Slide 9. Real-World Example:

Consider **shortest path in a graph with 1 million nodes**:

- **Discrete Math**: Teaches you Dijkstra's algorithm, graph representation, complexity O(V² or E log V)
- **CUDA**: Lets you explore thousands of nodes simultaneously instead of one at a time
- The combination makes previously intractable problems (huge graphs) solvable in seconds

## Example 03
Render 1920x1080 HD images (over 2 million pixels)
Create a zoom sequence from 1x to 1000x magnification
Both CPU (sequential) and GPU (parallel) versions
Timed both implementations accurately

- **CPU**: Several seconds per frame (~0.5-3 FPS)
- **GPU**: Milliseconds per frame (~30-300+ FPS)
- **Speedup**: Typically 50-500x faster on GPU

## Slide 10. The Bottom Line:

Discrete Math helps you **recognize which problems are parallelizable** and **design parallel algorithms**. CUDA gives you the **tool to execute** those parallel algorithms at massive scale.

They're complementary:

- **Discrete Math** → Understand the problem structure
- **CUDA** → Exploit that structure with parallel hardware

For CS students, discrete math provides the theoretical foundation to understand **why** certain problems benefit from GPU acceleration and **how** to design algorithms that leverage thousands of parallel threads effectively.

Does this connection make sense for your presentation? You could potentially include a slide showing how discrete math concepts map to CUDA applications.