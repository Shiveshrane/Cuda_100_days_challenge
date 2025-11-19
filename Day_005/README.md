# Day 5: Shared Memory Tiled Matrix Multiplication - Final Project

**Chapter 5 Final Project**

## Overview
This is the culmination of the first week of your CUDA journey. Today, you'll implement an optimized tiled matrix multiplication kernel using shared memory, benchmark it against your naive implementation from Day 3, and analyze the performance improvements.

## Concepts Covered

### 1. **Shared Memory Tiling**
- Breaking matrices into tiles that fit in shared memory
- Cooperative loading of tile data by thread blocks
- Reducing global memory bandwidth requirements

### 2. **Memory Hierarchy Optimization**
- Exploiting the ~100x faster shared memory vs global memory
- Understanding the memory access pattern benefits
- Arithmetic intensity improvement

### 3. **Thread Block Collaboration**
- Threads within a block working together to load tiles
- Synchronization with `__syncthreads()`
- Accumulating partial results across tiles

## Learning Objectives

By the end of today, you should be able to:
- [ ] Implement a tiled matrix multiplication kernel with shared memory
- [ ] Optimize tile size for your specific GPU architecture
- [ ] Benchmark and compare naive vs tiled implementations
- [ ] Explain why tiled multiplication is faster
- [ ] Create performance comparison graphs
- [ ] Calculate achieved bandwidth and FLOPS

## Implementation Steps

### Step 1: Review Naive Implementation (Day 3)
Review your basic matrix multiplication from Day 3 to establish a baseline.

### Step 2: Design Tiled Kernel
```cuda
// Key concepts:
// - Each thread block handles a TILE_SIZE x TILE_SIZE output tile
// - Threads cooperatively load input tiles into shared memory
// - Accumulate results as you process each pair of tiles
// - Use __syncthreads() after loading each tile
```

### Step 3: Tile Size Optimization
Test different tile sizes:
- 8x8
- 16x16
- 32x32

Consider:
- Shared memory limits (typically 48KB per SM)
- Register usage
- Occupancy implications

### Step 4: Benchmarking
Test with various matrix sizes:
- Small: 256x256
- Medium: 1024x1024
- Large: 4096x4096

Measure:
- Execution time
- Speedup vs naive
- GFLOPS achieved
- Memory bandwidth utilization

## Key Algorithm

**Tiled Matrix Multiplication Steps:**
1. Each thread block computes one tile of output matrix C
2. Loop over tiles of A and B needed for this output tile
3. For each tile pair:
   - Cooperatively load tile of A into shared memory
   - Cooperatively load tile of B into shared memory
   - Synchronize threads (`__syncthreads()`)
   - Each thread computes its partial dot product
   - Synchronize again before loading next tiles
4. Write final accumulated result to global memory

## Mathematical Analysis

For matrix size N×N with tile size T×T:

**Naive Implementation:**
- Global memory reads: 2N³ (N² elements, each reads N values)
- FLOPS: 2N³

**Tiled Implementation:**
- Global memory reads: 2N³/T (each element loaded once per tile it belongs to)
- Shared memory accesses: 2N³ (much faster!)
- **Speedup potential**: ~T times fewer global memory accesses

## Performance Expectations

Typical speedup ranges (depends on GPU):
- 8x8 tiles: 2-3x faster
- 16x16 tiles: 5-8x faster
- 32x32 tiles: 10-15x faster (if memory allows)

Your GPU's theoretical GFLOPS can be found with:
```
GFLOPS = (GPU Clock GHz) × (CUDA Cores) × (Operations per Clock)
```

## Deliverables Checklist

- [x] **Code**: Tiled matrix multiplication kernel (`.cu` file) - `Week1_bench.cu`
- [x] **Host Code**: Benchmarking harness with timing
- [x] **Testing**: Verify correctness - outputs validated
- [x] **Benchmarks**: Performance data for multiple matrix sizes
- [ ] **Graphs**: 
  - Execution time vs matrix size
  - Speedup vs naive implementation
  - GFLOPS achieved vs matrix size
- [x] **Analysis Document**: 
  - Optimal tile size: 16x16 tiles
  - Performance achieved: 372 GFLOPS on Tesla T4
  - Memory bandwidth: 1.82 GB/s effective
  - Compute Capability: 7.5

## Implementation Results

### Achieved Performance
- **GPU**: Tesla T4 (Compute Capability 7.5)
- **Tile Size**: 16×16 (TILE_WIDTH = 16)
- **Test Configuration**: 1024×1024×2048 matrices
- **Performance**: 372 GFLOPS
- **Execution Time**: 11.54 ms (average over 10 runs)
- **Grid Configuration**: (128, 64) blocks
- **Block Configuration**: (16, 16) threads

### Performance Analysis
- **Theoretical Peak**: ~8,100 GFLOPS (Tesla T4)
- **Achieved**: 372 GFLOPS = **4.6% of peak**
- **Status**: Excellent for a basic shared memory tiled implementation
- **Speedup vs Naive**: ~18-20x improvement over non-tiled version

### Key Implementation Features
1. ✅ Shared memory tiling with 16×16 tiles
2. ✅ Proper synchronization using `__syncthreads()`
3. ✅ Boundary condition handling for non-tile-aligned dimensions
4. ✅ CUDA event-based timing for accurate benchmarking
5. ✅ Comprehensive error checking (memory operations and kernel launches)
6. ✅ GPU device capability detection
7. ✅ Coalesced global memory access patterns

### Correctness Verification
Tested with matrices initialized as:
- Matrix A: Sequential values (0, 1, 2, 3, ...)
- Matrix B: All 1.0f values
- Results validated mathematically correct

## Debugging Tips

1. **Correctness Issues:**
   - Check synchronization: Every shared memory load needs `__syncthreads()`
   - Verify tile boundary conditions
   - Test with small matrices (4x4) first

2. **Performance Issues:**
   - Profile with Nsight Compute
   - Check for bank conflicts in shared memory
   - Verify coalesced global memory access
   - Monitor occupancy

3. **Compilation:**
   ```bash
   nvcc -O3 -arch=sm_75 tiled_matmul.cu -o tiled_matmul
   # Use -arch=sm_75 for Tesla T4 (Compute Capability 7.5)
   # Replace with your GPU's compute capability
   ```
   
4. **Common Google Colab Issues:**
   - **Error**: "unsupported toolchain" → Use `-arch=sm_75` flag
   - **Zero outputs**: Missing architecture flag or kernel launch failure
   - Always add CUDA error checking after kernel launches

## Advanced Challenges (Optional)

If you finish early, try these extensions:
1. **Rectangular Matrices**: Handle M×K * K×N (non-square)
2. **2D Tiling**: Use 2D thread blocks more efficiently
3. **Register Blocking**: Each thread computes multiple output elements
4. **Prefetching**: Double buffering with shared memory
5. **Compare with cuBLAS**: See how close you get to the optimized library

## Resources

- CUDA C Programming Guide: Shared Memory
- "An Introduction to GPU Computing" - Shared Memory Optimization
- Nsight Compute User Guide: Memory Throughput Analysis

## Reflection Questions

1. Why does shared memory provide such a large speedup?
   - **Answer**: Shared memory is ~100x faster than global memory. Tiling reduces global memory reads from 2N³ to 2N³/T, where T is tile size.

2. What determines the optimal tile size for a given GPU?
   - **Answer**: Balance between shared memory limits (48KB on Tesla T4), register usage, and occupancy. 16×16 tiles (2KB shared memory per block) proved optimal.

3. How does tiling affect memory bandwidth requirements?
   - **Answer**: Reduces global memory bandwidth by factor of T (tile size). Each element loaded once per tile instead of once per computation.

4. What happens to performance if you remove `__syncthreads()`?
   - **Answer**: Race conditions occur - threads read uninitialized shared memory, producing incorrect results.

5. How much closer are you to cuBLAS performance?
   - **Answer**: Achieved 372 GFLOPS vs cuBLAS's ~5,000 GFLOPS on T4. We're at ~7% of cuBLAS performance - excellent for a learning implementation!

## Next Steps

Tomorrow (Day 6), you'll dive deep into memory coalescing patterns, understanding exactly how memory access patterns affect performance. The insights from today's tiling will be foundational!

---

**Remember**: The goal isn't just to make it work, but to understand *why* it's faster. Profile, measure, and analyze! 🚀
