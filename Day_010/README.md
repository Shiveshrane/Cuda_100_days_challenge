# Day 10: Thread Block Size & Occupancy

**Chapter 6: Occupancy**

## Overview
Choosing the right thread block size is both art and science. Today you'll learn about GPU occupancy, how block size affects it, and how to find optimal configurations for your kernels.

## What is Occupancy?

**Occupancy** = Ratio of active warps to maximum possible warps on an SM

```
Occupancy = (Active Warps per SM) / (Maximum Warps per SM)
```

### Why It Matters:
- Higher occupancy → Better latency hiding
- More warps → More opportunities to hide memory latency
- **BUT**: Higher occupancy ≠ always faster!
- Goal: **Sufficient** occupancy, not maximum

### The Balance:
More threads per block → More shared memory/registers per block → Fewer blocks per SM → Lower occupancy

## Concepts Covered

### 1. **Occupancy Fundamentals**
- Active warps vs resident warps
- SM resource limits
- Latency hiding mechanism

### 2. **Limiting Factors**
- Block size
- Registers per thread
- Shared memory per block
- Maximum blocks per SM (hardware limit)

### 3. **Occupancy Calculation**
- Manual calculation
- CUDA Occupancy Calculator (spreadsheet)
- Runtime API: `cudaOccupancyMaxActiveBlocksPerMultiprocessor()`

### 4. **Configuration Optimization**
- Trade-offs between block size and occupancy
- When occupancy doesn't matter
- Architecture-specific considerations

## Learning Objectives

By the end of today, you should be able to:
- [ ] Define and calculate occupancy
- [ ] Identify limiting factors for a kernel
- [ ] Use CUDA Occupancy Calculator
- [ ] Implement kernels with variable block sizes
- [ ] Measure actual occupancy with profiling
- [ ] Find optimal block size for your kernels
- [ ] Explain when maximum occupancy isn't necessary

## GPU Resource Limits (Example: RTX 3080)

### Per SM:
- **Maximum Warps**: 48 (1536 threads)
- **Maximum Blocks**: 16
- **Registers**: 65,536
- **Shared Memory**: 48 KB (configurable: 48KB shared/16KB L1 or 32KB shared/32KB L1)

### Per Block Limits:
- **Maximum Threads**: 1024
- **Maximum Registers per Thread**: 255
- **Maximum Shared Memory**: 48 KB

*Note: Check your GPU specs—limits vary by architecture!*

## Occupancy Calculation Examples

### Example 1: Basic Kernel

```cuda
__global__ void simpleKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}
```

Assume:
- Block size: 256 threads (8 warps)
- Registers per thread: 16
- Shared memory: 0 bytes

**Calculate occupancy:**

1. **Warps per block**: 256 / 32 = 8 warps
2. **Max blocks limited by threads**: 1536 / 256 = 6 blocks per SM
3. **Max blocks limited by warps**: 48 / 8 = 6 blocks per SM
4. **Max blocks limited by registers**: 65536 / (256 * 16) = 16 blocks per SM
5. **Max blocks limited by shared mem**: No limit (using 0)
6. **Hardware limit**: 16 blocks per SM

**Limiting factor**: Threads (6 blocks)

**Actual occupancy**: (6 blocks * 8 warps) / 48 max warps = **48/48 = 100%** ✅

### Example 2: Memory-Heavy Kernel

```cuda
__global__ void memoryKernel(float *data, int n) {
    __shared__ float shared[1024];  // 4 KB
    int tid = threadIdx.x;
    // ... uses 40 registers per thread
}
```

Assume:
- Block size: 256 threads
- Registers per thread: 40
- Shared memory: 4 KB per block

**Calculate:**

1. **Max blocks by threads**: 1536 / 256 = 6
2. **Max blocks by warps**: 48 / 8 = 6
3. **Max blocks by registers**: 65536 / (256 * 40) = 6.4 → **6 blocks**
4. **Max blocks by shared mem**: 49152 / 4096 = 12 blocks
5. **Hardware limit**: 16

**Limiting factor**: Registers or threads (both give 6)

**Occupancy**: 48/48 = **100%** ✅

### Example 3: Low Occupancy

```cuda
__global__ void heavyKernel(float *data) {
    __shared__ float shared[2048];  // 8 KB
    int tid = threadIdx.x;
    // ... uses 64 registers per thread
}
```

Assume:
- Block size: 512 threads
- Registers: 64 per thread
- Shared memory: 8 KB

**Calculate:**

1. **Max blocks by threads**: 1536 / 512 = 3
2. **Max blocks by warps**: 48 / 16 = 3
3. **Max blocks by registers**: 65536 / (512 * 64) = 2 → **Limiting!**
4. **Max blocks by shared mem**: 49152 / 8192 = 6

**Limiting factor**: Registers (2 blocks)

**Occupancy**: (2 * 16) / 48 = **67%** ⚠️

## Implementation Tasks

### Task 1: Variable Block Size Kernel

```cuda
template<int BLOCK_SIZE>
__global__ void variableBlockKernel(float *in, float *out, int n) {
    __shared__ float shared[BLOCK_SIZE];
    
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < n) {
        shared[threadIdx.x] = in[idx];
    }
    __syncthreads();
    
    // Some computation...
    if (idx < n) {
        out[idx] = shared[threadIdx.x] * 2.0f;
    }
}
```

Test with block sizes: 64, 128, 256, 512, 1024

### Task 2: Measure Runtime Occupancy

```cuda
int numBlocks;
int blockSize = 256;

cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocks,
    variableBlockKernel<256>,
    blockSize,
    0  // dynamic shared memory size
);

int device;
cudaDeviceProp props;
cudaGetDevice(&device);
cudaGetDeviceProperties(&props, device);

float occupancy = (numBlocks * blockSize / 32) / (float)props.maxThreadsPerMultiProcessor * 32;
printf("Theoretical occupancy: %.2f%%\n", occupancy * 100);
```

### Task 3: Find Optimal Block Size Automatically

```cuda
int minGridSize;
int blockSize;

cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,
    &blockSize,
    variableBlockKernel<256>,
    0,  // dynamic shared memory
    0   // block size limit (0 = no limit)
);

printf("Suggested block size: %d\n", blockSize);
printf("Minimum grid size for max occupancy: %d\n", minGridSize);
```

### Task 4: Benchmark Different Configurations

```cuda
void benchmarkOccupancy(int n) {
    int blockSizes[] = {64, 128, 256, 512, 1024};
    
    for (int i = 0; i < 5; i++) {
        int blockSize = blockSizes[i];
        int gridSize = (n + blockSize - 1) / blockSize;
        
        // Time kernel
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        variableBlockKernel<blockSize><<<gridSize, blockSize>>>(d_in, d_out, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        
        // Calculate theoretical occupancy
        int numBlocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, 
            variableBlockKernel<blockSize>, blockSize, 0);
        
        printf("Block size: %4d, Time: %.3f ms, Blocks/SM: %d\n", 
               blockSize, ms, numBlocks);
    }
}
```

## Using CUDA Occupancy Calculator

NVIDIA provides an Excel spreadsheet:
1. Download from CUDA Toolkit documentation
2. Input your GPU architecture (Compute Capability)
3. Enter: threads per block, registers per thread, shared memory per block
4. Spreadsheet calculates occupancy and limiting factors

**Alternative**: Online calculator at developer.nvidia.com

## Profiling Occupancy with Nsight Compute

```bash
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./kernel

# Key metrics:
# - sm__warps_active.avg.pct_of_peak_sustained_active : Achieved occupancy
# - sm__maximum_warps_per_active_cycle : Theoretical maximum
# - launch__occupancy_limit_* : What's limiting occupancy
```

### Interpreting Results:
- **Occupancy < 50%**: Investigate limiting factors
- **Occupancy 50-75%**: Probably sufficient for memory-bound kernels
- **Occupancy > 75%**: Excellent, but diminishing returns

## When Occupancy Doesn't Matter

### High occupancy NOT critical when:
1. **Compute-bound kernels**: Fully utilizing ALUs regardless
2. **Memory-intensive with enough warps**: Even 50% occupancy can hide latency
3. **Small data sizes**: Occupancy optimization overkill
4. **Synchronization-heavy**: Frequent `__syncthreads()` limits benefit

### Better to optimize:
- **Memory coalescing** (bigger impact!)
- **Shared memory usage** (reduce conflicts)
- **Algorithm efficiency** (reduce work)

## Deliverables Checklist

- [ ] **Code**: Kernel with variable block size template
- [ ] **Occupancy Calculator**: Runtime computation for different configs
- [ ] **Benchmarks**: Performance vs block size graphs
- [ ] **Profiling**: Nsight Compute occupancy reports
- [ ] **Analysis Document**:
  - Occupancy calculations for 3+ kernel configs
  - Identify limiting factors for each
  - Explain why certain block sizes perform better
  - Discuss when occupancy is vs isn't critical
- [ ] **Optimization Guide**: Recommendations for your GPU

## Expected Insights

### Typical Findings:
1. **Block size 256-512**: Often optimal balance
2. **Power of 2**: Usually best (warp alignment)
3. **Very large blocks (1024)**: May reduce occupancy due to resource limits
4. **Very small blocks (64)**: May not fully utilize SM

### Performance vs Occupancy:
- 100% occupancy ≠ fastest kernel
- Often 50-75% occupancy is sufficient
- Diminishing returns beyond certain point

## Common Pitfalls

1. **Chasing 100% occupancy**: Sacrificing other optimizations
2. **Ignoring architecture**: Block size optimal on one GPU may not be on another
3. **Not profiling**: Theoretical vs actual occupancy can differ
4. **Over-tuning**: Occupancy last resort, not first optimization

## Register Pressure Reduction Techniques

If registers are limiting occupancy:

```cuda
// 1. Use __launch_bounds__ to hint compiler
__global__ void __launch_bounds__(256, 4)  // 256 threads, 4 blocks/SM minimum
myKernel(float *data) {
    // Compiler will try to use fewer registers
}

// 2. Reduce local variables
// Before:
float temp1 = a + b;
float temp2 = c + d;
float temp3 = temp1 * temp2;

// After (reuse variables):
float temp = a + b;
temp = temp * (c + d);

// 3. Move computation to shared memory
// Instead of register spilling, use shared memory explicitly
```

## Reflection Questions

1. What's your GPU's theoretical maximum occupancy?
2. Which resource limit affects your kernels most?
3. Did maximum occupancy give best performance?
4. How does block size affect your specific kernel?
5. When would you prioritize occupancy over other optimizations?

## Advanced Topics (Optional)

1. **Dynamic parallelism**: How child kernels affect occupancy
2. **Multi-GPU**: Per-GPU occupancy tuning
3. **Cooperative groups**: Impact on occupancy calculations
4. **Persistent threads**: Intentionally low occupancy strategy

## Next Steps

Tomorrow (Day 11), we'll explore register pressure in more depth and learn about local memory spilling—critical for understanding why some kernels have low occupancy!

---

**Key Insight**: Occupancy is about giving the GPU enough work to hide latency. "Enough" is often less than "maximum"! 📊
