# Day 6: Memory Coalescing - Theory & Basic Patterns

**Chapter 6: Performance Considerations**

## Overview
Memory coalescing is one of the most critical concepts for achieving high performance in CUDA. Today, you'll learn how the GPU's memory system works and how thread access patterns dramatically affect bandwidth utilization.

## What is Memory Coalescing?

**Memory Coalescing** occurs when consecutive threads in a warp access consecutive memory locations. The GPU can combine these accesses into fewer, larger transactions—dramatically improving bandwidth.

### Key Facts:
- Threads in a warp execute in lockstep (32 threads)
- Global memory is accessed in 32-byte, 64-byte, or 128-byte transactions
- Coalesced access: 32 threads access 128 consecutive bytes → 1 transaction
- Uncoalesced access: 32 threads access scattered locations → up to 32 transactions
- **Performance impact**: 10-30x difference!

## Concepts Covered

### 1. **Warp-Level Memory Transactions**
- How warps issue memory requests
- Memory transaction sizes and alignment
- Cache behavior (L1/L2)

### 2. **Access Patterns**
- **Sequential**: Perfect coalescing (thread i accesses element i)
- **Strided**: Threads skip elements (thread i accesses element i*stride)
- **Reversed**: Backward sequential access
- **Random**: No pattern, worst case

### 3. **Memory Alignment**
- Base address alignment requirements
- How misalignment affects transactions
- Structure padding considerations

## Learning Objectives

By the end of today, you should be able to:
- [ ] Explain memory coalescing and why it matters
- [ ] Implement vector addition with sequential access
- [ ] Implement vector addition with strided access
- [ ] Implement vector addition with reversed access
- [ ] Measure and compare bandwidth for each pattern
- [ ] Use CUDA events for accurate timing
- [ ] Calculate effective bandwidth utilization

## Implementation Tasks

### Task 1: Sequential Access (Baseline) ✅
```cuda
__global__ void mem_seq_kernel(const int *Input, int *Output, int N){
    __shared__ int shared_data[SHARED_MEM_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    shared_data[tid] = Input[i];  // Perfect coalescing - sequential access
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int j=1; j<blockDim.x; j*=2){
        shared_data[tid] += shared_data[tid + j];
        __syncthreads();
    }
    
    if (tid==0){
        Output[blockIdx.x] = shared_data[0];
    }
}
```

### Task 2: Strided Access ✅
```cuda
__global__ void mem_strided_kernel(const int *Input, int *Output, int N, int stride){
    __shared__ int shared_data[SHARED_MEM_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    
    shared_data[tid] = Input[i*stride];  // Non-coalesced - strided access
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int j=1; j<blockDim.x; j*=2){
        shared_data[tid] += shared_data[tid + j];
        __syncthreads();
    }
    
    if (tid==0){
        Output[blockIdx.x] = shared_data[0];
    }
}
```
Test with strides: 1, 2, 4, 8, 16, 32

### Task 3: Reversed Access ✅
```cuda
__global__ void mem_rev_kernel(const int *Input, int *Output, int N){
    __shared__ int shared_data[SHARED_MEM_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    
    int rev_idx = BLOCK_SIZE - tid - 1;  // Reverse within block
    
    shared_data[tid] = Input[blockIdx.x * blockDim.x + rev_idx];  // Still coalesced!
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int j=1; j<blockDim.x; j*=2){
        shared_data[tid] += shared_data[tid + j];
        __syncthreads();
    }
    
    if (tid==0){
        Output[blockIdx.x] = shared_data[0];
    }
}
```

### Task 4: Timing Infrastructure
Use CUDA events for precise timing:
```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>(args);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
```

## Mathematical Analysis

**Theoretical Bandwidth** (from GPU specs):
- Example: RTX 3080 = 760 GB/s
- Your GPU bandwidth: Check `nvidia-smi` or specs

**Effective Bandwidth Calculation:**
```
Effective BW (GB/s) = (Bytes Read + Bytes Written) / Time(s) / 1e9

For vector addition of N floats:
Bytes = 3 * N * sizeof(float)  // Read A, Read B, Write C
```

**Bandwidth Efficiency:**
```
Efficiency = (Effective BW / Theoretical BW) * 100%
```

## Expected Results

### Sequential Access:
- Time: ~0.028 ms (for 1024 elements with 32 threads/block)
- Perfect coalescing - all threads access consecutive memory
- Best-case scenario

### Strided Access:
- Time: ~0.134 ms (for stride=32)
- **~5-6x slower** than sequential
- Non-coalesced access creates memory transaction overhead
- Performance degrades as stride increases

### Reversed Access:
- Time: ~0.021 ms (similar to sequential)
- Should match sequential! (Surprise!)
- GPU can still coalesce backward sequential access
- Demonstrates that **direction doesn't break coalescing**

## Actual Benchmark Results

**Test Configuration:**
- Array size: 1024 elements
- Block size: 32 threads
- Stride (for strided kernel): 32

**Results:**
```
Strided Access Time:    0.134048 ms  (Non-coalesced)
Sequential Access Time: 0.027552 ms  (Coalesced)
Reversed Access Time:   0.020864 ms  (Coalesced)
```

**Key Takeaway:** Strided access is **~5-6x slower** than coalesced patterns, demonstrating the critical importance of memory access patterns in CUDA performance!

## Deliverables Checklist

- [x] **Code**: Three reduction kernels (sequential, strided, reversed)
- [x] **Timing**: Accurate measurements using CUDA events
- [x] **Testing**: Verified correctness for all kernels
- [x] **Benchmarks**: 
  - Test array size: 1024 elements
  - Test stride: 32 (worst case)
- [x] **Results**: Execution times demonstrating 5-6x performance difference
- [x] **Analysis**: Sequential and reversed show similar performance (both coalesced), strided shows significant degradation

## Debugging & Profiling

### Using Nsight Compute:
```bash
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum ./vector_add
```

Look for:
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` (L1 load transactions)
- `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` (L1 store transactions)
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld` (bytes per transaction)

### Compilation:
```bash
nvcc -O3 -arch=sm_XX Mem_Coalescing.cu -o Mem_Coalescing
```

### Run:
```bash
./Mem_Coalescing
```

**Expected Output:**
```
Strided Access Time: 0.134048 ms
Sequential Access Time: 0.027552 ms
Reversed Access Time: 0.020864 ms
```

## Key Insights to Document

1. **Why does stride matter so much?**
   - When stride=32, thread 0 accesses `Input[0]`, thread 1 accesses `Input[32]`, etc.
   - Creates 32-element gaps between consecutive thread accesses
   - GPU cannot combine these into single memory transaction
   - Result: 5-6x performance penalty observed in benchmarks

2. **Why doesn't reversed access hurt performance?**
   - GPU memory controller is smart enough to recognize sequential patterns
   - Can coalesce sequential accesses in **either direction**
   - Thread 0 reads `Input[31]`, thread 1 reads `Input[30]`, etc. - still consecutive!
   - Observed: reversed (0.021ms) ≈ sequential (0.028ms)

3. **When is coalescing impossible?**
   - Random access patterns (no predictable sequence)
   - Strided access with large strides (gaps in memory)
   - Irregular data structures without careful layout
   - Strategies to mitigate: shared memory buffering, data restructuring (AoS → SoA)

## Advanced Experiments (Optional)

1. **Misaligned Arrays**: Start arrays at offset +1 byte
2. **Mixed Patterns**: Half threads sequential, half strided
3. **Different Data Types**: Compare float vs double vs int
4. **Cache Effects**: Vary array size to fit/exceed L2 cache

## Real-World Applications

Memory coalescing matters in:
- Image processing (pixel access patterns)
- Matrix operations (row vs column major)
- Particle simulations (gather/scatter operations)
- Neural networks (weight access patterns)

## Reflection Questions

1. What percentage of theoretical bandwidth did you achieve with sequential access?
   - **Answer**: With small test size (1024 elements), absolute times are small. Sequential and reversed both show excellent coalescing behavior.

2. At what stride does performance drop significantly?
   - **Answer**: At stride=32, we observed a 5-6x performance degradation compared to sequential access.

3. Why might production code have uncoalesced access?
   - **Answer**: Complex data structures, irregular access patterns, or algorithm requirements that prioritize other factors over memory coalescing.

4. How does block size affect coalescing?
   - **Answer**: Block size should be multiple of warp size (32) to maximize coalescing opportunities within each warp.

5. What's the relationship between coalescing and occupancy?
   - **Answer**: Good coalescing reduces memory latency, which can help hide latency even with lower occupancy. Both are important but address different bottlenecks.

## Next Steps

Tomorrow (Day 7), you'll tackle advanced coalescing patterns with matrix transpose and Array-of-Structures vs Structure-of-Arrays—both classic coalescing challenges!

---

**Pro Tip**: Always think "consecutive threads, consecutive memory" when designing kernels! 🎯
