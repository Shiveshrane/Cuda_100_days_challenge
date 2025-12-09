# Day 8: Shared Memory Bank Conflicts - Detection

**Chapter 6: Shared Memory**

## Overview
Shared memory is fast—but only if you avoid bank conflicts! Today you'll learn what bank conflicts are, how to detect them, and how they impact performance. Tomorrow, you'll learn to fix them.

## What Are Bank Conflicts?

Shared memory is divided into **banks** (32 on modern GPUs). Multiple threads can access shared memory simultaneously IF they access different banks.

### Bank Structure:
- 32 banks (numbered 0-31)
- 4-byte word per bank
- Successive 4-byte words map to successive banks
- Address → Bank mapping: `bank = (address / 4) % 32`

### Access Scenarios:

**No Conflict** ✅
- 32 threads access 32 different banks
- All accesses happen in parallel
- Maximum bandwidth

**2-Way Bank Conflict** ⚠️
- 2 threads access the same bank (different addresses)
- Access serialized into 2 transactions
- 50% of maximum bandwidth

**32-Way Bank Conflict** ❌
- All 32 threads access the same bank
- Serialized into 32 transactions
- ~3% of maximum bandwidth (slowest case!)

**Broadcast** ✅ (Special Case)
- All threads read the SAME address in the same bank
- Single broadcast transaction (no conflict!)
- Full bandwidth

## Concepts Covered

### 1. **Bank Conflict Mechanics**
- How shared memory is organized into banks
- Bank address calculation
- Conflict serialization behavior
- Broadcast exception

### 2. **Common Conflict Patterns**
- Strided access (stride == multiple of 32)
- Diagonal patterns
- Non-power-of-2 dimensions

### 3. **Detection Methods**
- Manual calculation
- Nsight Compute metrics
- Visual analysis

## Learning Objectives

By the end of today, you should be able to:
- [ ] Calculate which bank an address maps to
- [ ] Identify bank conflicts by analyzing access patterns
- [ ] Implement kernels with intentional conflicts
- [ ] Use Nsight Compute to detect and quantify conflicts
- [ ] Understand conflict severity (2-way, 4-way, etc.)
- [ ] Read shared memory conflict metrics
- [ ] Create visual diagrams of access patterns

## Implementation Tasks

### Task 1: No Conflict Baseline
```cuda
__global__ void noConflictKernel(float *output) {
    __shared__ float shared[1024];
    int tid = threadIdx.x;
    
    // Sequential access - each thread accesses different bank
    shared[tid] = tid;
    __syncthreads();
    
    output[tid] = shared[tid];
}
```

Bank mapping for thread tid:
- Thread 0 → `shared[0]` → Bank 0
- Thread 1 → `shared[1]` → Bank 1
- Thread 31 → `shared[31]` → Bank 31
- Thread 32 → `shared[32]` → Bank 0
- ...

✅ **No conflicts!**

### Task 2: Stride-Based Conflicts
```cuda
__global__ void strideConflictKernel(float *output, int stride) {
    __shared__ float shared[1024];
    int tid = threadIdx.x;
    
    // Strided access
    shared[tid * stride] = tid;
    __syncthreads();
    
    output[tid] = shared[tid * stride];
}
```

Test with strides: 1, 2, 4, 8, 16, 32

**Analyze stride = 32:**
- Thread 0 → `shared[0]` → Bank 0
- Thread 1 → `shared[32]` → Bank 0
- Thread 2 → `shared[64]` → Bank 0
- ...
- All 32 threads → Bank 0!

❌ **32-way bank conflict!**

### Task 3: Matrix Transpose Conflicts
```cuda
#define TILE_SIZE 32

__global__ void transposeConflicts(float *in, float *out, int n) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Read with no conflicts
    tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    __syncthreads();
    
    // Write transposed - CONFLICTS HERE!
    // All threads in a warp access the same column
    // Column access means same bank for entire warp
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    out[y * n + x] = tile[threadIdx.x][threadIdx.y];  // ❌ Conflicts!
}
```

When thread 0-31 read `tile[threadIdx.x][0]`:
- Thread 0 reads `tile[0][0]` → Address 0 → Bank 0
- Thread 1 reads `tile[1][0]` → Address 32 → Bank 0
- Thread 2 reads `tile[2][0]` → Address 64 → Bank 0
- ...

❌ **32-way bank conflict!**

### Task 4: Broadcast (No Conflict)
```cuda
__global__ void broadcastKernel(float *output) {
    __shared__ float shared[32];
    int tid = threadIdx.x;
    
    if (tid == 0) {
        shared[0] = 3.14f;
    }
    __syncthreads();
    
    // All threads read the same address - broadcast!
    output[tid] = shared[0];  // ✅ No conflict, single broadcast
}
```

### Task 5: Diagonal Pattern
```cuda
__global__ void diagonalKernel(float *output) {
    __shared__ float shared[32][32];
    int tid = threadIdx.x;
    
    // Diagonal access
    shared[tid][tid] = tid;
    __syncthreads();
    
    output[tid] = shared[tid][tid];
}
```

Analyze: Does this have conflicts?

## Bank Conflict Calculation

**General Formula:**
```
Address = base_address + (row * row_size + col) * sizeof(element)
Bank = (Address / 4) % 32
```

For 2D array `shared[ROW][COL]`:
```
Bank = ((row * COL + col) * sizeof(element) / 4) % 32
```

**Example:** `shared[32][32]` of floats (4 bytes)
- `shared[0][0]` → Bank 0
- `shared[0][1]` → Bank 1
- `shared[1][0]` → Bank 8 (if COL=32: 1*32/1 % 32 = 0... wait, 32%32 = 0)

Actually: `shared[1][0]` → Bank 0 (conflict with `shared[0][0]`!)

## Profiling with Nsight Compute

### Basic Command:
```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./kernel
```

### Key Metrics:
1. **`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld`**
   - Number of shared memory load bank conflicts
   
2. **`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st`**
   - Number of shared memory store bank conflicts

3. **`l1tex__shared_bank_conflicts_per_inst`**
   - Average conflicts per shared memory instruction

4. **`smsp__sass_average_data_bytes_per_wavefront_mem_shared`**
   - Bytes per wavefront (shows serialization)

### Interpretation:
- **0 conflicts**: Perfect! ✅
- **< 1.0 average**: Minor conflicts ⚠️
- **> 2.0 average**: Significant performance hit ❌
- **~ 31.0 average**: Worst case (32-way conflicts) 🔥

## Visual Analysis

Create diagrams showing:
1. **Thread to Bank mapping** for your access pattern
2. **Conflict matrix**: Which threads access which banks
3. **Serialization visualization**: Show how accesses split into waves

Example table:
```
Thread | Address   | Bank | Conflicts With
-------|-----------|------|---------------
0      | shared[0] | 0    | -
1      | shared[32]| 0    | Thread 0
2      | shared[64]| 0    | Threads 0,1
...
```

## Deliverables Checklist

- [ ] **Code**: 5 kernel variants (no conflict, stride, transpose, broadcast, diagonal)
- [ ] **Timing**: Performance measurements for each
- [ ] **Nsight Reports**: Conflict metrics for all kernels
- [ ] **Calculations**: Manual bank conflict analysis for each pattern
- [ ] **Diagrams**: Visual representation of access patterns
- [ ] **Analysis Document**:
  - Explanation of bank conflict mechanism
  - Why different patterns have different conflicts
  - Performance impact correlation with conflict count
- [ ] **Comparison Table**: Conflicts vs execution time

## Expected Results

| Kernel      | Bank Conflicts | Relative Time |
|-------------|----------------|---------------|
| Sequential  | 0              | 1.0x          |
| Stride 2    | 0              | 1.0x          |
| Stride 32   | ~31 per access | 15-30x        |
| Transpose   | ~31 per access | 10-20x        |
| Broadcast   | 0              | 1.0x          |
| Diagonal    | Varies         | Varies        |

## Debugging Tips

1. **Start small**: 32 threads, easy to trace
2. **Print bank IDs**: Add printf in kernel (debugging builds)
3. **Use pencil and paper**: Draw the access pattern
4. **Check alignment**: Misaligned arrays can hide conflicts
5. **Profile single kernel**: Isolate the problematic operation

## Common Conflict Patterns to Memorize

❌ **Stride = 32 (or multiples)**: 32-way conflicts
❌ **Column-major access of [32][32] array**: 32-way conflicts
❌ **Access with offset that's multiple of 32**: N-way conflicts
✅ **Sequential access**: No conflicts
✅ **Same address read by all threads**: No conflict (broadcast)
✅ **Randomized access**: Unlikely conflicts (but slow for other reasons)

## Reflection Questions

1. Why does stride = 2 have no conflicts but stride = 32 does?
2. What's special about 32 banks and 32 threads per warp?
3. How do you calculate expected conflicts for a given pattern?
4. Why is broadcast not considered a conflict?
5. What's the relationship between conflicts and execution time?

## Advanced Analysis (Optional)

1. **Sub-warp analysis**: What if block size < 32?
2. **Multiple warps**: Do conflicts happen across warps?
3. **Different data types**: How do `double` (8 bytes) affect banks?
4. **Compute capability differences**: Bank count across architectures

## Next Steps

Tomorrow (Day 9), you'll learn the techniques to **fix** these conflicts! Padding, indexing tricks, and layout changes await.

---

**Remember**: Detection is the first step to optimization. Profile everything! 📊
