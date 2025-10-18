# Day 003: Matrix Multiplication in CUDA

## Overview
This project implements matrix multiplication in CUDA using two different approaches:
1. **Basic Matrix Multiplication** (without tiling)
2. **Tiled Matrix Multiplication** (with shared memory optimization)

## Problem Statement
Given two matrices A and B of size N×N, compute the product matrix C = A × B where:
```
C[i][j] = Σ(k=0 to N-1) A[i][k] × B[k][j]
```

## Implementation Details

### Matrix Dimensions
- Matrix size: **256 × 256**
- Tile size: **16 × 16** (defined by `TILE_WIDTH`)
- Grid dimensions: `(16, 16, 1)` blocks
- Block dimensions: `(16, 16, 1)` threads per block

### Input Data
- Matrix A: All elements initialized to `1.0f`
- Matrix B: All elements initialized to `2.0f`
- Expected result: Matrix C with all elements = `512.0f` (256 × 1.0 × 2.0)

---

## Kernel Implementations

### 1. `matrix_mult_without_tiling` (Basic Approach)

```cuda
__global__ void matrix_mult_without_tiling(const float *A, const float *B, float *C, int N)
```

**How it works:**
- Each thread computes one element of the output matrix C
- Thread position calculated using `blockDim`, `blockIdx`, and `threadIdx`
- Direct access to global memory for every operation
- No optimization for memory access patterns

**Memory Access Pattern:**
- **Matrix A**: Reads elements in row-major order (coalesced access)
- **Matrix B**: Reads elements in column-major order (strided/non-coalesced access)
- All memory accesses go directly to **global memory** (slow)

**Performance Characteristics:**
- ❌ High global memory latency (100-400 cycles per access)
- ❌ Poor cache utilization
- ❌ Non-coalesced reads from matrix B
- ✅ Simple implementation
- ✅ Easy to understand

**Time Complexity:** O(N) per thread for computing one element

---

### 2. `matrix_mult_with_tiling` (Current Implementation)

```cuda
__global__ void matrix_mult_with_tiling(const float *A, const float *B, float *C, int N)
```

**How it works:**
- Similar to the basic approach (currently)
- Thread position calculated using `TILE_WIDTH` constant
- Each thread still computes one element of output matrix C
- Direct global memory access (no shared memory yet)

**Current State:**
⚠️ **Note:** Despite the name "with_tiling", this kernel **does NOT currently use shared memory or tiling optimization**. It's structurally similar to the basic kernel but uses a different indexing scheme based on `TILE_WIDTH`.

**Actual Implementation:**
```cuda
int row_idx = blockIdx.x * TILE_WIDTH + threadIdx.x;
int col_idx = blockIdx.y * TILE_WIDTH + threadIdx.y;
```

This is equivalent to the basic kernel but with hardcoded tile width instead of using `blockDim`.

---

## Key Differences Between Kernels

| Feature | `matrix_mult_without_tiling` | `matrix_mult_with_tiling` |
|---------|------------------------------|---------------------------|
| **Indexing** | Uses `blockDim.x/y` (dynamic) | Uses `TILE_WIDTH` (compile-time constant) |
| **Memory Access** | Direct global memory | Direct global memory (no optimization yet) |
| **Shared Memory** | ❌ Not used | ❌ Not used (despite the name) |
| **Performance** | Baseline | Same as baseline |
| **Flexibility** | Can use different block sizes | Fixed to 16×16 tiles |

---

## What True Tiling Should Look Like

A proper tiled implementation would:

1. **Use Shared Memory:**
   ```cuda
   __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
   __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];
   ```

2. **Load Data in Tiles:**
   - Load a tile of A and B into shared memory
   - Synchronize threads (`__syncthreads()`)
   - Perform computations using fast shared memory
   - Repeat for all tiles

3. **Benefits:**
   - Reduces global memory accesses by factor of `TILE_WIDTH`
   - Exploits data reuse (each element used multiple times)
   - Much faster execution (10-20x speedup possible)

---

## Memory Management

### Host (CPU) Memory:
```cpp
float *A = new float[N*N];  // Input matrix A
float *B = new float[N*N];  // Input matrix B
float *C = new float[N*N];  // Output matrix C
```

### Device (GPU) Memory:
```cpp
cudaMalloc(&d_A, sizeof(float)*N*N);  // Device matrix A
cudaMalloc(&d_B, sizeof(float)*N*N);  // Device matrix B
cudaMalloc(&d_C, sizeof(float)*N*N);  // Device matrix C (REQUIRED!)
```

**⚠️ Common Bug:** Forgetting to allocate `d_C` causes segmentation fault!

---

## Compilation and Execution

### Compile:
```bash
nvcc matrix_mult.cu -o matrix_mult
```

### Run:
```bash
./matrix_mult
```

### Expected Output:
```
512 512 512 ... (256×256 matrix of 512.0)
```

---

## Performance Considerations

### Current Implementation:
- **Memory Bandwidth Limited:** Bottlenecked by global memory access
- **No Data Reuse:** Each element of A and B read multiple times from slow global memory
- **Arithmetic Intensity:** Low (more memory operations than compute)

### Potential Optimizations:
1. ✅ Implement true shared memory tiling
2. ✅ Use texture memory for read-only data
3. ✅ Optimize memory access patterns
4. ✅ Use cuBLAS library for production code

---

## Key Takeaways

1. **Memory Hierarchy Matters:** Global memory is 100x slower than shared memory
2. **Naming is Important:** The "with_tiling" kernel doesn't actually use tiling yet
3. **Index Calculation:** Both kernels achieve the same thread-to-element mapping
4. **Memory Allocation:** Always allocate device memory for all arrays (including output!)
5. **Next Steps:** Implement true shared memory tiling for significant performance gains

---

## References
- CUDA C Programming Guide: [Matrix Multiplication Example](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- TILE_WIDTH = 16 is chosen to balance:
  - Shared memory usage (16×16×4 bytes = 1KB per tile × 2 matrices = 2KB)
  - Thread occupancy (256 threads per block)
  - GPU warp size (32 threads)

---

## Future Improvements
- [ ] Implement actual shared memory tiling
- [ ] Add error checking for CUDA calls
- [ ] Benchmark and compare performance
- [ ] Support non-square matrices
- [ ] Handle matrices not divisible by TILE_WIDTH
- [ ] Add timing measurements using CUDA events
