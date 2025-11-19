# Day 4: Shared Memory Matrix Multiplication

## Overview
This program demonstrates **matrix multiplication using CUDA shared memory** optimization. Shared memory is a key optimization technique in CUDA programming that significantly improves performance by reducing global memory access latency.

## Concept Explanation

### What is Shared Memory?
- **Shared memory** is a special type of memory on the GPU that is shared among all threads within a block
- It's much faster than global memory (approximately 100x faster)
- Located on-chip, providing low-latency access
- Limited in size (typically 48-96 KB per SM)
- Acts as a user-managed cache

### Why Use Shared Memory for Matrix Multiplication?
In naive matrix multiplication, each thread reads the same elements from global memory multiple times:
- For an NxN matrix multiplication, each element is read N times from global memory
- This creates a memory bandwidth bottleneck

**Shared memory optimization solves this by:**
1. Loading data into shared memory once
2. Reusing it multiple times from the faster shared memory
3. Reducing global memory bandwidth requirements

### The Tiling Technique

This implementation uses **tiling** (also called blocking):

```
Matrix A (NxN)     Matrix B (NxN)     Matrix C (NxN)
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  [tile]     │   │  [tile]     │   │  [result]   │
│             │ × │             │ = │             │
│             │   │             │   │             │
└─────────────┘   └─────────────┘   └─────────────┘
```

- Matrices are divided into **TILE_WIDTH × TILE_WIDTH** tiles (16×16 in this code)
- Each tile is loaded into shared memory
- Threads compute partial results using shared memory
- Process repeats for all tiles along the K-dimension

## Algorithm Breakdown

### 1. Shared Memory Declaration
```cuda
__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
```
- Creates two 16×16 shared memory arrays
- One for tiles of matrix A, one for tiles of matrix B
- Shared among all threads in the block

### 2. Thread and Block Indexing
```cuda
int tx = threadIdx.x;  // Thread X index within block
int ty = threadIdx.y;  // Thread Y index within block
int bx = blockIdx.x;   // Block X index in grid
int by = blockIdx.y;   // Block Y index in grid
int row_idx = by * TILE_WIDTH + ty;  // Global row index
int col_idx = bx * TILE_WIDTH + tx;  // Global column index
```

### 3. Tiled Computation Loop
```cuda
for(int t = 0; t < N/TILE_WIDTH; t++)
```
- Iterates through all tiles needed to compute one output element
- Each iteration processes one tile from A and one tile from B

### 4. Loading Tiles into Shared Memory
```cuda
s_A[ty][tx] = A[row_idx*N + (t*TILE_WIDTH + tx)];
s_B[ty][tx] = B[(t*TILE_WIDTH + ty)*N + col_idx];
__syncthreads();
```
- Each thread loads one element from A and one from B into shared memory
- `__syncthreads()` ensures all threads have finished loading before computation

### 5. Computing Partial Results
```cuda
for (int k = 0; k < TILE_WIDTH; k++) {
    val += s_A[ty][k] * s_B[k][tx];
}
__syncthreads();
```
- Each thread computes its partial dot product using shared memory
- Second `__syncthreads()` ensures computation is complete before loading next tile

### 6. Writing Final Result
```cuda
C[row_idx*N + col_idx] = val;
```
- After processing all tiles, write the final result to global memory

## Key CUDA Concepts

### `__syncthreads()`
- **Critical synchronization barrier**
- Ensures all threads in a block reach this point before any continue
- Used here to:
  1. Ensure all data is loaded before computation
  2. Ensure computation is done before loading new data (prevents race conditions)

### Memory Hierarchy
```
Global Memory  ─────┐ (Slow, large, ~GB)
                    │
Shared Memory  ─────┤ (Fast, small, ~KB) ← We use this!
                    │
Registers      ─────┘ (Fastest, tiny, ~bytes)
```

### Thread Block Configuration
```cuda
dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);  // 16×16 = 256 threads per block
dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, 
             (N + TILE_WIDTH - 1) / TILE_WIDTH);
```
- Each block handles one 16×16 tile of the output matrix
- Grid size calculated to cover entire matrix

## Performance Benefits

### Memory Access Reduction
- **Without shared memory**: Each element accessed from global memory N times
- **With shared memory**: Each element loaded once per tile into shared memory
- **Speedup**: Can achieve 10-20x performance improvement

### Bandwidth Calculation
For N×N matrix multiplication:
- **Naive**: 2N³ global memory reads
- **Tiled**: 2N³/TILE_WIDTH global memory reads
- **Reduction**: TILE_WIDTH times fewer global memory accesses

## Compilation and Execution

### Compile
```bash
nvcc -o shared_matmul shared_matmul.cu
```

### Run
```bash
./shared_matmul
```

### Example Output
```
Enter the size of the square matrices: 64
Resultant Matrix C (first 10 elements):
2080 2144 2208 2272 2336 2400 2464 2528 2592 2656
```

## Limitations and Considerations

1. **Matrix Size**: Current implementation assumes N is divisible by TILE_WIDTH
2. **Shared Memory Size**: Limited by hardware (typically 48-96 KB)
3. **Tile Size**: TILE_WIDTH=16 is a good balance; larger tiles may exceed shared memory
4. **Boundary Conditions**: Code could be enhanced to handle matrices not divisible by tile size

## Learning Outcomes

After studying this code, you should understand:
- ✅ How shared memory improves performance in CUDA
- ✅ The tiling technique for matrix multiplication
- ✅ Proper use of `__syncthreads()` for thread synchronization
- ✅ Memory coalescing and access patterns
- ✅ Trade-offs between tile size and shared memory usage

## Next Steps

- Experiment with different TILE_WIDTH values (8, 32)
- Add timing code to measure performance improvement
- Handle non-square matrices
- Implement boundary checks for arbitrary matrix sizes
- Try rectangular tiles for further optimization

## References

- CUDA Programming Guide: Shared Memory
- NVIDIA CUDA Best Practices Guide
- "Programming Massively Parallel Processors" by Kirk and Hwu

---
**Day 4 of 100 Days CUDA Challenge** 🚀
