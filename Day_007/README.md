# Day 7: Memory Coalescing - Advanced Patterns

**Chapter 6 Continued**

## Overview
Today you'll tackle two classic memory access challenges: matrix transpose and data structure layout. Both are fundamental patterns that appear throughout GPU computing, and both require careful thought about coalescing.

## Concepts Covered

### 1. **Matrix Transpose Problem**
- Reading from rows, writing to columns (or vice versa)
- One direction will be uncoalesced in naive implementation
- Shared memory solution for both coalesced reads and writes

### 2. **Data Structure Layout**
- **AoS (Array of Structures)**: Traditional C style
- **SoA (Structure of Arrays)**: GPU-friendly style
- Cache line utilization
- When to use each pattern

### 3. **Shared Memory as Coalescing Buffer**
- Using shared memory to change access patterns
- Read coalesced → rearrange in shared memory → write coalesced
- The "staging area" technique

## Learning Objectives

By the end of today, you should be able to:
- [ ] Implement naive matrix transpose
- [ ] Implement optimized matrix transpose with shared memory
- [ ] Explain the coalescing issue in transpose
- [ ] Convert AoS to SoA layout
- [ ] Benchmark AoS vs SoA for vector operations
- [ ] Measure and explain performance differences
- [ ] Identify when coalescing requires data structure changes

## Problem 1: Matrix Transpose

### The Challenge
Transposing a matrix means: `Output[x][y] = Input[y][x]`

**Naive approach issue:**
- Reading Input row-wise: `Input[y * width + x]` -> ✅ Coalesced
- Writing Output column-wise: `Output[x * height + y]` -> ❌ Uncoalesced (Strided)

Strided writes are terrible for performance because adjacent threads write to memory addresses that are far apart (separated by `height`). This forces the memory controller to issue many separate transactions.

### Solution 1: Naive Transpose
We implemented a naive kernel to establish a baseline.
```cuda
__global__ void naiveTranspose(const float *Input, float *Output, int width, int height){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx < width && ty < height){
        int input_idx = ty * width + tx;
        int output_idx = tx * height + ty;
        Output[output_idx] = Input[input_idx];
    }
}
```
**Performance:** Slow due to uncoalesced writes.

### Solution 2: Shared Memory Transpose (Optimized)
To achieve **Coalesced Reads AND Coalesced Writes**, we use Shared Memory as a staging buffer.

**The Strategy:**
1.  **Read Coalesced:** Threads read a tile from Global Memory into Shared Memory (row-wise).
2.  **Transpose in Shared Memory:** We swap indices when reading *out* of Shared Memory.
3.  **Write Coalesced:** Threads write the tile from Shared Memory to Global Memory (row-wise).

**Key Optimization: Block Swapping**
To write row-wise to the transposed matrix, we must treat the Output matrix as if we are filling it row-by-row.
-   We swap `blockIdx.x` and `blockIdx.y` for the write phase.
-   `blockIdx.y` becomes the new "X" (column) index.
-   `blockIdx.x` becomes the new "Y" (row) index.
-   We keep `threadIdx.x` as the fast-moving index.

This ensures that adjacent threads write to adjacent memory addresses: `Output[row * stride + (base + threadIdx.x)]`.

**Key Optimization: Shared Memory Padding**
To avoid **Bank Conflicts** when reading column-wise from Shared Memory, we add padding to the tile width.
`__shared__ float tile[TILE_SIZE][TILE_SIZE + 1];`
This offsets the columns so they don't land in the same memory bank, allowing simultaneous access.

```cuda
__global__ void SharedMemTranspose(const float *Input, float *Output, int width, int height){
    // 1. READ PHASE (Coalesced)
    __shared__ float tileA[TILE_SIZE][TILE_SIZE+1]; // +1 Padding
    
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx < width && ty < height){
        tileA[threadIdx.y][threadIdx.x] = Input[ty * width + tx];
    }
    __syncthreads();

    // 2. WRITE PHASE (Coalesced)
    // Swap block indices to transpose the grid
    int tx2 = blockIdx.y * blockDim.x + threadIdx.x; 
    int ty2 = blockIdx.x * blockDim.y + threadIdx.y;

    if (tx2 < height && ty2 < width){
        // Coalesced Write: tx2 varies fast (0,1,2...)
        Output[ty2 * height + tx2] = tileA[threadIdx.x][threadIdx.y];
    }
}
```

## Problem 2: Array of Structures vs Structure of Arrays

### The AoS Problem

```cpp
// Array of Structures - NOT GPU friendly
struct Particle {
    float x, y, z;    // position
    float vx, vy, vz; // velocity
    float mass;
};

Particle particles[N];

// Kernel accessing just x-coordinates:
float x = particles[idx].x;  // Loads 7 floats, uses 1! ❌
```

Threads in a warp access scattered memory:
- Thread 0: particles[0].x (byte 0)
- Thread 1: particles[1].x (byte 28)
- Thread 2: particles[2].x (byte 56)
- → Non-coalesced!

### The SoA Solution

```cpp
// Structure of Arrays - GPU friendly
struct ParticlesSoA {
    float *x, *y, *z;
    float *vx, *vy, *vz;
    float *mass;
};

// Kernel accessing x-coordinates:
float x = x_array[idx];  // Coalesced! ✅
```

Threads access consecutive memory:
- Thread 0: x[0] (byte 0)
- Thread 1: x[1] (byte 4)
- Thread 2: x[2] (byte 8)
- → Perfect coalescing!

### Task 4: Implement AoS Particle System
```cuda
struct Particle {
    float3 pos;
    float3 vel;
    float mass;
};

__global__ void updateAoS(Particle *particles, int n, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        particles[idx].pos.x += particles[idx].vel.x * dt;
        particles[idx].pos.y += particles[idx].vel.y * dt;
        particles[idx].pos.z += particles[idx].vel.z * dt;
    }
}
```

### Task 5: Implement SoA Particle System
```cuda
__global__ void updateSoA(float *pos_x, float *pos_y, float *pos_z,
                          float *vel_x, float *vel_y, float *vel_z,
                          int n, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        pos_x[idx] += vel_x[idx] * dt;
        pos_y[idx] += vel_y[idx] * dt;
        pos_z[idx] += vel_z[idx] * dt;
    }
}
```

### Task 6: Hybrid Approach (AoSoA)
For small structures, array of small structures of arrays:
```cpp
struct Particle4 {  // 4 particles as a unit
    float4 pos_x;  // x for 4 particles
    float4 pos_y;  // y for 4 particles
    // ...
};
```

## Benchmarking Matrix Transpose

We implemented a benchmark that tests various matrix sizes, including non-square matrices.

**Test Cases:**
- Square: 1024x1024, 2048x2048, 4096x4096
- Rectangular: 1024x2048, 2048x1024, etc.

**Results:**
- Naive Transpose: ~0.5ms (for small sizes)
- Shared Memory Transpose: ~0.13ms
- **Speedup:** ~3.6x

The speedup comes from replacing the slow strided writes with fast coalesced writes.

## Benchmarking AoS vs SoA

Test with particle counts:
- 10K particles
- 100K particles
- 1M particles
- 10M particles

Measure:
- Execution time
- Effective bandwidth
- Speedup of SoA over AoS

Expected result: SoA should be 3-8x faster!

## Deliverables Checklist

- [ ] **Transpose Code**: Naive and optimized versions
- [ ] **Transpose Benchmarks**: Performance comparison across sizes
- [ ] **Transpose Analysis**: Explain coalescing patterns with diagrams
- [ ] **Particle Code**: AoS and SoA implementations
- [ ] **Particle Benchmarks**: Performance comparison
- [ ] **Data Layout Guide**: When to use AoS vs SoA
- [ ] **Graphs**: 
  - Transpose: Bandwidth vs matrix size
  - Particles: Time vs particle count for both layouts

## Profiling with Nsight Compute

For transpose:
```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./transpose
```

Look for:
- Global load transactions
- Global store transactions
- L2 cache hit rate

For AoS vs SoA:
```bash
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./particles
```

Compare DRAM throughput percentage between AoS and SoA.

## Common Pitfalls

1. **Forgetting `__syncthreads()`** in shared memory transpose
2. **Bank conflicts** in shared memory (we'll fix this tomorrow!)
3. **Incorrect boundary handling** for non-tile-aligned matrices
4. **Not padding SoA arrays** (can cause partition camping)

## Real-World Applications

### Matrix Transpose:
- Linear algebra routines
- Image rotation
- FFT algorithms
- CNN layer transformations

### SoA Pattern:
- Particle physics simulations
- Molecular dynamics
- Ray tracing (ray structures)
- Neural network batching

## Reflection Questions

1. Why can't naive transpose be fully coalesced?
2. How does shared memory solve the coalescing problem?
3. What's the memory overhead of SoA vs AoS?
4. When might AoS actually be better than SoA?
5. How do you decide tile size for transpose?

## Advanced Challenges (Optional)

1. **Optimize for rectangular matrices** (non-square)
2. **In-place transpose** for square matrices
3. **Block transpose** (transpose 4×4 blocks at a time)
4. **3D matrix transpose** (permute 3 dimensions)
5. **AoSoA optimal struct size** (find best grouping)

## Next Steps

Tomorrow (Day 8), we'll dive into shared memory bank conflicts—you might notice them in your transpose kernel today! We'll learn to detect and eliminate them.

---

**Key Takeaway**: Sometimes you need to "stage" data through shared memory to get optimal access patterns. Think of it as a coalescing adapter! 💡
