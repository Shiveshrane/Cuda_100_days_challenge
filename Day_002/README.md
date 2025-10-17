# Day 002: CUDA Matrix Addition

## Overview
This project implements matrix addition using CUDA, demonstrating 2D thread organization and grid configuration. The program adds two square matrices element-wise using parallel GPU computation.

## Topics Covered

### 1. **2D Thread Organization**
- Using 2D thread blocks and grids for matrix operations
- `blockIdx.x`, `blockIdx.y` for 2D block indexing
- `threadIdx.x`, `threadIdx.y` for 2D thread indexing within blocks
- Mapping 2D thread coordinates to 1D memory layout

### 2. **dim3 Data Type**
- `dim3 blockDim(x, y, z)`: Defines the dimensions of a thread block
- `dim3 gridDim(x, y, z)`: Defines the dimensions of the grid
- Organizing threads in multiple dimensions for natural problem mapping

### 3. **Row-Major Memory Layout**
- Storing 2D matrices in 1D arrays using row-major order
- Index calculation: `index = row * width + column`
- Understanding memory layout for efficient access

### 4. **2D Boundary Checking**
- Checking both row and column indices to prevent out-of-bounds access
- Handling cases where matrix dimensions don't evenly divide by block size

### 5. **Dynamic Memory Allocation**
- Using `new` and `delete` for dynamic host memory allocation
- Proper memory management for variable-sized matrices

## Code Explanation

### The Kernel Function
```cuda
__global__ void matrix_Add(const float *A, const float *B, float *C, int N){
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (row_idx < N && col_idx < N){
        C[row_idx * N + col_idx] = A[row_idx * N + col_idx] + B[row_idx * N + col_idx];
    }
}
```
- Each thread computes one element of the output matrix
- `row_idx`: Global row index for this thread
- `col_idx`: Global column index for this thread
- 2D boundary check ensures threads don't access invalid memory
- Row-major indexing: `row * N + col` converts 2D coordinates to 1D array index

### Execution Configuration
```cuda
int block_size = 256;
dim3 blockDim(block_size, block_size);
dim3 gridDim(rows/blockDim.x + 1, columns/blockDim.y + 1);
```
- **blockDim**: Each block has 256×256 threads (65,536 threads per block)
- **gridDim**: Number of blocks needed to cover the entire matrix
- The `+1` ensures coverage when dimensions don't divide evenly

### Main Program Flow
1. **Allocate host memory**: Create matrices A, B, and C using dynamic allocation
2. **Initialize data**: Fill matrices A and B with test values (1.0 and 2.0)
3. **Allocate device memory**: Reserve GPU memory for all three matrices
4. **Copy to GPU**: Transfer input matrices from host to device
5. **Launch kernel**: Execute parallel matrix addition with 2D grid
6. **Copy results back**: Transfer output matrix from device to host
7. **Display results**: Print the resulting matrix
8. **Clean up**: Free both host and device memory

## Compilation

To compile this CUDA program, use the NVIDIA CUDA Compiler (nvcc):

```bash
nvcc Matrix_add.cu -o matrix_add
```

## Execution

Run the compiled program:

```bash
./matrix_add
```

### Expected Output
The program will output a 10×10 matrix where each element is 3.0:
```
3 3 3 3 3 3 3 3 3 3
3 3 3 3 3 3 3 3 3 3
3 3 3 3 3 3 3 3 3 3
...
```

Where:
- A[i][j] = 1.0
- B[i][j] = 2.0
- C[i][j] = A[i][j] + B[i][j] = 3.0

## Key Concepts

### 2D vs 1D Thread Organization
- **1D organization** (Day 001): Used for vector operations
  - `threadIdx.x` only
  - Simple linear indexing
  
- **2D organization** (Day 002): Natural for matrix operations
  - `threadIdx.x` and `threadIdx.y`
  - Maps directly to row and column structure

### Thread Block Size Considerations
- Current implementation uses 256×256 = 65,536 threads per block
- **Important**: Most GPUs have a maximum of 1024 threads per block
- This code may not run on all devices due to block size limits
- Recommended: Use 16×16 or 32×32 thread blocks for better compatibility

### Memory Layout
```
2D Matrix View:          1D Memory Layout:
[0,0] [0,1] [0,2]   →   [0] [1] [2] [3] [4] [5] [6] [7] [8]
[1,0] [1,1] [1,2]
[2,0] [2,1] [2,2]
```

## Performance Considerations

### Current Implementation Issues
1. **Block size too large**: 256×256 exceeds typical GPU limits
2. **Small problem size**: 10×10 matrix too small to benefit from GPU
3. **No error checking**: Missing CUDA error checks for debugging

### Suggested Improvements
```cuda
// Better block size configuration
dim3 blockDim(16, 16);  // 256 threads per block
dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, 
             (columns + blockDim.y - 1) / blockDim.y);
```

### When to Use GPU for Matrix Operations
- GPUs excel with large matrices (1000×1000 or larger)
- Memory transfer overhead significant for small matrices
- For real applications, consider using cuBLAS library

## Comparison with Day 001

| Aspect | Day 001 (Vector Add) | Day 002 (Matrix Add) |
|--------|---------------------|---------------------|
| **Problem Type** | 1D vector operation | 2D matrix operation |
| **Thread Organization** | 1D blocks and grid | 2D blocks and grid |
| **Indexing** | Single index calculation | Row and column indices |
| **dim3 Usage** | Not required | Essential for 2D mapping |
| **Boundary Check** | Single condition | Two conditions |

## Next Steps
- Fix block size to stay within GPU limits (use 16×16 or 32×32)
- Add CUDA error checking (`cudaGetLastError()`, `cudaDeviceSynchronize()`)
- Test with larger matrices to see GPU performance benefits
- Implement matrix multiplication (more complex memory access patterns)
- Explore shared memory optimization for matrix operations
- Learn about memory coalescing for optimal performance

## Common Issues

### Block Size Limit Error
If you get an error about too many resources requested:
```
invalid configuration argument
```
**Solution**: Reduce block dimensions to 16×16 or 32×32

### Grid Size Calculation
The current `+1` approach can create extra threads. Better approach:
```cuda
dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, 
             (columns + blockDim.y - 1) / blockDim.y);
```
