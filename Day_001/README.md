# Day 001: CUDA Vector Addition

## Overview
This project implements a simple vector addition operation using CUDA to demonstrate parallel computing on the GPU. The program adds two vectors element-wise and stores the result in a third vector.

## Topics Covered

### 1. **CUDA Kernel Functions**
- `__global__` keyword to define kernel functions that run on the GPU
- Kernel launch syntax: `kernelName<<<gridSize, blockSize>>>(parameters)`

### 2. **Thread Indexing**
- Calculating global thread index: `blockIdx.x * blockDim.x + threadIdx.x`
- Using thread indices to map work to data elements
- Boundary checking to prevent out-of-bounds access

### 3. **Memory Management**
- `cudaMalloc()`: Allocating memory on the GPU
- `cudaFree()`: Freeing GPU memory to prevent memory leaks

### 4. **Data Transfer**
- `cudaMemcpy()`: Transferring data between host (CPU) and device (GPU)
- `cudaMemcpyHostToDevice`: Copy from CPU to GPU
- `cudaMemcpyDeviceToHost`: Copy from GPU to CPU

### 5. **Execution Configuration**
- **Block Size**: Number of threads per block (256 in this example)
- **Grid Size**: Number of blocks in the grid (calculated as `ceil(N/blockSize)`)
- Understanding the trade-offs in choosing block and grid dimensions

## Code Explanation

### The Kernel Function
```cuda
__global__ void vectoradd(const float* A, const float* B, float* C, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        C[i] = A[i] + B[i];
    }
}
```
- Each thread computes one element of the output vector
- The global index `i` determines which element this thread processes
- Boundary check ensures threads don't access memory out of bounds

### Main Program Flow
1. **Initialize data on host**: Create and populate input vectors A and B
2. **Allocate GPU memory**: Reserve space for vectors on the device
3. **Copy data to GPU**: Transfer input vectors from host to device
4. **Launch kernel**: Execute parallel vector addition on GPU
5. **Copy results back**: Transfer output vector from device to host
6. **Clean up**: Free GPU memory and display results

## Compilation

To compile this CUDA program, use the NVIDIA CUDA Compiler (nvcc):

```bash
nvcc vector_add.cu -o vector_add
```

## Execution

Run the compiled program:

```bash
./vector_add
```

### Expected Output
The program will output the element-wise sum of vectors A and B:
```
0 2 6 12 20 30 42 56 72 90
```

Where:
- A[i] = i
- B[i] = i²
- C[i] = A[i] + B[i] = i + i²

## Key Concepts

### Parallel Execution Model
- CUDA organizes threads into **blocks** and blocks into a **grid**
- All threads execute the same kernel code but work on different data (SIMT - Single Instruction, Multiple Thread)

### Memory Hierarchy
- **Host memory**: CPU RAM
- **Device memory**: GPU global memory
- Data must be explicitly transferred between host and device

### Performance Considerations
- This simple example uses N=10, which is too small to see GPU benefits
- GPUs excel when processing thousands or millions of elements in parallel
- Memory transfer overhead can dominate for small problem sizes

## Next Steps
- Experiment with larger vector sizes to observe GPU performance benefits
- Try different block sizes and measure performance
- Explore CUDA error checking for robust code
- Learn about shared memory for optimizing memory access patterns
