# Day 024: Sparse Matrix Operations - GEMV

**Chapter 14: Sparse Linear Algebra**

## Overview
Implement sparse matrix-vector multiplication (SpMV). Sparse matrices are common in scientific computing, machine learning, and graph algorithms. Learn sparse matrix storage formats and efficient GPU implementation.

## Sparse Matrix Formats

### 1. **COO (Coordinate Format)**
```
Matrix:     Values: [1, 2, 3, 4]
[1 0 2]     Rows:   [0, 0, 1, 2]
[0 0 0] →   Cols:   [0, 2, 1, 2]
[0 0 4]
```
- Simple, flexible
- High memory overhead
- Good for unstructured data

### 2. **CSR (Compressed Sparse Row)**
```
Values:     [1, 2, 3, 4]
ColIdx:     [0, 2, 1, 2]
RowPtr:     [0, 2, 3, 4]
```
- Most common format
- Efficient row access
- Lower memory overhead

### 3. **CSC (Compressed Sparse Column)**
- Column-based variant of CSR
- Efficient column operations
- Cache-friendly for column-wise multiplication

### 4. **ELL (Ellpack Format)**
```
Padded storage with fixed max non-zeros per row
Good for regular sparse patterns
```

## SpMV Algorithm: y = A × x

### Problem
Multiply sparse matrix by dense vector:

```
[1 0 2]   [x0]   [1*x0 + 2*x2]
[0 0 0] × [x1] = [0]
[0 0 4]   [x2]   [4*x2]
```

## GPU Implementation Strategies

### 1. **Row-Based Distribution**
- One thread per row
- Warp per row (if many non-zeros)
- Issues with load imbalance

### 2. **CSR Merge-Path Approach**
- Thread block processes multiple rows
- Segments assigned via binary search
- Balanced work distribution

### 3. **Warp-Level Operations**
- Warp per row for dense rows
- Thread per row for sparse rows
- Hybrid approach

## Concepts Covered

### 1. **Sparse Matrix Storage**
- Format selection trade-offs
- Conversion between formats
- Format-specific optimizations

### 2. **Load Balancing**
- Handling variable row sparsity
- Work stealing approaches
- Adaptive kernel selection

### 3. **Memory Access Patterns**
- SpMV: SparseMat global, Vector coalesced
- Cache reuse analysis
- L2 cache effectiveness

### 4. **Synchronization and Communication**
- Within-warp reductions
- Shared memory for partial sums
- Atomic operations for sparse patterns

## Implementation Considerations

- Matrix format conversion utilities
- Sparsity analysis and optimization
- Sparse vector support
- Transpose operations for format conversion

## Performance Metrics

- Effective bandwidth (GB/s)
- Operations per second (GFLOP/s)
- Load balance distribution
- Cache hit rates

## Applications

- **Linear Solvers**: Iterative methods (CG, GMRES)
- **Graph Algorithms**: Adjacency matrix operations
- **Physics Simulation**: Large sparse systems
- **Machine Learning**: Sparse neural networks
- **PDE Solvers**: Finite element/difference methods

## Optimization Techniques

- Reordering for cache efficiency
- Format selection based on sparsity pattern
- Hybrid CPU-GPU execution
- Multi-GPU distribution
- Batch SpMV for multiple vectors (SpMM)
