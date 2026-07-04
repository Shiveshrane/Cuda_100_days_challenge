# Day 018: Reduction Operations

**Chapter 10: Parallel Reductions**

## Overview
Implement efficient parallel reduction operations—combining all elements in an array into a single result (sum, max, min, product). Learn tree-based reductions and handle synchronization across blocks.

## What is a Reduction?

Combine all array elements into a single value:

```
Input:  [3, 1, 4, 1, 5, 9, 2, 6]
Sum:    31
Max:    9
Min:    1
```

## Reduction Strategies

### 1. **Single-Block Reduction**
- Shared memory tree reduction
- Multiple reduction steps with synchronization
- Suitable for smaller arrays or initial block-level reduction

### 2. **Multi-Block Reduction**
- Kernel launch per reduction level
- Global synchronization via multiple kernel launches
- Handles arbitrary array sizes

### 3. **Atomic Operations**
- `atomicAdd()` for simple reduction
- Lower occupancy due to atomic contention
- Acceptable for specific use cases

## Concepts Covered

- Tree-based reduction patterns
- Shared memory reduction
- Bank conflict avoidance in reductions
- Warp-level primitives (warp shuffle, warp reductions)
- Multi-pass reductions
- Atomic operations and their trade-offs

## Optimization Techniques

- **Address Stride**: Minimize warp divergence
- **Loop Unrolling**: Reduce instruction count
- **Warp Shuffles**: Eliminate shared memory access
- **Register Tiling**: Process multiple elements per thread

## Applications

- **Sum/Max/Min Operations**: Aggregation functions
- **Dot Product**: Vector operations
- **Norm Calculations**: Vector magnitudes
- **Statistics**: Mean, variance, standard deviation
- **Histogram Generation**: Count-based aggregations

## Performance Considerations

- Global memory access patterns during reduction
- Shared memory bank conflicts
- Warp efficiency and divergence
- Multi-block synchronization overhead
