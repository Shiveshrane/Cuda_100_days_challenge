# Day 020: Sorting - Bitonic Sort & Radix Sort

**Chapter 11: Advanced Parallel Sorting**

## Overview
Implement efficient parallel sorting algorithms: bitonic sort and radix sort. Bitonic sort is a highly parallelizable comparison-based sort, while radix sort is non-comparative and achieves linear time complexity relative to the number of bits.

## Bitonic Sort

### Algorithm Overview
Hierarchical sorting network with recursive structure:

```
Bitonic Sequence: First half is sorted ascending, second half descending
Bitonic Sorter: Recursively sorts bitonic sequences

Example of bitonic sequence: [3, 7, 4, 8, 9, 1, 5, 2]
```

### Key Properties
- Works on power-of-2 sized arrays
- **Complexity**: O(log² N) parallel steps, O(N log² N) work
- **High Parallelism**: Excellent GPU fit
- **Regular Communication Pattern**: Cache-friendly

### Concepts

1. **Bitonic Sequence Recognition**
- First half ascending, second half descending
- Natural for recursive divide-and-conquer

2. **Recursive Structure**
- Bitonic merge of two sorted sequences
- Compare and swap at increasing distances

3. **Sorting Networks**
- Fixed comparison topology
- Independent comparisons in each stage
- Predictable memory access patterns

## Radix Sort

### Algorithm Overview
Non-comparative sorting based on digit values:

```
Input:  [170, 45, 75, 90, 2, 8, 802, 24]
Radix 1: [90, 802, 2, 24, 45, 75, 170, 8]    (sorted by 1s place)
Radix 2: [2, 8, 24, 45, 75, 90, 170, 802]    (sorted by 10s place)
```

### Advantages
- Linear time: O(k·N) where k = number of bits/digits
- Stable sorting property
- Highly parallelizable counting phase

### Key Concepts

1. **Counting Phase**
- Count occurrences of each digit value
- Parallel histogram computation

2. **Prefix Sum Phase**
- Compute cumulative counts (scan operation)
- Determines output positions

3. **Scatter Phase**
- Place elements in output array based on counts
- May require multiple passes for multi-digit keys

## CUDA Implementation Considerations

### Bitonic Sort
- Tile-based implementation for large arrays
- Shared memory for tile sorting
- Multiple kernel launches for global merges
- Bit-reversal addressing patterns

### Radix Sort
- Histogram computation (reduction-based)
- Prefix sum integration (scan operation)
- Scatter phase optimization
- Multi-pass handling for large keys

## Performance Characteristics

| Algorithm | Steps | Work | Parallelism | Cache Efficiency |
|-----------|-------|------|-------------|------------------|
| Bitonic Sort | O(log² N) | O(N log² N) | Excellent | Good |
| Radix Sort | O(k log N) | O(k·N) | Excellent | Excellent |

## Applications

- **GPU-Accelerated Databases**: Sorting large datasets
- **Ray Tracing**: Primitive sorting
- **Particle Systems**: Position-based sorting
- **Graph Processing**: Edge/vertex sorting
- **Physics Simulations**: Spatial partitioning

## Hybrid Approaches

- Bitonic sort for small arrays
- Radix sort for large datasets
- Adaptive sorting based on data characteristics
- Combining with CPU for memory-limited scenarios
