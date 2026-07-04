# Day 019: Sorting - Odd-Even Sort

**Chapter 11: Parallel Sorting Algorithms**

## Overview
Implement parallel sorting algorithms, starting with odd-even sort (Batcher's sort). This is a simple comparative sort that's easy to parallelize and understand. Learn bitonic sort foundations.

## Odd-Even Sort (Brick Sort)

### Algorithm Overview
```
Repeat until sorted:
  Odd Phase:    Compare elements at (0,1), (2,3), (4,5)...
  Even Phase:   Compare elements at (1,2), (3,4), (5,6)...
```

Example progression:
```
Initial:     [3, 1, 4, 1, 5, 9, 2, 6]
After odd:   [1, 3, 1, 4, 5, 9, 2, 6]
After even:  [1, 1, 3, 4, 2, 5, 6, 9]
... continues
```

## Concepts Covered

### 1. **Comparison-Based Sorting**
- Comparator networks
- Sorting networks theory
- Correctness via zero-one principle

### 2. **Parallelization Strategy**
- Phase-based execution
- Independent comparisons in each phase
- Synchronization between phases

### 3. **CUDA Implementation**
- Kernel organization for phases
- Thread mapping to comparisons
- Efficient data arrangement
- Bank conflict considerations

### 4. **Bitonic Sort Introduction**
- More efficient sorting network
- Recursive structure
- Better parallelism properties

## Algorithms Hierarchy

1. **Bitonic Sort**: O(log²N) parallel steps, O(N log²N) work
2. **Odd-Even Sort**: O(N) worst case, simpler structure
3. **Radix Sort**: Non-comparative, O(log N) with preprocessing

## Implementation Focus

- Comparison function design
- Thread-to-comparison mapping
- Global synchronization between phases
- Memory layout for efficient comparison
- In-place vs. additional memory trade-offs

## Performance Metrics

- Comparisons per element
- Parallel steps required
- Memory bandwidth utilization
- Speedup over sequential sort

## Applications

- Sorting for duplicate removal
- Data rearrangement preprocessing
- Stream compaction support
- Building blocks for more complex algorithms
