# Day 017: Prefix Sum (Scan) Operations

**Chapter 9: Parallel Scan Algorithms**

## Overview
Implement parallel prefix sum (scan) operations—a fundamental parallel algorithm used in sorting, stream compaction, and many GPU applications. Learn work-efficient and work-inefficient scan algorithms.

## What is a Prefix Sum?

Compute cumulative sums in parallel:

```
Input:  [3, 1, 4, 1, 5, 9, 2, 6]
Output: [3, 4, 8, 9, 14, 23, 25, 31]
```

Each output element is the sum of all input elements up to and including that position.

## Scan Algorithms

### 1. **Work-Inefficient Parallel Scan (Blelloch)**
- Logarithmic steps: O(log N) steps
- Work complexity: O(N log N)
- Higher parallelism
- Simpler implementation

### 2. **Work-Efficient Parallel Scan**
- Logarithmic steps: O(log N) steps
- Work complexity: O(N)
- Lower bandwidth consumption
- More complex implementation

### 3. **Variations**
- **Inclusive scan**: Output includes current element
- **Exclusive scan**: Output excludes current element
- **Segmented scan**: Multiple independent scans

## Concepts Covered

- Parallel algorithm design
- Synchronization and data dependencies
- Shared memory communication
- Hierarchical scan (multi-level for large arrays)
- Work analysis and efficiency metrics

## Applications

- **Sorting**: Radix sort, quicksort foundations
- **Stream Compaction**: Removing elements efficiently
- **Load Balancing**: Histogram-based operations
- **Graph Algorithms**: BFS with compact queues
- **Physics Simulations**: Particle interactions

## Implementation Considerations

- Single-block vs. multi-block scans
- Shared memory management
- Bank conflict avoidance in scan arrays
- Boundary handling for non-power-of-2 sizes
