# Day 023: Merge and Merge Sort

**Chapter 13: Merge-Based Algorithms**

## Overview
Implement merge operations and merge sort on the GPU. Merge is combining two sorted sequences into one, fundamental to divide-and-conquer sorting and many other algorithms.

## The Merge Operation

### Problem Definition
Combine two sorted sequences into one sorted sequence:

```
Array A: [1, 3, 5, 7]
Array B: [2, 4, 6, 8]
Merged:  [1, 2, 3, 4, 5, 6, 7, 8]
```

## Merge Algorithms

### 1. **Serial Merge**
- Classic two-pointer approach
- O(N) comparisons
- Not GPU-friendly due to data dependencies

### 2. **Parallel Merge (Rank-Based)**
- Each thread finds position in opposite array (binary search)
- Rank: position where element would be inserted
- Output position = rank in A + rank in B

```
Element 3 from A:
  - Rank in A: 1 (1 element less than 3)
  - Binary search in B: between 2 and 4, rank 2
  - Output position: 1 + 2 = 3
```

### 3. **Bitonic Merge**
- Sorting network-based merge
- Regular communication pattern
- Excellent for coalesced access

## Merge Sort Implementation

### Algorithm Overview
1. Divide array into small tiles
2. Sort tiles locally (shared memory)
3. Merge sorted tiles hierarchically
4. Each level doubles the sorted tile size

```
Unsorted:  [3, 1, 4, 1, 5, 9, 2, 6]

Tile sort: [1, 3] [1, 4] [5, 9] [2, 6]

Merge 1:   [1, 1, 3, 4] [2, 5, 6, 9]

Merge 2:   [1, 1, 2, 3, 4, 5, 6, 9]
```

## Concepts Covered

### 1. **Rank-Based Merge**
- Binary search integration
- Parallel rank computation
- Output position calculation

### 2. **Two-Phase Merge**
- Forward phase: Compute ranks
- Backward phase: Fill output array
- Register-efficient implementation

### 3. **Hierarchical Merge**
- Multi-level merging for large arrays
- Tile size vs. merge efficiency trade-offs
- Occupancy considerations

### 4. **Memory Access Patterns**
- Coalesced access in merge
- Cache efficiency
- Bank conflicts in shared memory

## Implementation Considerations

- Tile size selection for local sorting
- Shared memory for temporary storage
- Binary search kernel parameters
- Output buffer allocation

## Performance Characteristics

| Operation | Complexity | Work | Steps | GPU Fit |
|-----------|-----------|------|-------|---------|
| Serial Merge | O(N) | O(N) | O(N) | Poor |
| Parallel Merge | O(log²N) | O(N log N) | O(log²N) | Excellent |
| Merge Sort | O(N log²N) | O(N log²N) | O(log²N) | Good |

## Applications

- **Sorting**: Merge sort, hybrid merge-quicksort
- **Joining**: Database join operations
- **Set Operations**: Union, intersection, difference
- **Sparse Data**: Merging sparse arrays
- **Graph Algorithms**: Edge merging in graph construction

## Optimization Techniques

- Minimize binary search overhead
- Warp-efficient parallelization
- Shared memory optimization
- Multi-pass merging for large arrays
- Adaptive tile sizing
