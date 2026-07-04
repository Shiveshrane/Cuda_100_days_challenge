# Day 022: Stream Compaction & Partition

**Chapter 12: Selective Data Manipulation**

## Overview
Implement stream compaction (removing unwanted elements) and partition operations. These are fundamental for filtering data and supporting algorithms like quicksort and sparse data processing.

## Stream Compaction

### Problem Definition
Remove elements that don't satisfy a condition:

```
Input:      [3, 1, 4, 1, 5, 9, 2, 6]
Predicate:  x > 2
Output:     [3, 4, 5, 9, 6]
```

### Algorithm Steps
1. Mark/flag elements meeting the condition
2. Compute prefix sum of flags (exclusive scan)
3. Scatter marked elements to new positions

```
Input:       [3, 1, 4, 1, 5, 9, 2, 6]
Flags:       [1, 0, 1, 0, 1, 1, 0, 1]
Scan (excl): [0, 1, 1, 2, 2, 3, 4, 4]
Output:      [3, 4, 5, 9, 6]
```

## Partition Operation

### Problem Definition
Divide elements into groups based on a predicate:

```
Input:      [3, 1, 4, 1, 5, 9, 2, 6]
Predicate:  x > 2
True:       [3, 4, 5, 9, 6]
False:      [1, 1, 2]
```

### Algorithm Approaches
1. **Two-Pass Partition**: Scan for true, scan for false positions
2. **Single-Pass Partition**: Count true elements, use dual pointers
3. **In-Place Partition**: Two-pointer approach with swaps

## Concepts Covered

### 1. **Scan-Based Algorithms**
- Using inclusive/exclusive scans for compact operations
- Stream compaction via scan + scatter
- Multi-way partitioning

### 2. **Predicate Evaluation**
- Efficient predicate functions
- Branching vs. predicate predicates
- Predicate specialization

### 3. **Scatter Operations**
- Direct scatter to output buffer
- Handling sparse data patterns
- Memory coalescing in scatter

### 4. **Size Determination**
- Computing output size from scan
- Allocating dynamic output buffers
- Host-device synchronization for sizes

## Implementation Details

- Separate flag computation kernel
- Integration with scan primitives
- Scatter kernel design
- Output size tracking

## Applications

- **Filtering**: Remove outliers, invalid data
- **Compression**: Sparse data extraction
- **Quicksort**: Partition around pivot
- **Ray Tracing**: Culling rays, element filtering
- **Physics Simulation**: Active particle tracking
- **Data Cleaning**: Invalid record removal

## Performance Considerations

- Memory bandwidth for scatter phase
- Load balancing with sparse predicates
- Output buffer size prediction
- Cache efficiency of scatter patterns
- Comparison with CPU alternatives

## Hybrid Approaches

- CPU-GPU work division based on predicate selectivity
- Multi-level partitioning for complex predicates
- Load imbalance mitigation strategies
