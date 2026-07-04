# Day 025: Atomics, Synchronization & Memory Consistency

**Chapter 15: Advanced Synchronization and Memory Model**

## Overview
Master atomic operations and synchronization primitives for coordinating work across threads and blocks. Understand CUDA's memory consistency model and inter-block synchronization strategies.

## Atomic Operations

### Basic Atomic Functions

```cuda
atomicAdd(address, val)       // Supported on all types
atomicSub(address, val)       // Subtraction
atomicInc(address, wrap)      // Increment with wrap
atomicDec(address, wrap)      // Decrement with wrap
atomicExch(address, val)      // Exchange (swap)
atomicMin(address, val)       // Minimum
atomicMax(address, val)       // Maximum
atomicCAS(addr, cmp, val)     // Compare and swap
```

### Type Support
- **32-bit**: All operations on int, unsigned int, float
- **64-bit**: Limited operations on long long, unsigned long long
- **Shared Memory**: Most operations on shared memory too

## Atomic Operation Characteristics

### 1. **Global Memory Atomics**
- Fastest at GPU level
- Global atomics serialize through GPU atomic unit
- Variable latency due to L2 cache

### 2. **Shared Memory Atomics**
- Lower latency for intra-block operations
- More operations available
- No global serialization

### 3. **Performance Considerations**
- Atomic contention (high contention = low throughput)
- ~5-20 clock cycles per atomic operation
- Poor performance with high thread counts on same address

## Memory Consistency Model

### 1. **Memory Ordering**
- **Within-thread**: Sequential consistency
- **Between-threads in block**: Guaranteed by `__syncthreads()`
- **Between blocks**: No guarantee without external sync

### 2. **Memory Hierarchy Effects**
```
L1 Cache (per SM)
    ↕
L2 Cache (shared)
    ↕
Global Memory
```
- Writes visible after L2 writeback
- Atomics force consistency

### 3. **Visibility Rules**
- `__threadfence()`: Waits for all writes to global memory
- `__threadfence_block()`: Waits for writes to shared memory
- `__syncthreads()`: Block-level synchronization barrier

## Synchronization Patterns

### 1. **Intra-Block Synchronization**
```cuda
__syncthreads()          // Barrier for all threads
__syncthreads_count()    // Count predicate across block
__syncthreads_and()      // Logical AND
__syncthreads_or()       // Logical OR
```

### 2. **Warp-Level Primitives**
```cuda
__ballot_sync(mask, pred)      // Ballot across warp
__shfl_sync(mask, var, lane)   // Shuffle within warp
__shfl_xor_sync(mask, var, lane) // Shuffle XOR
```

### 3. **Inter-Block Synchronization**
- Grid-level synchronization via kernel launch
- Cooperative groups for fine-grained control
- External synchronization via host-device barrier

## Concepts Covered

### 1. **Lock-Free Programming**
- Compare-and-swap loops
- Atomic increment for allocators
- Wait-free algorithms

### 2. **Race Conditions**
- Data races vs. low-level races
- Benign races and synchronization
- Tools for race detection

### 3. **Cooperative Groups (CUDA 9+)**
- Grid-level synchronization without kernel launch
- Hierarchical group operations
- Flexible synchronization patterns

### 4. **Memory Barriers**
- Acquire/release semantics
- Sequential consistency enforcement
- Cost-benefit analysis

## Implementation Patterns

### Pattern 1: Atomic Counter
```cuda
unsigned int idx = atomicInc(&counter, limit);
// Use idx for work assignment
```

### Pattern 2: Compare-and-Swap Loop
```cuda
unsigned int old_val, new_val;
do {
    old_val = *address;
    new_val = old_val + increment;
} while (atomicCAS(address, old_val, new_val) != old_val);
```

### Pattern 3: Block-Wide Reduction
```cuda
__shared__ int sdata[256];
int val = threadIdx.x;
__syncthreads();
// Perform tree reduction with synchronization
```

## Common Pitfalls

- Excessive atomic contention
- Insufficient synchronization causing races
- Synchronization overhead exceeding benefits
- Warp divergence with synchronization predicates
- Deadlock with circular synchronization

## Performance Optimization

- Minimize atomic operations on hot paths
- Use atomic operations sparingly
- Consider lock-free queues/stacks
- Profile synchronization overhead
- Alternative: Temporary overflow buffers

## Applications

- **Histogram Computation**: Atomic increments
- **Memory Allocation**: Atomic counters
- **Load Balancing**: Work stealing queues
- **Graph Processing**: Concurrent updates
- **Physical Simulation**: Particle interactions

## Debugging Techniques

- Print-based debugging with atomics
- CUDA-GDB for breakpoint debugging
- Memory access pattern analysis
- Synchronization validation tools
