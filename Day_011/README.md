# Day 11: Register Pressure & Local Memory

**Chapter 6: Register Usage**

## Overview
Registers are the fastest memory on the GPU, but they're limited. When kernels use too many registers, performance suffers through "register spilling" to slower local memory. Today you'll learn to manage register pressure and optimize register usage.

## What is Register Pressure?

**Register Pressure**: When a kernel needs more registers than available, causing:
- Register spilling to local memory (slow!)
- Reduced occupancy (fewer blocks fit on SM)
- Performance degradation

### Memory Hierarchy Speed (Relative):
- **Registers**: 1x (fastest)
- **Shared Memory**: ~1-2x
- **Local Memory**: ~100x (slow! It's actually global memory)
- **Global Memory**: ~100-400x

## Concepts Covered

### 1. **Register Allocation**
- How compiler assigns registers
- Register file organization
- Per-thread vs per-warp registers

### 2. **Register Spilling**
- What triggers spilling
- Local memory characteristics
- Performance impact

### 3. **Optimization Techniques**
- `__launch_bounds__` directive
- Variable reuse strategies
- Loop unrolling trade-offs

### 4. **Profiling Register Usage**
- Using ptxas verbose output
- Nsight Compute metrics
- cuobjdump analysis

## Learning Objectives

By the end of today, you should be able to:
- [ ] Identify register pressure in kernels
- [ ] Use ptxas to check register usage
- [ ] Apply `__launch_bounds__` effectively
- [ ] Reduce register usage through code optimization
- [ ] Detect local memory spilling with profiling
- [ ] Balance register usage vs occupancy
- [ ] Make informed trade-offs

## Implementation Tasks

### Task 1: High Register Pressure Kernel

```cuda
__global__ void highRegisterKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Many local variables = high register usage
    float a = input[idx];
    float b = a * 2.0f;
    float c = b + 3.0f;
    float d = c * c;
    float e = d - 1.0f;
    float f = e / 2.0f;
    float g = f + a;
    float h = g * b;
    float i = h - c;
    float j = i + d;
    float k = j * e;
    float l = k - f;
    
    output[idx] = l;
}
```

Compile and check register usage:
```bash
nvcc -arch=sm_XX --ptxas-options=-v high_regs.cu

# Output will show: "Used X registers, Y bytes smem"
```

### Task 2: Optimized Version (Reuse Variables)

```cuda
__global__ void optimizedRegisterKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Reuse variables - fewer registers!
    float temp = input[idx];
    float original = temp;  // Keep one copy
    
    temp = temp * 2.0f;         // b
    temp = temp + 3.0f;         // c
    temp = temp * temp;         // d
    temp = temp - 1.0f;         // e
    temp = temp / 2.0f;         // f
    temp = temp + original;     // g
    // Continue reusing...
    
    output[idx] = temp;
}
```

### Task 3: Using `__launch_bounds__`

```cuda
// Tell compiler: optimize for 256 threads/block, at least 4 blocks/SM
__global__ void __launch_bounds__(256, 4)
launchBoundsKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Compiler will try to use fewer registers
        // to achieve 4 blocks/SM with 256 threads
        float result = 0.0f;
        for (int i = 0; i < 10; i++) {
            result += input[idx] * i;
        }
        output[idx] = result;
    }
}
```

**`__launch_bounds__` parameters:**
```cuda
__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
```

### Task 4: Detect Local Memory Spilling

```cuda
__global__ void spillingKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Large array - likely to spill to local memory
    float array[128];
    
    for (int i = 0; i < 128; i++) {
        array[i] = input[idx] + i;
    }
    
    float sum = 0.0f;
    for (int i = 0; i < 128; i++) {
        sum += array[i];
    }
    
    output[idx] = sum;
}
```

Check for spilling:
```bash
nvcc -arch=sm_XX --ptxas-options=-v spilling.cu

# Look for: "X bytes stack frame, Y bytes spill stores, Z bytes spill loads"
```

## Checking Register Usage

### Method 1: Compile-Time (ptxas)

```bash
nvcc -arch=sm_75 --ptxas-options=-v kernel.cu

# Output example:
# ptxas info: Used 32 registers, 0 bytes smem, 328 bytes cmem[0]
# ptxas info: Used 48 bytes local memory
```

- **Registers**: Number per thread
- **Local memory**: Indicates spilling!

### Method 2: cuobjdump

```bash
nvcc -arch=sm_75 -c kernel.cu -o kernel.o
cuobjdump -sass kernel.o | grep -A 50 "Function : _Z.*kernel"

# Shows SASS (assembly) with register usage
```

### Method 3: Nsight Compute

```bash
ncu --metrics launch__registers_per_thread,launch__registers_per_thread_allocated ./kernel

# Also check:
ncu --metrics smsp__sass_inst_executed_op_ldsm.sum ./kernel  # Local memory loads
ncu --metrics smsp__sass_inst_executed_op_stsm.sum ./kernel  # Local memory stores
```

## Profiling Local Memory Access

Key metrics in Nsight Compute:
- `l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum`: Local memory loads
- `l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum`: Local memory stores
- `launch__registers_per_thread`: Actual register usage

**If you see local memory loads/stores > 0 → Spilling is happening!**

## Optimization Strategies

### 1. **Reduce Local Variables**
```cuda
// Bad: Many variables
float a, b, c, d, e, f, g;
a = x + 1; b = a * 2; c = b + 3; ...

// Good: Reuse
float temp;
temp = x + 1; temp *= 2; temp += 3; ...
```

### 2. **Move Large Arrays to Shared Memory**
```cuda
// Bad: Local array (spills)
float local[100];

// Good: Shared array (if pattern allows)
__shared__ float shared[100 * BLOCK_SIZE];
float *mySlice = &shared[threadIdx.x * 100];
```

### 3. **Reduce Loop Unrolling**
```cuda
// Compiler might unroll and use many registers
#pragma unroll 1  // Disable unrolling
for (int i = 0; i < 100; i++) {
    // ...
}
```

### 4. **Use Restrict Keyword**
```cuda
__global__ void kernel(float * __restrict__ a, float * __restrict__ b) {
    // Tells compiler pointers don't alias, may improve register use
}
```

### 5. **Adjust Optimization Level**
```bash
nvcc -O2 kernel.cu  # Instead of -O3
# Sometimes lower optimization uses fewer registers
```

## Calculate Occupancy Impact

**Example**: Your kernel uses 64 registers per thread

GPU with 65,536 registers per SM:
- **Block size 256**: 256 × 64 = 16,384 registers per block
  - Max blocks: 65,536 / 16,384 = 4 blocks per SM
  - Occupancy: (4 × 8 warps) / 48 max warps = 67%

- **After optimization to 48 registers**:
  - 256 × 48 = 12,288 registers per block
  - Max blocks: 65,536 / 12,288 = 5 blocks per SM
  - Occupancy: (5 × 8) / 48 = 83% ✅

**Reducing 25% of registers → 16% occupancy increase!**

## Deliverables Checklist

- [ ] **High-Register Kernel**: Intentional register pressure
- [ ] **Optimized Version**: Reduced register usage
- [ ] **`__launch_bounds__` Example**: Properly configured
- [ ] **Spilling Example**: Large local array
- [ ] **Profiling Data**: ptxas output for all kernels
- [ ] **Nsight Reports**: Local memory metrics
- [ ] **Performance Comparison**: Before/after optimization
- [ ] **Analysis Document**:
  - Register usage for each kernel variant
  - Occupancy calculations
  - Performance impact of spilling
  - Optimization strategy effectiveness

## Performance Impact Analysis

Create comparison table:

| Kernel Variant | Registers/Thread | Local Mem (bytes) | Occupancy | Time (ms) |
|----------------|------------------|-------------------|-----------|-----------|
| Naive          | 64               | 0                 | 67%       | 10.5      |
| Optimized      | 48               | 0                 | 83%       | 8.2       |
| Spilling       | 32               | 256               | 100%      | 15.3      |
| Launch Bounds  | 42               | 0                 | 75%       | 8.7       |

## Real-World Scenarios

### When Register Pressure Matters:
1. **Complex kernels**: Many intermediate calculations
2. **Unrolled loops**: Compiler expands loops into many instructions
3. **Inline functions**: Each inlined function adds registers
4. **Recursive algorithms**: Stack frames consume registers

### When to Accept Higher Usage:
1. **Compute-bound**: If ALU usage is already 100%
2. **Small problem sizes**: Occupancy less critical
3. **One-time kernels**: Optimization time not worth it

## Common Patterns

### ✅ Register-Friendly:
- Variable reuse
- Streaming computations (input → output)
- Minimal state tracking

### ❌ Register-Hungry:
- Large local arrays
- Many live variables
- Deeply nested calculations
- Heavy loop unrolling

## Reflection Questions

1. At what point does register spilling occur for your kernel?
2. How much does `__launch_bounds__` help occupancy?
3. What's the performance cost of local memory access?
4. Can you achieve 100% occupancy with 64 registers/thread on your GPU?
5. When is register pressure optimization worth the effort?

## Advanced Topics (Optional)

1. **Register Caching**: Some local memory cached in registers
2. **Warp Scheduling**: How register usage affects scheduler
3. **Volta+ Features**: Independent thread scheduling implications
4. **Mixed Precision**: Using half-precision to reduce registers

## Next Steps

Tomorrow (Day 12), we'll bring together everything from Week 1 into an integration project: an optimized 2D convolution kernel using all the techniques you've learned!

---

**Pro Tip**: Profile first! Don't optimize register usage unless it's actually limiting performance. 🎯
