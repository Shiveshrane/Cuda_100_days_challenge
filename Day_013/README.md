# Day 13: 1D Convolution - Basics

**Chapter 7: Convolution Introduction**

## Overview
Today you begin Week 2 with 1D convolution—a simpler problem than 2D, but perfect for understanding convolution fundamentals, boundary conditions, and optimization strategies. 1D convolution is used in audio processing, time series analysis, and as a building block for separable 2D convolution.

## What is 1D Convolution?

Apply a filter (mask/kernel) to a 1D signal:

```
Output[i] = Σ Input[i+j] * Mask[j]
            j=-radius to +radius
```

**Example**: 3-element mask `[1, 2, 1]` (radius=1, normalized by 1/4)
```
Input:  [5, 3, 7, 2, 9, 4]
Mask:   [1, 2, 1] / 4
Output[2] = (5*1 + 3*2 + 7*1) / 4 = 18/4 = 4.5
```

## Applications

- **Audio**: Echo, reverb, filtering
- **Signal Processing**: Smoothing, differentiation
- **Time Series**: Moving averages, trend detection
- **Computer Vision**: Separable filters (first step of 2D)

## Concepts Covered

### 1. **Basic 1D Convolution**
- Kernel application
- Loop structure
- Index calculations

### 2. **Boundary Conditions**
- Zero padding
- Clamp to edge
- Wrap around
- Reduce size (valid convolution)

### 3. **Coalescing in 1D**
- Thread-to-data mapping
- Sequential access patterns
- Output write patterns

### 4. **Correctness Testing**
- CPU reference implementation
- Floating-point comparison
- Edge case handling

## Learning Objectives

By the end of today, you should be able to:
- [ ] Implement basic 1D convolution on GPU
- [ ] Handle multiple boundary condition strategies
- [ ] Ensure coalesced memory access
- [ ] Test with various kernel sizes (3, 5, 7, 11)
- [ ] Validate correctness against CPU implementation
- [ ] Benchmark performance
- [ ] Understand trade-offs of different boundary handling

## Implementation Tasks

### Task 1: CPU Reference Implementation

```cpp
void convolve1D_CPU(float *input, float *output, float *mask, 
                    int length, int maskRadius) {
    for (int i = 0; i < length; i++) {
        float sum = 0.0f;
        for (int j = -maskRadius; j <= maskRadius; j++) {
            int idx = i + j;
            // Zero padding
            if (idx >= 0 && idx < length) {
                sum += input[idx] * mask[j + maskRadius];
            }
        }
        output[i] = sum;
    }
}
```

### Task 2: Basic GPU Kernel (Global Memory)

```cuda
__global__ void convolve1D_basic(float *input, float *output, float *mask,
                                  int length, int maskRadius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < length) {
        float sum = 0.0f;
        
        for (int j = -maskRadius; j <= maskRadius; j++) {
            int inputIdx = idx + j;
            
            // Zero padding boundary condition
            if (inputIdx >= 0 && inputIdx < length) {
                sum += input[inputIdx] * mask[j + maskRadius];
            }
        }
        
        output[idx] = sum;
    }
}
```

**Analysis:**
- Each thread reads `(2*maskRadius + 1)` elements from global memory
- Consecutive threads access consecutive elements ✅ Coalesced!
- But lots of redundant reads (overlapping neighborhoods)

### Task 3: Boundary Condition Variants

**Clamp to Edge:**
```cuda
int inputIdx = min(max(idx + j, 0), length - 1);
sum += input[inputIdx] * mask[j + maskRadius];
```

**Wrap Around (Periodic):**
```cuda
int inputIdx = (idx + j + length) % length;
sum += input[inputIdx] * mask[j + maskRadius];
```

**Valid Convolution (No Padding):**
```cuda
if (idx >= maskRadius && idx < length - maskRadius) {
    // Only compute where full mask fits
    for (int j = -maskRadius; j <= maskRadius; j++) {
        sum += input[idx + j] * mask[j + maskRadius];
    }
    output[idx] = sum;
}
```

### Task 4: Test with Various Mask Sizes

**3-element (Smoothing):**
```cpp
float mask3[3] = {1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f};  // Box filter
```

**5-element (Gaussian-like):**
```cpp
float mask5[5] = {1.0f/16.0f, 4.0f/16.0f, 6.0f/16.0f, 4.0f/16.0f, 1.0f/16.0f};
```

**7-element:**
```cpp
float mask7[7] = {1, 6, 15, 20, 15, 6, 1};  // Binomial (normalize by 64)
```

**11-element (Large kernel):**
```cpp
float mask11[11] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};  // Custom (normalize by 36)
```

## Memory Access Pattern Analysis

For array of length N with mask radius R:

**Global Memory Reads:**
- Each of N threads reads `(2R+1)` elements
- Total reads: `N * (2R+1)`
- But only N unique elements in array!
- **Redundancy factor**: `(2R+1)/1 = 2R+1`

For R=5 (11-element mask): Each element read 11 times on average!

**Tomorrow** we'll use shared memory to fix this redundancy.

## Benchmarking Setup

```cuda
// Timing
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
convolve1D_basic<<<gridSize, blockSize>>>(d_input, d_output, d_mask, length, radius);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float ms = 0;
cudaEventElapsedTime(&ms, start, stop);

// Calculate bandwidth
float bytesAccessed = length * sizeof(float) * (2 * radius + 2);  // reads + write
float bandwidth = bytesAccessed / (ms / 1000.0) / 1e9;  // GB/s
printf("Bandwidth: %.2f GB/s\n", bandwidth);
```

## Test Configurations

| Signal Length | Mask Radius | Block Size | Expected Time |
|---------------|-------------|------------|---------------|
| 1,000         | 1           | 256        | < 0.1 ms      |
| 10,000        | 2           | 256        | < 0.5 ms      |
| 100,000       | 3           | 256        | < 1 ms        |
| 1,000,000     | 5           | 256        | < 5 ms        |

## Correctness Verification

```cpp
bool verify(float *gpu_output, float *cpu_output, int length, float tolerance = 1e-4) {
    for (int i = 0; i < length; i++) {
        float diff = fabs(gpu_output[i] - cpu_output[i]);
        if (diff > tolerance) {
            printf("Mismatch at index %d: GPU=%.6f, CPU=%.6f, diff=%.6f\n",
                   i, gpu_output[i], cpu_output[i], diff);
            return false;
        }
    }
    return true;
}
```

## Deliverables Checklist

- [ ] **CPU Implementation**: Reference for correctness
- [ ] **GPU Basic Kernel**: Global memory version
- [ ] **Boundary Variants**: Zero pad, clamp, wrap, valid
- [ ] **Test Masks**: 3, 5, 7, 11 elements
- [ ] **Timing Code**: Accurate benchmarking
- [ ] **Verification**: GPU vs CPU comparison
- [ ] **Benchmarks**: Time vs length and mask size
- [ ] **Graphs**:
  - Execution time vs signal length
  - Time vs mask radius
- [ ] **Analysis**: Memory access patterns and efficiency

## Expected Results

**Performance Characteristics:**
- Linear scaling with signal length
- Linear scaling with mask size
- Memory-bound (not compute-bound)
- Bandwidth: 20-40% of theoretical peak (due to redundant reads)

## Common Pitfalls

1. **Incorrect indexing**: Off-by-one in mask array
2. **Boundary bugs**: Missing edge cases
3. **Normalization**: Forgetting to normalize mask weights
4. **Integer overflow**: Using `int` for large indices
5. **Synchronization**: Not synchronizing before copying results

## Debugging Tips

1. **Test tiny signals**: 10-20 elements, print full arrays
2. **Hand calculate**: Verify first few output values manually
3. **Boundary focus**: Test signals where boundary matters (small N)
4. **Single mask element**: Use [0, 1, 0] to test identity
5. **Symmetry**: Use symmetric masks to check for indexing errors

## Real-World Examples

### Audio Echo Effect:
```cpp
// Simple echo: mix current sample with delayed sample
float echoMask[201];  // 200ms delay at 1kHz sample rate
for (int i = 0; i < 201; i++) {
    echoMask[i] = (i == 0) ? 0.7f : (i == 200) ? 0.3f : 0.0f;
}
```

### Moving Average:
```cpp
// Smooth out noise with 5-point average
float avgMask[5] = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};
```

## Reflection Questions

1. Why is this kernel memory-bound rather than compute-bound?
2. Which boundary condition is most appropriate for audio signals?
3. How does mask size affect performance?
4. What's the redundancy factor for your largest mask?
5. How would you optimize memory access patterns?

## Advanced Challenges (Optional)

1. **Multi-channel audio**: Stereo or 5.1 surround
2. **Integer signals**: Optimize for int16_t audio samples
3. **Streaming**: Process very long signals in chunks
4. **Asymmetric masks**: Non-centered masks (FIR filters)
5. **Complex convolution**: Complex numbers for frequency domain

## Next Steps

Tomorrow (Day 14), you'll move the mask to constant memory and add shared memory tiling to eliminate redundant global memory reads. Expect 5-10x speedup!

---

**Key Insight**: Even basic kernels benefit from proper boundary handling and correctness testing. Profile, measure, optimize—in that order! 📊
