# Day 12: Week 1 Integration Project - Optimized Convolution

**Memory Optimization Capstone**

## Overview
This is where everything comes together! You'll implement a 2D convolution kernel that incorporates ALL the optimization techniques from Week 1: memory coalescing, shared memory tiling, bank conflict elimination, and proper occupancy management.

## Project Goal

Build a production-quality 2D convolution kernel that:
- ✅ Achieves coalesced memory access
- ✅ Uses shared memory with tiling
- ✅ Has zero bank conflicts
- ✅ Maintains good occupancy
- ✅ Handles boundary conditions correctly
- ✅ Outperforms naive implementation by 10-30x

## What is 2D Convolution?

Convolution applies a small kernel (filter) to an image:

```
Output[y][x] = Σ Σ Input[y+j][x+i] * Kernel[j][i]
               j i
```

Common applications:
- **Blur filters**: Averaging neighboring pixels
- **Edge detection**: Sobel, Laplacian operators
- **Sharpening**: Enhance edges
- **CNN layers**: Deep learning convolution

## Concepts Applied

### 1. **Memory Coalescing** (Day 6-7)
- Input image reads must be coalesced
- Output writes must be coalesced
- Struct layout for filter coefficients

### 2. **Shared Memory Tiling** (Day 5, 8-9)
- Load image tiles with halos (ghost cells)
- Avoid redundant global memory loads
- Pad shared memory to prevent bank conflicts

### 3. **Occupancy Optimization** (Day 10-11)
- Choose tile size for good occupancy
- Balance shared memory vs blocks per SM
- Minimize register usage

### 4. **Boundary Handling**
- Zero padding
- Clamp to edge
- Proper halo cell loading

## Learning Objectives

By the end of today, you should be able to:
- [ ] Implement multi-stage convolution kernel
- [ ] Load input tiles with halo regions into shared memory
- [ ] Apply filter coefficients efficiently
- [ ] Handle image boundaries correctly
- [ ] Optimize for coalescing, bank conflicts, and occupancy
- [ ] Benchmark against naive implementation
- [ ] Profile and analyze bottlenecks
- [ ] Create visual outputs showing filter effects

## Implementation Phases

### Phase 1: Naive Convolution (Baseline)

```cuda
__global__ void convolution2D_naive(float *input, float *output, 
                                     float *kernel, 
                                     int width, int height, 
                                     int kernelRadius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        
        for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
            for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
                int ix = x + kx;
                int iy = y + ky;
                
                // Boundary check (zero padding)
                if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                    int inputIdx = iy * width + ix;
                    int kernelIdx = (ky + kernelRadius) * (2 * kernelRadius + 1) + (kx + kernelRadius);
                    sum += input[inputIdx] * kernel[kernelIdx];
                }
            }
        }
        
        output[y * width + x] = sum;
    }
}
```

### Phase 2: Constant Memory for Kernel

```cuda
#define MAX_KERNEL_SIZE 11
__constant__ float d_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

// Copy kernel to constant memory on host:
cudaMemcpyToSymbol(d_kernel, h_kernel, kernelSize * kernelSize * sizeof(float));
```

Why? Constant memory is:
- Cached
- Broadcast efficiently
- Perfect for filter coefficients

### Phase 3: Shared Memory Tiling

```cuda
#define TILE_SIZE 32
#define KERNEL_RADIUS 2
#define BLOCK_SIZE (TILE_SIZE + 2 * KERNEL_RADIUS)  // Include halo

__global__ void convolution2D_tiled(float *input, float *output,
                                    int width, int height) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1];  // +1 for padding!
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;
    
    // Load tile with halo (each thread may load multiple elements)
    // Main tile
    int inputX = blockIdx.x * TILE_SIZE + tx - KERNEL_RADIUS;
    int inputY = blockIdx.y * TILE_SIZE + ty - KERNEL_RADIUS;
    
    if (inputX >= 0 && inputX < width && inputY >= 0 && inputY < height) {
        tile[ty][tx] = input[inputY * width + inputX];
    } else {
        tile[ty][tx] = 0.0f;  // Zero padding
    }
    
    __syncthreads();
    
    // Compute convolution using shared memory
    if (x < width && y < height && tx < TILE_SIZE && ty < TILE_SIZE) {
        float sum = 0.0f;
        for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ky++) {
            for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; kx++) {
                int tileX = tx + kx + KERNEL_RADIUS;
                int tileY = ty + ky + KERNEL_RADIUS;
                int kernelIdx = (ky + KERNEL_RADIUS) * (2 * KERNEL_RADIUS + 1) + 
                                (kx + KERNEL_RADIUS);
                sum += tile[tileY][tileX] * d_kernel[kernelIdx];
            }
        }
        output[y * width + x] = sum;
    }
}
```

**Key Optimization**: `BLOCK_SIZE + 1` padding to prevent bank conflicts!

### Phase 4: Advanced Halo Loading

Each thread loads multiple pixels to fill halo:

```cuda
// Load multiple elements to cover halo region
for (int i = ty; i < BLOCK_SIZE; i += blockDim.y) {
    for (int j = tx; j < BLOCK_SIZE; j += blockDim.x) {
        int inputX = blockIdx.x * TILE_SIZE + j - KERNEL_RADIUS;
        int inputY = blockIdx.y * TILE_SIZE + i - KERNEL_RADIUS;
        
        if (inputX >= 0 && inputX < width && inputY >= 0 && inputY < height) {
            tile[i][j] = input[inputY * width + inputX];
        } else {
            tile[i][j] = 0.0f;
        }
    }
}
```

## Test Kernels

### Gaussian Blur (3×3)
```
1  2  1
2  4  2  ×  1/16
1  2  1
```

### Box Blur (3×3)
```
1  1  1
1  1  1  ×  1/9
1  1  1
```

### Sobel X (Edge Detection)
```
-1  0  1
-2  0  2
-1  0  1
```

### Sharpening
```
 0  -1   0
-1   5  -1
 0  -1   0
```

## Benchmarking Plan

### Test Configurations:
- **Image sizes**: 512×512, 1024×1024, 2048×2048, 4096×4096
- **Kernel sizes**: 3×3, 5×5, 7×7, 11×11
- **Variants**: Naive, constant memory, tiled, fully optimized

### Metrics to Measure:
1. **Execution time** (milliseconds)
2. **Speedup** vs naive
3. **Effective bandwidth** (GB/s)
4. **GFLOPS**
5. **Occupancy** (from profiling)
6. **Bank conflicts** (should be 0!)

### Create Performance Graphs:
1. Time vs image size (for each kernel variant)
2. Speedup vs kernel size
3. Bandwidth utilization comparison
4. Occupancy across configurations

## Profiling Checklist

Use Nsight Compute to verify:

```bash
# Memory coalescing
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./convolution

# Bank conflicts (should be 0!)
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./convolution

# Occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./convolution

# Shared memory efficiency
ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_shared.pct ./convolution
```

## Deliverables Checklist

- [ ] **Code**: All 4 kernel variants (naive, constant, tiled, optimized)
- [ ] **Host Code**: Timing harness and correctness checking
- [ ] **Test Suite**: Multiple filter types
- [ ] **Visual Output**: Before/after images for each filter
- [ ] **Benchmarks**: Comprehensive performance data
- [ ] **Graphs**: 
  - Execution time vs image size
  - Speedup comparison
  - Bandwidth utilization
- [ ] **Profiling Reports**: Nsight Compute analysis
- [ ] **Detailed Report**:
  - Explanation of each optimization
  - Performance impact of each technique
  - Analysis of bottlenecks
  - Comparison with expectations
  - Lessons learned

## Success Criteria

- [ ] Correctness: Output matches CPU/naive version
- [ ] Speedup: At least 10x over naive for large images
- [ ] Zero bank conflicts in shared memory
- [ ] Coalesced global memory access
- [ ] Occupancy > 50%
- [ ] Code is well-documented
- [ ] Professional-quality report

## Image I/O (Use STB or similar)

```cpp
// stb_image.h and stb_image_write.h for image loading/saving
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Load image
int width, height, channels;
unsigned char* image = stbi_load("input.png", &width, &height, &channels, 1);

// Convert to float
float *h_input = new float[width * height];
for (int i = 0; i < width * height; i++) {
    h_input[i] = image[i] / 255.0f;
}

// After convolution, convert back and save
unsigned char *output_img = new unsigned char[width * height];
for (int i = 0; i < width * height; i++) {
    output_img[i] = (unsigned char)(h_output[i] * 255.0f);
}
stbi_write_png("output.png", width, height, 1, output_img, width);
```

## Debugging Tips

1. **Start small**: Test on tiny images (16×16) first
2. **Visualize**: Print tile loading pattern
3. **Check boundaries**: Verify halo loading is correct
4. **Compare pixel-by-pixel**: CPU vs GPU outputs
5. **Use cuda-memcheck**: Detect out-of-bounds access

## Common Pitfalls

1. **Forgetting `__syncthreads()`** after loading tile
2. **Incorrect halo region indexing**
3. **Bank conflicts from unpaded shared memory**
4. **Boundary condition bugs** (off-by-one errors)
5. **Thread vs block index confusion**

## Reflection Questions

1. Which optimization provided the biggest speedup?
2. How does kernel size affect the benefit of tiling?
3. Why does padding help with bank conflicts in this kernel?
4. What's the optimal tile size for your GPU?
5. How close did you get to theoretical bandwidth?
6. What would you do differently if starting over?

## Advanced Challenges (Optional)

1. **Separable Filters**: Decompose 2D filter into two 1D passes
2. **Multiple Channels**: RGB images instead of grayscale
3. **Dynamic Kernel Size**: Support any kernel size at runtime
4. **Fused Operations**: Convolution + activation function
5. **Multi-Stream**: Process multiple images concurrently
6. **Compare with cuDNN**: `cudnnConvolutionForward()`

## Week 1 Reflection

Take time to write about:
- Most surprising insight from this week
- Most challenging concept to understand
- Most impactful optimization technique
- Confidence level with CUDA fundamentals
- Areas that need more practice

## Next Steps

Week 2 starts tomorrow with advanced tiling techniques for convolution! You'll dive deeper into 1D/2D convolution optimizations and learn about constant memory strategies.

---

**Congratulations on completing Week 1!** You now have a solid foundation in GPU memory optimization. These principles will apply to EVERY kernel you write. 🎉

**Pro Tip**: Keep this convolution kernel as a reference implementation. It demonstrates best practices you'll use repeatedly! 💎
