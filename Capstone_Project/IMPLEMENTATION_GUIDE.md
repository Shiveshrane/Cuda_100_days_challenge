# Project Structure & Implementation Guide

## Quick Start

### Prerequisites
- CUDA Toolkit 11.0+
- CMake 3.10+
- C++14 compatible compiler
- NVIDIA GPU with compute capability 3.5+

### Build Instructions

```bash
# Clone/navigate to project
cd Capstone_Project

# Create build directory
mkdir build && cd build

# Configure and build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Run
./image_processor ../test_images/input.ppm output.ppm --filter blur
```

## File Descriptions

### Main Application Files

#### `main.cu`
Entry point for the image processing pipeline. Handles:
- Command-line argument parsing
- Pipeline orchestration
- Timing and benchmarking
- Error handling

**Key Functions:**
```cuda
int main(int argc, char** argv);
void printUsage();
void parseArguments(int argc, char** argv, ProcessingOptions& opts);
```

#### `image_io.cu / image_io.h`
Image loading and saving utilities. Supports:
- PPM format (P6 binary)
- Memory allocation/deallocation
- Error checking

**Key Functions:**
```cuda
struct Image { int width, height; uint8_t* data; };

Image loadImage(const char* filename);
void saveImage(const char* filename, const Image& img);
void freeImage(Image& img);
Image allocateImage(int width, int height);
void copyImageToDevice(Image& host_img, Image& device_img);
void copyImageFromDevice(Image& device_img, Image& host_img);
```

#### `pipeline.cu / pipeline.h`
Orchestrates the complete processing pipeline. Manages:
- Kernel launch sequencing
- Memory allocation for intermediate results
- Filter composition
- Performance monitoring

**Key Functions:**
```cuda
struct Pipeline { 
    std::vector<std::string> filters;
    Image intermediate_buffers[MAX_FILTERS];
};

Pipeline createPipeline(const std::vector<std::string>& filter_names);
Image processPipeline(Pipeline& pipeline, const Image& input);
void destroyPipeline(Pipeline& pipeline);
```

#### `performance.cu / performance.h`
Performance analysis and benchmarking utilities:
- CUDA event-based timing
- Bandwidth calculation
- Occupancy analysis
- Result validation

**Key Functions:**
```cuda
struct PerformanceMetrics {
    float execution_time_ms;
    float bandwidth_gb_s;
    float occupancy_percent;
};

PerformanceMetrics benchmarkKernel(const char* kernel_name, 
                                   std::function<void()> kernel_launch,
                                   int iterations);
void validateResults(const Image& gpu_result, const Image& cpu_result);
```

### Kernel Files

#### `kernels/blur_kernels.cu`
Gaussian blur implementation using separable convolution.

**Kernels:**
```cuda
// Horizontal blur pass with shared memory halos
__global__ void gaussianBlurHorizontal(
    const uint8_t* input,
    uint8_t* output,
    int width, int height,
    float* kernel, int kernel_size);

// Vertical blur pass with shared memory halos
__global__ void gaussianBlurVertical(
    const uint8_t* input,
    uint8_t* output,
    int width, int height,
    float* kernel, int kernel_size);
```

**Concepts Applied:**
- Day 13-14: 1D convolution with optimization
- Shared memory tiling with halo regions
- Memory coalescing for row-major access
- Bank conflict avoidance

#### `kernels/edge_kernels.cu`
Edge detection using Sobel operator.

**Kernels:**
```cuda
// Sobel edge detection with 2D shared memory tiling
__global__ void sobelEdgeDetection(
    const uint8_t* input,
    uint8_t* output,
    int width, int height);
```

**Concepts Applied:**
- Day 15-16: 2D convolution with shared memory
- Multiple kernel applications simultaneously
- Efficient gradient computation

#### `kernels/histogram_kernels.cu`
Histogram computation and equalization.

**Kernels:**
```cuda
// Compute histogram using atomics
__global__ void computeHistogram(
    const uint8_t* input,
    unsigned int* histogram,
    int width, int height);

// Compute cumulative histogram (CDF)
__global__ void cumulativeHistogram(
    unsigned int* histogram,
    float* cdf,
    int bins);

// Apply histogram equalization
__global__ void applyHistogramEqualization(
    const uint8_t* input,
    uint8_t* output,
    const float* cdf,
    int width, int height);
```

**Concepts Applied:**
- Day 21: Atomic operations for histogram
- Day 17: Prefix scan for cumulative histogram
- Day 22: Scatter operation for output

#### `kernels/transform_kernels.cu`
Image transformations (rotation, transpose).

**Kernels:**
```cuda
// Image rotation using shared memory tiling
__global__ void rotateImage(
    const uint8_t* input,
    uint8_t* output,
    int width, int height,
    float angle_radians);

// Matrix transpose (useful for separable convolution)
__global__ void transposeImage(
    const uint8_t* input,
    uint8_t* output,
    int width, int height);
```

**Concepts Applied:**
- Day 16: 2D shared memory tiling
- Day 007: Transpose optimization
- Handling non-square dimensions

#### `kernels/compact_kernels.cu`
Stream compaction for sparse operations.

**Kernels:**
```cuda
// Mark pixels meeting predicate
__global__ void markPixels(
    const uint8_t* input,
    uint8_t* flags,
    int width, int height,
    uint8_t threshold);

// Scatter marked pixels to compact representation
__global__ void scatterCompact(
    const uint8_t* input,
    const uint8_t* flags,
    const unsigned int* scan,
    uint8_t* output,
    int width, int height);
```

**Concepts Applied:**
- Day 22: Stream compaction pattern
- Day 17: Integration with scan operation
- Efficient memory layout for sparse data

#### `kernels/utility_kernels.cu`
Utility kernels for reductions and scans.

**Kernels:**
```cuda
// Block-level reduction (sum)
__global__ void blockReduce(
    const unsigned int* input,
    unsigned int* output,
    int n);

// Exclusive scan using work-efficient algorithm
__global__ void exclusiveScan(
    const unsigned int* input,
    unsigned int* output,
    int n);
```

**Concepts Applied:**
- Day 18: Parallel reduction patterns
- Day 17: Work-efficient scan algorithm
- Day 25: Synchronization and atomics

## Implementation Strategy

### Phase 1: Foundation
```
main.cu → image_io.cu → Load image → Allocate GPU memory
                                   → Copy to device
```

### Phase 2: Basic Filters
```
Simple kernels → Gaussian Blur (separate H/V passes)
              → Sobel Edge Detection
              → Performance measurement
```

### Phase 3: Advanced Operations
```
Utility kernels → Histogram computation
              → Stream compaction
              → Image transformation
```

### Phase 4: Integration
```
Pipeline orchestration → Compose filters
                     → Memory management
                     → Error handling
```

## Common Patterns in Implementation

### 1. Shared Memory Tiling Pattern (Days 15-16)
```cuda
extern __shared__ uint8_t tile[];
// Load tile + halo
int gx = blockIdx.x * blockDim.x + threadIdx.x;
int gy = blockIdx.y * blockDim.y + threadIdx.y;
int lx = threadIdx.x;
int ly = threadIdx.y;

// Load main tile
tile[ly * blockDim.x + lx] = input[gy * width + gx];

// Load halos if needed
if (lx < HALO_SIZE) {
    tile[ly * blockDim.x + (lx - HALO_SIZE)] = input[gy * width + (gx - HALO_SIZE)];
}
__syncthreads();

// Process using shared memory data
```

### 2. Atomic Histogram Pattern (Day 21)
```cuda
__global__ void computeHistogram(const uint8_t* input, unsigned int* hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint8_t pixel = input[idx];
        atomicAdd(&hist[pixel], 1);
    }
}
```

### 3. Scan + Scatter Pattern (Days 17, 22)
```cuda
// Mark: flag[i] = 1 if predicate(input[i])
// Scan: compute prefix sum of flags
// Scatter: output[scan[i]] = input[i] if flag[i]
```

### 4. Synchronization Pattern (Day 25)
```cuda
__global__ void multiStageKernel(...) {
    // Stage 1: Compute something
    int result = computeStage1();
    __syncthreads();
    
    // Stage 2: Use results from stage 1
    int final = computeStage2(result);
    __syncthreads();
}
```

## Testing Strategy

### Correctness Testing
```cpp
// For each filter:
1. Run on GPU with test image
2. Run on CPU with same image
3. Compare outputs (with floating point tolerance)
4. Print diff statistics
```

### Performance Testing
```cpp
// Benchmark each component:
1. Measure kernel execution time
2. Calculate effective bandwidth
3. Compare against peak theoretical bandwidth
4. Profile with NVIDIA Nsight
```

### Edge Case Testing
```cpp
// Test with:
- Various image sizes (not powers of 2)
- Different kernel sizes
- Boundary pixels
- Maximum/minimum value pixels
```

## Build Configuration (CMakeLists.txt)

```cmake
cmake_minimum_required(VERSION 3.10)
project(ImageProcessor CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CUDA_ARCHITECTURES 60 61 70 75 80)

add_executable(image_processor
    src/main.cu
    src/image_io.cu
    src/pipeline.cu
    src/performance.cu
    src/kernels/blur_kernels.cu
    src/kernels/edge_kernels.cu
    src/kernels/histogram_kernels.cu
    src/kernels/transform_kernels.cu
    src/kernels/compact_kernels.cu
    src/kernels/utility_kernels.cu
)

target_link_libraries(image_processor PUBLIC cuda)
```

## Debugging Tips

### Common Issues

1. **CUDA Out of Memory**
   - Check image dimensions
   - Profile intermediate buffer sizes
   - Use unified memory for testing

2. **Incorrect Results**
   - Verify against CPU reference
   - Check boundary conditions
   - Debug shared memory bank conflicts

3. **Performance Issues**
   - Profile with Nsight Systems
   - Check memory bandwidth utilization
   - Analyze occupancy per SM
   - Verify coalescing patterns

### Debugging Tools
```bash
# Profiling
nsys profile ./image_processor input.ppm output.ppm

# Memory check
cuda-memcheck ./image_processor input.ppm output.ppm

# Generate SASS code
nvcc -arch=sm_75 -keep main.cu
```

## Next Steps After Capstone

1. Extend with video processing
2. Add real-time visualization (OpenGL)
3. Integrate with deep learning inference
4. Optimize for mobile GPUs
5. Multi-GPU implementation
