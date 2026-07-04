# Day 015: 2D Convolution - Basics

**Chapter 8: 2D Convolution Introduction**

## Overview
Extend convolution concepts to two dimensions. 2D convolution is fundamental in image processing, convolutional neural networks (CNNs), and computer vision applications. Understand how to map 2D data to 1D thread indexing and manage 2D boundary conditions.

## What is 2D Convolution?

Apply a 2D filter (kernel/mask) to a 2D image:

```
Output[i,j] = Σ Σ Input[i+u,j+v] * Kernel[u,v]
              u  v
```

**Example**: 3×3 kernel applied to image region

```
Input Region:      Kernel:        Output Element:
[1  2  3]          [1  0  -1]     computed from
[4  5  6]    *     [2  0  -2]  =  weighted sum
[7  8  9]          [1  0  -1]
```

## Applications

- **Image Filtering**: Blur, sharpen, edge detection, Sobel, Laplacian
- **Computer Vision**: Feature extraction, template matching
- **Deep Learning**: First layer of CNNs
- **Image Enhancement**: Denoising, super-resolution

## Key Concepts

### 1. **2D Thread Indexing**
- Mapping 2D threads to 2D image data
- Calculating 2D indices from 1D thread ID
- Block and grid dimensions in 2D

### 2. **Boundary Conditions**
- Zero padding
- Clamp to edge
- Wrap around
- Reduce output size

### 3. **Kernel Layout**
- Row-major storage
- Kernel flipping in convolution
- Constant memory kernel storage

### 4. **Global Memory Access Patterns**
- Row-major access coalescing
- Cache line utilization
- Memory transaction efficiency

## Implementation Considerations

- Input image representation (linear memory)
- Kernel storage strategy
- Output buffer allocation
- Thread block organization (e.g., 16×16, 32×8)

## Performance Baseline

- Naive global memory implementation
- Memory bandwidth analysis
- Occupancy calculation
