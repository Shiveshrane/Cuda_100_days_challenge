#include <iostream>
using namespace std;
#include <cuda_runtime.h>

__global__ void vectoradd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 10;
    float A[N], B[N], C[N];

    float *d_a, *d_b, *d_c;

    // Initialize vectors on host (CPU)
    for (int i = 0; i < N; i++) {
        A[i] = float(i);
        B[i] = float(i * i);
    }

    // Allocate memory on device (GPU)
    cudaError_t err;
    err = cudaMalloc(&d_a, N * sizeof(float));
    if (err != cudaSuccess) {
        cerr << "Error allocating d_a: " << cudaGetErrorString(err) << endl;
        return 1;
    }
    err = cudaMalloc(&d_b, N * sizeof(float));
    if (err != cudaSuccess) {
        cerr << "Error allocating d_b: " << cudaGetErrorString(err) << endl;
        cudaFree(d_a);
        return 1;
    }
    err = cudaMalloc(&d_c, N * sizeof(float));
    if (err != cudaSuccess) {
        cerr << "Error allocating d_c: " << cudaGetErrorString(err) << endl;
        cudaFree(d_a);
        cudaFree(d_b);
        return 1;
    }

    // Transfer data from host to device (GPU)
    cudaMemcpy(d_a, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    // Fix: Use proper integer ceiling division
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch kernel
    vectoradd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    // Wait for GPU to finish and check for errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cerr << "Kernel execution failed: " << cudaGetErrorString(err) << endl;
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return 1;
    }

    // Transfer result from device to host
    cudaMemcpy(C, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Display results
    cout << "Result (C = A + B):" << endl;
    for (int i = 0; i < N; i++) {
        cout << C[i] << " ";
    }
    cout << endl;

    return 0;
}