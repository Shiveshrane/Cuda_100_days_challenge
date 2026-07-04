#include <cuda_runtime.h>
#include <iostream>
#define TILE_DIM 32
using namespace std;

__global__ void XORTransposeKernel(const float *input, float *output, int width, int height){
    __shared__ float SharedTile[TILE_DIM][TILE_DIM];
    int x=blockDim.x*blockIdx.x + threadIdx.x;
    int y=blockDim.y*blockIdx.y + threadIdx.y;

    if (x<width && y<height){
        int index_in=y*width + x;
        SharedTile[threadIdx.y][threadIdx.x ^ threadIdx.y]=input[index_in]; //XOR on store to avoid bank conflicts
    }
    __syncthreads();

    x=blockDim.x*blockIdx.y + threadIdx.x; //Swap blockIdx for transpose
    y=blockDim.y*blockIdx.x + threadIdx.y;

    if (x<height && y<width){
        int index_out=y*height + x;
        output[index_out]=SharedTile[threadIdx.x][threadIdx.y ^ threadIdx.x]; //XOR on read to retrieve correct value
    }



}

int main(){
    int width=1024;
    int height=1024; 
    size_t size=width*height*sizeof(float);
    float *h_input=(float*)malloc(size);
    float *h_output=(float*)malloc(size);

    for (int i=0; i<width*height; i++){
        h_input[i]=static_cast<float>(i);
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    

    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    XORTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds=0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout<<"Time taken for XOR Transpose: "<<milliseconds<<" ms"<<endl;

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    return 0;

}