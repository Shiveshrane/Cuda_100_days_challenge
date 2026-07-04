#include <iostream>
#include <cuda_runtime.h>
 
#define TILE_SIZE 32
using namespace std;


__global__ void MatTransposeKernel(const float *input, float *output, int width, int height){
    __shared__ float tile[TILE_SIZE][TILE_SIZE+1];
    int tx=blockIdx.x*blockDim.x + threadIdx.x;
    int ty=blockIdx.y*blockDim.y + threadIdx.y;

    if(tx<width && ty<height){
        tile[threadIdx.y][threadIdx.x]=input[ty*width + tx];
    }
    __syncthreads();

    tx=blockIdx.y*blockDim.x + threadIdx.x;
    ty=blockIdx.x*blockDim.y + threadIdx.y;

    if (tx<height && ty<width){
        output[ty*height + tx]=tile[threadIdx.x][threadIdx.y];
    }
}


int main(){
    int width, height;
    cout<<"Enter matrix width and height: ";
    cin>>width>>height;

    size_t size=width*height*sizeof(float);

    float *h_input=(float*)malloc(size);
    float *h_output=(float*)malloc(size);

    for(int i=0; i<width*height; i++){
        h_input[i]=float(i);
    }
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((width+TILE_SIZE-1)/TILE_SIZE, (height+TILE_SIZE-1)/TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    MatTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout<<"Time taken for matrix transpose: "<<milliseconds<<" ms"<<endl;
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    return 0;

}