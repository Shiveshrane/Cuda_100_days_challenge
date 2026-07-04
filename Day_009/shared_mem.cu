#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void SharedMemKernelConf(const float *input, float *output, int size){
    __shared__ float sharedData[256];

    int index=blockDim.x*blockIdx.x + threadIdx.x;
    sharedData[threadIdx.x]= (index < size) ? input[index] : 0.0f;
    __syncthreads();

    for (int s=1; s<blockDim.x; s*=2){
        if (threadIdx.x%(2*s)==0){
            sharedData[threadIdx.x] += sharedData[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x==0){
        output[blockIdx.x]=sharedData[0];
    }
}


__global__ void SharedMemKernelNoConf(const float *input, float *output, int size){
    __shared__ float sharedData[256];

    int tx=blockDim.x*blockIdx.x + threadIdx.x;
    sharedData[threadIdx.x]= (tx < size) ? input[tx] : 0.0f;
    __syncthreads();


    for (int s=blockDim.x/2;s>0;s>>=1){ // We start from half the block size and halve s each iteration, by doing bitwise right shift.
        if (threadIdx.x<s){
            sharedData[threadIdx.x] += sharedData[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x==0){
        output[blockIdx.x]=sharedData[0];
    }
}


int main(){
    int size;
    cout<<"Enter size of array: ";
    cin>>size;

    float *h_input = new float[size];
    float *h_output= new float[size];

    for (int i=0;i<size;i++){
        h_input[i]=float(i);
    }

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size*sizeof(float));
    cudaMalloc((void**)&d_output, size*sizeof(float));

    cudaMemcpy(d_input, h_input, size*sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(256);
    dim3 grid((size + block.x - 1)/block.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    SharedMemKernelConf<<<grid, block>>>(d_input, d_output, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float millisecondsConf = 0;
    cudaEventElapsedTime(&millisecondsConf, start, stop);
    cout<<"Time with configuration (ms): "<<millisecondsConf<<endl;

    cudaEventRecord(start);
    SharedMemKernelNoConf<<<grid, block>>>(d_input, d_output, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float millisecondsNoConf = 0;
    cudaEventElapsedTime(&millisecondsNoConf, start, stop);
    cout<<"Time without configuration (ms): "<<millisecondsNoConf<<endl;

    cudaMemcpy(h_output, d_output, grid.x*sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    return 0;
}