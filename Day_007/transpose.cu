#include <iostream>
#include <cuda_runtime.h>
#define TILE_SIZE 32

using namespace std;

__global__ void naiveTranspose(const float *Input, float *Output, int width, int height){
    // Width is the number of elements in a row, while height is the number of element in column. 
    int tx=blockIdx.x * blockDim.x + threadIdx.x;
    int ty=blockIdx.y * blockDim.y + threadIdx.y;

    if (tx<width && ty<height){
        int input_idx=ty*width+tx; 
        int output_idx=tx*height+ty;

        Output[output_idx]=Input[input_idx];
    }
}

__global__ void SharedMemTranspose(const float *Input, float *Output, int width, int height){

    int tx=blockIdx.x*blockDim.x+threadIdx.x;
    int ty=blockIdx.y*blockDim.y+threadIdx.y;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE+1]; // +1 to avoid bank conflicts. 

    
    // columns will map to the same bank when we read column-wise.
    
    if (tx<width && ty<height){
        int input_idx=ty*width + tx;
        tileA[threadIdx.y][threadIdx.x]=Input[input_idx];
    }

    __syncthreads();

    int tx2=blockIdx.y*blockDim.x + threadIdx.x;
    int ty2=blockIdx.x*blockDim.y + threadIdx.y;
    if (ty2<width && tx2<height){
        int output_idx=ty2*height + tx2;
        Output[output_idx]=tileA[threadIdx.x][threadIdx.y];
    }

}


int main(){
    cout<<"Transpose using Shared Memory"<<endl;
    cout<< "Add height and weight"<<endl;
    int width, height;
    cin>>width>>height;

    float *h_Input, *h_Output;
    size_t size=width*height*sizeof(float);
    h_Input=(float *)malloc(size);
    h_Output=(float *)malloc(size);

    for (int i=0;i<width*height;i++){
        h_Input[i]=(float)i;
    }

    float *d_Input, *d_Output;
    cudaMalloc((void **)&d_Input, size);
    cudaMalloc((void **)&d_Output, size);


    cudaMemcpy(d_Input, h_Input, size, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((width+TILE_SIZE-1)/TILE_SIZE, (height+TILE_SIZE-1)/TILE_SIZE);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Naive
    cudaEventRecord(start);
    naiveTranspose<<<gridSize, blockSize>>>(d_Input, d_Output, width, height);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float msNaive=0;
    cudaEventElapsedTime(&msNaive, start, stop);

    cout<<"Time taken by Naive Transpose: "<<msNaive<<" ms"<<endl;

    //Shared Memory
    cudaEventRecord(start);
    SharedMemTranspose<<<gridSize, blockSize>>>(d_Input, d_Output, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msShared=0;
    cudaEventElapsedTime(&msShared, start, stop);
    cout<<"Time taken by Shared Memory Transpose: "<<msShared<<" ms"<<endl;
    cudaMemcpy(h_Output, d_Output, size, cudaMemcpyDeviceToHost);

    cout<<"Input Matrix: "<<endl;
    for (int i=0;i<height;i++){
        for (int j=0;j<width;j++){
            cout<<h_Input[i*width + j]<<" ";
        }
        cout<<endl;
    }

    cout<<"Output Matrix: "<<endl;
    for (int i=0;i<width;i++){
        for (int j=0;j<height;j++){
            cout<<h_Output[i*height + j]<<" ";
        }
        cout<<endl;
    }
    //free memory

    cudaFree(d_Input);
    cudaFree(d_Output);
    free(h_Input);
    free(h_Output);
    return 0;
}