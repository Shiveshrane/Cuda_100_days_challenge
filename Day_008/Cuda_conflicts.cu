#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 32
using namespace std;


__global__ void noconflictK(float *output, int N){
    __shared__ float tile[TILE_SIZE];
    int tx=blockIdx.x*blockDim.x + threadIdx.x;
    if (tx<N){
    tile[threadIdx.x]=tx*0.5f;
    }
    __syncthreads();
    if (tx<N){
        output[tx]=tile[threadIdx.x];
    }
}


__global__ void ConflictK(float *output, int stride, int N){
    extern __shared__ float tile[];
    int tx=blockIdx.x*blockDim.x + threadIdx.x;
    if (tx<N){
        tile[threadIdx.x*stride]=tx*0.5f;
    }
    __syncthreads();

    if (tx<N){
        output[tx]=tile[threadIdx.x*stride];
    }
    
}


__global__ void MatTransposeConflict(float *output, float *input, int rows, int cols){
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int tx=blockIdx.x*blockDim.x + threadIdx.x;
    int ty=blockIdx.y*blockDim.y+threadIdx.y;


    if (tx<cols && ty<rows){
        tile[threadIdx.y][threadIdx.x]=input[ty*cols+tx];
    }
    __syncthreads();


    int tx2=blockIdx.y*blockDim.x+threadIdx.x;
    int ty2=blockIdx.x*blockDim.y+threadIdx.y;
    if (ty2<cols && tx2<rows){
       output[ty2*rows+tx2]=tile[threadIdx.x][threadIdx.y];
    }
}


__global__ void broadcastKernel(float *output){
    __shared__ float tile[TILE_SIZE];
    if (threadIdx.x==0){
        tile[0]=30.0f;
    }
    __syncthreads();
    output[threadIdx.x]=tile[0];
}


int main(){
    cout<<"Transpose using Shared Memory"<<endl;
    cout<< "Add row size and colsize"<<endl;
    int rows, cols;
    cin>>rows>>cols;
    int N=rows*cols;

    float *h_mat_input, *h_mat_output;

    size_t mat_size=rows*cols*sizeof(float);
    h_mat_input=(float *)malloc(mat_size);
    h_mat_output=(float *)malloc(mat_size);
    float *d_mat_input, *d_mat_output;
    cudaMalloc((void **)&d_mat_input, mat_size);
    cudaMalloc((void **)&d_mat_output, mat_size);
    for (int i=0; i<rows; i++){
        for (int j=0; j<cols; j++){
            h_mat_input[i*cols + j]=i*cols + j;
        }
    }


    float *h_array_output;
    h_array_output=(float *)malloc(N*sizeof(float));
    float *d_array_output;
    cudaMalloc((void **)&d_array_output, N*sizeof(float));

    for (int i=0;i<N;i++){
        h_array_output[i]=float(i);
    }

    cudaMemcpy(d_mat_input, h_mat_input, mat_size, cudaMemcpyHostToDevice);
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((cols + TILE_SIZE -1)/TILE_SIZE, (rows + TILE_SIZE -1)/TILE_SIZE);
    dim3 gridSize1D((N + TILE_SIZE -1)/TILE_SIZE);
    dim3 blockSize1D(TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //warmup
    noconflictK<<<1, TILE_SIZE>>>(d_array_output, N);
    cudaDeviceSynchronize();
    //No conflict kernel launch

    cudaEventRecord(start);
    noconflictK<<<gridSize1D, blockSize1D>>>(d_array_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds=0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout<<"No conflict kernel time: "<<milliseconds<<" ms"<<endl;

    //Conflict kernel launch
    int stride=32;
    size_t sharedMemSize=stride*TILE_SIZE*sizeof(float);
    cudaEventRecord(start);
    ConflictK<<<gridSize1D, blockSize1D, sharedMemSize>>>(d_array_output, stride, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    milliseconds=0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout<<"Conflict kernel time: "<<milliseconds<<" ms"<<endl;

    //Matrix Transpose kernel launch
    cudaEventRecord(start);
    MatTransposeConflict<<<gridSize, blockSize>>>(d_mat_output, d_mat_input, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds=0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout<<"Matrix Transpose kernel time: "<<milliseconds<<" ms"<<endl;
    
    //Broadcast kernel launch
    cudaEventRecord(start);
    broadcastKernel<<<1, TILE_SIZE>>>(d_array_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds=0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout<<"Broadcast kernel time: "<<milliseconds<<" ms"<<endl;
    cudaFree(d_mat_input);
    cudaFree(d_mat_output);
    cudaFree(d_array_output);
    free(h_mat_input);
    free(h_mat_output);
    free(h_array_output);
    return 0;
}