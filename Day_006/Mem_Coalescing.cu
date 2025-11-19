#include <cuda.h>
#include <iostream>
#define BLOCK_SIZE 32

#define SHARED_MEM_SIZE BLOCK_SIZE * 32

using namespace std;

__global__ void mem_strided_kernel(const int *Input, int *Output, int N, int stride){
    __shared__ int shared_data[SHARED_MEM_SIZE];
    long long int op=0;

    unsigned int tid=threadIdx.x;
    unsigned int i=blockIdx.x*blockDim.x + tid;
        shared_data[tid]=Input[i*stride];
        __syncthreads();
        for (unsigned int j=1; j<blockDim.x;j*=2){
            shared_data[tid] += shared_data[tid + j];
            __syncthreads();

        }

        if (tid==0){
            Output[blockIdx.x]=shared_data[0];
        }

}



__global__ void mem_seq_kernel(const int *Input, int *Output, int N){
    __shared__ int shared_data[SHARED_MEM_SIZE];

    unsigned int tid=threadIdx.x;
    unsigned int i=blockIdx.x*blockDim.x + tid;

    shared_data[tid]=Input[i];
    __syncthreads();
    for (unsigned int j=1;j<blockDim.x;j*=2){
        shared_data[tid] += shared_data[tid + j];
        __syncthreads();
    }
    if (tid==0){
        Output[blockIdx.x]=shared_data[0];
    }
    
}

__global__ void mem_rev_kernel(const int *Input, int *Output, int N){
    __shared__ int shared_data[SHARED_MEM_SIZE];

    unsigned int tid=threadIdx.x;
    unsigned int i=blockIdx.x*blockDim.x + tid;

    int rev_idx=BLOCK_SIZE - tid - 1;

    shared_data[tid]=Input[blockIdx.x * blockDim.x + rev_idx];
    __syncthreads();
    for (unsigned int j=1;j<blockDim.x;j*=2){
        shared_data[tid] += shared_data[tid + j];
        __syncthreads();
    }
    if (tid==0){
        Output[blockIdx.x]=shared_data[0];
    }
}


int main(){
    const int N=1024;

    int *h_Input, *h_Output;

    int *d_Input, *d_Output;

    h_Input=(int *)malloc(N*sizeof(int));
    h_Output=(int *)malloc((N/BLOCK_SIZE)*sizeof(int));

    for (int i=0;i<N;i++){
        h_Input[i]=1;
    }
    cudaMalloc((void **)&d_Input, N*sizeof(int));
    cudaMalloc((void **)&d_Output, (N/BLOCK_SIZE)*sizeof(int));

    cudaMemcpy(d_Input, h_Input, N*sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(N/BLOCK_SIZE);

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    mem_strided_kernel<<<dimGrid, dimBlock>>>(d_Input, d_Output, N, 32);

    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start1, stop1);
    cout<<"Strided Access Time: "<<milliseconds1<<" ms"<<endl;

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);
    mem_seq_kernel<<<dimGrid, dimBlock>>>(d_Input, d_Output, N);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds2, start2, stop2);
    cout<<"Sequential Access Time: "<<milliseconds2<<" ms"<<endl;



    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);

    cudaEventRecord(start3);    
    mem_rev_kernel<<<dimGrid, dimBlock>>>(d_Input, d_Output, N);
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);
    float milliseconds3 = 0;
    cudaEventElapsedTime(&milliseconds3, start3, stop3);

    cout<<"Reversed Access Time: "<<milliseconds3<<" ms"<<endl;
    cudaFree(d_Input);
    cudaFree(d_Output);
    free(h_Input);
    free(h_Output);
    return 0;
}