#include <iostream>
#include <cuda.h>

#define TILE_WIDTH 16


using namespace std;



__global__ void shared_matmul_kernel(const float *A, const float *B, float *C, int K, int M, int N ){
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int row=blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col=blockIdx.x * TILE_WIDTH + threadIdx.x;


    float val=0.0f;

    int numTiles=(K+TILE_WIDTH-1)/TILE_WIDTH; // We need to make sure that the tiles are enough to cover K, so we take ceiling of K/TILE_WIDTH to get the required number of tiles. 
    for (int i=0;i<numTiles; i++){
        int tileCol=i*TILE_WIDTH + threadIdx.x; // This is to traverse the columns of A for the current tile, and load the appropriate element into shared memory. We here do for tileCol instead of tileRow because we are loading row-wise elements of A into shared memory.
        if ((row<M) && (tileCol<K)){
            tile_A[threadIdx.y][threadIdx.x]=A[row*K+tileCol];
        }
        else{
            tile_A[threadIdx.y][threadIdx.x]=0.0f;
        }
        int tilerow=i*TILE_WIDTH+threadIdx.y; // This is to traverse the rows of B for the current tile, and load the appropriate element into shared memory. We here do for tileRow instead of tileCol because we are loading column-wise elements of B into shared memory.
        if ((tilerow<K) &&(col<N)){
            tile_B[threadIdx.y][threadIdx.x]=B[tilerow*N+col];
        }
        else{
            tile_B[threadIdx.y][threadIdx.x]=0.0f;
        }
        __syncthreads();


        for (int j=0; j<TILE_WIDTH; j++){
            val+=tile_A[threadIdx.y][j]*tile_B[j][threadIdx.x];
        }
        __syncthreads();
    }

    if ((row<M) && (col<N)){
        C[row*N+col]=val;
    }

}




int main(){
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        cout << "No CUDA devices found!" << endl;
        return -1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "Using GPU: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    
    int M,N,K;
    cout<<"Enter dimensions M N K: ";
    cin>>M>>N>>K;

    size_t size_A=M*K*sizeof(float);
    size_t size_B=K*N*sizeof(float);
    size_t size_C=M*N*sizeof(float);

    float *h_A=(float *)malloc(size_A);
    float *h_B=(float *)malloc(size_B);
    float *h_C=(float *)malloc(size_C);
    
    
    for (int i=0;i<M*K;i++){
        h_A[i]=i;
    }

    for (int i=0;i<K*N;i++){
        h_B[i]=1.0f;
    }



    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);


    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "CUDA Error after memory operations: " << cudaGetErrorString(err) << endl;
        return -1;
    }


    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

    dim3 gridDim((N+TILE_WIDTH-1)/TILE_WIDTH, ((M+TILE_WIDTH-1)/TILE_WIDTH));

    // BENCHMARK CODE

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    const int NUM_RUNS=10;
    float total_time=0.0f;

    for (int run=0; run<NUM_RUNS; run++){
        cudaEventRecord(start);
        shared_matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, K, M, N);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cout << "Kernel launch failed: " << cudaGetErrorString(err) << endl;
            return -1;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds=0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time+=milliseconds;
    }
    float avg_time=total_time/NUM_RUNS;

    cout<<"Average Kernel Execution Time over "<<NUM_RUNS<<" runs: "<<avg_time<<" ms"<<endl;
    // END BENCHMARK CODE


    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        cout << "CUDA Error after device to host copy: " << cudaGetErrorString(err2) << endl;
        return -1;
    }

    //performance
    double num_ops=2.0*(double)M*(double)N*(double)K;
    double gflops=(num_ops/(avg_time/1000.0f))/1e9;

    //bandwidth
    size_t total_bytes=size_A + size_B + size_C;

    double bandwidth= ((double)total_bytes/(avg_time/1000.0f))/1e9;
    cout << "\n=== Benchmark Results ===" << endl;
    cout << "Matrix dimensions: " << M << " × " << K << " × " << N << endl;
    cout << "Grid: (" << gridDim.x << ", " << gridDim.y << ")" << endl;
    cout << "Block: (" << blockDim.x << ", " << blockDim.y << ")" << endl;
    cout << "Average time: " << avg_time << " ms" << endl;
    cout << "Performance: " << gflops << " GFLOPS" << endl;
    cout << "Bandwidth: " << bandwidth << " GB/s" << endl;


    for (int i=0;i<min(M,10);i++){
        for (int j=0;j<min(N,10);j++){
            cout<<h_C[i*N+j]<<" ";
        }
        cout<<endl;
    }
    return 0;
}