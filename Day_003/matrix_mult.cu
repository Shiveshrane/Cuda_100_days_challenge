#include<iostream>
#include<cuda_runtime.h>
#define TILE_WIDTH 16
using namespace std;


__global__ void matrix_mult_without_tiling(const float *A, const float *B, float *C, int N){
    int row_idx=blockDim.x*blockIdx.x+threadIdx.x;
    int col_idx=blockDim.y*blockIdx.y+threadIdx.y;

    if (row_idx<N && col_idx<N){
        float sum=0.0;
        for (int i=0; i<N; i++){
            sum+=A[row_idx*N+i]*B[i*N+col_idx]; 
        }
        C[row_idx*N+col_idx]=sum;
    }
}

__global__ void matrix_mult_with_tiling(const float *A, const float *B, float *C, int N){
    int row_idx=blockIdx.x*TILE_WIDTH+threadIdx.x;
    int col_idx=blockIdx.y*TILE_WIDTH+threadIdx.y;

    if (row_idx<N && col_idx<N){
        float sum=0.0;
        for (int t=0; t<N; ++t){
            sum+=A[row_idx*TILE_WIDTH+t]*B[t*TILE_WIDTH+col_idx];
        }
        C[row_idx*N+col_idx]=sum;
    }
}



int main(){
    float *d_A, *d_B, *d_C;
    int N=16;
    float *A=new float[N*N];
    float *B=new float[N*N];
    float *C=new float[N*N];

    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            A[i*N+j]=float(i);
            B[i*N+j]=float(j);
        }
    }


    cudaMalloc(&d_A, sizeof(float)*N*N);
    cudaMalloc(&d_B, sizeof(float)*N*N);
    cudaMalloc(&d_C, sizeof(float)*N*N);
    
    cudaMemcpy(d_A, A, sizeof(float)*N*N, cudaMemcpyHostToDevice);  
    cudaMemcpy(d_B, B, sizeof(float)*N*N, cudaMemcpyHostToDevice);

    dim3 dimGrid(N/TILE_WIDTH, N/TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    matrix_mult_with_tiling<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, sizeof(float)*N*N, cudaMemcpyDeviceToHost);

    
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            cout<<C[i*N+j]<<" ";
        }
        cout<<endl;
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
