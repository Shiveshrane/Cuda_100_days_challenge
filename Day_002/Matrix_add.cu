#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void matrix_Add(const float *A, const float *B, float *C, int N){
    int row_idx=blockIdx.x*blockDim.x+threadIdx.x;
    int col_idx=blockIdx.y*blockDim.y+threadIdx.y;

    if (row_idx<N && col_idx<N){
        C[row_idx*N+col_idx]=A[row_idx*N+col_idx]+B[row_idx*N+col_idx];

    }
}

int main(){
    int rows=10;
    int columns=10;

    float *A = new float[rows*columns];
    float *B = new float[rows*columns];
    float *C = new float[rows*columns];
    for (int i=0; i<rows; i++){
        for (int j=0; j<columns; j++){
            A[i*columns+j]=1.0f;
            B[i*columns+j]=2.0f;
        }
    }

    float *da, *db, *dc;
    cudaMalloc(&da, sizeof(float)*rows*columns);
    cudaMalloc(&db, sizeof(float)*rows*columns);
    cudaMalloc(&dc, sizeof(float)*rows*columns);

    cudaMemcpy(da, A, sizeof(float)*rows*columns, cudaMemcpyHostToDevice);
    cudaMemcpy(db, B, sizeof(float)*rows*columns, cudaMemcpyHostToDevice);
    int block_size=256;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim(rows/blockDim.x+1, columns/blockDim.y+1);
    matrix_Add<<<gridDim, blockDim>>>(da, db, dc, rows);
    cudaMemcpy(C, dc, sizeof(float)*rows*columns, cudaMemcpyDeviceToHost);
    for (int i=0; i<rows; i++){
        for (int j=0; j<columns; j++){
            cout<<C[i*columns+j]<<" ";
        }
        cout<<endl;
    }
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return 0;
}
