#include<iostream>
#include<cuda_runtime.h>
#define TILE_WIDTH 16

using namespace std;


__global__ void matrix_mult_shared(const float *A, const float *B, float *C, int N){
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int bx=blockIdx.x;
    int by=blockIdx.y;
    int row_idx=by*TILE_WIDTH+ty;
    int col_idx=bx*TILE_WIDTH+tx;

    if(row_idx<N && col_idx<N){
        float val=0.0;
        for(int t=0;t<N/TILE_WIDTH;t++){
            s_A[ty][tx]=A[row_idx*N+(t*TILE_WIDTH+tx)];
            s_B[ty][tx]=B[(t*TILE_WIDTH+ty)*N+col_idx];
            __syncthreads();
            for (int k=0;k<TILE_WIDTH;k++){
                val+=s_A[ty][k]*s_B[k][tx];

            }
            __syncthreads();
        }
        C[row_idx*N+col_idx]=val;

        }
    }

int main(){
    int N;
    cout<<"Enter the size of the square matrices: ";
    cin>>N;

    size_t size=N*N*sizeof(float);
    float *h_A=(float *)malloc(size);
    float *h_B=(float *)malloc(size);
    float *h_C=(float *)malloc(size);

    for(int i=0;i<N*N;i++){
        h_A[i]=float(i);
        h_B[i]=float(i+1);
    }

    float *d_A,*d_B,*d_C;
    cudaMalloc((void **)&d_A,size);
    cudaMalloc((void **)&d_B,size);
    cudaMalloc((void **)&d_C,size);

    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    matrix_mult_shared<<<dimGrid,dimBlock>>>(d_A,d_B,d_C,N);

    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);

    cout<<"Resultant Matrix C (first 10 elements):"<<endl;
    for(int i=0;i<min(10,N*N);i++){
        cout<<h_C[i]<<" ";
    }
    cout<<endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}