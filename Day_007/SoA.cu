#include <iostream>
#include <cuda_runtime.h>
#define TILE_SIZE 32

using namespace std;

struct AoS{
    float x,y,z;
    float vx,vy,vz;
};

__global__ void  updatePAoS(AoS *particles, float dt, int n){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<n){
        particles[idx].vx += particles[idx].x * dt;
        particles[idx].vy += particles[idx].y * dt;
        particles[idx].vz += particles[idx].z * dt;
    }
}


__global__ void updatePSoA(float *x, float *y, float *z, float *vx, float *vy, float *vz, float dt, int n){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx<n){
        vx[idx]+= x[idx]*dt;
        vy[idx]+= y[idx]*dt;
        vz[idx]+= z[idx]*dt;
    }
}

int main(){
    int n=1<<20;
    size_t sizeAoS=n*sizeof(AoS);
    size_t sizeSoA=n*sizeof(float);

    AoS* d_particlesAoS;
    cudaMalloc(&d_particlesAoS, sizeAoS);

    float *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz;
    cudaMalloc(&d_x, sizeSoA);
    cudaMalloc(&d_y, sizeSoA);
    cudaMalloc(&d_z, sizeSoA);
    cudaMalloc(&d_vx, sizeSoA);
    cudaMalloc(&d_vy, sizeSoA);
    cudaMalloc(&d_vz, sizeSoA);

    int blockSize=256;
    int numBlocks=(n+blockSize-1)/blockSize;

    dim3 grid(numBlocks, 1,1);
    dim3 block(blockSize, 1,1);

    cudaEvent_t start, stop;
    float dt=0.01f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //AoS

    cudaEventRecord(start);
    updatePAoS<<<grid,block>>>(d_particlesAoS, dt, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float millisecAOS=0.0f;
    cudaEventElapsedTime(&millisecAOS, start, stop);
    cout<<"AoS Time: "<<millisecAOS<<" ms"<<endl;

    //SoA
    cudaEventRecord(start);
    updatePSoA<<<grid, block>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, dt, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float millisecSoA=0.0f;
    cudaEventElapsedTime(&millisecSoA, start, stop);
    cout<<"SoA Time: "<<millisecSoA<<" ms"<<endl;
    cudaFree(d_particlesAoS);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);
    return 0;
}