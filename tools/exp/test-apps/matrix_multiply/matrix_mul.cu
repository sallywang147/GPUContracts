#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>

#define TILE_WIDTH 16
#define MATRIX_SIZE 1024
#define THREADS_PER_BLOCK 16
#define BLOCK_SIZE 16

__managed__ float A[MATRIX_SIZE][MATRIX_SIZE];
__managed__ float B[MATRIX_SIZE][MATRIX_SIZE];
__managed__ float C[MATRIX_SIZE][MATRIX_SIZE];

///////////////////////////////////////////////////////////////////////////////
// The is the core function of this program. Matrix Multiplication with 16-way
// bank conflicts.
///////////////////////////////////////////////////////////////////////////////
__global__ void matrixMulKernel(float* A, float* B, float* C, int width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH]; // Shared memory for A
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH]; // Shared memory for B

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0;

    for (int m = 0; m < width/TILE_WIDTH; ++m) {
        // Introduce 16-way bank conflict by accessing shared memory with a stride
        int shared_index_x = (tx * 16) % TILE_WIDTH;  // Stride to create conflict
        int shared_index_y = ty;

        Mds[shared_index_y][shared_index_x] = A[Row * width + (m * TILE_WIDTH + shared_index_x)];
        Nds[shared_index_y][shared_index_x] = B[(m * TILE_WIDTH + shared_index_y) * width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    C[Row * width + Col] = Pvalue;
}

///////////////////////////////////////////////////////////////////////////////
// This is a wrapper to call the matrix multiplication kernel.
///////////////////////////////////////////////////////////////////////////////
void matrixMul(int width) {
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(width / TILE_WIDTH, width / TILE_WIDTH);
    matrixMulKernel<<<dimGrid, dimBlock>>>(reinterpret_cast<float*>(A), reinterpret_cast<float*>(B), reinterpret_cast<float*>(C), width);
}

///////////////////////////////////////////////////////////////////////////////
// Main function to setup and call the kernel
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
    setbuf(stdout, NULL); // Disable stdout buffering

    // Set the device
    int device = 0;
    cudaSetDevice(device);
    cudaDeviceProp cudaDevicePropForChoosing;
    cudaGetDeviceProperties(&cudaDevicePropForChoosing, device);

    printf("Device %d (%s) is being used\n", device, cudaDevicePropForChoosing.name);
    printf("memory: %.4f GB %s %d SMs x%d\n", cudaDevicePropForChoosing.totalGlobalMem/(1024.f*1024.f*1024.f), (cudaDevicePropForChoosing.ECCEnabled)?"ECC on":"ECC off", cudaDevicePropForChoosing.multiProcessorCount, cudaDevicePropForChoosing.clockRate );

    // Initialize the matrices with arbitrary values for A and B
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            A[i][j] = 1.0f;  // Arbitrary initialization
            B[i][j] = 1.0f;  // Arbitrary initialization
            C[i][j] = 0.0f;  // Initialize output matrix
        }
    }

    matrixMul(MATRIX_SIZE);

    // Print a few values for validation
    printf("C[0][0] = %f\n", C[0][0]);
    printf("C[10][10] = %f\n", C[10][10]);

    return 0;
}
