#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>

#define THREADS_PER_WARP 32
#define WARPS_PER_CTA 32
#define DEFAULT_CTAS 10
#define DEFAULT_NREPS 10
#define SHARED_MEM_SIZE 1024

__managed__ uint32_t global_sum;

///////////////////////////////////////////////////////////////////////////////
// Kernel with intentional memory bank conflicts.
///////////////////////////////////////////////////////////////////////////////
__global__ void add_with_bank_conflict(int nreps)
{
    __shared__ int shared_data[SHARED_MEM_SIZE];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize shared memory to ensure consistent results
    shared_data[threadIdx.x] = threadIdx.x;

    __syncthreads(); // Ensure all threads have written their values

    int local_sum = 0;

    for (int i = 0; i < nreps; i++) {
        // Introduce intentional bank conflicts by accessing shared memory
        // with a stride that maps to the same bank
        local_sum += shared_data[(threadIdx.x * 8) % SHARED_MEM_SIZE];
    }

    atomicAdd(&global_sum, local_sum);
}

///////////////////////////////////////////////////////////////////////////////
// Wrapper function for the kernel.
///////////////////////////////////////////////////////////////////////////////
void add_with_bank_conflict_wrapper(int ctas, int nreps)
{
    dim3 block(WARPS_PER_CTA * THREADS_PER_WARP, 1);
    dim3 grid(ctas, 1);

    cudaDeviceSynchronize(); 
    add_with_bank_conflict<<<grid, block, 0>>>(nreps);
    cudaDeviceSynchronize(); 
    cudaError_t error = cudaGetLastError(); 
    if (error != cudaSuccess) {
        printf("Error: kernel failed %s\n", cudaGetErrorString(error));
    }
}

int main(int argc, char *argv[])
{
    setbuf(stdout, NULL); // Disable stdout buffering

    // Set the device
    int device = 0;
    cudaSetDevice(device);
    cudaDeviceProp cudaDevicePropForChoosing;
    cudaGetDeviceProperties(&cudaDevicePropForChoosing, device);

    printf("Device %d (%s) is being used\n", device, cudaDevicePropForChoosing.name);
    printf("memory: %.4f GB %s %d SMs x%d\n", cudaDevicePropForChoosing.totalGlobalMem/(1024.f*1024.f*1024.f), (cudaDevicePropForChoosing.ECCEnabled)?"ECC on":"ECC off", cudaDevicePropForChoosing.multiProcessorCount, cudaDevicePropForChoosing.clockRate );

    int nreps = DEFAULT_NREPS;
    int ctas = DEFAULT_CTAS;
    printf("#CTAs=%d, nreps=%d, threads/CTA=%d\n", ctas, nreps, THREADS_PER_WARP*WARPS_PER_CTA);

    global_sum = 0; // Initialize the sum to 0

    // Call the main function now
    add_with_bank_conflict_wrapper(ctas, nreps);

    printf("global sum = %d \n", global_sum); 

    return 0;
}