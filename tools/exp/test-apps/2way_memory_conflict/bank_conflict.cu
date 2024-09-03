#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>

#define THREADS_PER_WARP 32
#define WARPS_PER_CTA 32
#define DEFAULT_CTAS 10
#define DEFAULT_NREPS 10
#define SHARED_MEM_SIZE 2048  // Define shared memory size for the conflict

__managed__ uint32_t global_sum;
__managed__ uint32_t global_data[SHARED_MEM_SIZE];

///////////////////////////////////////////////////////////////////////////////
// The is the core function of this program. 
///////////////////////////////////////////////////////////////////////////////
__global__ void bank_conflict_kernel(int nreps)
{
    __shared__ uint32_t shared_data[SHARED_MEM_SIZE];  // Shared memory array
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize global data
    if (tid < SHARED_MEM_SIZE) {
        global_data[tid] = tid;
    }

    __syncthreads();  // Synchronize before loading data into shared memory

    // Introduce 2-way bank conflicts by accessing memory with strides
    for (int i = 0; i < nreps; i++) {
        int index = (2 * tid) % SHARED_MEM_SIZE;
        shared_data[index] = global_data[index];
        shared_data[index + 1] = global_data[index + 1];
    }

    __syncthreads();  // Ensure all threads complete the above operation

    // Perform some reduction in shared memory and write back to global memory
    uint32_t local_sum = 0;
    if (tid < SHARED_MEM_SIZE) {
        local_sum = shared_data[tid];
    }

    atomicAdd(&global_sum, local_sum);
}

///////////////////////////////////////////////////////////////////////////////
// This is a wrapper to call the bank_conflict_kernel.
///////////////////////////////////////////////////////////////////////////////
void bank_conflict_wrapper(int ctas, int nreps)
{
    dim3 block(WARPS_PER_CTA * THREADS_PER_WARP, 1);
    dim3 grid(ctas, 1);

    cudaDeviceSynchronize(); 
    bank_conflict_kernel<<<grid, block, 0>>>(nreps);
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
    bank_conflict_wrapper(ctas, nreps);

    printf("global sum = %d \n", global_sum); 

    return 0;
}
 