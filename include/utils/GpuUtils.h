#pragma once

#ifdef GPU
/** 
* __global__ function are kernels, executed on the device
* and called by the host.
*/
#define cudaGlobal __global__
/// __device__ functions are called from and executed in the device.
#define cudaDevice __device__
/** 
* __host__ __device__ functions are compiled for both the host
* and the device.
*/
#define cudaHostDevice __host__ __device__
#define WARP_SIZE 32
#define THREADS_PER_MULTIPROCESSOR 2048
#define BLOCK_PER_MULTIPROCESSOR 16
#define DEFAULT_BLOCK_SIZE (THREADS_PER_MULTIPROCESSOR / BLOCK_PER_MULTIPROCESSOR)
#define HEAP_SIZE 512 * 1024 * 1024
#else
#define cudaGlobal
#define cudaDevice
#define cudaHostDevice
#endif
