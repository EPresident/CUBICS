#pragma once

#ifdef GPU
#define cudaGlobal __global__
#define cudaDevice __device__
#define cudaHostDevice __host__ __device__
#else
#define cudaGlobal
#define cudaDevice
#define cudaHostDevice
#endif
