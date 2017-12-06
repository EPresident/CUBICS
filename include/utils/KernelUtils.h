#ifdef GPU
#pragma once

#include <utils/GpuUtils.h>

#define THREAD_ID (blockDim.x * blockIdx.x) + threadIdx.x

namespace KernelUtils
{
    cudaDevice int getBlockCount(int taskCount, int blockSize = DEFAULT_BLOCK_SIZE, bool divergence = false);

    cudaDevice int getTaskIndex(int threadId, bool divergence = false);
}
#endif
