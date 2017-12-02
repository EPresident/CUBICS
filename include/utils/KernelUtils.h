#ifdef GPU
#pragma once

#include <utils/GpuUtils.h>

namespace KernelUtils
{
    cudaHostDevice inline int getBlockCount(int taskCount, int blockSize, bool divergence = false)
    {
        if (divergence)
        {
            return ceil(static_cast<float>(taskCount * WARP_SIZE) / blockSize);
        }
        else
        {
            return ceil(static_cast<float>(taskCount) / blockSize);
        }
    }

    cudaDevice inline int getTaskIndex(bool divergence = false)
    {
        int threadIndex = (blockDim.x * blockIdx.x) + threadIdx.x;

        if (divergence)
        {
            if (threadIndex % WARP_SIZE != 0)
            {
                return -1;
            }
            else
            {
                return threadIndex / WARP_SIZE;
            }
        }
        else
        {
            return threadIndex;
        }
    }
}
#endif
