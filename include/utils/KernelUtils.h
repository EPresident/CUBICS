#ifdef GPU
#pragma once

#include <utils/GpuUtils.h>

#define THREAD_ID ((blockDim.x * blockIdx.x) + threadIdx.x)

namespace KernelUtils
{
    /**
    * Get the number of n-sized ("groupSize") thread blocks needed to perform 
    * m tasks ("taskCount").
    * \param divergence indicates whether the threads within a warp diverge,
    * i.e. if branching can cause them to execute different instructions.
    * A warp is a "hardware" group of threads, typically of size 32. CUDA can't
    * schedule a thread group smaller than a warp of threads.
    */
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

    /**
    * Get the task index for a given thread ("threadIndex").
    * \param divergence indicates whether the threads within a warp diverge,
    * i.e. if branching can cause them to execute different instructions.
    * A warp is a "hardware" group of threads, typically of size 32. CUDA can't
    * schedule a thread group smaller than a warp of threads.
    */
    cudaDevice inline int getTaskIndex(int threadIndex, bool divergence = false)
    {
        //int threadIndex = (blockDim.x * blockIdx.x) + threadIdx.x;

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
