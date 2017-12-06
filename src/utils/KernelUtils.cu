#ifdef GPU

#include <utils/KernelUtils.h>

cudaDevice int KernelUtils::getBlockCount(int taskCount, int blockSize, bool divergence)
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

cudaDevice int KernelUtils::getTaskIndex(int threadId, bool divergence)
{

    if (divergence)
    {
        if (threadId % WARP_SIZE != 0)
        {
            return -1;
        }
        else
        {
            return threadId / WARP_SIZE;
        }
    }
    else
    {
        return threadId;
    }
}
#endif
