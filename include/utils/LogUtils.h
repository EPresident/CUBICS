#pragma once

#include <utils/GpuUtils.h>

namespace LogUtils
{
    cudaHostDevice void error(const char* function, const char* msg);
#ifdef GPU
    void cudaAssert(const char* function, cudaError_t code);
#endif
}
