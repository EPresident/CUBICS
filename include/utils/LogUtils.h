#pragma once

#include <utils/GpuUtils.h>

namespace LogUtils
{
    /**
    * This function prints an error message to the standard output,
    * defined by the "function" and "msg" parameters.
    * It also launches a device-level trap (on GPU).
    */
    cudaHostDevice void error(const char* function, const char* msg);
#ifdef GPU
    /** 
    * This function calls error() if the "code" parameter is not
    * equal to cudaSuccess. That is, the second parameter is usually a
    * call to a function that returns a "cudaError_t" error code.
    */
    void cudaAssert(const char* function, cudaError_t code);
#endif
}
