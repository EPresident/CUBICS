#include <cstdlib>
#include <cstdio>

#include <utils/LogUtils.h>

/**
* This function prints an error message to the standard output,
* defined by the "function" and "msg" parameters.
* It also launches a device-level trap (on GPU).
*/
cudaHostDevice void LogUtils::error(const char* function, const char* msg)
{
    printf("%s:%s\n", function, msg);
#ifdef __CUDA_ARCH__
    asm("trap;");
#else
    abort();
#endif
}

#ifdef GPU
/** 
* This function calls error() if the "code" parameter is not
* equal to cudaSuccess. That is, the second parameter is usually a
* call to a function that returns a "cudaError_t" error code.
*/
void LogUtils::cudaAssert(const char* function, cudaError_t code)
{
    if (code != cudaSuccess)
    {
        error(function, cudaGetErrorString(code));
    }
}
#endif
