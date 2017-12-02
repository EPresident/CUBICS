#include <cstdlib>
#include <cstdio>

#include <utils/LogUtils.h>

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
void LogUtils::cudaAssert(const char* function, cudaError_t code)
{
    if (code != cudaSuccess)
    {
        error(function, cudaGetErrorString(code));
    }
}
#endif
