#include <cstdlib>
#include <cstdio>

#include <utils/LogUtils.h>

void LogUtils::error(const char* function, const char* msg)
{
    printf("%s:%s\n", function, msg);
#ifdef __CUDA_ARCH__
    asm("trap;");
#else
    abort();
#endif
}

