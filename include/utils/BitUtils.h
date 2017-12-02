#pragma once

#include <climits>

#define UINT_BIT_SIZE 32

namespace BitsUtils
{
    inline int getLeftmostOneIndex(unsigned int val)
    {
#ifdef __CUDA_ARCH__
        return __clz(val);
#else
        return __builtin_clz(val);
#endif
    }

    inline int getRightmostOneIndex(unsigned int val)
    {
#ifdef __CUDA_ARCH__
        return UINT_BIT_SIZE - __ffs(val);
#else
        return UINT_BIT_SIZE - __builtin_ffs(val);
#endif
    }

    inline int getOnesCount(unsigned int val)
    {
#ifdef GPU
        return __popc(val);
#else
        return __builtin_popcount(val);
#endif
    }

    inline unsigned int getMask(int bitIndex)
    {
        return 1 << (UINT_BIT_SIZE - 1 - bitIndex);
    }

    inline unsigned int getLeftFilledMask(int bitIndex)
    {
        return UINT_MAX << (UINT_BIT_SIZE - 1 - bitIndex);
    }

    inline unsigned int getRightFilledMask(int bitIndex)
    {
        return UINT_MAX >> bitIndex;
    }
}
