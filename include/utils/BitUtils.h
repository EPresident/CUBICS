#pragma once

#include <climits>

#define UINT_BIT_SIZE 32

namespace BitsUtils
{
    inline int getLeftmostOneIndex(unsigned int val)
    {
        return __builtin_clz(val);
    }

    inline int getRightmostOneIndex(unsigned int val)
    {
        return UINT_BIT_SIZE - __builtin_ffs(val);
    }

    inline int getOnesCount(unsigned int val)
    {
        return __builtin_popcount(val);
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
