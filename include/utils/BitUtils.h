#pragma once

#include <climits>

#define UINT_BIT_SIZE 32

namespace BitsUtils
{
    /// \return the index of the leftmost 1 bit in an integer. 
    cudaDevice inline int getLeftmostOneIndex(unsigned int val)
    {
        /* 
        * clz() returns the number of consecutive high-order zero bits 
        * in a 32 bit integer.
        */
#ifdef __CUDA_ARCH__
        return __clz(val);
#else
        return __builtin_clz(val);
#endif
    }

    /// \return the index of the rightmost 1 bit in an integer.
    cudaDevice inline int getRightmostOneIndex(unsigned int val)
    {
        /* 
        * ffs() finds the position of the least significant bit set
        * to 1 in a 32 bit integer. 
        */ 
#ifdef __CUDA_ARCH__
        return UINT_BIT_SIZE - __ffs(val);
#else
        return UINT_BIT_SIZE - __builtin_ffs(val);
#endif
    }
    
    /// \return the number of bits that are set to 1 in a 32 bit integer.
    cudaDevice inline int getOnesCount(unsigned int val)
    {
#ifdef GPU
        return __popc(val);
#else
        return __builtin_popcount(val);
#endif
    }

    /** Returns the bitmask necessary to retrieve the i-th bit
    *   from an unsigned integer.
    *   The bits are numbered incrementally from 0, left to right.
    *   A bitwise AND operation between the integer and the bitmask
    *   yields the value of the i-th bit.
    *
    *   For example, the bitmask for retrieving the third bit from
    *   10011010 is 00000100.
    */
    cudaDevice inline unsigned int getMask(int bitIndex)
    {
        return 1 << (UINT_BIT_SIZE - 1 - bitIndex);
    }

    /** Returns the bitmask necessary to retrieve the i-th bit
    *   from an unsigned integer, with all bits left of the i-th
    *   set to 1.
    *   The bits are numbered incrementally from 0, left to right.
    *
    *   For example, using one byte, the left filled mask for i=3 of
    *   10011010 is 11111100.
    */
    cudaHostDevice inline unsigned int getLeftFilledMask(int bitIndex)
    {
        return UINT_MAX << (UINT_BIT_SIZE - 1 - bitIndex);
    }

    /** Returns the bitmask necessary to retrieve the i-th bit
    *   from an unsigned integer, with all bits right of the i-th
    *   set to 1.
    *   The bits are numbered incrementally from 0, left to right.
    *
    *   For example, using one byte, the left filled mask for i=3 of
    *   10011010 is 00000111.
    */
    cudaDevice inline unsigned int getRightFilledMask(int bitIndex)
    {
        return UINT_MAX >> bitIndex;
    }
}
