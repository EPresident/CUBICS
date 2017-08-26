#pragma once

#include <cstdlib>
#include <cstring>
#include <cassert>

namespace MemUtils
{
    template<typename T>
    void malloc(T** ptr, int count = 1)
    {
#if defined(__CUDA_ARCH__) || !defined(GPU)
        *ptr = static_cast<T*>(std::malloc(sizeof(T) * count));
#else
        LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaMallocManaged(ptr, sizeof(T) * count));
#endif
        assert(*ptr != nullptr);
    }

    template<typename T>
    void free(T* ptr)
    {
#if defined(__CUDA_ARCH__) || !defined(GPU)
        std::free(ptr);
#else
        LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaFree(ptr));
#endif
    }

    template<typename T>
    void memcpy(T* dst, T* src, int count)
    {
        std::memcpy(dst, src, sizeof(T) * count);
    }

    template<typename T>
    void realloc(T** ptr, int newCount, int oldCount)
    {
#ifdef GPU
        T* ptrTmp;
        MemUtils::malloc<T>(&ptrTmp, newCount);
        MemUtils::memcpy<T>(ptrTmp, *ptr, oldCount);
        MemUtils::free<T>(*ptr);
        *ptr = ptrTmp;
#else
        *ptr = static_cast<T*>(std::realloc(*ptr, sizeof(T) * newCount));
#endif
    }
}
