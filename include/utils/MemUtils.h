#pragma once

#include <cstdlib>
#include <cstring>
#include <cassert>

namespace MemUtils
{
    /**
    * Allocate "count" times the memory size of T (the type of the data).
    * The address of the newly allocated memory is copied in "*ptr".
    */
    template<typename T>
    cudaHostDevice void malloc(T** ptr, int count = 1)
    {
        /*
        * The #if-#else directives in this namespace have the following meaning:
        * the first branch holds code meant to be launched on from the CPU
        * (respectively, GPU) and executed on the CPU (resp., GPU) itself.
        * that code are typically standard library calls, which are automatically
        * substituted by CUDA if executed on the GPU.
        * The second branch is code to be launched from the CPU and executed on the
        * GPU, which needs a data format compatible for both. So, for example, the
        * first branch of "malloc()" is a call to the standard library, while the
        * second branch is a call to "cudaMallocManaged()", which allocates memory
        * accessible by both the CPU and the GPU.
        */

#if defined(__CUDA_ARCH__) || !defined(GPU)
        *ptr = static_cast<T*>(std::malloc(sizeof(T) * count));
#else
        LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaMallocManaged(ptr, sizeof(T) * count));
#endif
        assert(*ptr != nullptr);
    }

    /**
    * Free the memory referenced by "ptr".
    */
    template<typename T>
    cudaHostDevice void free(T* ptr)
    {
#if defined(__CUDA_ARCH__) || !defined(GPU)
        std::free(ptr);
#else
        LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaFree(ptr));
#endif
    }

    /**
    * Copy n bytes from the memory referenced by "src" to that
    * referenced by "dst", where n is "count" times the size of T,
    * i.e. the type of the data.
    */
    template<typename T>
    cudaHostDevice void memcpy(T* dst, T* src, int count)
    {
        std::memcpy(dst, src, sizeof(T) * count);
    }

    /**
    * Reallocates the memory referenced by "ptr", increasing
    * it from "oldCount" times n to "newCount" times n, where
    * n is the size of the data (of type T).
    *
    * Device reallocation (i.e. with CUDA, on the GPU) is performed by 
    * allocating a new batch of memory of the required size, copying the 
    * old data within, and freeing the previously allocated memory.
    */
    template<typename T>
    cudaHostDevice void realloc(T** ptr, int newCount, int oldCount)
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
