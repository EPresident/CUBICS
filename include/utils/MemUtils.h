#pragma once

#include <cstdlib>
#include <cstring>
#include <cassert>

namespace MemUtils
{
    template<typename T>
    void malloc(T** ptr, int count = 1)
    {
        *ptr = static_cast<T*>(std::malloc(sizeof(T) * count));
        assert(*ptr != nullptr);
    }

    template<typename T>
    void free(T* ptr)
    {
        std::free(ptr);
    }

    template<typename T>
    void memcpy(T* dst, T* src, int count)
    {
        std::memcpy(dst, src, sizeof(T) * count);
    }

    template<typename T>
    void realloc(T** ptr, int newCount)
    {
        *ptr = static_cast<T*>(std::realloc(*ptr, sizeof(T) * newCount));
    }
}
