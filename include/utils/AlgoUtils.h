#pragma once

#include <data_structures/Vector.h>

namespace AlgoUtils
{
    template<typename T>
    cudaHostDevice void fill(Vector<T>* vector, T value)
    {
        for (int i = 0; i < vector->size; i += 1)
        {
            vector->at(i) = value;
        }
    }
}
