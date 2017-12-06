#pragma once

#include <cassert>

#include <data_structures/Vector.h>
#include <data_structures/Lock.h>
#include <utils/Utils.h>

struct MonotonicIntVector
{
    int maxSize;
	Vector<bool> booleanMask;
	Vector<int> vector;

    void initialize(int maxSize);

    void deinitialize();

#ifdef GPU
    Lock lock;
#endif

    cudaHostDevice inline bool contain(int value)
    {
        assert(value < booleanMask.size);

        return booleanMask[value];
    }

    cudaHostDevice inline int operator[](int index)
    {
        return vector[index];
    }

    cudaHostDevice inline int at(int index)
    {
        return vector[index];
    }

    cudaHostDevice int getSize()
    {
        return vector.size;
    }

    cudaHostDevice void add(int value);

    cudaHostDevice void add(MonotonicIntVector* other);

    cudaHostDevice void clear();

    cudaHostDevice void copy(MonotonicIntVector* other);

    cudaHostDevice void reinitialize(int maxSize);
};
