#pragma once

#include <cassert>

#include <data_structures/Vector.h>
#include <utils/Utils.h>

struct MonotonicIntVector
{
    int maxSize;
	Vector<bool> booleanMask;
	Vector<int> vector;

    void initialize(int maxSize);

    void deinitialize();

    inline bool contain(int value)
    {
        assert(value < booleanMask.size);

        return booleanMask[value];
    }

    inline int operator[](int index)
    {
        return vector[index];
    }

    inline int at(int index)
    {
        return vector[index];
    }

    int getSize()
    {
        return vector.size;
    }

    void add(int value);

    void add(MonotonicIntVector* other);

    void clear();

    void copy(MonotonicIntVector* other);

    void reinitialize(int maxSize);
};
