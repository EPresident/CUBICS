#pragma once

#include <data_structures/Vector.h>

struct IntDomainsRepresentations
{
    Vector<int> minimums;
    Vector<int> maximums;
    Vector<int> offsets;
    Vector<unsigned int> versions;
    Vector<Vector<unsigned int>> bitvectors;

    void initialize(int count);
    void deinitialize();

    void push(int min, int max);
    void push(int min, int max, int offset, unsigned int version, Vector<unsigned int>* bitvector);
    void pop();

    inline int getChunkIndex(int index, int val)
    {
        return abs(val - offsets[index]) / UINT_BIT_SIZE;
    }

    inline int getBitIndex(int index, int val)
    {
        return abs(val - offsets[index]) % UINT_BIT_SIZE;
    }

    inline int getValue(int index, int chunkIndex, int bitIndex)
    {
        return offsets[index] + (UINT_BIT_SIZE * chunkIndex) + bitIndex;
    }

    inline bool isEmpty(int index)
    {
        return minimums[index] > maximums[index];
    }

    inline bool isSingleton(int index)
    {
        return minimums[index] == maximums[index];
    }

    inline unsigned int getApproximateCardinality(int index)
    {
        return abs(maximums[index] - minimums[index]) + 1;
    }

    bool contain(int index, int val);
    bool getNextValue(int index, int val, int* nextVal);
    bool getPrevValue(int index, int val, int* prevVal);

    void remove(int index, int val);
    void removeAnyLesserThan(int index, int val);
    void removeAnyGreaterThan(int index, int val);

    void removeAll(int index);
    void keepOnly(int index, int val);
};
