#pragma once

#include <data_structures/Vector.h>
#include <utils/Utils.h>

struct IntDomainsRepresentations
{
    Vector<int> minimums;
    Vector<int> maximums;
    Vector<int> offsets;
    Vector<unsigned int> versions;
    Vector<Vector<unsigned int>> bitvectors;

    void initialize(int count);
    void deinitialize();

    cudaHostDevice void push(int min, int max);
    cudaHostDevice void push(int min, int max, int offset, unsigned int version, Vector<unsigned int>* bitvector);
    cudaDevice void pop();

    cudaHostDevice inline int getChunkIndex(int index, int val)
    {
        return abs(val - offsets[index]) / UINT_BIT_SIZE;
    }

    cudaHostDevice inline int getBitIndex(int index, int val)
    {
        return abs(val - offsets[index]) % UINT_BIT_SIZE;
    }

    cudaDevice inline int getValue(int index, int chunkIndex, int bitIndex)
    {
        return offsets[index] + (UINT_BIT_SIZE * chunkIndex) + bitIndex;
    }

    cudaDevice inline bool isEmpty(int index)
    {
        return minimums[index] > maximums[index];
    }

    cudaDevice  bool isSingleton(int index)
    {
        return minimums[index] == maximums[index];
    }

    cudaDevice inline unsigned int getApproximateCardinality(int index)
    {
        return abs(maximums[index] - minimums[index]) + 1;
    }

    cudaDevice bool contain(int index, int val);
    cudaDevice bool getNextValue(int index, int val, int* nextVal);
    cudaDevice bool getPrevValue(int index, int val, int* prevVal);

    cudaDevice void remove(int index, int val);
    cudaDevice void removeAnyLesserThan(int index, int val);
    cudaDevice void removeAnyGreaterThan(int index, int val);

    cudaDevice void removeAll(int index);
    cudaDevice void keepOnly(int index, int val);
};
