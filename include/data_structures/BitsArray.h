#pragma once

#include <cassert>

#include <utils/Utils.h>

struct BitsArray
{
    int size;
    int chunkCount;
    unsigned int* data;

    cudaHostDevice void initialize(int size, bool fill = true);

    cudaHostDevice void initialize(BitsArray* other);

    cudaHostDevice void deinitialize();

    cudaHostDevice void copy(BitsArray* other);

    cudaHostDevice inline bool get(int index)
    {
        assert(index < size);

        return data[BitsUtils::getChunkIndex(index)] & BitsUtils::getMask(BitsUtils::getBitIndex(index));
    }

    cudaHostDevice inline void set(int index)
    {
        assert(index < size);

        data[BitsUtils::getChunkIndex(index)] |= BitsUtils::getMask(BitsUtils::getBitIndex(index));
    }

    cudaHostDevice inline void reset(int index)
    {
        assert(index < size);

        data[BitsUtils::getChunkIndex(index)] &= ~BitsUtils::getMask(BitsUtils::getBitIndex(index));
    }

    cudaHostDevice void setAll();

    cudaHostDevice void resetAll();

    cudaDevice int getLeftmostOneIndexIn(int minIndex, int maxIndex);

    cudaDevice int getRightmostOneIndexIn(int minIndex, int maxIndex);
};
