#include <cassert>

#include <data_structures/BitsArray.h>

cudaHostDevice void BitsArray::initialize(int size, bool fill)
{
    this->size = size;
    chunkCount = BitsUtils::getChunkCount(size);
    MemUtils::malloc(&data, chunkCount);

    if (fill)
    {
        setAll();
    }
    else
    {
        resetAll();
    }
}

cudaHostDevice void BitsArray::initialize(BitsArray* other)
{
    this->size = other->size;
    this->chunkCount = other->chunkCount;

    MemUtils::malloc(&data, chunkCount);

    copy(other);
}

cudaHostDevice void BitsArray::deinitialize()
{
    MemUtils::free(data);
}

cudaHostDevice void BitsArray::copy(BitsArray* other)
{
    assert(chunkCount == other->chunkCount and size == other->size);

    MemUtils::memcpy(data, other->data, chunkCount);
}

cudaHostDevice void BitsArray::setAll()
{
    for(int i = 0; i < chunkCount; i += 1)
    {
        data[i] = UINT_MAX;
    }

    int bitIndex = BitsUtils::getBitIndex(size - 1);
    unsigned int mask = BitsUtils::getLeftFilledMask(bitIndex);
    data[chunkCount - 1] &= mask;
}

cudaHostDevice void BitsArray::resetAll()
{
    for(int i = 0; i < chunkCount; i += 1)
    {
        data[i] = 0;
    }
}


cudaDevice int BitsArray::getLeftmostOneIndexIn(int minIndex, int maxIndex)
{
    int result = -1;

    int bitIndexMin = BitsUtils::getBitIndex(minIndex);
    unsigned int maskMin = BitsUtils::getRightFilledMask(bitIndexMin);

    int chunkIndex = BitsUtils::getChunkIndex(minIndex);
    unsigned int chunk = data[chunkIndex] & maskMin;

    int chunkIndexMax = BitsUtils::getChunkIndex(maxIndex);

    if(not chunk)
    {
        for(chunkIndex += 1; chunkIndex <= chunkIndexMax; chunkIndex += 1)
        {
            chunk = data[chunkIndex];

            if(chunk)
            {
                break;
            }
        }
    }

    if(chunkIndex == chunkIndexMax)
    {
        int bitIndexMax = BitsUtils::getBitIndex(maxIndex);
        unsigned int maskMax = BitsUtils::getLeftFilledMask(bitIndexMax);
        chunk &= maskMax;
    }

    if(chunk)
    {
        result = (chunkIndex * UINT_BIT_SIZE) + BitsUtils::getLeftmostOneIndex(chunk);
    }

    return result;
}

cudaDevice int BitsArray::getRightmostOneIndexIn(int minIndex, int maxIndex)
{
    int result = -1;

    int bitIndexMax = BitsUtils::getBitIndex(maxIndex);
    unsigned int maskMax = BitsUtils::getLeftFilledMask(bitIndexMax);

    int chunkIndex = BitsUtils::getChunkIndex(maxIndex);
    unsigned int chunk = data[chunkIndex] & maskMax;

    int chunkIndexMin = BitsUtils::getChunkIndex(minIndex);

    if(not chunk)
    {
        for(chunkIndex -= 1; chunkIndex >= chunkIndexMin; chunkIndex -= 1)
        {
            chunk = data[chunkIndex];

            if(chunk)
            {
                break;
            }
        }
    }

    if(chunkIndex == chunkIndexMin)
    {
        int bitIndexMin = BitsUtils::getBitIndex(minIndex);
        unsigned int maskMin = BitsUtils::getRightFilledMask(bitIndexMin);
        chunk &= maskMin;
    }

    if(chunk)
    {
        result = (chunkIndex * UINT_BIT_SIZE) + BitsUtils::getRightmostOneIndex(chunk);
    }

    return result;
}

