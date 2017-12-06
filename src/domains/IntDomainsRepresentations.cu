#include <domains/IntDomainsRepresentations.h>
#include <utils/Utils.h>

void IntDomainsRepresentations::initialize(int count)
{
    minimums.initialize(count);
    maximums.initialize(count);
    offsets.initialize(count);
    versions.initialize(count);
    bitvectors.initialize(count);
}

void IntDomainsRepresentations::deinitialize()
{
    minimums.deinitialize();
    maximums.deinitialize();
    offsets.deinitialize();
    versions.deinitialize();

    for (int i = 0; i < bitvectors.size; i += 1)
    {
        bitvectors[i].deinitialize();
    }
    bitvectors.initialize();
}

cudaHostDevice void IntDomainsRepresentations::push(int min, int max)
{
    minimums.push_back(min);
    maximums.push_back(max);
    offsets.push_back(min);
    versions.push_back(0);

    bitvectors.resize_by_one();
    int index = minimums.size - 1;
    int maxChunkIndex = getChunkIndex(index, maximums[index]);
    int chunksCount = maxChunkIndex + 1;
    bitvectors.back().initialize(chunksCount);
    bitvectors.back().resize(chunksCount);
    AlgoUtils::fill(&bitvectors.back(), UINT_MAX);
    int maxBitIndex = getBitIndex(index, maximums[index]);
    unsigned int mask = BitsUtils::getLeftFilledMask(maxBitIndex);
    bitvectors.back()[maxChunkIndex] &= mask;
}

cudaHostDevice void IntDomainsRepresentations::push(int min, int max, int offset, unsigned int version, Vector<unsigned int>* bitvector)
{
    minimums.push_back(min);
    maximums.push_back(max);
    offsets.push_back(offset);
    versions.push_back(version);
    bitvectors.resize_by_one();
    bitvectors.back().initialize(bitvector);
}

cudaDevice void IntDomainsRepresentations::pop()
{
    minimums.pop_back();
    maximums.pop_back();
    offsets.pop_back();
    versions.pop_back();
    bitvectors.back().deinitialize();
    bitvectors.pop_back();
}

cudaDevice bool IntDomainsRepresentations::contain(int index, int val)
{
    if (minimums[index] <= val and val <= maximums[index])
    {
        int valChunkIndex = getChunkIndex(index, val);
        int valBitIndex = getBitIndex(index, val);

        unsigned int mask = BitsUtils::getMask(valBitIndex);
        return (bitvectors[index][valChunkIndex] & mask) != 0;
    }
    else
    {
        return false;
    }
}

cudaDevice bool IntDomainsRepresentations::getNextValue(int index, int val, int* nextVal)
{
    if (val < maximums[index])
    {
        int nextValChunkIndex = getChunkIndex(index, val);
        int valBitIndex = getBitIndex(index, val);

        unsigned int mask = ~BitsUtils::getLeftFilledMask(valBitIndex);
        unsigned int nextValChunk = bitvectors[index][nextValChunkIndex] & mask;

        if (nextValChunk == 0)
        {
            int maxChunkIndex = getChunkIndex(index, maximums[index]);
            for (nextValChunkIndex = nextValChunkIndex + 1; nextValChunkIndex <= maxChunkIndex; nextValChunkIndex += 1)
            {
                if (bitvectors[index][nextValChunkIndex] != 0)
                {
                    break;
                }
            }

            nextValChunk = bitvectors[index][nextValChunkIndex];
        }

        *nextVal = getValue(index, nextValChunkIndex, BitsUtils::getLeftmostOneIndex(nextValChunk));
        return true;
    }
    else
    {
        return false;
    }
}

cudaDevice bool IntDomainsRepresentations::getPrevValue(int index, int val, int* prevVal)
{
    if (minimums[index] < val)
    {
        int prevValChunkIndex = getChunkIndex(index, val);
        int valBitIndex = getBitIndex(index, val);

        unsigned int mask = ~BitsUtils::getRightFilledMask(valBitIndex);
        unsigned int prevValChunk = bitvectors[index][prevValChunkIndex] & mask;

        if (prevValChunk == 0)
        {
            int minChunkIndex = getChunkIndex(index, minimums[index]);
            for (prevValChunkIndex = prevValChunkIndex - 1; prevValChunkIndex >= minChunkIndex; prevValChunkIndex -= 1)
            {
                if (bitvectors[index][prevValChunkIndex] != 0)
                {
                    break;
                }
            }

            prevValChunk = bitvectors[index][prevValChunkIndex];
        }

        *prevVal = getValue(index, prevValChunkIndex, BitsUtils::getRightmostOneIndex(prevValChunk));
        return true;
    }
    else
    {
        return false;
    }
}

cudaDevice void IntDomainsRepresentations::remove(int index, int val)
{
    if (contain(index, val))
    {
        int valChunkIndex = getChunkIndex(index, val);
        int valBitIndex = getBitIndex(index, val);

        unsigned int mask = ~BitsUtils::getMask(valBitIndex);

        bitvectors[index][valChunkIndex] &= mask;

        if (not isEmpty(index))
        {
            if (val == minimums[index])
            {
                if(not getNextValue(index, val, &minimums[index]))
                {
                  removeAll(index);
                }
            }

            if (val == maximums[index])
            {
                if(not getPrevValue(index, val, &maximums[index]))
                {
                    removeAll(index);
                }
            }
        }

        versions[index] += 1;
    }
}

cudaDevice void IntDomainsRepresentations::removeAnyLesserThan(int index, int val)
{
    if (minimums[index] < val and val <= maximums[index])
    {
        int newMinimum = minimums[index];

        if (contain(index, val))
        {
            newMinimum = val;
        }
        else
        {
            getNextValue(index, val, &newMinimum);
        }

        minimums[index] = newMinimum;

        versions[index] += 1;
    }
    else if (val > maximums[index])
    {
        removeAll(index);
    }
}


cudaDevice void IntDomainsRepresentations::removeAnyGreaterThan(int index, int val)
{
    if (minimums[index] <= val and val < maximums[index])
    {
        int newMaximum = maximums[index];

        if (contain(index, val))
        {
            newMaximum = val;
        }
        else
        {
            getPrevValue(index, val, &newMaximum);
        }

        maximums[index] = newMaximum;

        versions[index] += 1;
    }
    else if (val < minimums[index])
    {
        removeAll(index);
    }
}

cudaDevice void IntDomainsRepresentations::keepOnly(int index, int val)
{
    if (contain(index, val))
    {
        minimums[index] = val;
        maximums[index] = val;
        versions[index] += 1;
    }
    else
    {
        removeAll(index);
    }
}

cudaDevice void IntDomainsRepresentations::removeAll(int index)
{
    minimums[index] = INT_MAX;
    maximums[index] = INT_MIN;
    versions[index] += 1;
}
