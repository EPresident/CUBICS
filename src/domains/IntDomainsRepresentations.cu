#include <domains/IntDomainsRepresentations.h>
#include <utils/Utils.h>

/// Initialize the domain representations for "count" variables.
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
    bitvectors.deinitialize();
}

/** 
* Add the interval ["min","max"] to a new domain (for a new variable), 
* using "min" as offset. 
*/
void IntDomainsRepresentations::push(int min, int max)
{
    minimums.push_back(min);
    maximums.push_back(max);
    offsets.push_back(min);
    versions.push_back(0);

    bitvectors.resize_by_one();
    int index = minimums.size - 1;
    // Find how many bitvectors (intervals) are needed to represent the domain
    int maxChunkIndex = getChunkIndex(index, maximums[index]);
    int chunksCount = maxChunkIndex + 1;
    bitvectors.back().initialize(chunksCount);
    for(int ci = 0; ci < chunksCount; ci += 1)
    {
        bitvectors.back().push_back(UINT_MAX);
    }
    int maxBitIndex = getBitIndex(index, maximums[index]);
    /* 
    * Use a mask to remove the elements outside the domain. This is
    * necessary when the last chunk represents more values than needed.
    * All bits have been set to 1 above, so we need to flip the invalid ones. 
    */
    unsigned int mask = BitsUtils::getLeftFilledMask(maxBitIndex);
    bitvectors.back()[maxChunkIndex] &= mask;
}

/** 
* Add an interval to a new domain (for a new variable).
* \param min is the lower bound of the interval
* \param max is the upper bound of the interval
* \param offset the first value represented by the bitvector
* \param version the number of modification the domain has had
* \param cardinality the number of values in the interval
* \param bitvector a bitvector indicating which values are to be in the domain;
* \see IntDomainsRepresentations::bitvectors
*/
cudaHostDevice void IntDomainsRepresentations::push(int min, int max, int offset, unsigned int version, Vector<unsigned int>* bitvector)
{
    minimums.push_back(min);
    maximums.push_back(max);
    offsets.push_back(offset);
    versions.push_back(version);
    bitvectors.resize_by_one();
    bitvectors.back().initialize(bitvector);
}

/// Remove the last representation added.
cudaHostDevice void IntDomainsRepresentations::pop()
{
    minimums.pop_back();
    maximums.pop_back();
    offsets.pop_back();
    versions.pop_back();
    bitvectors.back().deinitialize();
    bitvectors.pop_back();
}

/** 
* Returns true if the "index"-th chunk contains "val", i.e. the bit
* representing "val" is 1.
*/
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

/** 
* Set "nextVal" to point to the next value in the "index"-th domain, after "val".
* \param nextVal a pointer that will be set to the next value.
* \return false if no such value exists, true otherwise.
*/
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

/** 
* Set "prevVal" to point to the previous value in the "index"-th domain,
* after "val".
* \param prevVal a pointer that will be set to the previous value.
* \return false if no such value exists, true otherwise.
*/
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

/**
* Remove the value "val" from the domain of the "index"-th variable, updating
* the domain representation accordingly (minimum, maximum, cardinality).
*/
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

/**
* Remove all values lesser than "val" from the domain of the "index"-th 
* variable, updating the domain representation accordingly (minimum, 
* maximum, cardinality).
*/
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

/**
* Remove all values greater than "val" from the domain of the "index"-th 
* variable, updating the domain representation accordingly (minimum, 
* maximum, cardinality).
*/
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

/**
* Remove all values other than "val" from the domain of the "index"-th 
* variable, thus making it a singleton.
*/
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

/**
* Remove all values from the domain of the "index"-th variable, thus making it 
* an empty set.
*/
cudaDevice void IntDomainsRepresentations::removeAll(int index)
{
    minimums[index] = INT_MAX;
    maximums[index] = INT_MIN;
    versions[index] += 1;
}
