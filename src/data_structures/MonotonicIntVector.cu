#include <data_structures/MonotonicIntVector.h>

void MonotonicIntVector::initialize(int maxSize)
{
    this->maxSize = maxSize;

    booleanMask.initialize(maxSize);
    booleanMask.resize(maxSize);
    AlgoUtils::fill(&booleanMask, false);

    vector.initialize(maxSize);
}

void  MonotonicIntVector::deinitialize()
{
    booleanMask.deinitialize();
    vector.deinitialize();
}

void MonotonicIntVector::add(int value)
{
    if(not contain(value))
    {
        booleanMask[value] = true;
        vector.push_back(value);
    }
}

void MonotonicIntVector::add(MonotonicIntVector* other)
{
    assert(this->maxSize == other->maxSize);

    for(int i = 0; i < other->vector.size; i += 1)
    {
        add(other->at(i));
    }
}

void MonotonicIntVector::clear()
{
    for(int i = 0; i < vector.size; i += 1)
    {
        booleanMask[vector[i]] = false;
    }

    vector.clear();
}

void MonotonicIntVector::copy(MonotonicIntVector* other)
{
    assert(this->maxSize == other->maxSize);

    this->booleanMask.copy(&other->booleanMask);
    this->vector.copy(&other->vector);
}

void MonotonicIntVector::reinitialize(int maxSize)
{
    this->maxSize = maxSize;
    booleanMask.resize(maxSize);
    AlgoUtils::fill(&booleanMask, false);
    vector.reserve(maxSize);
    vector.clear();
}
