#include <data_structures/MonotonicIntVector.h>

void MonotonicIntVector::initialize(int maxSize)
{
    this->maxSize = maxSize;

    booleanMask.initialize(maxSize);
    booleanMask.resize(maxSize);
    AlgoUtils::fill(&booleanMask, false);

    vector.initialize(maxSize);

#ifdef GPU
    lock.initialize();
#endif
}

void  MonotonicIntVector::deinitialize()
{
    booleanMask.deinitialize();
    vector.deinitialize();
}

cudaHostDevice void MonotonicIntVector::add(int value)
{

#if defined(GPU) && defined(__CUDA_ARCH__)
    lock.lock();
#endif

    if(not contain(value))
    {
        booleanMask[value] = true;
        vector.push_back(value);
    }
#if defined(GPU) && defined(__CUDA_ARCH__)
    lock.unlock();
#endif


}

cudaHostDevice void MonotonicIntVector::add(MonotonicIntVector* other)
{
    assert(this->maxSize == other->maxSize);

    for(int i = 0; i < other->vector.size; i += 1)
    {
        add(other->at(i));
    }
}

cudaHostDevice void MonotonicIntVector::clear()
{
    for(int i = 0; i < vector.size; i += 1)
    {
        booleanMask[vector[i]] = false;
    }

    vector.clear();
}

cudaHostDevice void MonotonicIntVector::copy(MonotonicIntVector* other)
{
    assert(this->maxSize == other->maxSize);

    this->booleanMask.copy(&other->booleanMask);
    this->vector.copy(&other->vector);
}

cudaHostDevice void MonotonicIntVector::reinitialize(int maxSize)
{
    this->maxSize = maxSize;
    booleanMask.resize(maxSize);
    AlgoUtils::fill(&booleanMask, false);
    vector.reserve(maxSize);
    vector.clear();
}
