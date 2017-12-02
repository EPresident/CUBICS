#pragma once

#include <domains/IntDomainsRepresentations.h>
#include <domains/IntDomainsActions.h>
#include <data_structures/MonotonicIntVector.h>

struct IntDomains
{
    enum EventTypes
    {
        None,
        Changed
    };

    Vector<int> events;

    IntDomainsRepresentations representations;
    IntDomainsActions actions;
    MonotonicIntVector changes;

    void initialize(int count);
    void deinitialize();

    void push(int min, int max);


    cudaDevice inline bool isEmpty(int index)
    {
        return representations.isEmpty(index);
    }

    cudaDevice inline bool isSingleton(int index)
    {
        return representations.isSingleton(index);
    }

    cudaDevice inline unsigned int getApproximateCardinality(int index)
    {
        return representations.getApproximateCardinality(index);
    }

    cudaHostDevice inline int getMin(int index)
    {
        return representations.minimums[index];
    }

    cudaDevice inline int getMax(int index)
    {
        return representations.maximums[index];
    }

    cudaDevice void fixValue(int index, int value);

    cudaDevice void update(int index);
};
