#pragma once

#include <domains/IntDomainsRepresentations.h>
#include <domains/IntDomainsActions.h>

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

    void initialize(int count);
    void deinitialize();

    void push(int min, int max);


    inline bool isEmpty(int index)
    {
        return representations.isEmpty(index);
    }

    inline bool isSingleton(int index)
    {
        return representations.isSingleton(index);
    }

    inline unsigned int getApproximateCardinality(int index)
    {
        return representations.getApproximateCardinality(index);
    }

    inline int getMin(int index)
    {
        return representations.minimums[index];
    }

    inline int getMax(int index)
    {
        return representations.maximums[index];
    }

    void fixValue(int index, int value);

    void updateDomain(int index);
};
