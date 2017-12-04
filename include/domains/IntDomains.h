#pragma once

#include <domains/IntDomainsRepresentations.h>
#include <domains/IntDomainsActions.h>

struct IntDomains
{
    enum EventTypes
    {
        None = 0x0,
        ValueRemoved = 0x1,
        IncreasedMinimum = 0x10,
        DecreasedMaximums = 0x100,
        Istantiated = 0x1000
    };

    Vector<unsigned int> events;

    IntDomainsRepresentations representations;
    IntDomainsActions actions;

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

    cudaDevice void setEvent(int index, unsigned int event);

    cudaDevice inline void clearEvent(int index)
    {
        events[index] = None;
    }

    cudaDevice static inline bool isEventOccurred(unsigned int event, EventTypes eventType)
    {
        return event & eventType;
    }

    cudaDevice void fixValue(int index, int value);

    cudaDevice void updateDomain(int index);


};
