#pragma once

#include <domains/IntDomainsRepresentations.h>
#include <domains/IntDomainsActions.h>
#include <variables/IntNeighborhood.h>

struct IntDomains
{
    enum EventTypes
    {
        None,
        Changed ///< A domain has been changed
    };

    /// A list of domain events (domain changed) for each variable.
    Vector<int> events;

    IntDomainsRepresentations representations;
    IntDomainsActions actions;

    void initialize(int count);
    void deinitialize();

    /// Add a new domain (for a new variable), ranging from "min" to "max".
    void push(int min, int max);


    cudaDevice inline bool isEmpty(int index)
    {
        return representations.isEmpty(index);
    }
    cudaDevice bool isEmpty(int index, IntNeighborhood* nbh, int reprIdx = -1);

    cudaDevice inline bool isSingleton(int index)
    {
        return representations.isSingleton(index);
    }
    cudaDevice bool isSingleton(int index, IntNeighborhood* nbh, int reprIdx = -1);

    /**
    * Get an upper bound to the cardinality of the domain.
    * Any gaps in the domain are not counted.
    * \return the difference between the max and min values in the domain.
    */
    cudaDevice inline unsigned int getApproximateCardinality(int index)
    {
        return representations.getApproximateCardinality(index);
    }
    cudaDevice unsigned int getApproximateCardinality(int index, IntNeighborhood* nbh, int reprIdx = -1);

    cudaHostDevice inline int getMin(int index)
    {
        return representations.minimums[index];
    }
    cudaDevice int getMin(int index, IntNeighborhood* nbh, int reprIdx = -1);

    cudaDevice inline int getMax(int index)
    {
        return representations.maximums[index];
    }
    cudaDevice int getMax(int index, IntNeighborhood* nbh, int reprIdx = -1);

    /// Reduce the domain on the "index"-th variable to "value" (singleton).
    cudaDevice void fixValue(int index, int value);
    cudaDevice void fixValue(int index, int value, IntNeighborhood* nbh);

    /**
    * Perform the domain reduction actions pertaining the "index"-th 
    * variable/domain.
    * Values outside the bounds and inside the "remove list" are dropped.
    */
    cudaDevice void updateDomain(int index);
    cudaDevice void updateDomain(int index, IntNeighborhood* nbh);
};
