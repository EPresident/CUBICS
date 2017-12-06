#pragma once

#include <domains/IntDomainsRepresentations.h>
#include <domains/IntDomainsActions.h>

struct IntDomains
{
    enum EventTypes
    {
        None,
        Changed ///< A domain has been changed
    };

    /// A list of domain events (domain changed) in chronological order.
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

    cudaDevice inline bool isSingleton(int index)
    {
        return representations.isSingleton(index);
    }

    /**
    * Get an upper bound to the cardinality of the domain.
    * Any gaps in the domain are not counted.
    * \return the difference between the max and min values in the domain.
    */
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

    /// Reduce the domain on the "index"-th variable to "value" (singleton).
    cudaDevice void fixValue(int index, int value);

    /**
    * Perform the domain reduction actions pertaining the "index"-th 
    * variable/domain.
    * Values outside the bounds and inside the "remove list" are dropped.
    */
    cudaDevice void updateDomain(int index);
};
