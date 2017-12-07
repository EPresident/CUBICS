#pragma once

#include <data_structures/Vector.h>

struct IntDomainsRepresentations
{
    /// Lower bounds for the variable domains.
    Vector<int> minimums;
    /// Upper bounds for the variable domains.
    Vector<int> maximums;
    /**
    * The first value for the i-th variable is offsets[i].
    * So minimums[i]-offsets[i] should be zero, and 
    * maximums[i]-offsets[i] should be the theoretical max
    * cardinality.
    * The offset can (technically) be assigned independent
    * of the bounds, so the above might not apply (but it really should).
    */
    Vector<int> offsets;
    /**
    * A counter of the number of modifications to the domain of the
    * i-th variable.
    */
    Vector<unsigned int> versions;
    /**
    * Each variable has a list of bitvectors that indicate whether a
    * certain value is in domain or not. In a sense, bitvectors can be seen
    * as value intervals, whose union is the variable domain.
    *
    * These bitvectors/intervals are referred to as "chunks" for brevity.
    * (as in, they represent a chunk of the domain)
    *
    * Let's make the example of a variable X, whose domain is made by a single
    * chunk, i.e. the byte 11101011; this means X's domain is {0,1,2,4,6,7},
    * assuming the offset for X is zero.
    */
    Vector<Vector<unsigned int>> bitvectors;

    /// Initialize the domain representations for "count" variables.
    void initialize(int count);
    void deinitialize();

    /** 
    * Add the interval ["min","max"] to a new domain (for a new variable), 
    * using "min" as offset. 
    */
    void push(int min, int max);
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
    cudaDevice void push(int min, int max, int offset, unsigned int version, Vector<unsigned int>* bitvector);
    /// Remove the last representation added.
    cudaDevice void pop();

    /** 
    * Get the index of the bitvector representing the interval which
    * contains the value "val", for the domain of variable "index".
    * In other words, find i such that val is in the interval bitvectors[index][i].
    */
    cudaHostDevice inline int getChunkIndex(int index, int val)
    {
        return abs(val - offsets[index]) / UINT_BIT_SIZE;
    }

    /**
    * Get the index of the bit (belonging to one of the bitvectors/intervals) 
    * indicating whether "val" is part of the domain or not.
    */
    cudaHostDevice inline int getBitIndex(int index, int val)
    {
        return abs(val - offsets[index]) % UINT_BIT_SIZE;
    }

    cudaDevice inline int getValue(int index, int chunkIndex, int bitIndex)
    {
        return offsets[index] + (UINT_BIT_SIZE * chunkIndex) + bitIndex;
    }

    cudaDevice inline bool isEmpty(int index)
    {
        return minimums[index] > maximums[index];
    }

    cudaDevice  bool isSingleton(int index)
    {
        return minimums[index] == maximums[index];
    }

    /**
    * Get an upper bound to the cardinality of the domain.
    * Any gaps in the domain are not counted.
    * \return the difference between the max and min values in the domain.
    */
    cudaDevice inline unsigned int getApproximateCardinality(int index)
    {
        return abs(maximums[index] - minimums[index]) + 1;
    }

    /** 
    * Returns true if the "index"-th chunk contains "val", i.e. the bit
    * representing "val" is 1.
    */
    cudaDevice bool contain(int index, int val);
    /** 
    * Set "nextVal" to point to the next value in the "index"-th domain, after "val".
    * \param nextVal a pointer that will be set to the next value.
    * \return false if no such value exists, true otherwise.
    */
    cudaDevice bool getNextValue(int index, int val, int* nextVal);
    /** 
    * Set "prevVal" to point to the previous value in the "index"-th domain,
    * after "val".
    * \param prevVal a pointer that will be set to the previous value.
    * \return false if no such value exists, true otherwise.
    */
    cudaDevice bool getPrevValue(int index, int val, int* prevVal);

    /**
    * Remove the value "val" from the domain of the "index"-th variable, updating
    * the domain representation accordingly (minimum, maximum, cardinality).
    */
    cudaDevice void remove(int index, int val);
    /**
    * Remove all values lesser than "val" from the domain of the "index"-th 
    * variable, updating the domain representation accordingly (minimum, 
    * maximum, cardinality).
    */
    cudaDevice void removeAnyLesserThan(int index, int val);
    /**
    * Remove all values greater than "val" from the domain of the "index"-th 
    * variable, updating the domain representation accordingly (minimum, 
    * maximum, cardinality).
    */
    cudaDevice void removeAnyGreaterThan(int index, int val);

    /**
    * Remove all values from the domain of the "index"-th variable, thus making it 
    * an empty set.
    */
    cudaDevice void removeAll(int index);
    /**
    * Remove all values other than "val" from the domain of the "index"-th 
    * variable, thus making it a singleton.
    */
    cudaDevice void keepOnly(int index, int val);
};
