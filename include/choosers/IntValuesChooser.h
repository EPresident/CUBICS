#pragma once

#include <variables/IntVariables.h>

struct IntValuesChooser
{
    enum Type
    {
        InOrder,
        Random
    };

    int type;
    /// Seed for the PRNG
    long randSeed;
    const static long DEFAULT_SEED = 543219876;
    
    IntVariables* variables;
    #ifdef GPU
        /// State for the cuRAND PRNG library (GPU)
        curandState* cuRANDstate;
    #endif

    void initialzie(int type, IntVariables* variables, long seed = DEFAULT_SEED);

    /**
    * Get the first value (following the chooser's criteria) for a variable.
    * \return true if successful.
    */
    cudaDevice bool getFirstValue(int variable, int* firstValue);
    /**
    * Get the next value (following the chooser's criteria) for a variable.
    * \return true if successful.
    */
    cudaDevice bool getNextValue(int variable, int currentValue, int* nextValue);
};
