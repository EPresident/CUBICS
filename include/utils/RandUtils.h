#pragma once

#ifdef GPU
#include <utils/GpuUtils.h>
#include <curand_kernel.h>
#include <cassert>


/**
 * \file Utilities for PRNG (Pseudo-Random Number Generation) 
 */
namespace RandUtils
{
    /**
    * Generate a random integer between \a min and \a max (both included),
    * using a uniform distribution.
    * 
    * \param state the curandState that will be used. Must be created
    * and initialized with \a curand_init().
    * \param min the minumum value allowed.
    * \param max the maximum value allowed.
    */
    cudaDevice inline int uniformRand(curandState *state, int min, int max)
    {
        // Want to be able to choose from at least 2 values
        assert(max-min > 1);
        
        int idx = threadIdx.x + blockDim.x*blockIdx.x;

        float randFloat = curand_uniform(state+idx);
        randFloat *= (max - min + 0.999999); // Multiply by n° of values
        randFloat += min; // Add offset
        int randInt = (int)truncf(randFloat);
        return randInt;
    }

}
#endif
