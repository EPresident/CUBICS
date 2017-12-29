#pragma once

#include <utils/GpuUtils.h>
#include <curand_kernel.h>
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
    cudaDevice int uniformRand(curandState *state, int min, int max);
}
