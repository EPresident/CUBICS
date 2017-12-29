#include <utils/RandUtils.h>
#include <cmath>

/**
* Generate a random integer between \a min and \a max (both included),
* using a uniform distribution.
* 
* \param state the curandState that will be used. Must be created
* and initialized with \a curand_init().
* \param min the minumum value allowed.
* \param max the maximum value allowed.
*/
cudaDevice int uniformRand(curandState *state, int min, int max)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    float randFloat = curand_uniform(state+idx);
    randFloat *= (max - min + 0.999999); // Multiply by n° of values
    randFloat += min; // Add offset
    int randInt = (int)truncf(randFloat);
    return randInt;
}