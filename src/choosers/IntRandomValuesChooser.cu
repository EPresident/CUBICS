#include <choosers/IntRandomValuesChooser.h>

/**
 * Initializes the curand/PRNG state, then behaves like \a getNextValue().
 */
cudaDevice bool IntRandomValuesChooser::getFirstValue(IntValuesChooser* valuesChooser, int variable, int* firstValue)
{
    #ifdef GPU
        // init cuRAND state with the given seed and no offset
        MemUtils::malloc(&valuesChooser->cuRANDstate);
        curand_init(valuesChooser->randSeed, threadIdx.x + blockIdx.x * blockDim.x,
            0, valuesChooser->cuRANDstate);
    #endif
    return getNextValue(valuesChooser, variable, 0, firstValue);
}

/**
 * If multiple values are available to choose from, do so randomly.
 * A random value is valid only if present in the domain.
 * A limited number of attempts is made to avoid looping indefinitely.
 * If only a single value is available, it is returned deterministically.
 * It is supposed that variables examined here have a non empty domain.
 * 
 * Of course this function has no way of knowing when all values have been
 * returned!!
 * 
 * \return false if PRNG fails to get a valid value in (1000+approx.cardinality)
 * attempts, true otherwise.
 */
cudaDevice bool IntRandomValuesChooser::getNextValue(IntValuesChooser* valuesChooser, int variable, int currentValue, int* nextValue)
{
    IntDomainsRepresentations* intDomRepr  = &valuesChooser->variables->domains.representations;
    
    int val;
    int max = intDomRepr->maximums[variable];
    int min = intDomRepr->minimums[variable];
    long attempts = max - min + 1000;
    if (max - min > 1)
    {
        do{
            val = RandUtils::uniformRand(valuesChooser->cuRANDstate, min, max);
            --attempts;
        }while(attempts > 0 && !intDomRepr->contain(variable,val));
        *nextValue = val;
        return attempts > 0;
    }
    else
    {
        // Only a single value to choose from...
        *nextValue = min;
        return true;
    }
    
}
