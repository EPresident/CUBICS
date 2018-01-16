#include <choosers/IntRandomValuesChooser.h>

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

cudaDevice bool IntRandomValuesChooser::getNextValue(IntValuesChooser* valuesChooser, int variable, int currentValue, int* nextValue)
{
    IntDomainsRepresentations* intDomRepr  = &valuesChooser->variables->domains.representations;
    
    int val;
    unsigned int attempts = 1000000000;
    do{
        val = RandUtils::uniformRand(valuesChooser->cuRANDstate, intDomRepr->minimums[variable],
            intDomRepr->maximums[variable]);
        --attempts;
    }while(attempts > 0 && !intDomRepr->contain(variable,val));
    *nextValue = val;
    return attempts > 0;
}
