#include <choosers/IntRandomValuesChooser.h>

cudaDevice bool IntRandomValuesChooser::getFirstValue(IntValuesChooser* valuesChooser, int variable, int* firstValue)
{
    // init cuRAND state with the given seed and no offset
    MemUtils::malloc(&cuRANDstate);
    curand_init(randSeed, threadIdx.x + blockIdx.x * blockDim.x, 0, cuRANDstate);
    *firstValue = valuesChooser->variables->domains.representations.minimums[variable];
    return true;
}

cudaDevice bool IntRandomValuesChooser::getNextValue(IntValuesChooser* valuesChooser, int variable, int currentValue, int* nextValue)
{
    return valuesChooser->variables->domains.representations.getNextValue(variable, currentValue, nextValue);
}
