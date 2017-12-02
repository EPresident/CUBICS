#pragma once

#include <choosers/IntValuesChooser.h>

namespace IntInOrderValuesChooser
{
    cudaDevice bool getFirstValue(IntValuesChooser* valuesChooser, int variable, int* firstValue);
    cudaDevice bool getNextValue(IntValuesChooser* valuesChooser, int variable, int currentValue, int* nextValue);
}
