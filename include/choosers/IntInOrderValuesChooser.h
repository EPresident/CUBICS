#pragma once

#include <choosers/IntValuesChooser.h>

namespace IntInOrderValuesChooser
{
    cudaDevice bool getFirstValue(IntValuesChooser* valuesChooser, int variable, int* firstValue);
    cudaDevice bool getFirstValue(IntValuesChooser* valuesChooser, 
        int variable, int* firstValue, IntNeighborhood* nbh);
    cudaDevice bool getNextValue(IntValuesChooser* valuesChooser, 
        int variable, int currentValue, int* nextValue);
    cudaDevice bool getNextValue(IntValuesChooser* valuesChooser, 
        int variable, int currentValue, int* nextValue, IntNeighborhood* nbh);
}
