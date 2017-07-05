#pragma once

#include <choosers/IntValuesChooser.h>

namespace IntInOrderValuesChooser
{
    bool getFirstValue(IntValuesChooser* valuesChooser, int variable, int* firstValue);
    bool getNextValue(IntValuesChooser* valuesChooser, int variable, int currentValue, int* nextValue);
}
