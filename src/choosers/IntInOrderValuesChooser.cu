#include <choosers/IntInOrderValuesChooser.h>

bool IntInOrderValuesChooser::getFirstValue(IntValuesChooser* valuesChooser, int variable, int* firstValue)
{
    *firstValue = valuesChooser->variables->domains.representations.minimums[variable];
    return true;
}

bool IntInOrderValuesChooser::getNextValue(IntValuesChooser* valuesChooser, int variable, int currentValue, int* nextValue)
{
    return valuesChooser->variables->domains.representations.getNextValue(variable, currentValue, nextValue);
}
