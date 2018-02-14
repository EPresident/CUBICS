#include <choosers/IntInOrderValuesChooser.h>
#include <cassert>

cudaDevice bool IntInOrderValuesChooser::getFirstValue(IntValuesChooser* valuesChooser, int variable, int* firstValue)
{
    *firstValue = valuesChooser->variables->domains.representations.minimums[variable];
    return true;
}
cudaDevice bool IntInOrderValuesChooser::getFirstValue(IntValuesChooser* valuesChooser, int variable, int* firstValue, IntNeighborhood* nbh)
{
    assert(nbh->isNeighbor(variable));
    *firstValue = valuesChooser->variables->domains.getMin(variable, nbh);
    return true;
}

cudaDevice bool IntInOrderValuesChooser::getNextValue(IntValuesChooser* valuesChooser, int variable, int currentValue, int* nextValue)
{
    return valuesChooser->variables->domains.representations.getNextValue(variable, currentValue, nextValue);
}
cudaDevice bool IntInOrderValuesChooser::getNextValue(IntValuesChooser* valuesChooser, int variable, int currentValue, int* nextValue, IntNeighborhood* nbh)
{
    assert(nbh->isNeighbor(variable));
    return nbh->neighRepr.getNextValue(nbh->getRepresentationIndex(variable), currentValue, nextValue);
}
