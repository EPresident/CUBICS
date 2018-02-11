#include <constraints/IntOptUb.h>
#include <constraints/IntConstraints.h>

cudaDevice void IntOptUb::propagate(IntConstraints* constraints, int index, IntVariables* variables)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    Vector<int>* constraintParameters = &constraints->parameters[index];

    variables->domains.actions.removeAnyGreaterThan(constraintVariables->at(0), constraintParameters->at(0));
}
cudaDevice void IntOptUb::propagate(IntConstraints* constraints, int index, IntVariables* variables, IntNeighborhood* nbh)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    Vector<int>* constraintParameters = &constraints->parameters[index];

    nbh->neighActions.removeAnyGreaterThan(nbh->getRepresentationIndex(constraintVariables->at(0)), constraintParameters->at(0));
}

cudaDevice bool IntOptUb::satisfied(IntConstraints* constraints, int index, IntVariables* variables)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    Vector<int>* constraintParameters = &constraints->parameters[index];

    return variables->domains.representations.minimums[constraintVariables->at(0)] <= constraintParameters->at(0);
}
cudaDevice bool IntOptUb::satisfied(IntConstraints* constraints, int index, IntVariables* variables, IntNeighborhood* nbh)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    Vector<int>* constraintParameters = &constraints->parameters[index];

    return variables->domains.getMin(constraintVariables->at(0), nbh) <= constraintParameters->at(0);
}
