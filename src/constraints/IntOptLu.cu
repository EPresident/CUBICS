#include <constraints/IntOptLb.h>
#include <constraints/IntConstraints.h>

cudaDevice void IntOptLb::propagate(IntConstraints* constraints, int index, IntVariables* variables)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    Vector<int>* constraintParameters = &constraints->parameters[index];

    variables->domains.actions.removeAnyLesserThan(constraintVariables->at(0), constraintParameters->at(0));
}
cudaDevice void IntOptLb::propagate(IntConstraints* constraints, int index, IntVariables* variables, IntNeighborhood* nbh)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    Vector<int>* constraintParameters = &constraints->parameters[index];

    nbh->neighActions.removeAnyLesserThan(nbh->getRepresentationIndex(constraintVariables->at(0)),
            constraintParameters->at(0));
}


cudaDevice bool IntOptLb::satisfied(IntConstraints* constraints, int index, IntVariables* variables)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    Vector<int>* constraintParameters = &constraints->parameters[index];

    return variables->domains.representations.minimums[constraintVariables->at(0)] >= constraintParameters->at(0);
}
cudaDevice bool IntOptLb::satisfied(IntConstraints* constraints, int index, IntVariables* variables, IntNeighborhood* nbh)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    Vector<int>* constraintParameters = &constraints->parameters[index];

    return variables->domains.getMin(constraintVariables->at(0), nbh) >= constraintParameters->at(0);
}
