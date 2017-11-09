#include <constraints/IntOptLb.h>
#include <constraints/IntConstraints.h>

cudaDevice void IntOptLb::propagate(IntConstraints* constraints, int index, IntVariables* variables)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    Vector<int>* constraintParameters = &constraints->parameters[index];

    variables->domains.actions.removeAnyLesserThan(constraintVariables->at(0), constraintParameters->at(0));
}

cudaDevice bool IntOptLb::satisfied(IntConstraints* constraints, int index, IntVariables* variables)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    Vector<int>* constraintParameters = &constraints->parameters[index];

    return variables->domains.representations.minimums[constraintVariables->at(0)] >= constraintParameters->at(0);
}
