#include <constraints/IntLinNe.h>
#include <data_structures/Vector.h>
#include <constraints/IntConstraints.h>

cudaDevice void IntLinNe::propagate(IntConstraints* constraints, int index, IntVariables* variables)
{
    //Constraint propagation is performed when only one variable is not ground

    Vector<int>* constraintVariables = &constraints->variables[index];
    Vector<int>* constraintParameters = &constraints->parameters[index];

    int notFixedVariablesCount = 0;
    int notFixedVariableIndex = 0;
    for (int vi = 0; vi < constraintVariables->size; vi += 1)
    {
        if (not variables->domains.isSingleton(constraintVariables->at(vi)))
        {
            notFixedVariablesCount += 1;
            notFixedVariableIndex = vi;
        }
    }

    if (notFixedVariablesCount == 1)
    {
        int value = 0;
        for (int i = 0; i < constraintVariables->size; i += 1)
        {
            if (i != notFixedVariableIndex)
            {
                value += constraintParameters->at(i) * variables->domains.getMin(constraintVariables->at(i));
            }
        }
        int toRemove = -value + constraintParameters->back();

        if (toRemove % constraintParameters->at(notFixedVariableIndex) == 0)
        {
            toRemove /= constraintParameters->at(notFixedVariableIndex);
            variables->domains.actions.removeElement(constraintVariables->at(notFixedVariableIndex), toRemove);
        }
    }
}

cudaDevice bool IntLinNe::satisfied(IntConstraints* constraints, int index, IntVariables* variables)
{
    //Satisfaction check is performed only when all variables is ground

    Vector<int>* constraintVariables = &constraints->variables[index];
    Vector<int>* constraintParameters = &constraints->parameters[index];

    for (int i = 0; i < constraintVariables->size; i += 1)
    {
        if (not variables->domains.isSingleton(constraintVariables->at(i)))
        {
            return true;
        }
    }

    int value = 0;
    for (int i = 0; i < constraintVariables->size; i += 1)
    {
        value += constraintParameters->at(i) * variables->domains.getMin(constraintVariables->at(i));
    }

    return value != constraintParameters->back();
}
