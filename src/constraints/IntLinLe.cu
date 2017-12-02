#include <cmath>

#include <constraints/IntLinLe.h>
#include <data_structures/Vector.h>
#include <constraints/IntConstraints.h>

cudaDevice void IntLinLe::propagate(IntConstraints* constraints, int index, IntVariables* variables)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    Vector<int>* constraintParameters = &constraints->parameters[index];

    int sumNegCoeffHighValue = 0;
    int sumPosCoeffLowValue = 0;
    for (int i = 0; i < constraintVariables->size; i += 1)
    {
        int variableIndex = constraintVariables->at(i);
        int variableCoefficient = constraintParameters->at(i);

        if (variableCoefficient > 0)
        {
            int variableValue = variables->domains.getMin(variableIndex);
            sumPosCoeffLowValue += variableCoefficient * variableValue;
        }
        else
        {
            int variableValue = variables->domains.getMax(variableIndex);
            sumNegCoeffHighValue += (-variableCoefficient) * variableValue;
        }
    }

    int b = constraintParameters->back();
    for (int i = 0; i < constraintVariables->size; i += 1)
    {
        int variableIndex = constraintVariables->at(i);
        int variableCoefficient = constraintParameters->at(i);

        if(variables->domains.isSingleton(variableIndex))
        {
            continue;
        }

        if (variableCoefficient > 0)
        {
            int variableLowContribution = variableCoefficient * variables->domains.getMin(variableIndex);
            float alpha = static_cast<float>(b - (sumPosCoeffLowValue - variableLowContribution) + sumNegCoeffHighValue) / static_cast<float>(variableCoefficient);
            variables->domains.actions.removeAnyGreaterThan(variableIndex, static_cast<int>(floor(alpha)));
        }
        else
        {
            int variableLowContribution = (-variableCoefficient) * variables->domains.getMax(variableIndex);
            float beta = static_cast<float>(-b + sumPosCoeffLowValue - (sumNegCoeffHighValue - variableLowContribution)) / static_cast<float>(-variableCoefficient);
            variables->domains.actions.removeAnyLesserThan(variableIndex, static_cast<int>(ceil(beta)));
        }
    }
}


cudaDevice bool IntLinLe::satisfied(IntConstraints* constraints, int index, IntVariables* variables)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    Vector<int>* constraintParameters = &constraints->parameters[index];

    //Satisfaction check is performed only when all variables is ground
    for (int i = 0; i < constraintVariables->size; i += 1)
    {
        if (not variables->domains.isSingleton(constraintVariables->at(i)))
        {
            return true;
        }
    }

    int sum = 0;
    for (int vi = 0; vi < constraintVariables->size; vi += 1)
    {
        sum += constraintParameters->at(vi) * variables->domains.getMin(constraintVariables->at(vi));
    }

    return sum <= constraintParameters->back();
}
