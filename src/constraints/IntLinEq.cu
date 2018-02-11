#include <cmath>

#include <constraints/IntLinEq.h>
#include <data_structures/Vector.h>
#include <constraints/IntConstraints.h>
#include <utils/LogUtils.h>

cudaDevice void IntLinEq::propagate(IntConstraints* constraints, int index, IntVariables* variables)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    Vector<int>* constraintParameters = &constraints->parameters[index];

    int sumPosCoeffLowValue = 0;
    int sumPosCoeffHighValue = 0;
    int sumNegCoeffLowValue = 0;
    int sumNegCoeffHighValue = 0;
    for (int i = 0; i < constraintVariables->size; i += 1)
    {
        int variableIndex = constraintVariables->at(i);
        int variableCoefficient = constraintParameters->at(i);

        if (variableCoefficient > 0)
        {
            sumPosCoeffLowValue += variableCoefficient * variables->domains.getMin(variableIndex);
            sumPosCoeffHighValue += variableCoefficient * variables->domains.getMax(variableIndex);
        }
        else
        {
            sumNegCoeffLowValue += (-variableCoefficient) * variables->domains.getMin(variableIndex);
            sumNegCoeffHighValue += (-variableCoefficient) * variables->domains.getMax(variableIndex);
        }
    }

    int b = constraintParameters->back();
    for (int i = 0; i < constraintVariables->size; i += 1)
    {
        int variableIndex = constraintVariables->at(i);
        int variableCoefficient = constraintParameters->at(i);

        if (variables->domains.isSingleton(variableIndex))
        {
            continue;
        }

        if (variableCoefficient > 0)
        {
            int variableLowContribution = variableCoefficient * variables->domains.getMin(variableIndex);
            int variableHightContribution = variableCoefficient * variables->domains.getMax(variableIndex);

            float alpha = (b - (sumPosCoeffLowValue - variableLowContribution) + sumNegCoeffHighValue) / static_cast<float>(variableCoefficient);
            float gamma = (b - (sumPosCoeffHighValue - variableHightContribution) + sumNegCoeffLowValue) / static_cast<float>(variableCoefficient);

            variables->domains.actions.removeAnyGreaterThan(variableIndex, static_cast<int>(floor(alpha)));
            variables->domains.actions.removeAnyLesserThan(variableIndex, static_cast<int>(ceil(gamma)));

        }
        else
        {
            int variableLowContribution = (-variableCoefficient) * variables->domains.getMin(variableIndex);
            int variableHightContribution = (-variableCoefficient) * variables->domains.getMax(variableIndex);

            float beta = (-b + sumPosCoeffLowValue - (sumNegCoeffHighValue - variableHightContribution)) / static_cast<float>(-variableCoefficient);
            float delta = (-b + sumPosCoeffHighValue - (sumNegCoeffLowValue - variableLowContribution)) / static_cast<float>(-variableCoefficient);

            variables->domains.actions.removeAnyLesserThan(variableIndex, static_cast<int>(ceil(beta)));
            variables->domains.actions.removeAnyGreaterThan(variableIndex, static_cast<int>(floor(delta)));
        }
    }
}
cudaDevice void IntLinEq::propagate(IntConstraints* constraints, int index, IntVariables* variables, IntNeighborhood* nbh)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    Vector<int>* constraintParameters = &constraints->parameters[index];

    int sumPosCoeffLowValue = 0;
    int sumPosCoeffHighValue = 0;
    int sumNegCoeffLowValue = 0;
    int sumNegCoeffHighValue = 0;
    for (int i = 0; i < constraintVariables->size; i += 1)
    {
        int variableIndex = constraintVariables->at(i);
        int variableCoefficient = constraintParameters->at(i);
        // Index in the neighborhood
        int neighVarIndex = -1;
        if(nbh->isNeighbor(variableIndex))
        {
            neighVarIndex = nbh->getRepresentationIndex(variableIndex);
        }
        
        if (variableCoefficient > 0)
        {
            sumPosCoeffLowValue += variableCoefficient * variables->domains.getMin(variableIndex, nbh, neighVarIndex);
            sumPosCoeffHighValue += variableCoefficient * variables->domains.getMax(variableIndex, nbh, neighVarIndex);
        }
        else
        {
            sumNegCoeffLowValue += (-variableCoefficient) * variables->domains.getMin(variableIndex, nbh, neighVarIndex);
            sumNegCoeffHighValue += (-variableCoefficient) * variables->domains.getMax(variableIndex, nbh, neighVarIndex);
        }
    }

    int b = constraintParameters->back();
    for (int i = 0; i < constraintVariables->size; i += 1)
    {
        int variableIndex = constraintVariables->at(i);
        int variableCoefficient = constraintParameters->at(i);
        // Index in the neighborhood
        int neighVarIndex = -1;
        if(nbh->isNeighbor(variableIndex))
        {
            neighVarIndex = nbh->getRepresentationIndex(variableIndex);
        }

        if (variables->domains.isSingleton(variableIndex, nbh, neighVarIndex))
        {
            continue;
        }

        if (variableCoefficient > 0)
        {
            int variableLowContribution = variableCoefficient * variables->domains.getMin(variableIndex, nbh, neighVarIndex);
            int variableHightContribution = variableCoefficient * variables->domains.getMax(variableIndex, nbh, neighVarIndex);

            float alpha = (b - (sumPosCoeffLowValue - variableLowContribution) + sumNegCoeffHighValue) / static_cast<float>(variableCoefficient);
            float gamma = (b - (sumPosCoeffHighValue - variableHightContribution) + sumNegCoeffLowValue) / static_cast<float>(variableCoefficient);

            if(neighVarIndex > 0)
            {
                // Variable in the neighborhood
                nbh->neighActions.removeAnyGreaterThan(neighVarIndex, static_cast<int>(floor(alpha)));
                nbh->neighActions.removeAnyLesserThan(neighVarIndex, static_cast<int>(ceil(gamma)));
            }
            else
            {
                // This should NEVER happen
                LogUtils::error(__PRETTY_FUNCTION__, "Trying to change a non-neighbor!");
            }
            
        }
        else
        {
            int variableLowContribution = (-variableCoefficient) * variables->domains.getMin(variableIndex, nbh, neighVarIndex);
            int variableHightContribution = (-variableCoefficient) * variables->domains.getMax(variableIndex, nbh, neighVarIndex);

            float beta = (-b + sumPosCoeffLowValue - (sumNegCoeffHighValue - variableHightContribution)) / static_cast<float>(-variableCoefficient);
            float delta = (-b + sumPosCoeffHighValue - (sumNegCoeffLowValue - variableLowContribution)) / static_cast<float>(-variableCoefficient);

            if(neighVarIndex > 0)
            {
                // Variable in the neighborhood
                nbh->neighActions.removeAnyLesserThan(neighVarIndex, static_cast<int>(ceil(beta)));
                nbh->neighActions.removeAnyGreaterThan(neighVarIndex, static_cast<int>(floor(delta)));
            }
            else
            {
                // This should NEVER happen
                LogUtils::error(__PRETTY_FUNCTION__, "Trying to change a non-neighbor!");
            }
        }
    }
}

cudaDevice bool IntLinEq::satisfied(IntConstraints* constraints, int index, IntVariables* variables)
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

    int sum = 0;
    for (int i = 0; i < constraintVariables->size; i += 1)
    {
        sum += constraintParameters->at(i) * variables->domains.getMin(constraintVariables->at(i));
    }

    return sum == constraintParameters->back();
}
cudaDevice bool IntLinEq::satisfied(IntConstraints* constraints, int index, IntVariables* variables, IntNeighborhood* nbh)
{
    //Satisfaction check is performed only when all variables is ground

    Vector<int>* constraintVariables = &constraints->variables[index];
    Vector<int>* constraintParameters = &constraints->parameters[index];

    for (int i = 0; i < constraintVariables->size; i += 1)
    {
        if (not variables->domains.isSingleton(constraintVariables->at(i), nbh))
        {
            return true;
        }
    }

    int sum = 0;
    for (int i = 0; i < constraintVariables->size; i += 1)
    {
        sum += constraintParameters->at(i) * variables->domains.getMin(constraintVariables->at(i), nbh);
    }

    return sum == constraintParameters->back();
}
