#include <choosers/IntRestrictInOrdVarChooser.h>
/// \file \author Elia

/**
* Choose a variable from a restricted pool, in order. *variable is its index.
* \return true if successful.
*/
cudaDevice bool IntRestrictInOrdVarChooser::getVariable(IntVariablesChooser* variablesChooser, int backtrackLevel, int* variable)
{
    if(variables->size == 0)
    {
        // pool empty, all variables allowed
        *variable = backtrackLevel;
    } else
    {
        // Select only from the pool
        if(backtrackLevel >= variables->size)
        {
            LogUtils::error(__PRETTY_FUNCTION__, "Invalid backtrack level (higher" + 
                            " than variable pool).");
            return false;
        }
        *variable = variables[backtracklevel];
    }
    return true;
}

void IntRestrictInOrdVarChooser::initialize(int expectedVarCount)
{
    if (expectedVarCount < 1)
    {
        variables.initialize();
    } else
    {
        variables.initialize(expectedVarCount);
    }
}

void IntRestrictInOrdVarChooser::addVariable(int variable)
{
    variables.push_back(variable);   
}