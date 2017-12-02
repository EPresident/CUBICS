#include <searchers/IntBacktrackSearcher.h>
#include <utils/Utils.h>
#include <wrappers/Wrappers.h>

void IntBacktrackSearcher::initialize(IntVariables* variables, IntConstraints* constraints)
{
    this->variables = variables;
    this->constraints = constraints;

    chosenVariables.initialize(variables->count);
    chosenValues.initialize(variables->count);

    stack.initialize(&variables->domains.representations);

    variablesChooser.initialzie(IntVariablesChooser::InOrder, variables, &chosenVariables);
    valuesChooser.initialzie(IntValuesChooser::InOrder, variables);

    propagator.initialize(variables, constraints);

    backtrackingLevel = 0;
    backtrackingState = VariableNotChosen;

#ifdef GPU
    varibalesBlockCount = KernelUtils::getBlockCount(variables->count, DEFAULT_BLOCK_SIZE);
#endif
}

void IntBacktrackSearcher::deinitialize()
{
    chosenVariables.deinitialize();

    chosenValues.deinitialize();

    stack.deinitialize();

    propagator.deinitialize();
}


cudaDevice bool IntBacktrackSearcher::getNextSolution()
{
    bool solutionFound = false;

    while (backtrackingLevel >= 0 and (not solutionFound))
    {
        switch (backtrackingState)
        {
            case VariableNotChosen:
            {
#ifdef GPU
                Wrappers::saveState<<<varibalesBlockCount, DEFAULT_BLOCK_SIZE>>>(&stack, backtrackingLevel);
                cudaDeviceSynchronize();
#else
                stack.saveState(backtrackingLevel, &variables->domains.changes);
#endif
                variables->domains.changes.clear();

                if (variablesChooser.getVariable(backtrackingLevel, &chosenVariable))
                {
                    chosenVariables.push_back(chosenVariable);
                    backtrackingState = VariableChosen;
                }
                else
                {
                    LogUtils::error(__PRETTY_FUNCTION__, "Failed to set variable");
                }
            }
                break;

            case VariableChosen:
            {
                if (not variables->domains.isSingleton(chosenVariables.back()))
                  {

                    if (valuesChooser.getFirstValue(chosenVariables.back(), &chosenValue))
                    {
                        chosenValues.push_back(chosenValue);
                        variables->domains.fixValue(chosenVariables.back(), chosenValues.back());
                        backtrackingState = ValueChosen;
                    }
                    else
                    {
                        LogUtils::error(__PRETTY_FUNCTION__, "Failed to set first value");
                    }
                }
                else
                {
                    chosenValues.push_back(variables->domains.getMin(chosenVariables.back()));
                    backtrackingState = SuccessfulPropagation;
                }
            }
                break;
            case ValueChosen:
            {
                bool noEmptyDomains = propagator.propagateConstraints();

                if (noEmptyDomains)
                {
                    backtrackingState = SuccessfulPropagation;
                }
                else
                {
                    backtrackingState = ValueChecked;
                }
            }
                break;

            case SuccessfulPropagation:
            {
                if (backtrackingLevel < variables->count - 1)
                {
                    backtrackingLevel += 1;
                    backtrackingState = VariableNotChosen;
                }
                else
                {
                    backtrackingState = ValueChecked;

                    if (propagator.verifyConstraints())
                    {
                        solutionFound = true;
                    }
                }
            }
                break;

            case ValueChecked:
            {
#ifdef GPU
<<<<<<< HEAD
                Wrappers::restoreState<<<1, 1>>>(&stack, &variables->domains.changes);
=======
                Wrappers::restoreState<<<varibalesBlockCount, DEFAULT_BLOCK_SIZE>>>(&stack, backtrackingLevel);
>>>>>>> 0d1cbffb... [Stack] Parallelize integer backtrack stack
                cudaDeviceSynchronize();
#else
                stack.resetState(&variables->domains.changes);
#endif
                variables->domains.changes.clear();

                if (valuesChooser.getNextValue(chosenVariables.back(), chosenValues.back(), &chosenValue))
                {
                    chosenValues.back() = chosenValue;
                    variables->domains.fixValue(chosenVariables.back(), chosenValues.back());
                    backtrackingState = ValueChosen;
                }
                else
                {

                    if(backtrackingLevel > 0)
                    {
#ifdef GPU
                        Wrappers::clearState<<<varibalesBlockCount, DEFAULT_BLOCK_SIZE>>>(&stack, backtrackingLevel);
                        cudaDeviceSynchronize();
#else
                        stack.restorePreviousState(backtrackingLevel);
#endif
                    }


                    backtrackingLevel -= 1;
                    chosenVariables.pop_back();
                    chosenValues.pop_back();
                }
            }
                break;
        }
    }

    return solutionFound;
}
