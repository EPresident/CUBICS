#include <searchers/IntBacktrackSearcher.h>
#include <utils/Utils.h>
#include <wrappers/Wrappers.h>

void IntBacktrackSearcher::initialize(FlatZinc::FlatZincModel* fzModel)
{
    variables = fzModel->intVariables;
    constraints = fzModel->intConstraints;

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
    timer.initialize();
#endif

    switch (fzModel->method())
    {
        case FlatZinc::FlatZincModel::Meth::SAT:
        {
            searchType = Satisfiability;
        }
            break;
        case FlatZinc::FlatZincModel::Meth::MAX:
        {
            searchType = Maximization;
        }
            break;
        case FlatZinc::FlatZincModel::Meth::MIN:
        {
            searchType = Minimization;
        }
            break;
    }

    if (searchType == Maximization or searchType == Minimization)
    {
        optVariable = fzModel->optVar();
        optConstraint = fzModel->optConst();
    }
    else
    {
        optVariable = -1;
        optConstraint = -1;
    }
}

void IntBacktrackSearcher::deinitialize()
{
    chosenVariables.deinitialize();

    chosenValues.deinitialize();

    stack.deinitialize();

    propagator.deinitialize();
}

/**
* Find the next solution, backtracking when needed.
* \return true if a solution is found, false otherwise.
*/
cudaDevice bool IntBacktrackSearcher::getNextSolution(long long timeout)
{
    bool solutionFound = false;
    #ifndef NDEBUG
        assert(backtrackingState < 5);
        assert(backtrackingLevel < variables->count);
    #endif

    while (backtrackingLevel >= 0 and (not solutionFound) and timeout > 0)
    {
        // Setup timer to compute this iteration's duration
        timer.setStartTime();
        
        switch (backtrackingState)
        {
            case VariableNotChosen:
            {
                // Backup current state (GPU/CPU)
#ifdef GPU
                Wrappers::saveState<<<varibalesBlockCount, DEFAULT_BLOCK_SIZE>>>(&stack, backtrackingLevel);
                cudaDeviceSynchronize();
#else
                stack.saveState(backtrackingLevel);
#endif
                // Choose a variable to assign
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
                // Choose a value for the variable
                if (not variables->domains.isSingleton(chosenVariables.back()))
                {
                    // Not a singleton, use the chooser
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
                    // Domain's a singleton, choose the only possible value.
                    chosenValues.push_back(variables->domains.getMin(chosenVariables.back()));
                    // Nothing has been changed, so no need to propagate.
                    backtrackingState = SuccessfulPropagation;
                }
            }
                break;
            case ValueChosen:
            {
                // A domain has been changed, need to propagate.
                bool noEmptyDomains = propagator.propagateConstraints();

                if (noEmptyDomains)
                {
                    backtrackingState = SuccessfulPropagation;
                }
                else
                {
                    // A domain has been emptied, try another value.
                    backtrackingState = ValueChecked;
                }
            }
                break;

            case SuccessfulPropagation:
            {
                if (backtrackingLevel < variables->count - 1)
                {
                    // Not all variables have been assigned, move to the next
                    backtrackingLevel += 1;
                    backtrackingState = VariableNotChosen;
                }
                else
                {
                    // All variables assigned
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
                // Revert last value choice on the stack (GPU/CPU)
#ifdef GPU
                Wrappers::restoreState<<<varibalesBlockCount, DEFAULT_BLOCK_SIZE>>>(&stack, backtrackingLevel);
                cudaDeviceSynchronize();
#else
                stack.restoreState(backtrackingLevel);
#endif
                // Choose another value, if possible
                if (valuesChooser.getNextValue(chosenVariables.back(), chosenValues.back(), &chosenValue))
                {
                    // There's another value
                    chosenValues.back() = chosenValue;
                    variables->domains.fixValue(chosenVariables.back(), chosenValues.back());
                    backtrackingState = ValueChosen;
                }
                else
                {
                    // All values have been used, backtrack.
                    // Clear current level state from the stack (GPU/CPU)
#ifdef GPU
                    Wrappers::clearState<<<varibalesBlockCount, DEFAULT_BLOCK_SIZE>>>(&stack, backtrackingLevel);
                    cudaDeviceSynchronize();
#else
                    stack.clearState(backtrackingLevel);
#endif
                    // Return to the previous backtrack level
                    // and resume the search from there
                    backtrackingLevel -= 1;
                    chosenVariables.pop_back();
                    chosenValues.pop_back();
                }
            }
                break;
        }
        
        // Compute elapsed time and subtract it from timeout
        timeout -= timer.getElapsedTime();
    }

    // Make sure the next solution is better (for optimization problems)
    if (solutionFound and (searchType == Maximization or searchType == Minimization))
    {
        shrinkOptimizationBound();
    }
    
    if(timeout <= 0)
    {
        printf(">>> GPU Timed out! <<<\n");
    }

    return solutionFound;
}

/**
* Require that the optimization variable ("optVariable") take a value
* greater/smaller than its minumum/maximum value (for a 
* maximization/minimization problem).
* 
* In other words, once a solution with value x has been found, calling this
* function will require the next solution to have value (at least) x(+/-)1.
*/
cudaDevice void IntBacktrackSearcher::shrinkOptimizationBound()
{
    if (searchType == Maximization)
    {
        constraints->parameters[optConstraint][0] = variables->domains.getMin(optVariable) + 1;
    }
    else if (searchType == Minimization)
    {
        constraints->parameters[optConstraint][0] = variables->domains.getMin(optVariable) - 1;
    }
}
