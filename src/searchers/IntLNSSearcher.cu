#include <searchers/IntLNSSearcher.h>
#include <utils/Utils.h>
#include <wrappers/Wrappers.h>

void IntLNSSearcher::initialize(FlatZinc::FlatZincModel* fzModel, double unassignRate)
{
    variables = fzModel->intVariables;
    constraints = fzModel->intConstraints;

    chosenVariables.initialize(variables->count);
    chosenValues.initialize(variables->count);

    BTSearcher.initialize(fzModel);

    LNSState = VariableNotChosen;
    unassignmentRate = unassignRate;
    iterationsDone = 0;
    
#ifdef GPU
    varibalesBlockCount = KernelUtils::getBlockCount(variables->count, DEFAULT_BLOCK_SIZE);
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
        LogUtils::error(__PRETTY_FUNCTION__, "Large Neighborhood Search is only"+
                        " possible on optimization problems!");
    }
}

void IntLNSSearcher::deinitialize()
{
    chosenVariables.deinitialize();
    
    BTSearcher.deinitialize();
}

/**
* Find the next solution, backtracking when needed.
* \return true if a solution is found, false otherwise.
*/
cudaDevice bool IntLNSSearcher::getNextSolution()
{
    bool solutionFound = false;

    while (not solutionFound)
    {
        switch (backtrackingState)
        {
            case Initialized:
            {
                // Find first solution
            }
                break;

            case FirstSolutionFound:
            {
                // Unassign variables
            }
                break;
            case VariablesUnassigned:
            {
                // Begin exploring the new neighborhood
                ++iterationsDone;
            }
                break;

            case NeighborhoodExplored:
            {
                // Subtree explored
            }
                break;
        }
    }

    return solutionFound;
}

