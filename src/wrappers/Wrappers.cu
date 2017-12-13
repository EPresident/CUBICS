#ifdef GPU
#include <wrappers/Wrappers.h>

//Integer backtracking stack
cudaGlobal void Wrappers::saveState(IntBacktrackStack* backtrackStack, int backtrackingLevel)
{
    backtrackStack->saveState(backtrackingLevel);
}

cudaGlobal void Wrappers::restoreState(IntBacktrackStack* backtrackStack, int backtrackingLevel)
{
    backtrackStack->restoreState(backtrackingLevel);
}

cudaGlobal void Wrappers::clearState(IntBacktrackStack* backtrackStack, int backtrackingLevel)
{
    backtrackStack->clearState(backtrackingLevel);
}

//Integer backtracking searcher
cudaGlobal void Wrappers::getNextSolution(IntBacktrackSearcher* backtrackSearcher, bool* solutionFound)
{
    *solutionFound = backtrackSearcher->getNextSolution();
}

//Integer LNS searcher
cudaGlobal void Wrappers::getNextSolution(IntLNSSearcher* LNSSearcher, bool* solutionFound)
{
    *solutionFound = LNSSearcher->getNextSolution();
}

//Integer constraints propagator
cudaGlobal void Wrappers::propagateConstraints(IntConstraintsPropagator* propagator, bool* satisfiableModel)
{
    *satisfiableModel = propagator->propagateConstraints();
}

cudaGlobal void Wrappers::setConstraintsToPropagate(IntConstraintsPropagator* propagator)
{
    propagator->setConstraintsToPropagate();
}

cudaGlobal void Wrappers::collectActions(IntConstraintsPropagator* propagator)
{
    propagator->collectActions();
}

cudaGlobal void Wrappers::clearDomainsEvents(IntConstraintsPropagator* propagator)
{
    propagator->clearDomainsEvents();
}

cudaGlobal void Wrappers::updateDomains(IntConstraintsPropagator* propagator)
{
    propagator->updateDomains();
}

cudaGlobal void Wrappers::clearConstraintsToPropagate(IntConstraintsPropagator* propagator)
{
    propagator->clearConstraintsToPropagate();
}

cudaGlobal void Wrappers::checkEmptyDomains(IntConstraintsPropagator* propagator)
{
    propagator->checkEmptyDomains();
}

cudaGlobal void Wrappers::checkSatisfiedConstraints(IntConstraintsPropagator* propagator)
{
    propagator->checkSatisfiedConstraints();
}
#endif 
