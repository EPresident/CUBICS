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
cudaGlobal void Wrappers::getNextSolution(IntBacktrackSearcher* backtrackSearcher, bool* solutionFound, long timeout)
{
    *solutionFound = backtrackSearcher->getNextSolution(timeout);
}

//----------------------
// Integer LNS searcher
//----------------------
cudaGlobal void Wrappers::getNextSolution(IntLNSSearcher* LNSSearcher, bool* solutionFound, long timeout)
{
    *solutionFound = LNSSearcher->getNextSolution(timeout);
}

cudaGlobal void Wrappers::saveBestSolution(IntLNSSearcher* LNSSearcher)
{
    LNSSearcher->saveBestSolution();
}

cudaGlobal void Wrappers::restoreBestSolution(IntLNSSearcher* LNSSearcher)
{
    LNSSearcher->restoreBestSolution();
}

//----------------------
// Integer SNB searcher
//----------------------
cudaGlobal void Wrappers::getNextSolution(IntSNBSearcher* SNBSearcher, bool* solutionFound, long timeout)
{
    *solutionFound = SNBSearcher->getNextSolution(timeout);
}

cudaGlobal void Wrappers::saveBestSolution(IntSNBSearcher* SNBSearcher)
{
    SNBSearcher->saveBestSolution();
}

cudaGlobal void Wrappers::restoreBestSolution(IntSNBSearcher* SNBSearcher)
{
    SNBSearcher->restoreBestSolution();
}
//----------------------------------------------
//Integer constraints propagator
//----------------------------------------------
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

cudaGlobal void Wrappers::propagateConstraints(IntConstraintsPropagator* propagator, IntNeighborhood* nbh, bool* satisfiableModel)
{
    *satisfiableModel = propagator->propagateConstraints(nbh);
}

cudaGlobal void Wrappers::setConstraintsToPropagate(IntConstraintsPropagator* propagator, IntNeighborhood* nbh)
{
    propagator->setConstraintsToPropagate(nbh);
}

cudaGlobal void Wrappers::collectActions(IntConstraintsPropagator* propagator, IntNeighborhood* nbh)
{
    propagator->collectActions(nbh);
}

cudaGlobal void Wrappers::clearDomainsEvents(IntConstraintsPropagator* propagator, IntNeighborhood* nbh)
{
    propagator->clearDomainsEvents(nbh);
}

cudaGlobal void Wrappers::updateDomains(IntConstraintsPropagator* propagator, IntNeighborhood* nbh)
{
    propagator->updateDomains(nbh);
}

cudaGlobal void Wrappers::clearConstraintsToPropagate(IntConstraintsPropagator* propagator, IntNeighborhood* nbh)
{
    propagator->clearConstraintsToPropagate(nbh);
}

cudaGlobal void Wrappers::checkEmptyDomains(IntConstraintsPropagator* propagator, IntNeighborhood* nbh)
{
    propagator->checkEmptyDomains(nbh);
}

cudaGlobal void Wrappers::checkSatisfiedConstraints(IntConstraintsPropagator* propagator, IntNeighborhood* nbh)
{
    propagator->checkSatisfiedConstraints(nbh);
}

//----------------------
// Integer neighborhoods
//----------------------
cudaGlobal void Wrappers::getBinding(IntNeighborhood* nbh, int variable, int* reprIdx)
{
    nbh->getBinding(variable, reprIdx);
}
#endif 
