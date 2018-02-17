#ifdef GPU
#pragma once

#include <searchers/IntBacktrackSearcher.h>
#include <searchers/IntLNSSearcher.h>
#include <searchers/IntSNBSearcher.h>
#include <propagators/IntConstraintsPropagator.h>

namespace Wrappers
{
    //Integer backtracking stack
    cudaGlobal void saveState(IntBacktrackStack* backtrackStack, int backtrackingkLevel);
    cudaGlobal void restoreState(IntBacktrackStack* backtrackStack, int backtrackingkLevel);
    cudaGlobal void clearState(IntBacktrackStack* backtrackStack, int backtrackingkLevel);

    //Integer backtracking searcher
    cudaGlobal void getNextSolution(IntBacktrackSearcher* backtrackSearcher, bool* solutionFound, long long timeout);
    
    //Integer LNS searcher
    cudaGlobal void getNextSolution(IntLNSSearcher* LNSSearcher, long long timeout);
    cudaGlobal void saveBestSolution(IntLNSSearcher* LNSSearcher, IntNeighborhood* nbh);
    cudaGlobal void restoreBestSolution(IntLNSSearcher* LNSSearcher);
    
    //Integer SNB searcher
    cudaGlobal void getNextSolution(IntSNBSearcher* SNBSearcher, bool* solutionFound, long timeout);
    cudaGlobal void saveBestSolution(IntSNBSearcher* SNBSearcher);
    cudaGlobal void restoreBestSolution(IntSNBSearcher* SNBSearcher);
    
    //Integer constraints propagator
    cudaGlobal void propagateConstraints(IntConstraintsPropagator* propagator, bool* satisfiableModel);
    cudaGlobal void setConstraintsToPropagate(IntConstraintsPropagator* propagator);
    cudaGlobal void collectActions(IntConstraintsPropagator* propagator);
    cudaGlobal void clearDomainsEvents(IntConstraintsPropagator* propagator);
    cudaGlobal void updateDomains(IntConstraintsPropagator* propagator);
    cudaGlobal void clearConstraintsToPropagate(IntConstraintsPropagator* propagator);
    cudaGlobal void checkEmptyDomains(IntConstraintsPropagator* propagator);
    cudaGlobal void checkSatisfiedConstraints(IntConstraintsPropagator* propagator);
    
    cudaGlobal void propagateConstraints(IntConstraintsPropagator* propagator, IntNeighborhood* nbh, bool* satisfiableModel);
    cudaGlobal void setConstraintsToPropagate(IntConstraintsPropagator* propagator, IntNeighborhood* nbh);
    cudaGlobal void collectActions(IntConstraintsPropagator* propagator, IntNeighborhood* nbh);
    cudaGlobal void clearDomainsEvents(IntConstraintsPropagator* propagator, IntNeighborhood* nbh);
    cudaGlobal void updateDomains(IntConstraintsPropagator* propagator, IntNeighborhood* nbh);
    cudaGlobal void clearConstraintsToPropagate(IntConstraintsPropagator* propagator, IntNeighborhood* nbh);
    cudaGlobal void checkEmptyDomains(IntConstraintsPropagator* propagator, IntNeighborhood* nbh);
    cudaGlobal void checkSatisfiedConstraints(IntConstraintsPropagator* propagator, IntNeighborhood* nbh);
    
    // Int neighborhoods
    cudaGlobal void getBinding(IntNeighborhood* nbh, int variable, int* reprIdx);
}
#endif 
