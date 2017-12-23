#ifdef GPU
#pragma once

#include <searchers/IntBacktrackSearcher.h>
#include <searchers/IntLNSSearcher.h>
#include <propagators/IntConstraintsPropagator.h>

namespace Wrappers
{
    //Integer backtracking stack
    cudaGlobal void saveState(IntBacktrackStack* backtrackStack, int backtrackingkLevel);
    cudaGlobal void restoreState(IntBacktrackStack* backtrackStack, int backtrackingkLevel);
    cudaGlobal void clearState(IntBacktrackStack* backtrackStack, int backtrackingkLevel);

    //Integer backtracking searcher
    cudaGlobal void getNextSolution(IntBacktrackSearcher* backtrackSearcher, bool* solutionFound);
    
    //Integer LNS searcher
    cudaGlobal void getNextSolution(IntLNSSearcher* LNSSearcher, bool* solutionFound);
    cudaGlobal void saveBestSolution(IntLNSSearcher* LNSSearcher);
    cudaGlobal void restoreBestSolution(IntLNSSearcher* LNSSearcher);
    
    //Integer constraints propagator
    cudaGlobal void propagateConstraints(IntConstraintsPropagator* propagator, bool* satisfiableModel);
    cudaGlobal void setConstraintsToPropagate(IntConstraintsPropagator* propagator);
    cudaGlobal void collectActions(IntConstraintsPropagator* propagator);
    cudaGlobal void clearDomainsEvents(IntConstraintsPropagator* propagator);
    cudaGlobal void updateDomains(IntConstraintsPropagator* propagator);
    cudaGlobal void clearConstraintsToPropagate(IntConstraintsPropagator* propagator);
    cudaGlobal void checkEmptyDomains(IntConstraintsPropagator* propagator);
    cudaGlobal void checkSatisfiedConstraints(IntConstraintsPropagator* propagator);
}
#endif 
