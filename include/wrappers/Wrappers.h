#ifdef GPU
#pragma once

#include <searchers/IntBacktrackSearcher.h>
#include <propagators/IntConstraintsPropagator.h>

namespace Wrappers
{
    //Integer backtracking stack
    cudaGlobal void saveState(IntBacktrackStack* backtrackStack, int backtrackingkLevel, MonotonicIntVector* changedDomains);
    cudaGlobal void resetState(IntBacktrackStack* backtrackStack, MonotonicIntVector* changedDomains);
    cudaGlobal void restorePreviousState(IntBacktrackStack* backtrackStack, int backtrackingkLevel);

    //Integer backtracking searcher
    cudaGlobal void getNextSolution(IntBacktrackSearcher* backtrackSearcher, bool* solutionFound);

    //Integer constraints propagator
    cudaGlobal void propagateConstraints(IntConstraintsPropagator* propagator, bool* satisfiableModel);
    cudaGlobal void setConstraintsToPropagate(IntConstraintsPropagator* propagator);
    cudaGlobal void collectActions(IntConstraintsPropagator* propagator);
    cudaGlobal void updateDomains(IntConstraintsPropagator* propagator);
    cudaGlobal void clearConstraintsToPropagate(IntConstraintsPropagator* propagator);
    cudaGlobal void checkEmptyDomains(IntConstraintsPropagator* propagator);
    cudaGlobal void checkSatisfiedConstraints(IntConstraintsPropagator* propagator);

}
#endif 
