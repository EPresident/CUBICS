#ifdef GPU
#pragma once

#include <searchers/IntBacktrackSearcher.h>

namespace Wrappers
{
    //Integer backtracking stack
    cudaGlobal void saveState(IntBacktrackStack* backtrackStack, int backtrackingkLevel);
    cudaGlobal void restoreState(IntBacktrackStack* backtrackStack, int backtrackingkLevel);
    cudaGlobal void clearState(IntBacktrackStack* backtrackStack, int backtrackingkLevel);

    //Integer backtracking searcher
    cudaGlobal void getNextSolution(IntBacktrackSearcher* backtrackSearcher, bool* solutionFound);
}
#endif 
