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
#endif 
