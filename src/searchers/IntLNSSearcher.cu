#include <searchers/IntLNSSearcher.h>
#include <utils/Utils.h>
#include <wrappers/Wrappers.h>
#include <random>
#include <algorithm>

void IntLNSSearcher::initialize(FlatZinc::FlatZincModel* fzModel, double unassignRate,
                                int iterations)
{
    variables = fzModel->intVariables;
    constraints = fzModel->intConstraints;

    BTSearcher.initialize(fzModel);

    LNSState = Initialized;
    unassignmentRate = unassignRate;
    iterationsDone = 0;
    LNSState = IntLNSSearcher::Initialized;
    unassignAmount = variables->count*unassignRate;
    randSeed = 1337;
    maxIterations = iterations;
    
    chosenVariables.initialize(unassignAmount);
    
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
        LogUtils::error(__PRETTY_FUNCTION__, "Large Neighborhood Search is only possible on optimization problems!");
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

    while (not solutionFound && iterationsDone < maxIterations)
    {
        switch (LNSState)
        {
            case Initialized:
            {
                // Find first solution
                solutionFound = BTSearcher.getNextSolution();
                LNSState = DoUnassignment;
            }
                break;

            case DoUnassignment:
            {               
                if(unassignAmount < 1)
                {
                    return false;
                }
                // Choose variables to unassign
                //chooseVariables();
                // Mersenne Twister PRNG
                std::mt19937 mt_rand(randSeed);

                // Fill variables vector to be shuffled
                Vector<int> shuffledVars;
                shuffledVars.initialize(variables->count);
                for(int i = 0; i < variables->count; i += 1)
                {
                    shuffledVars.push_back(i);
                }
                // Shuffle (Fisher-Yates/Knuth)
                for(int i = 0; i < variables->count-1; i += 1)
                {
                    // We want a random variable index
                    std::uniform_int_distribution<int> rand_dist(i, variables->count);
                    int j{rand_dist(mt_rand)};
                    int tmp{shuffledVars[i]};
                    shuffledVars[i] = shuffledVars[j];
                    shuffledVars[j] = tmp;
                }
                // Store the chosen variables
                chosenVariables.clear();
                for(int i = 0; i < unassignAmount; i += 1)
                {
                    chosenVariables.push_back(shuffledVars[i]);
                }
                shuffledVars.deinitialize();
            
                // Unassign variables
                for (int i = 0; i < unassignAmount; i += 1)
                {
                    int vi = chosenVariables[i];
                    BTSearcher.stack.representations->minimums[vi] =
                        BTSearcher.stack.backupsStacks[vi].minimums[0];
                    BTSearcher.stack.representations->maximums[vi] =
                        BTSearcher.stack.backupsStacks[vi].maximums[0];
                    BTSearcher.stack.representations->offsets[vi] =
                        BTSearcher.stack.backupsStacks[vi].offsets[0];
                    // Versions are reset below, for all variables             
                    BTSearcher.stack.representations->bitvectors[vi].
                        copy(&BTSearcher.stack.backupsStacks[vi].bitvectors[0]);
                }
                
                // Reset backtrack searcher stack
                // We don't want it to backtrack our unassignment    
                for (int i = 0; i < variables->count; i += 1)
                {
                    BTSearcher.stack.backupsStacks[i].minimums.clear();
                    BTSearcher.stack.backupsStacks[i].maximums.clear();
                    BTSearcher.stack.backupsStacks[i].offsets.clear();
                    BTSearcher.stack.backupsStacks[i].versions.clear();
                    BTSearcher.stack.backupsStacks[i].bitvectors.clear();
                    BTSearcher.stack.levelsStacks[i].clear();
                    // Reset versions (n° of modifications to domains)
                    BTSearcher.stack.representations->versions[i] = 0;
                }
                BTSearcher.backtrackingState = 0; // Reset backtracker state
                BTSearcher.backtrackingLevel = 0;
                BTSearcher.chosenVariables.clear();
                BTSearcher.chosenValues.clear();
            
                // Update LNS state
                ++iterationsDone;
                LNSState = VariablesUnassigned;
            }
                break;
            case VariablesUnassigned:
            {
                // Begin exploring the new neighborhood
                solutionFound = BTSearcher.getNextSolution();
                if(not solutionFound)
                {
                    // Subtree exhausted
                    LNSState = DoUnassignment;
                }
            }
                break;
        }
    }

    return solutionFound;
}

