#include <searchers/IntLNSSearcher.h>
#include <utils/Utils.h>
#include <wrappers/Wrappers.h>
#include <algorithm>
#include <cassert>
#include <iostream>

void IntLNSSearcher::initialize(FlatZinc::FlatZincModel* fzModel, double unassignRate,
                                int iterations)
{
    variables = fzModel->intVariables;
    constraints = fzModel->intConstraints;
    
    unassignAmount = variables->count*unassignRate;
    if(unassignAmount < 1) unassignAmount = 1;

    BTSearcher.initialize(fzModel);
    chosenVariables.initialize(unassignAmount);
    domainsBackup.initialize(variables->count);
    domainsBackup.resize(variables->count);
    for(int i = 0; i < variables->count; i += 1)
    {
        domainsBackup[i].initialize(2);
    }
    #ifdef GPU
        timer.initialize();
    #endif

    LNSState = Initialized;
    unassignmentRate = unassignRate;
    iterationsDone = 0;
    LNSState = IntLNSSearcher::Initialized;
    
    randSeed = 1337;
    maxIterations = iterations;  
    
#ifdef GPU
    varibalesBlockCount = KernelUtils::getBlockCount(variables->count, DEFAULT_BLOCK_SIZE);
#else
    mt_rand = std::mt19937(randSeed);
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
    for(int i = 0; i < variables->count; i += 1)
    {
        domainsBackup[i].deinitialize();
    }
    domainsBackup.deinitialize();
    chosenVariables.deinitialize();
    BTSearcher.deinitialize();
}

/**
* Find the next solution, backtracking when needed.
* \return true if a solution is found, false otherwise.
*/
cudaDevice bool IntLNSSearcher::getNextSolution(long long timeout)
{
    bool solutionFound = false;

    while (not solutionFound 
            && iterationsDone < maxIterations
            && timeout > 0
          )
    {
        // Setup timer to compute this iteration's duration
        timer.setStartTime();
        
        switch (LNSState)
        {
            case Initialized:
            {
                // Backup initial domains
                backupInitialDomains();
                
                // Find first solution
                solutionFound = BTSearcher.getNextSolution(timeout);
                LNSState = DoUnassignment;
                // Save solution
                #ifdef GPU
                    Wrappers::saveBestSolution
                        <<<varibalesBlockCount, DEFAULT_BLOCK_SIZE>>>(this);
                    cudaDeviceSynchronize();
                    
                    // init cuRAND state with the given seed and no offset
                    MemUtils::malloc(&cuRANDstate);
                    curand_init(randSeed, threadIdx.x + blockIdx.x * blockDim.x, 0, cuRANDstate);
                #else
                    saveBestSolution();
                #endif
            }
                break;

            case DoUnassignment:
            {               
                if(unassignAmount < 1)
                {
                    return false;
                }
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
                    // We want a random variable index (bar the optVariable)
                    #ifdef GPU
                        int j {RandUtils::uniformRand(cuRANDstate, i,
                                variables->count-2)};
                    #else
                        std::uniform_int_distribution<int> rand_dist(i,
                            variables->count-2);
                        int j{rand_dist(mt_rand)};
                    #endif
                    
                    int tmp{shuffledVars[i]};
                    shuffledVars[i] = shuffledVars[j];
                    shuffledVars[j] = tmp;
                }
                // Store the chosen variables
                chosenVariables.clear();
                for(int i = 0; i < unassignAmount; i += 1)
                {
                    chosenVariables.push_back(shuffledVars[i]);
                    #ifndef GPU
                        std::cerr << "LNS[" << iterationsDone+1 << 
                            "]: unassigning variable " << 
                            shuffledVars[i] << "\n";
                    #endif
                }
                shuffledVars.deinitialize();
                // Unassign optimization variable
                chosenVariables.push_back(optVariable);
            
                // Unassignment will be performed starting from the
                // best solution found so far
                #ifdef GPU
                    Wrappers::restoreBestSolution
                        <<<varibalesBlockCount, DEFAULT_BLOCK_SIZE>>>(this);
                    cudaDeviceSynchronize();
                #else
                    restoreBestSolution();
                #endif
            
                // Unassign variables
                for (int i = 0; i < chosenVariables.size; i += 1)
                {
                    int vi = chosenVariables[i];
                    BTSearcher.stack.representations->minimums[vi] =
                        domainsBackup[vi].minimums[0];
                    BTSearcher.stack.representations->maximums[vi] =
                        domainsBackup[vi].maximums[0];
                    BTSearcher.stack.representations->offsets[vi] =
                        domainsBackup[vi].offsets[0];
                    // Versions are reset below, for all variables             
                    BTSearcher.stack.representations->bitvectors[vi].
                        copy(&domainsBackup[vi].bitvectors[0]);
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
                solutionFound = BTSearcher.getNextSolution(timeout);
                if(not solutionFound)
                {
                    // Subtree exhausted
                    LNSState = DoUnassignment;
                }
                else
                {
                    #ifndef GPU
                        std::cerr << "LNS[" << iterationsDone << 
                            "]: improving solution found.\n";
                    #endif
                    // Backup improving solution
                    #ifdef GPU
                        Wrappers::saveBestSolution
                            <<<varibalesBlockCount, DEFAULT_BLOCK_SIZE>>>(this);
                        cudaDeviceSynchronize();
                    #else
                        saveBestSolution();
                    #endif
                }
            }
                break;
        }
        
        // Compute elapsed time and subtract it from timeout
        timeout -= timer.getElapsedTime();
        
    }

    return solutionFound;
}

/**
* Back up the initial domains.
*/
cudaDevice void IntLNSSearcher::backupInitialDomains()
{
/*#ifdef GPU
    int vi = KernelUtils::getTaskIndex();
    if (vi >= 0 and vi < variables->count)
#else*/
    for (int vi = 0; vi < variables->count; vi += 1)
//#endif
    {   
        IntDomainsRepresentations* intDomRepr  = &variables->domains.representations;
        int min = intDomRepr->minimums[vi];
        int max = intDomRepr->maximums[vi];
        int offset = intDomRepr->offsets[vi];
        int version = intDomRepr->versions[vi];
        Vector<unsigned int>* bitvector = &intDomRepr->bitvectors[vi];
        domainsBackup[vi].push(min, max, offset, version, bitvector);
    }
}

/**
* Back up the best solution found so far.
* Beware that no optimality checks are performed.
*/
cudaDevice void IntLNSSearcher::saveBestSolution()
{
#ifdef GPU
    int vi = KernelUtils::getTaskIndex();
    if (vi >= 0 and vi < variables->count)
#else
    for (int vi = 0; vi < variables->count; vi += 1)
#endif
    {   
        // Make sure there are at most two entries per variable,
        // i.e. the initial domain and the best solution.
        if(domainsBackup[vi].minimums.size > 1)
        {
            assert(domainsBackup[vi].minimums.size <= 2);
            domainsBackup[vi].pop();
        }
        IntDomainsRepresentations* intDomRepr  = &variables->domains.representations;
        int min = intDomRepr->minimums[vi];
        int max = intDomRepr->maximums[vi];
        int offset = intDomRepr->offsets[vi];
        int version = intDomRepr->versions[vi];
        Vector<unsigned int>* bitvector = &intDomRepr->bitvectors[vi];
        domainsBackup[vi].push(min, max, offset, version, bitvector);
    }
}

/**
* Restore the best solution found so far, by overwriting the current
* domains representation.
*/
cudaDevice void IntLNSSearcher::restoreBestSolution()
{
#ifdef GPU
    int vi = KernelUtils::getTaskIndex();
    if (vi >= 0 and vi < variables->count)
#else
    for (int vi = 0; vi < variables->count; vi += 1)
#endif
    {
        IntDomainsRepresentations* intDomRepr  = &variables->domains.representations;
        intDomRepr->minimums[vi] = domainsBackup[vi].minimums.back();
        intDomRepr->maximums[vi] = domainsBackup[vi].maximums.back();
        intDomRepr->offsets[vi] = domainsBackup[vi].offsets.back();
        intDomRepr->versions[vi] = domainsBackup[vi].versions.back();
        intDomRepr->bitvectors[vi].copy(&domainsBackup[vi].bitvectors.back());
    }
}
