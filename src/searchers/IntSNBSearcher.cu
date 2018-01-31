#include <searchers/IntSNBSearcher.h>
#include <utils/Utils.h>
#include <wrappers/Wrappers.h>
#include <algorithm>
#include <cassert>
#include <iostream>

void IntSNBSearcher::initialize(FlatZinc::FlatZincModel* fzModel, int unassignAmount,
                                int iterations)
{
    variables = fzModel->intVariables;
    constraints = fzModel->intConstraints;
    
    this->unassignAmount = unassignAmount;

    BTSearcher.initialize(fzModel);
    valuesChooser.initialzie(IntValuesChooser::InOrder, variables);
    chosenVariables.initialize(unassignAmount+1);
    chosenValues.initialize(unassignAmount+1);
    propagator.initialize(variables, constraints);
    domainsBackup.initialize(variables->count);
    domainsBackup.resize(variables->count);
    for(int i = 0; i < variables->count; i += 1)
    {
        domainsBackup[i].initialize(2);
    }
    #ifdef GPU
        timer.initialize();
    #endif

    iterationsDone = 0;
    SNBSState = IntSNBSearcher::Initialized;
    
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

void IntSNBSearcher::deinitialize()
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
cudaDevice bool IntSNBSearcher::getNextSolution(long timeout)
{
    bool solutionFound = false;

    while (not solutionFound 
            && iterationsDone < maxIterations
            && timeout > 0
          )
    {
        // Setup timer to compute this iteration's duration
        timer.setStartTime();
        
        switch (SNBSState)
        {
            case Initialized:
            {
                // Backup initial domains
                backupInitialDomains();
                
                // Find first solution (there has to be at least one)
                solutionFound = BTSearcher.getNextSolution();
                
                //BTSearcher.deinitialize(); // searcher no longer needed
                SNBSState = NewNeighborhood;
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

            case NewNeighborhood:
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
                    unassignVariable(vi);
                }
            
                // Update SNBS state
                chosenValues.clear();
                neighVarsAssigned = 0;
                SNBSState = Generate;
            }
            break;
            case Generate:
            {
                // Generate a candidate solution naively
                // First-time value assignment to a variable
                int currentVar {chosenVariables[neighVarsAssigned]};
                if (not variables->domains.isSingleton(currentVar))
                {
                    // Not a singleton, use the chooser
                    if (valuesChooser.getFirstValue(currentVar, &chosenValue))
                    {
                        chosenValues.push_back(chosenValue);
                        variables->domains.fixValue(currentVar, chosenValues.back());
                    }
                    else
                    {
                        LogUtils::error(__PRETTY_FUNCTION__, "Failed to set first value");
                    }
                }
                else
                {
                    // Domain's a singleton, choose the only possible value.
                    chosenValues.push_back(variables->domains.getMin(currentVar));
                    
                }
                
                ++neighVarsAssigned;
                
                if(neighVarsAssigned == unassignAmount)
                {
                    // First candidate has been generated
                    SNBSState = Test;
                } // else more variables to assign

            }
            break;
            case NextCandidate:
            {
                // Get the next candidate solution
                // Undo assignments until a new value can be set
                --neighVarsAssigned;
                int currentVar = chosenVariables[neighVarsAssigned];
                unassignVariable(currentVar);
                
                if(not valuesChooser.getNextValue(currentVar, 
                        chosenValues[neighVarsAssigned], &chosenValue) )
                {
                    do
                    {
                        currentVar = chosenVariables[--neighVarsAssigned];
                        unassignVariable(currentVar);
                    }while( neighVarsAssigned-1 >= 0 and
                            not valuesChooser.getNextValue(currentVar, 
                            chosenValues[neighVarsAssigned], &chosenValue) 
                          );
                }
                currentVar = chosenVariables[neighVarsAssigned];
                if(neighVarsAssigned >= 0)
                {
                    chosenValues[neighVarsAssigned] = chosenValue;
                    
                    assert(variables->domains.representations.contain(currentVar, chosenValue));
                    // Start assigning next value(s)
                    variables->domains.fixValue(currentVar, chosenValue);
                    while(++neighVarsAssigned < unassignAmount)
                    {
                        currentVar = chosenVariables[neighVarsAssigned];
                        assert(variables->domains.representations.contain(currentVar, chosenValue));
                        variables->domains.fixValue(currentVar,
                            chosenValues[neighVarsAssigned]);
                    }
                    
                    // Next candidate has been generated
                    SNBSState = Test;
                }
                else
                {
                    // Done exploring the neighborhood
                    SNBSState = NewNeighborhood;
                    ++iterationsDone;
                }
            }
            break;
            case Test:
            {
                // Check if the generated solution is good
                // Also need to compute the cost function!
                // (but it should be a singleton after propagation, so...)
                bool noEmptyDomains = propagator.propagateConstraints();
                
                if (noEmptyDomains)
                {
                    assert(variables->domains.isSingleton(chosenVariables.back()));
                    solutionFound=true;
                    // Backup improving solution
                    #ifdef GPU
                        Wrappers::saveBestSolution
                            <<<varibalesBlockCount, DEFAULT_BLOCK_SIZE>>>(this);
                        cudaDeviceSynchronize();
                    #else
                        saveBestSolution();
                    #endif
                }
                    
                // Try another value.
                SNBSState = FreeOptVar;
            }
            break;
            case FreeOptVar:
            {
                // Free optimization variable
                unassignVariable(optVariable);
                
                SNBSState = NextCandidate;
            }
            break;
        }
        
        // Compute elapsed time and subtract it from timeout
        timeout -= timer.getElapsedTime();
        
    }
    
    // Make sure the next solution is better
    if (solutionFound)
    {
        if (searchType == Maximization)
        {
            constraints->parameters[optConstraint][0] = variables->domains.getMin(optVariable) + 1;
        }
        else if (searchType == Minimization)
        {
            constraints->parameters[optConstraint][0] = variables->domains.getMin(optVariable) - 1;
        }
    }

    return solutionFound;
}

/**
* Back up the initial domains.
*/
cudaDevice void IntSNBSearcher::backupInitialDomains()
{
    for (int vi = 0; vi < variables->count; vi += 1)
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
cudaDevice void IntSNBSearcher::saveBestSolution()
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
cudaDevice void IntSNBSearcher::restoreBestSolution()
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

/**
 * Unassign the given variable, restoring its domain representation
 * to the way it was before the search began.
 * 
 * WARNING: does not restore the version.
 * \see IntDomainsRepresentations
 */
cudaDevice void IntSNBSearcher::unassignVariable(int variable)
{
    IntDomainsRepresentations* intDomRepr  = &variables->domains.representations;
    intDomRepr->minimums[variable] = domainsBackup[variable].minimums[0];
    intDomRepr->maximums[variable] = domainsBackup[variable].maximums[0];
    intDomRepr->offsets[variable] = domainsBackup[variable].offsets[0];
    intDomRepr->versions[variable] += 1;
    intDomRepr->bitvectors[variable].copy(&domainsBackup[variable].bitvectors[0]);
}
