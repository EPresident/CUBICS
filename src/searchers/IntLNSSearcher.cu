#include <searchers/IntLNSSearcher.h>
#include <utils/Utils.h>
#include <wrappers/Wrappers.h>
#include <algorithm>
#include <cassert>
#include <iostream>

void IntLNSSearcher::initialize(FlatZinc::FlatZincModel* fzModel, double unassignRate,
                                int numNeighborhoods, IntDomainsRepresentations* originalDomains)
{
    variables = fzModel->intVariables;
    constraints = fzModel->intConstraints;
    randSeed = 1337 /*1234*/; // FIXME
    
    this->originalDomains = originalDomains;
    // bestSolution is set by main ATM
    
    unassignAmount = variables->count*unassignRate;
    if(unassignAmount < 1) unassignAmount = 1;
    
    valuesChooser.initialzie(IntValuesChooser::InOrder, variables);
    
    // Check problem search type
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
    //------------------------------------------------------------------
    //------------------------------------------------------------------
    // Generate neighborhoods
    //------------------------------------------------------------------
    //------------------------------------------------------------------
    neighborhoods.initialize(numNeighborhoods);
    std::mt19937 mt_rand = std::mt19937(randSeed);

    Vector<int> neighVars;
    neighVars.initialize(unassignAmount+1);
    Vector<int> shuffledVars;
    shuffledVars.initialize(fzModel->intVariables->count);
    
    for(int nbh = 0; nbh < numNeighborhoods; nbh += 1)
    {
        // Fill variables vector to be shuffled
        for(int i = 0; i < fzModel->intVariables->count; i += 1)
        {
            shuffledVars.push_back(i);
        }
        
        // Shuffle (Fisher-Yates/Knuth)
        for(int i = 0; i < fzModel->intVariables->count-1; i += 1)
        {
            // We want a random variable index (bar the optVariable)
            std::uniform_int_distribution<int> rand_dist(i, fzModel->intVariables->count-2);
            int j{rand_dist(mt_rand)};
            int tmp{shuffledVars[i]};
            shuffledVars[i] = shuffledVars[j];
            shuffledVars[j] = tmp;
        }
        // Copy the required subset of the shuffled variables
        for(int i = 0; i < unassignAmount; i++)
        {
            neighVars.push_back(shuffledVars[i]);
        }
        neighVars.push_back(optVariable);
        // Init neighborhood
        IntNeighborhood* newNeigh;
        MemUtils::malloc(&newNeigh);
        newNeigh->initialize(&neighVars, originalDomains, constraints->count);

        neighborhoods.push_back(newNeigh);
        // Clear vectors for reuse
        shuffledVars.clear();
        neighVars.clear();
    }
    neighVars.deinitialize();
    shuffledVars.deinitialize();
    //------------------------------------------------------------------
    //------------------------------------------------------------------

    // init chosen variables/values
    chosenVariables.initialize(numNeighborhoods);
    chosenVariables.resize(numNeighborhoods);
    for(int i = 0; i < numNeighborhoods; i++)
    {
        chosenVariables[i].initialize(unassignAmount+1);
    }
    chosenValues.initialize(numNeighborhoods);
    chosenValues.resize(numNeighborhoods);
    for(int i = 0; i < numNeighborhoods; i++)
    {
        chosenValues[i].initialize(unassignAmount+1);
    }
    
    propagator.initialize(variables, constraints);

    #ifdef GPU
        timers.initialize(numNeighborhoods);
        timers.resize(numNeighborhoods);
        for(int i = 0; i < numNeighborhoods; i++)
        {
            timers[i].initialize();
        }
    #endif

    unassignmentRate = unassignRate;
    
    LNSStates.initialize(numNeighborhoods);
    for(int i = 0; i < numNeighborhoods; i++)
    {
        LNSStates.push_back(IntLNSSearcher::Initialized);
    }

#ifdef GPU
    variablesBlockCount = KernelUtils::getBlockCount(variables->count, DEFAULT_BLOCK_SIZE);
    variablesBlockCountDivergence = KernelUtils::getBlockCount(variables->count, DEFAULT_BLOCK_SIZE, true);
    neighborsBlockCount = KernelUtils::getBlockCount(numNeighborhoods, DEFAULT_BLOCK_SIZE);
#endif

    //------------------------------------------------------------------
    //------------------------------------------------------------------
    // Initialize stacks
    //------------------------------------------------------------------
    //------------------------------------------------------------------
    stacks.initialize(numNeighborhoods);
    for(int s = 0; s < stacks.capacity; s++)
    {
        IntBacktrackStack* newStack;
        MemUtils::malloc(&newStack);
        newStack->initialize(&neighborhoods.at(s)->neighRepr);
        stacks.push_back(newStack);
    }
    assert(stacks.size = numNeighborhoods); 
    //------------------------------------------------------------------
    //------------------------------------------------------------------
}

void IntLNSSearcher::deinitialize()
{
    originalDomains->deinitialize();
    bestSolution->deinitialize();
    for(int n = 0; n < neighborhoods.size; n++)
    {
        neighborhoods.at(n)->deinitialize();
    }
    neighborhoods.deinitialize();
    
    // deinit stacks
    for(int s = 0; s < stacks.size; s++)
    {
        stacks[s]->deinitialize();
    }
    stacks.deinitialize();
    
    // deinit chosen variables/values
    for(int i = 0; i < chosenVariables.size; i++)
    {
        chosenVariables[i].deinitialize();
    }
    chosenVariables.deinitialize();
    
    for(int i = 0; i < chosenValues.size; i++)
    {
        chosenValues[i].deinitialize();
    }
    chosenValues.deinitialize();
    
    propagator.deinitialize();
}

/**
* Find the next solution, backtracking when needed.
* \return true if a solution is found, false otherwise.
*/
cudaDevice bool IntLNSSearcher::getNextSolution(long long timeout)
{
    #ifdef GPU
    int taskIndex = KernelUtils::getTaskIndex(true);
    if(taskIndex < 0 or taskIndex >= neighborhoods.size) return false;
    assert(taskIndex >= 0 and taskIndex < neighborhoods.size);
    printf("I'm LNS-boi N°%d\n",taskIndex);
    #endif
    bool solutionFound = false;
    IntNeighborhood* neighborhood = neighborhoods.at(taskIndex);
    IntBacktrackStack* stack = stacks[taskIndex];
    int backtrackingLevel = 0;
    int chosenValue;
    int currentVar;
    
    long long timeLeft = timeout;
    
    if(taskIndex == 0) bestSolLock.initialize();
    
    while ( backtrackingLevel >= 0
            and timeLeft > 0
          )
    {
        // Setup timer to compute this iteration's duration
        timers[taskIndex].setStartTime();
        switch (LNSStates[taskIndex])
        {
            case Initialized:
            {
                bool noEmptyDomains = propagator.propagateConstraints(neighborhood);
                
                if (noEmptyDomains)
                {
                    LNSStates[taskIndex] = VariableNotChosen;
                    printf("LNS-boi N°%d ready\n",taskIndex);
                }
                else
                {
                    // A domain has been emptied, try another value.
                    printf("LNS-boi %d has an empty starting domain!\n", taskIndex);
                    return false;
                }
            }
            break;
            case VariableNotChosen:
            {
                // Backup current state (GPU/CPU)
                #ifdef GPU
                Wrappers::saveState
                    <<<neighborsBlockCount, DEFAULT_BLOCK_SIZE>>>
                    (stack, backtrackingLevel);
                cudaDeviceSynchronize();
                #else
                stack->saveState(backtrackingLevel);
                #endif
                // "Choose" a variable to assign
                currentVar = neighborhood->map[backtrackingLevel];
                chosenVariables[taskIndex].push_back(neighborhood->map[backtrackingLevel]);

                LNSStates[taskIndex] = VariableChosen;

            }
            break;

            case VariableChosen:
            {
                // Choose a value for the variable
                if (not variables->domains.isSingleton(chosenVariables[taskIndex].back(), neighborhood))
                {
                    // Not a singleton, use the chooser
                    if (valuesChooser.getFirstValue(chosenVariables[taskIndex].back(), &chosenValue, neighborhood))
                    {
                        chosenValues[taskIndex].push_back(chosenValue);
                        variables->domains.fixValue(chosenVariables[taskIndex].back(), chosenValues[taskIndex].back(), neighborhood);
                        LNSStates[taskIndex] = ValueChosen;
                    }
                    else
                    {
                        LogUtils::error(__PRETTY_FUNCTION__, "Failed to set first value");
                    }
                }
                else
                {
                    // Domain's a singleton, choose the only possible value.
                    chosenValue = variables->domains.getMin(chosenVariables[taskIndex].back(), neighborhood);
                    chosenValues[taskIndex].push_back(variables->domains.getMin(chosenVariables[taskIndex].back(), neighborhood));
                    // Nothing has been changed, so no need to propagate.
                    LNSStates[taskIndex] = SuccessfulPropagation;
                }
            }
            break;
            
            case ValueChosen:
            {
                // A domain has been changed, need to propagate.
                bool noEmptyDomains = propagator.propagateConstraints(neighborhood);
                
                if (noEmptyDomains)
                {
                    LNSStates[taskIndex] = SuccessfulPropagation;
                }
                else
                {
                    // A domain has been emptied, try another value.
                    LNSStates[taskIndex] = ValueChecked;
                }
            }
            break;

            case SuccessfulPropagation:
            {
                if (backtrackingLevel < unassignAmount)
                {
                    // Not all variables have been assigned, move to the next
                    backtrackingLevel += 1;
                    LNSStates[taskIndex] = VariableNotChosen;
                }
                else
                {
                    // All variables assigned
                    LNSStates[taskIndex] = ValueChecked;
                    if (propagator.verifyConstraints(neighborhood))
                    {
                        solutionFound = true;
                        printf("LNS-boi %d found solution with cost %d\n", taskIndex,
                         variables->domains.getMin(optVariable, neighborhood));
                        // Shrink optimization bounds
                        bestSolLock.lock();
                        if (searchType == Maximization)
                        {
                            constraints->parameters[optConstraint][0] = variables->domains.getMin(optVariable, neighborhood) + 1;
                        }
                        else if (searchType == Minimization)
                        {
                            constraints->parameters[optConstraint][0] = variables->domains.getMin(optVariable, neighborhood) - 1;
                        }
                        
                        // Record best solution
                        #ifdef GPU
                        Wrappers::saveBestSolution
                            <<<variablesBlockCountDivergence, DEFAULT_BLOCK_SIZE>>>
                            (this, neighborhood);
                        cudaDeviceSynchronize();
                        bestSolLock.unlock();
                        #else
                        saveBestSolution(neighborhood);
                        #endif
                        printf("LNS-boi %d saved solution with cost %d\n", taskIndex,
                         variables->domains.getMin(optVariable, neighborhood));
                    }
                }
            }
            break;

            case ValueChecked:
            {
                // Revert last value choice on the stack (GPU/CPU)
                #ifdef GPU
                Wrappers::restoreState<<<neighborsBlockCount, DEFAULT_BLOCK_SIZE>>>(stack, backtrackingLevel);
                cudaDeviceSynchronize();
                #else
                stack->restoreState(backtrackingLevel);
                #endif
                // Choose another value, if possible
                if (valuesChooser.getNextValue(chosenVariables[taskIndex].back(),
                    chosenValues[taskIndex].back(), &chosenValue, neighborhood))
                {
                    // There's another value
                    chosenValues[taskIndex].back() = chosenValue;
                    variables->domains.fixValue(chosenVariables[taskIndex].back(),
                        chosenValues[taskIndex].back(), neighborhood);
                    LNSStates[taskIndex] = ValueChosen;
                }
                else
                {
                    // All values have been used, backtrack.
                    // Clear current level state from the stack (GPU/CPU)
                    #ifdef GPU
                    Wrappers::clearState<<<neighborsBlockCount, DEFAULT_BLOCK_SIZE>>>(stack, backtrackingLevel);
                    cudaDeviceSynchronize();
                    #else
                    stack->clearState(backtrackingLevel);
                    #endif
                    // Return to the previous backtrack level
                    // and resume the search from there
                    backtrackingLevel -= 1;
                    if(backtrackingLevel < 0) printf("Neighborhood %d explored.\n",taskIndex);
                    chosenVariables[taskIndex].pop_back();
                    chosenValues[taskIndex].pop_back();
                }
            }
            break;
        }

        // Compute elapsed time and subtract it from timeout
        timeLeft -= timers[taskIndex].getElapsedTime();
        
    }
    
    if(timeLeft <= 0)
    {
        printf(">>> GPU Timed out! (%d)<<<\n", taskIndex);
    }

    return solutionFound;
}

/**
* Back up the best solution found so far.
* Beware that no optimality checks are performed.
*/
cudaDevice void IntLNSSearcher::saveBestSolution(IntNeighborhood* neighborhood)
{
#ifdef GPU
    int vi = KernelUtils::getTaskIndex(true);
    if (vi >= 0 and vi < variables->count)
#else
    for (int vi = 0; vi < variables->count; vi += 1)
#endif
    {   
        int varIdx {vi};
        IntDomainsRepresentations* intDomRepr;
        if(neighborhood->isNeighbor(varIdx))
        {
             intDomRepr = &neighborhood->neighRepr;
             varIdx = neighborhood->getRepresentationIndex(vi);
        }
        else
        {
            intDomRepr = &variables->domains.representations;
        }
        bestSolution->minimums[vi] = intDomRepr->minimums[varIdx];
        bestSolution->maximums[vi] = intDomRepr->maximums[varIdx];
        bestSolution->offsets[vi] = intDomRepr->offsets[varIdx];
        bestSolution->versions[vi] = intDomRepr->versions[varIdx];
        bestSolution->bitvectors[vi].copy(&intDomRepr->bitvectors[varIdx]);
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
        intDomRepr->minimums[vi] = bestSolution->minimums[vi];
        intDomRepr->maximums[vi] = bestSolution->maximums[vi];
        intDomRepr->offsets[vi] = bestSolution->offsets[vi];
        intDomRepr->versions[vi] = bestSolution->versions[vi];
        intDomRepr->bitvectors[vi].copy(&bestSolution->bitvectors[vi]);
    }
}
