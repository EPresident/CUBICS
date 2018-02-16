#include <propagators/IntConstraintsPropagator.h>
#include <utils/Utils.h>
#include <wrappers/Wrappers.h>

void IntConstraintsPropagator::initialize(IntVariables* variables, IntConstraints* constraints)
{
    this->variables = variables;
    this->constraints = constraints;

    constraintToPropagate.initialize(constraints->count);
    constraintToPropagate.resize(constraints->count);
    clearConstraintsToPropagate();

#ifdef GPU
    constraintsBlockCountDivergence = KernelUtils::getBlockCount(constraints->count, DEFAULT_BLOCK_SIZE, true);
    constraintsBlockCount = KernelUtils::getBlockCount(constraints->count, DEFAULT_BLOCK_SIZE);
    variablesBlockCount = KernelUtils::getBlockCount(variables->count, DEFAULT_BLOCK_SIZE);
#endif
}

void IntConstraintsPropagator::deinitialize()
{
    constraintToPropagate.deinitialize();
}

cudaDevice bool IntConstraintsPropagator::propagateConstraints()
{
    someEmptyDomain = false;
    someConstraintsToPropagate = false;
#ifdef GPU
    Wrappers::setConstraintsToPropagate<<<constraintsBlockCount, DEFAULT_BLOCK_SIZE>>>(this);
    cudaDeviceSynchronize();
#else
    setConstraintsToPropagate();
#endif

    while (someConstraintsToPropagate and (not someEmptyDomain))
    {
#ifdef GPU
        Wrappers::collectActions<<<constraintsBlockCountDivergence, DEFAULT_BLOCK_SIZE>>>(this);
        cudaDeviceSynchronize();
#else
        collectActions();
#endif

#ifdef GPU
        Wrappers::clearDomainsEvents<<<variablesBlockCount, DEFAULT_BLOCK_SIZE>>>(this);
        cudaDeviceSynchronize();
#else
        clearDomainsEvents();
#endif

#ifdef GPU
        Wrappers::updateDomains<<<variablesBlockCount, DEFAULT_BLOCK_SIZE>>>(this);
        cudaDeviceSynchronize();
#else
        updateDomains();
#endif

#ifdef GPU
        Wrappers::clearConstraintsToPropagate<<<constraintsBlockCount, DEFAULT_BLOCK_SIZE>>>(this);
        cudaDeviceSynchronize();
#else
        clearConstraintsToPropagate();
#endif

        someEmptyDomain = false;
#ifdef GPU
        Wrappers::checkEmptyDomains<<<variablesBlockCount, DEFAULT_BLOCK_SIZE>>>(this);
        cudaDeviceSynchronize();
#else
        checkEmptyDomains();
#endif

        if (not someEmptyDomain)
        {
            someConstraintsToPropagate = false;
#ifdef GPU
            Wrappers::setConstraintsToPropagate<<<constraintsBlockCount, DEFAULT_BLOCK_SIZE>>>(this);
            cudaDeviceSynchronize();
#else
            setConstraintsToPropagate();
#endif
        }
    }

    return (not someEmptyDomain);
}
cudaDevice bool IntConstraintsPropagator::propagateConstraints(IntNeighborhood* nbh)
{
    nbh->someEmptyDomain = false;
    nbh->someConstraintsToPropagate = false;
#ifdef GPU
    Wrappers::setConstraintsToPropagate<<<constraintsBlockCount, DEFAULT_BLOCK_SIZE>>>(this, nbh);
    cudaDeviceSynchronize();
#else
    setConstraintsToPropagate(nbh);
#endif

    while (nbh->someConstraintsToPropagate and (not nbh->someEmptyDomain))
    {
#ifdef GPU
        Wrappers::collectActions<<<constraintsBlockCountDivergence, DEFAULT_BLOCK_SIZE>>>(this, nbh);
        cudaDeviceSynchronize();
#else
        collectActions(nbh);
#endif

#ifdef GPU
        Wrappers::clearDomainsEvents<<<variablesBlockCount, DEFAULT_BLOCK_SIZE>>>(this, nbh);
        cudaDeviceSynchronize();
#else
        clearDomainsEvents(nbh);
#endif

#ifdef GPU
        Wrappers::updateDomains<<<variablesBlockCount, DEFAULT_BLOCK_SIZE>>>(this, nbh);
        cudaDeviceSynchronize();
#else
        updateDomains(nbh);
#endif

#ifdef GPU
        Wrappers::clearConstraintsToPropagate<<<constraintsBlockCount, DEFAULT_BLOCK_SIZE>>>(this, nbh);
        cudaDeviceSynchronize();
#else
        clearConstraintsToPropagate(nbh);
#endif

        nbh->someEmptyDomain = false;
#ifdef GPU
        Wrappers::checkEmptyDomains<<<variablesBlockCount, DEFAULT_BLOCK_SIZE>>>(this, nbh);
        cudaDeviceSynchronize();
#else
        checkEmptyDomains(nbh);
#endif

        if (not nbh->someEmptyDomain)
        {
            nbh->someConstraintsToPropagate = false;
#ifdef GPU
            Wrappers::setConstraintsToPropagate<<<constraintsBlockCount, DEFAULT_BLOCK_SIZE>>>(this, nbh);
            cudaDeviceSynchronize();
#else
            setConstraintsToPropagate(nbh);
#endif
        }
    }

    return (not nbh->someEmptyDomain);
}

/**
* Check if any constraint needs to be propagated, updating the appropriate
* flags.
* Propagation is required if any variable of a constraint has a "Changed" domain
* event.
* \see IntDomains
*/
cudaDevice void IntConstraintsPropagator::setConstraintsToPropagate()
{
#ifdef GPU
    int ci = KernelUtils::getTaskIndex();
    if (ci >= 0 and ci < constraints->count)
#else
    for (int ci = 0; ci < constraints->count; ci += 1)
#endif
    {
        for (int vi = 0; vi < constraints->variables[ci].size; vi += 1)
        {
            int event = variables->domains.events[constraints->variables[ci][vi]];

            if (event == IntDomains::EventTypes::Changed)
            {
                constraintToPropagate[ci] = true;
                someConstraintsToPropagate = true;
            }
        }
    }
}
cudaDevice void IntConstraintsPropagator::setConstraintsToPropagate(IntNeighborhood* nbh)
{
#ifdef GPU
    int ci = KernelUtils::getTaskIndex();
    if (ci >= 0 and ci < constraints->count)
#else
    for (int ci = 0; ci < constraints->count; ci += 1)
#endif
    {
        for (int vi = 0; vi < constraints->variables[ci].size  and nbh->isNeighbor(constraints->variables[ci][vi]); vi += 1)
        {
            int event = nbh->events[nbh->getRepresentationIndex(constraints->variables[ci][vi])];

            if (event == IntDomains::EventTypes::Changed)
            {
                nbh->constraintToPropagate[ci] = true;
                nbh->someConstraintsToPropagate = true;
            }
        }
    }
}

/**
* Propagates all constraints flagged in "constraintToPropagate", and flips
* the respective flag.
*/
cudaDevice void IntConstraintsPropagator::collectActions()
{
#ifdef GPU
    int ci = KernelUtils::getTaskIndex(true);
    if (ci >= 0 and ci < constraints->count)
#else
    for (int ci = 0; ci < constraints->count; ci += 1)
#endif
    {
        if (constraintToPropagate[ci])
        {
            constraints->propagate(ci, variables);
            constraintToPropagate[ci] = false;
        }
    }
}
cudaDevice void IntConstraintsPropagator::collectActions(IntNeighborhood* nbh)
{
#ifdef GPU
    int ci = KernelUtils::getTaskIndex(true);
    if (ci >= 0 and ci < constraints->count)
#else
    for (int ci = 0; ci < constraints->count; ci += 1)
#endif
    {
        if (nbh->constraintToPropagate[ci])
        {
            constraints->propagate(ci, variables, nbh);
            nbh->constraintToPropagate[ci] = false;
        }
    }
}

/** 
* Clears the domain events list.
* \see IntDomains
*/
cudaDevice void IntConstraintsPropagator::clearDomainsEvents()
{
#ifdef GPU
    int vi = KernelUtils::getTaskIndex();
    if (vi >= 0 and vi < variables->count)
#else
    for (int vi = 0; vi < variables->count; vi += 1)
#endif
    {
        variables->domains.events[vi] = IntDomains::EventTypes::None;
    }
}
cudaDevice void IntConstraintsPropagator::clearDomainsEvents(IntNeighborhood* nbh)
{
#ifdef GPU
    int vi = KernelUtils::getTaskIndex();
    if (vi >= 0 and vi < variables->count and nbh->isNeighbor(vi))
#else
    for (int vi = 0; vi < variables->count and nbh->isNeighbor(vi); vi += 1)
#endif
    {
        nbh->events[nbh->getRepresentationIndex(vi)] = IntDomains::EventTypes::None;
    }
}

/**
* Perform the domain reduction actions on all variables.
* \see IntDomains
*/
cudaDevice void IntConstraintsPropagator::updateDomains()
{
#ifdef GPU
    int vi = KernelUtils::getTaskIndex();
    if (vi >= 0 and vi < variables->count)
#else
    for (int vi = 0; vi < variables->count; vi += 1)
#endif
    {
        variables->domains.updateDomain(vi);
    }
}
cudaDevice void IntConstraintsPropagator::updateDomains(IntNeighborhood* nbh)
{
#ifdef GPU
    int vi = KernelUtils::getTaskIndex();
    if (vi >= 0 and vi < variables->count and nbh->isNeighbor(vi))
#else
    for (int vi = 0; vi < variables->count and nbh->isNeighbor(vi); vi += 1)
#endif
    {
        variables->domains.updateDomain(vi, nbh);
    }
}

/// Clears the "constraintToPropagate" vector.
cudaHostDevice void IntConstraintsPropagator::clearConstraintsToPropagate()
{
#if defined(GPU) && defined (__CUDA_ARCH__)
    int ci = KernelUtils::getTaskIndex();
    if (ci >= 0 and ci < constraints->count)
#else
    for (int ci = 0; ci < constraints->count; ci += 1)
#endif
    {
        constraintToPropagate[ci] = false;
    }
}
cudaHostDevice void IntConstraintsPropagator::clearConstraintsToPropagate(IntNeighborhood* nbh)
{
#if defined(GPU) && defined (__CUDA_ARCH__)
    int ci = KernelUtils::getTaskIndex();
    if (ci >= 0 and ci < constraints->count)
#else
    for (int ci = 0; ci < constraints->count; ci += 1)
#endif
    {
        nbh->constraintToPropagate[ci] = false;
    }
}

/// Checks if any domain has become empty, updating the "someEmptyDomain" flag.
cudaDevice void IntConstraintsPropagator::checkEmptyDomains()
{
#ifdef GPU
    int vi = KernelUtils::getTaskIndex();
    if (vi >= 0 and vi < variables->count)
#else
    for (int vi = 0; vi < variables->count; vi += 1)
#endif
    {
        if (variables->domains.isEmpty(vi))
        {
            someEmptyDomain = true;
        }
    }
}
cudaDevice void IntConstraintsPropagator::checkEmptyDomains(IntNeighborhood* nbh)
{
#ifdef GPU
    int vi = KernelUtils::getTaskIndex();
    if (vi >= 0 and vi < variables->count)
#else
    for (int vi = 0; vi < variables->count; vi += 1)
#endif
    {
        if (nbh->isNeighbor(vi) and variables->domains.isEmpty(vi, nbh))
        {
            nbh->someEmptyDomain = true;
        }
    }
}

/// \return true if all constraints are satisfied.
cudaDevice bool IntConstraintsPropagator::verifyConstraints()
{
    allConstraintsSatisfied = true;
#ifdef GPU
    Wrappers::checkSatisfiedConstraints<<<constraintsBlockCountDivergence, DEFAULT_BLOCK_SIZE>>>(this);
    cudaDeviceSynchronize();
#else
    checkSatisfiedConstraints();
#endif

    return allConstraintsSatisfied;
}
cudaDevice bool IntConstraintsPropagator::verifyConstraints(IntNeighborhood* nbh)
{
    nbh->allConstraintsSatisfied = true;
#ifdef GPU
    Wrappers::checkSatisfiedConstraints<<<constraintsBlockCountDivergence, DEFAULT_BLOCK_SIZE>>>(this, nbh);
    cudaDeviceSynchronize();
#else
    checkSatisfiedConstraints(nbh);
#endif

    return nbh->allConstraintsSatisfied;
}

/// Updates the "allConstraintsSatisfied" flag, scanning all constraints.
cudaDevice void IntConstraintsPropagator::checkSatisfiedConstraints()
{
#ifdef GPU
    int ci = KernelUtils::getTaskIndex(true);
    if (ci >= 0 and ci < constraints->count)
#else
    for (int ci = 0; ci < constraints->count; ci += 1)
#endif
    {
        if (not constraints->satisfied(ci, variables))
        {
            allConstraintsSatisfied = false;
        }
    }
}
cudaDevice void IntConstraintsPropagator::checkSatisfiedConstraints(IntNeighborhood* nbh)
{
#ifdef GPU
    int ci = KernelUtils::getTaskIndex(true);
    if (ci >= 0 and ci < constraints->count)
#else
    for (int ci = 0; ci < constraints->count; ci += 1)
#endif
    {
        if (not constraints->satisfied(ci, variables, nbh))
        {
            nbh->allConstraintsSatisfied = false;
        }
    }
}
