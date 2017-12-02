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
