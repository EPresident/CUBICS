#include <cassert>

#include <propagators/IntConstraintsPropagator.h>
#include <utils/Utils.h>
#include <wrappers/Wrappers.h>

void IntConstraintsPropagator::initialize(IntVariables* variables, IntConstraints* constraints, Statistics* stats)
{
    this->variables = variables;
    this->constraints = constraints;

    constraintToPropagate.initialize(constraints->count);

#ifdef GPU
    constraintsBlockCountDivergence = KernelUtils::getBlockCount(constraints->count, DEFAULT_BLOCK_SIZE, true);
    constraintsBlockCount = KernelUtils::getBlockCount(constraints->count, DEFAULT_BLOCK_SIZE);
    variablesBlockCount = KernelUtils::getBlockCount(variables->count, DEFAULT_BLOCK_SIZE);
#endif

    this->stats = stats;
}

void IntConstraintsPropagator::deinitialize()
{
    constraintToPropagate.deinitialize();
}

cudaDevice bool IntConstraintsPropagator::propagateConstraints()
{
    someEmptyDomain = false;

    int domainsWithActionsCount = variables->domains.actions.domainsWithActions.getSize();

    //Instantiated variable
    if(domainsWithActionsCount == 1)
    {
#ifdef GPU
    Wrappers::setConstraintsToPropagate<<<constraintsBlockCount, DEFAULT_BLOCK_SIZE>>>(this);
    cudaDeviceSynchronize();
#else
    setConstraintsToPropagate();
#endif
    }
    else if(domainsWithActionsCount == 0) //Preprocessing
    {
        setAllConstraintsToPropagate();
    }
    else
    {
        LogUtils::error(__PRETTY_FUNCTION__, "Not expected domains state");
    }

    while (constraintToPropagate.getSize() > 0 and (not someEmptyDomain))
    {
        variables->domains.actions.domainsWithActions.clear();
#ifdef GPU
        Wrappers::collectActions<<<constraintsBlockCountDivergence, DEFAULT_BLOCK_SIZE>>>(this);
        cudaDeviceSynchronize();
#else
        collectActions();
#endif

#ifdef GPU
        Wrappers::resetDomainsEvents<<<1,1>>>(this);
        cudaDeviceSynchronize();
#else
        resetDomainsEvents();
#endif

#ifdef GPU
        Wrappers::updateDomains<<<variablesBlockCount, DEFAULT_BLOCK_SIZE>>>(this);
        cudaDeviceSynchronize();
#else
        updateDomains();
#endif
#ifdef GPU
        Wrappers::checkEmptyDomains<<<variablesBlockCount, DEFAULT_BLOCK_SIZE>>>(this);
        cudaDeviceSynchronize();
#else
        checkEmptyDomains();
#endif
        stats->propagationsCount += constraintToPropagate.getSize();
        clearConstraintsToPropagate();
        if (not someEmptyDomain)
        {
#ifdef GPU
            Wrappers::setConstraintsToPropagate<<<constraintsBlockCount, DEFAULT_BLOCK_SIZE>>>(this);
            cudaDeviceSynchronize();
#else
            setConstraintsToPropagate();
#endif
        }
    }

    variables->domains.actions.domainsWithActions.clear();
    resetDomainsEvents();

    return (not someEmptyDomain);
}

cudaDevice void IntConstraintsPropagator::setConstraintsToPropagate()
{
#ifdef GPU
    int vi = KernelUtils::getTaskIndex();
    if (vi >= 0 and vi < variables->domains.actions.domainsWithActions.getSize())
#else
     for (int vi = 0; vi < variables->domains.actions.domainsWithActions.getSize(); vi += 1)
#endif
    {
        int variable = variables->domains.actions.domainsWithActions[vi];

        if(variables->domains.events[variable] == IntDomains::EventTypes::Changed)
        {

            for(int ci = 0; ci < variables->constraints[variable].size; ci += 1)
            {
                int constraint = variables->constraints[variable][ci];
                constraintToPropagate.add(constraint);
            }
        }
    }
}

cudaDevice void IntConstraintsPropagator::setAllConstraintsToPropagate()
{
#ifdef GPU
    int ci = KernelUtils::getTaskIndex(true);
    if (ci >= 0 and ci < constraints->count)
#else
    for (int ci = 0; ci < constraints->count; ci += 1)
#endif
    {
        constraintToPropagate.add(ci);
    }
}


cudaDevice void IntConstraintsPropagator::collectActions()
{
#ifdef GPU
    int ci = KernelUtils::getTaskIndex();
    if (ci >= 0 and ci < constraintToPropagate.getSize())
#else
    for (int ci = 0; ci < constraintToPropagate.getSize(); ci += 1)
#endif
    {
        constraints->propagate(constraintToPropagate[ci], variables);
    }
}


cudaDevice void IntConstraintsPropagator::resetDomainsEvents()
{
    AlgoUtils::fill(&variables->domains.events, static_cast<int>(IntDomains::EventTypes::None));
}

cudaDevice void IntConstraintsPropagator::updateDomains()
{
#ifdef GPU
    int vi = KernelUtils::getTaskIndex();
    if (vi >= 0 and vi < variables->domains.actions.domainsWithActions.getSize())
#else
    for (int i = 0; i < variables->domains.actions.domainsWithActions.getSize(); i += 1)
#endif
    {
        int vi = variables->domains.actions.domainsWithActions[i];
        variables->domains.update(vi);
    }
}

cudaHostDevice void IntConstraintsPropagator::clearConstraintsToPropagate()
{
    constraintToPropagate.clear();
}

cudaDevice void IntConstraintsPropagator::checkEmptyDomains()
{
#ifdef GPU
    int i = KernelUtils::getTaskIndex();
    if (i >= 0 and i < variables->domains.actions.domainsWithActions.getSize())
#else
    for (int i = 0; i < variables->domains.actions.domainsWithActions.getSize(); i += 1)
#endif
    {
        int vi = variables->domains.actions.domainsWithActions[i];

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
