#include <cassert>

#include <propagators/IntConstraintsPropagator.h>
#include <utils/Utils.h>
#include <wrappers/Wrappers.h>

void IntConstraintsPropagator::initialize(IntVariables* variables, IntConstraints* constraints, Statistics* stats)
{
    this->variables = variables;
    this->constraints = constraints;

    constraintToPropagate.initialize(constraints->count);

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
        setConstraintsToPropagate();
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
#ifdef AGPU
        int blockCount = KernelUtils::getBlockCount(constraintToPropagate.getSize(), DEFAULT_BLOCK_SIZE, true );
        Wrappers::collectActions<<<1, 1>>>(this);
        cudaDeviceSynchronize();
#else
        collectActions();
#endif

        resetDomainsEvents();

#ifdef AGPU
        blockCount = KernelUtils::getBlockCount(variables->domains.actions.domainsWithActions.getSize());
        Wrappers::updateDomains<<<1, 1>>>(this);
        cudaDeviceSynchronize();
#else
        updateDomains();
#endif

#ifdef GPU
        int blockCount = KernelUtils::getBlockCount(variables->domains.actions.domainsWithActions.getSize());
        Wrappers::checkEmptyDomains<<<blockCount, DEFAULT_BLOCK_SIZE>>>(this);
        cudaDeviceSynchronize();
#else
        checkEmptyDomains();
#endif
        stats->propagationsCount += constraintToPropagate.getSize();
        clearConstraintsToPropagate();
        if (not someEmptyDomain)
        {
#ifdef AGPU
            Wrappers::setConstraintsToPropagate<<<blockCount, DEFAULT_BLOCK_SIZE>>>(this);
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
#ifdef AGPU
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
#ifdef AGPU
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
#ifdef AGPU
    int ci = KernelUtils::getTaskIndex(true);
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
#ifdef AGPU
    int i = KernelUtils::getTaskIndex();
    if (i >= 0 and i < variables->domains.actions.domainsWithActions.getSize())
#else
    for (int i = 0; i < variables->domains.actions.domainsWithActions.getSize(); i += 1)
#endif
    {
        int vi = variables->domains.actions.domainsWithActions[i];
        variables->domains.update(vi);
    }
}

cudaDevice void IntConstraintsPropagator::clearConstraintsToPropagate()
{
    constraintToPropagate.clear();
}

cudaDevice void IntConstraintsPropagator::checkEmptyDomains()
{
#ifdef GPU
    int i = KernelUtils::getTaskIndex(THREAD_ID);
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
#ifdef AGPU
    int blockCount = KernelUtils::getBlockCount(constraints->count, true);
    Wrappers::checkSatisfiedConstraints<<<blockCount, DEFAULT_BLOCK_SIZE>>>(this);
    cudaDeviceSynchronize();
#else
    checkSatisfiedConstraints();
#endif

    return allConstraintsSatisfied;
}

cudaDevice void IntConstraintsPropagator::checkSatisfiedConstraints()
{
#ifdef AGPU
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
