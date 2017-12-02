#include <cassert>

#include <propagators/IntConstraintsPropagator.h>
#include <utils/Utils.h>
#include <wrappers/Wrappers.h>

void IntConstraintsPropagator::initialize(IntVariables* variables, IntConstraints* constraints)
{
    this->variables = variables;
    this->constraints = constraints;

    constraintToPropagate.initialize(constraints->count);
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
    Wrappers::setConstraintsToPropagate<<<1,1>>>(this);
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
        Wrappers::collectActions<<<1,1>>>(this);
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
        Wrappers::updateDomains<<<1,1>>>(this);
        cudaDeviceSynchronize();
#else
        updateDomains();
#endif

#ifdef GPU
        Wrappers::checkEmptyDomains<<<1,1>>>(this);
        cudaDeviceSynchronize();
#else
        checkEmptyDomains();
#endif

        clearConstraintsToPropagate();
        if (not someEmptyDomain)
        {
#ifdef GPU
            Wrappers::setConstraintsToPropagate<<<1,1>>>(this);
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
    for (int vi = 0; vi < variables->domains.actions.domainsWithActions.getSize(); vi += 1)
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
    for(int ci = 0; ci < constraints->count; ci += 1)
    {
        constraintToPropagate.add(ci);
    }
}


cudaDevice void IntConstraintsPropagator::collectActions()
{
    for (int ci = 0; ci < constraintToPropagate.getSize(); ci += 1)
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
    for (int i = 0; i < variables->domains.actions.domainsWithActions.getSize(); i += 1)
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
    for (int i = 0; i < variables->domains.actions.domainsWithActions.getSize(); i += 1)
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
    Wrappers::checkSatisfiedConstraints<<<1,1>>>(this);
    cudaDeviceSynchronize();
#else
    checkSatisfiedConstraints();
#endif

    return allConstraintsSatisfied;
}

cudaDevice void IntConstraintsPropagator::checkSatisfiedConstraints()
{
    for (int ci = 0; ci < constraints->count; ci += 1)
    {
        if (not constraints->satisfied(ci, variables))
        {
            allConstraintsSatisfied = false;
        }
    }
}
