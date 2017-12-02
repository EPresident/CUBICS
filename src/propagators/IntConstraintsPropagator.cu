#include <propagators/IntConstraintsPropagator.h>
#include <utils/Utils.h>

void IntConstraintsPropagator::initialize(IntVariables* variables, IntConstraints* constraints)
{
    this->variables = variables;
    this->constraints = constraints;

    constraintToPropagate.initialize(constraints->count);
    constraintToPropagate.resize(constraints->count);
    clearConstraintsToPropagate();
}

void IntConstraintsPropagator::deinitialize()
{
    constraintToPropagate.deinitialize();
}

cudaDevice bool IntConstraintsPropagator::propagateConstraints()
{
    someEmptyDomain = false;
    someConstraintsToPropagate = false;
    setConstraintsToPropagate();

    while (someConstraintsToPropagate and (not someEmptyDomain))
    {
        collectActions();

        clearDomainsEvents();

        updateDomains();

        clearConstraintsToPropagate();

        someEmptyDomain = false;
        checkEmptyDomains();

        if (not someEmptyDomain)
        {
            someConstraintsToPropagate = false;
            setConstraintsToPropagate();
        }
    }

    return (not someEmptyDomain);
}

cudaDevice void IntConstraintsPropagator::setConstraintsToPropagate()
{
    for (int ci = 0; ci < constraints->count; ci += 1)
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
    for (int ci = 0; ci < constraints->count; ci += 1)
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
    for (int vi = 0; vi < variables->count; vi += 1)
    {
        variables->domains.events[vi] = IntDomains::EventTypes::None;
    }
}

cudaDevice void IntConstraintsPropagator::updateDomains()
{
    for (int vi = 0; vi < variables->count; vi += 1)
    {
        variables->domains.updateDomain(vi);
    }
}

cudaHostDevice void IntConstraintsPropagator::clearConstraintsToPropagate()
{
    for (int ci = 0; ci < constraints->count; ci += 1)
    {
        constraintToPropagate[ci] = false;
    }
}

cudaDevice void IntConstraintsPropagator::checkEmptyDomains()
{
    for (int vi = 0; vi < variables->count; vi += 1)
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
    checkSatisfiedConstraints();

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
