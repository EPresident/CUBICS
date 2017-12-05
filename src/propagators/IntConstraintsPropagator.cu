#include <cassert>

#include <propagators/IntConstraintsPropagator.h>
#include <utils/Utils.h>

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

bool IntConstraintsPropagator::propagateConstraints()
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
        collectActions();

        resetDomainsEvents();
        updateDomains();

        checkEmptyDomains();

        clearConstraintsToPropagate();
        if (not someEmptyDomain)
        {
            setConstraintsToPropagate();
        }
    }

    variables->domains.actions.domainsWithActions.clear();
    resetDomainsEvents();

    return (not someEmptyDomain);
}

void IntConstraintsPropagator::setConstraintsToPropagate()
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

void IntConstraintsPropagator::setAllConstraintsToPropagate()
{
    for(int ci = 0; ci < constraints->count; ci += 1)
    {
        constraintToPropagate.add(ci);
    }
}

void IntConstraintsPropagator::collectActions()
{
    for (int ci = 0; ci < constraintToPropagate.getSize(); ci += 1)
    {
        constraints->propagate(constraintToPropagate[ci], variables);
    }
}

void IntConstraintsPropagator::resetDomainsEvents()
{
    AlgoUtils::fill(&variables->domains.events, static_cast<int>(IntDomains::EventTypes::None));
}

void IntConstraintsPropagator::updateDomains()
{
    for (int i = 0; i < variables->domains.actions.domainsWithActions.getSize(); i += 1)
    {
        int vi = variables->domains.actions.domainsWithActions[i];
        variables->domains.update(vi);
    }
}

void IntConstraintsPropagator::clearConstraintsToPropagate()
{
    constraintToPropagate.clear();
}

void IntConstraintsPropagator::checkEmptyDomains()
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

bool IntConstraintsPropagator::verifyConstraints()
{
    allConstraintsSatisfied = true;
    checkSatisfiedConstraints();

    return allConstraintsSatisfied;
}

void IntConstraintsPropagator::checkSatisfiedConstraints()
{
    for (int ci = 0; ci < constraints->count; ci += 1)
    {
        if (not constraints->satisfied(ci, variables))
        {
            allConstraintsSatisfied = false;
        }
    }
}
