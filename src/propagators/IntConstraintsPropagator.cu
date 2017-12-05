#include <iostream>

#include <propagators/IntConstraintsPropagator.h>
#include <utils/Utils.h>
#include <wrappers/Wrappers.h>

void IntConstraintsPropagator::initialize(IntVariables* variables, IntConstraints* constraints, Statistics* stats)
{
    this->variables = variables;
    this->constraints = constraints;

    constraintToPropagate.initialize(constraints->count);

    checkedConstraintToPropagateMask.initialize(constraints->count);
    checkedConstraintToPropagateMask.resize(constraints->count);
    checkedConstraintToPropagate.initialize(constraints->count);

    changedDomainsMask.initialize(constraints->count);
    changedDomainsMask.resize(constraints->count);
    changedDomains.initialize(constraints->count);

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
    checkedConstraintToPropagateMask.deinitialize();
}

cudaDevice bool IntConstraintsPropagator::propagateConstraints(bool forcePropagation)
{
    someEmptyDomain = false;

    if(not forcePropagation)
    {
    	setConstraintsToPropagate();
    }
    else
    {
    	setAllConstraintsToPropagate();
    }

    while (constraintToPropagate.size > 0 and (not someEmptyDomain))
    {
    	clearDomainsEvents();
    	variables->domains.actions.clearAll();

        collectActions();
        this->updateChangedDomains();

        clearConstraintsToPropagate();

        updateDomains();

        checkEmptyDomains();


        if (not someEmptyDomain)
        {

            setConstraintsToPropagate();
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
    for (int vi = 0; vi < variables->domains.actions.changedDomains.size; vi += 1)
#endif
    {
    	int variable = variables->domains.actions.changedDomains[vi];
    	if(variables->domains.events[variable] != IntDomains::EventTypes::None)
    	{
    		for(int ci = 0; ci < variables->constraints[variable].size; ci += 1)
    		{
    			int constraint = variables->constraints[variable][ci];
    			if(not checkedConstraintToPropagateMask[constraint])
    			{
    				checkedConstraintToPropagateMask[constraint] = true;
    				checkedConstraintToPropagate.push_back(constraint);

    				if (constraints->toPropagate(constraint, variables))
					{
						constraintToPropagate.push_back(constraint);
						stats->propagationsCount += 1;
					}
    			}
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
    for (int ci = 0; ci < constraintToPropagate.size; ci += 1)
#endif
    {
		int constraint = constraintToPropagate[ci];
		constraints->propagate(constraint, variables);
    }
}

cudaDevice void IntConstraintsPropagator::clearDomainsEvents()
{
#ifdef GPU
    int vi = KernelUtils::getTaskIndex();
    if (vi >= 0 and vi < variables->count)
#else
    for (int vi = 0; vi < variables->domains.actions.changedDomains.size; vi += 1)
#endif
    {
        variables->domains.clearEvent(variables->domains.actions.changedDomains[vi]);
    }
}

cudaDevice void IntConstraintsPropagator::updateDomains()
{
#ifdef GPU
    int vi = KernelUtils::getTaskIndex();
    if (vi >= 0 and vi < variables->count)
#else
    for (int vi = 0; vi < variables->domains.actions.changedDomains.size; vi += 1)
#endif
    {
    	int variable = variables->domains.actions.changedDomains[vi];
        variables->domains.updateDomain(variable);
    }
    variables->domains.actions.clearActions();

}

cudaHostDevice void IntConstraintsPropagator::clearConstraintsToPropagate()
{
#if defined(GPU) && defined (__CUDA_ARCH__)
    int ci = KernelUtils::getTaskIndex();
    if (ci >= 0 and ci < constraints->count)
#else
    for (int ci = 0; ci < checkedConstraintToPropagate.size; ci += 1)
#endif
    {
        checkedConstraintToPropagateMask[checkedConstraintToPropagate[ci]] = false;

    }
    checkedConstraintToPropagate.clear();
    constraintToPropagate.clear();
}

cudaHostDevice void IntConstraintsPropagator::setAllConstraintsToPropagate()
{
#if defined(GPU) && defined (__CUDA_ARCH__)
    int ci = KernelUtils::getTaskIndex();
    if (ci >= 0 and ci < constraints->count)
#else
    for (int ci = 0; ci < constraints->count; ci += 1)
#endif
    {
    	checkedConstraintToPropagateMask[ci] = true;
    	checkedConstraintToPropagate.push_back(ci);
        constraintToPropagate.push_back(ci);
    }
}

cudaDevice void IntConstraintsPropagator::clearChangedDomains()
{
	for(int vi = 0; vi < changedDomains.size; vi += 1)
	{
		int index = changedDomains[vi];
		changedDomainsMask[index] = false;
	}

	changedDomains.clear();
}

cudaDevice void IntConstraintsPropagator::updateChangedDomains()
{
    for (int vi = 0; vi < variables->domains.actions.changedDomains.size; vi += 1)
    {
    	int variable = variables->domains.actions.changedDomains[vi];

    	if(not changedDomainsMask[variable])
    	{
    		changedDomainsMask[variable] = true;
    		changedDomains.push_back(variable);
    	}
    }
}

cudaDevice void IntConstraintsPropagator::checkEmptyDomains()
{
#ifdef GPU
    int vi = KernelUtils::getTaskIndex();
    if (vi >= 0 and vi < variables->count)
#else
    for (int vi = 0; vi < variables->domains.actions.changedDomains.size; vi += 1)
#endif
    {
        if (variables->domains.isEmpty(variables->domains.actions.changedDomains[vi]))
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
