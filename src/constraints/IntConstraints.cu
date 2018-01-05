#include <constraints/IntConstraints.h>
#include <utils/Utils.h>

void IntConstraints::initialize()
{
    count = 0;

    types.initialize();

    variables.initialize();
    parameters.initialize();
}

/** 
* Add a new constraint of type "type", whose variables
* and parameters still have to be set.
*/
void IntConstraints::push(int type)
{
    types.push_back(type);

    variables.resize_by_one();
    variables.back().initialize();

    parameters.resize_by_one();
    parameters.back().initialize();

    count += 1;
}

void IntConstraints::deinitialize()
{
    types.deinitialize();

    variables.deinitialize();
    parameters.deinitialize();
}

/**
* Propagate the "index"-th constraint. 
* Propagating means trimming the domains so that the constraint
* is satisfied.
*/
cudaDevice void IntConstraints::propagate(int index, IntVariables* variables)
{
    switch (types[index])
    {
        case IntLinNe:
            IntLinNe::propagate(this, index, variables);
            break;
        case IntLinLe:
            IntLinLe::propagate(this, index, variables);
            break;
        case IntOptLb:
            IntOptLb::propagate(this, index, variables);
            break;
        case IntOptUb:
            IntOptUb::propagate(this, index, variables);
            break;
        case IntLinEq:
            IntLinEq::propagate(this, index, variables);
            break;
        case IntAbs:
            IntAbs::propagate(this, index, variables);
            break;
        default:
            LogUtils::error(__PRETTY_FUNCTION__, "Invalid constraint type");
    }
}

/**
* Returns true if the "index"-th constraint is satisfied.
*/
cudaDevice bool IntConstraints::satisfied(int index, IntVariables* variables)
{
    switch (types[index])
    {
        case IntLinNe:
            return IntLinNe::satisfied(this, index, variables);
        case IntLinLe:
            return IntLinLe::satisfied(this, index, variables);
        case IntOptLb:
            return IntOptLb::satisfied(this, index, variables);
        case IntOptUb:
            return IntOptUb::satisfied(this, index, variables);
        case IntLinEq:
            return IntLinEq::satisfied(this, index, variables);
        case IntAbs:
            return IntAbs::satisfied(this, index, variables);
        default:
            LogUtils::error(__PRETTY_FUNCTION__, "Invalid constraint type");
            return false;
    }
}
