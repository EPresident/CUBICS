#include <constraints/IntConstraints.h>
#include <utils/Utils.h>

void IntConstraints::initialize()
{
    count = 0;

    types.initialize();

    variables.initialize();
    parameters.initialize();
}

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

void IntConstraints::propagate(int index, IntVariables* variables)
{
    switch (types[index])
    {
        case IntLinNe:
            IntLinNe::propagate(this, index, variables);
            break;
        default:
            LogUtils::error(__PRETTY_FUNCTION__, "Invalid constraint type");
    }
}

bool IntConstraints::satisfied(int index, IntVariables* variables)
{
    switch (types[index])
    {
        case IntLinNe:
            return IntLinNe::satisfied(this, index, variables);
        default:
            LogUtils::error(__PRETTY_FUNCTION__, "Invalid constraint type");
            return false;
    }
}
