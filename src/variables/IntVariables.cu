#include <variables/IntVariables.h>

void IntVariables::initialize(int count)
{
   this->count = 0;

   domains.initialize(count);
   constraints.initialize(count);

}

void IntVariables::deinitialize()
{
    domains.deinitialize();

    for(int i = 0; i < constraints.size; i += 1)
    {
        constraints[i].deinitialize();
    }
    constraints.deinitialize();
}

void IntVariables::push(int min, int max)
{
    domains.push(min, max);

    constraints.resize_by_one();
    constraints.back().initialize();

    count += 1;
}
