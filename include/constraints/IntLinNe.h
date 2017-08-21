#pragma once

struct IntConstraints;
struct IntVariables;

namespace IntLinNe
{
    void propagate(IntConstraints* constraints, int index, IntVariables* variables);
    bool satisfied(IntConstraints* constraints, int index, IntVariables* variables);
}
