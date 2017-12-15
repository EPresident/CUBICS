#pragma once

#include <utils/GpuUtils.h>

struct IntConstraints;
struct IntVariables;

namespace IntAbs
{
    cudaDevice void propagate(IntConstraints* constraints, int index, IntVariables* variables);
    cudaDevice bool satisfied(IntConstraints* constraints, int index, IntVariables* variables);
}
