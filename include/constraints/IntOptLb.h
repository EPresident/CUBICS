#pragma once

#include <utils/GpuUtils.h>

struct IntConstraints;
struct IntVariables;

namespace IntOptLb
{
    cudaDevice void propagate(IntConstraints* constraints, int index, IntVariables* variables);
    cudaDevice bool satisfied(IntConstraints* constraints, int index, IntVariables* variables);
    cudaDevice void propagate(IntConstraints* constraints, int index, IntVariables* variables, IntNeighborhood* nbh);
    cudaDevice bool satisfied(IntConstraints* constraints, int index, IntVariables* variables, IntNeighborhood* nbh);
}
