#pragma once

#include <choosers/IntVariablesChooser.h>

namespace IntInOrderVariablesChooser
{
    cudaDevice bool getVariable(IntVariablesChooser* variablesChooser, int backtrackLevel, int* variable);
}
