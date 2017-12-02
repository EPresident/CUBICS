#include <choosers/IntInOrderVariablesChooser.h>

cudaDevice bool IntInOrderVariablesChooser::getVariable(IntVariablesChooser* variablesChooser, int backtrackLevel, int* variable)
{
    *variable = backtrackLevel;
    return true;
}
