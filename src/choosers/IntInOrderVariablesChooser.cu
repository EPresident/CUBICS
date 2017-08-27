#include <choosers/IntInOrderVariablesChooser.h>

bool IntInOrderVariablesChooser::getVariable(IntVariablesChooser* variablesChooser, int backtrackLevel, int* variable)
{
    *variable = backtrackLevel;
    return true;
}
