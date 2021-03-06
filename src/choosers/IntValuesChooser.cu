#include <choosers/IntValuesChooser.h>

#include <utils/Utils.h>
#include <choosers/IntInOrderValuesChooser.h>

void IntValuesChooser::initialzie(int type, IntVariables* variables)
{
    this->type = type;
    this->variables = variables;
}

cudaDevice bool IntValuesChooser::getFirstValue(int variable, int* firstValue)
{
    switch (type)
    {
        case InOrder:
            return IntInOrderValuesChooser::getFirstValue(this, variable, firstValue);
        default:
            LogUtils::error(__PRETTY_FUNCTION__, "Invalid value chooser type");
            return false;
    }
}

cudaDevice bool IntValuesChooser::getNextValue(int variable, int currentValue, int* nextValue)
{
    switch (type)
    {
        case InOrder:
            return IntInOrderValuesChooser::getNextValue(this, variable, currentValue, nextValue);
        default:
            LogUtils::error(__PRETTY_FUNCTION__, "Invalid value chooser type");
            return false;
    }
}
