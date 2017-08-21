#include <choosers/IntValuesChooser.h>
#include <utils/Utils.h>

void IntValuesChooser::initialzie(int type, IntVariables* variables)
{
    this->type = type;
    this->variables = variables;
}

bool IntValuesChooser::getFirstValue(int variable, int* firstValue)
{
    switch (type)
    {
        default:
            LogUtils::error(__PRETTY_FUNCTION__, "Invalid value chooser type");
            return false;
    }
}

bool IntValuesChooser::getNextValue(int variable, int currentValue, int* nextValue)
{
    switch (type)
    {
        default:
            LogUtils::error(__PRETTY_FUNCTION__, "Invalid value chooser type");
            return false;
    }
}
