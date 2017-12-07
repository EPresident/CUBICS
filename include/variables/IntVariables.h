#pragma once

#include <data_structures/Vector.h>
#include <domains/IntDomains.h>

/**
* This struct represents integer variables, as in domains + constraints.
*/
struct IntVariables
{
    /// Number of integer variables.
    int count;

    IntDomains domains;
    Vector<Vector<int>> constraints;

    /// Allocate memory for "count" variables
    void initialize(int count);
    void deinitialize();

    /// Add a new variable with ["min","max"] domain.
    void push(int min, int max);
};
