#pragma once

#include <data_structures/Vector.h>
#include <variables/IntVariables.h>

#include <constraints/IntLinNe.h>
#include <constraints/IntLinLe.h>
#include <constraints/IntOptLb.h>
#include <constraints/IntOptUb.h>
#include <constraints/IntLinEq.h>

struct IntConstraints
{
    enum Type
    {
        IntLinNe,
        IntLinLe,
        IntOptLb,
        IntOptUb,
        IntLinEq
    };

    /// Number of constraints.
    int count;

    /// List of constraints and respective type.
    Vector<int> types;

    /// List of variables involved in each constraint.
    Vector<Vector<int>> variables;
    /// List of (numerical) parameters against which the variables are tested.
    Vector<Vector<int>> parameters;

    void initialize();
    void deinitialize();

    /** 
    * Add a new constraint of type "type", whose variables
    * and parameters still have to be set.
    */
    void push(int type);

    /**
    * Propagate the "index"-th constraint. 
    * Propagating means trimming the domains so that the constraint
    * is satisfied.
    */
    cudaDevice void propagate(int index, IntVariables* variables);
    /**
    * Returns true if the "index"-th constraint is satisfied.
    */
    cudaDevice bool satisfied(int index, IntVariables* variables);
};
