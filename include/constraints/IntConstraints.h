#pragma once

#include <data_structures/Vector.h>
#include <variables/IntVariables.h>
#include <variables/IntNeighborhood.h>

#include <constraints/IntLinNe.h>
#include <constraints/IntLinLe.h>
#include <constraints/IntOptLb.h>
#include <constraints/IntOptUb.h>
#include <constraints/IntLinEq.h>
#include <constraints/IntAbs.h>
#include <constraints/IntTimes.h>

struct IntConstraints
{
    enum Type
    {
        IntLinNe,
        IntLinLe,
        IntOptLb,
        IntOptUb,
        IntLinEq,
        IntAbs,
        IntTimes
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
    * Add a new constraint of type \a type, whose variables
    * and parameters still have to be set.
    */
    void push(int type);

    /**
    * Propagate the \a index-th constraint. 
    * Propagating means trimming the domains so that the constraint
    * is satisfied.
    */
    cudaDevice void propagate(int index, IntVariables* variables);
    cudaDevice void propagate(int index, IntVariables* variables, IntNeighborhood* nbh);
    /**
    * \return true if :
    *  - All variables of the constraint are ground, and the constraint is satisfied.
    *  - At least one variable is not ground (domain not singleton).
    *
    * True is returned in the second case because of how \a satisfied() is used 
    * inside \a propagate().
    */
    cudaDevice bool satisfied(int index, IntVariables* variables);
    cudaDevice bool satisfied(int index, IntVariables* variables, IntNeighborhood* nbh);
};
