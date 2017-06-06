#pragma once

#include <data_structures/Vector.h>
#include <variables/IntVariables.h>
#include <constraints/IntConstraints.h>

struct IntConstraintsPropagator
{
    IntVariables* variables;
    IntConstraints* constraints;

    Vector<bool> constraintToPropagate;

    bool someEmptyDomain;
    bool someConstraintsToPropagate;
    bool allConstraintsSatisfied;

    void initialize(IntVariables* variables, IntConstraints* constraints);
    void deinitialize();

    bool propagateConstraints();
    void setConstraintsToPropagate();
    void collectActions();
    void clearDomainsEvents();
    void updateDomains();
    void clearConstraintsToPropagate();
    void checkEmptyDomains();

    bool verifyConstraints();
    void checkSatisfiedConstraints();
};
