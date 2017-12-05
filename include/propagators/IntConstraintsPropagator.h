#pragma once

#include <data_structures/MonotonicIntVector.h>
#include <variables/IntVariables.h>
#include <constraints/IntConstraints.h>

struct IntConstraintsPropagator
{
    IntVariables* variables;
    IntConstraints* constraints;

    MonotonicIntVector constraintToPropagate;

    bool someEmptyDomain;
    bool allConstraintsSatisfied;

    void initialize(IntVariables* variables, IntConstraints* constraints);
    void deinitialize();

    bool propagateConstraints();
    void setConstraintsToPropagate();
    void setAllConstraintsToPropagate();
    void collectActions();
    void resetDomainsEvents();
    void updateDomains();
    void clearConstraintsToPropagate();
    void checkEmptyDomains();

    bool verifyConstraints();
    void checkSatisfiedConstraints();
};
