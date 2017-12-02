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

#ifdef GPU
    int constraintsBlockCountDivergence;
    int constraintsBlockCount;
    int variablesBlockCount;
#endif

    void initialize(IntVariables* variables, IntConstraints* constraints);
    void deinitialize();

    cudaDevice bool propagateConstraints();
    cudaDevice void setConstraintsToPropagate();
    cudaDevice void collectActions();
    cudaDevice void clearDomainsEvents();
    cudaDevice void updateDomains();
    cudaHostDevice void clearConstraintsToPropagate();
    cudaDevice void checkEmptyDomains();

    cudaDevice bool verifyConstraints();
    cudaDevice void checkSatisfiedConstraints();
};
