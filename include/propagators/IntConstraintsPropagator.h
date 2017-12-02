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

#ifdef GPU
    int constraintsBlockCountDivergence;
    int constraintsBlockCount;
    int variablesBlockCount;
#endif

    void initialize(IntVariables* variables, IntConstraints* constraints);
    void deinitialize();

    cudaDevice bool propagateConstraints();
    cudaDevice void setConstraintsToPropagate();
    cudaDevice void setAllConstraintsToPropagate();
    cudaDevice void collectActions();
    cudaDevice void resetDomainsEvents();
    cudaDevice void updateDomains();
    cudaDevice void clearConstraintsToPropagate();
    cudaDevice void checkEmptyDomains();
    cudaDevice bool verifyConstraints();
    cudaDevice void checkSatisfiedConstraints();
};
