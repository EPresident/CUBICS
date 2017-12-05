#pragma once

#include <data_structures/Vector.h>
#include <variables/IntVariables.h>
#include <constraints/IntConstraints.h>
#include <statistics/Statistics.h>


struct IntConstraintsPropagator
{
    IntVariables* variables;
    IntConstraints* constraints;

    Vector<int> constraintToPropagate;
    Vector<bool> checkedConstraintToPropagateMask;
    Vector<int> checkedConstraintToPropagate;

    Vector<bool> changedDomainsMask;
    Vector<int> changedDomains;

    bool someEmptyDomain;
    bool allConstraintsSatisfied;

#ifdef GPU
    int constraintsBlockCountDivergence;
    int constraintsBlockCount;
    int variablesBlockCount;
#endif

    Statistics* stats;

    void initialize(IntVariables* variables, IntConstraints* constraints, Statistics* stats);
    void deinitialize();

    cudaDevice bool propagateConstraints(bool forcePropagation = false);
    cudaDevice void setConstraintsToPropagate();
    cudaDevice void collectActions();
    cudaDevice void clearDomainsEvents();
    cudaDevice void updateDomains();
    cudaHostDevice void clearConstraintsToPropagate();
    cudaHostDevice void setAllConstraintsToPropagate();
    cudaDevice void checkEmptyDomains();

    cudaDevice void clearChangedDomains();
    cudaDevice void updateChangedDomains();


    cudaDevice bool verifyConstraints();
    cudaDevice void checkSatisfiedConstraints();
};
