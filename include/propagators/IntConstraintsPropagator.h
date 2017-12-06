#pragma once

#include <data_structures/MonotonicIntVector.h>
#include <variables/IntVariables.h>
#include <constraints/IntConstraints.h>
#include <statistics/Statistics.h>


struct IntConstraintsPropagator
{
    IntVariables* variables;
    IntConstraints* constraints;

    MonotonicIntVector constraintToPropagate;

    bool someEmptyDomain;
    bool allConstraintsSatisfied;

    Statistics* stats;

    void initialize(IntVariables* variables, IntConstraints* constraints, Statistics* stats);
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
