#include <cmath>
#include <cassert>

#include <constraints/IntTimes.h>
#include <data_structures/Vector.h>
#include <constraints/IntConstraints.h>
/**
 * \file Variables multiplication constraint.
 * 
 * x * y = z
 * x,y and z are variables.
 */
 
/** 
* Enforce bounds-consistency for this constraint.
* x * y = z
* Minimums and maximums of the variables must be supported by the others.
* 
* DEV NOTE: the code for the support verification of the variables
* could be refactored in a function.
*/
cudaDevice void IntTimes::propagate(IntConstraints* constraints, int index, IntVariables* variables)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    IntDomainsRepresentations& intDomRepr  = variables->domains.representations;
    IntDomainsActions& intDomAct = variables->domains.actions;
    
    // Indices of the variables
    int varX {constraintVariables->at(0)};
    int varY {constraintVariables->at(1)};
    int varZ {constraintVariables->at(2)};
    
    int minX {variables->domains.getMin(varX)};
    int minY {variables->domains.getMin(varY)};
    int minZ {variables->domains.getMin(varZ)};
    int maxX {variables->domains.getMax(varX)};
    int maxY {variables->domains.getMax(varY)};
    int maxZ {variables->domains.getMax(varZ)};
    
    // Fast truncations: remove all values that are obviously unsupported
    // e.g. if z = {100} then x = 1 can't be supported if max(y)=50
    
    // ---------------------------------------------------------
    // Fast truncations - minimums
    // ---------------------------------------------------------
    int q;
    (minY != 0) ? q = minZ / minY : q = 0;
    intDomAct.removeAnyLesserThan(varX, q);
    (minY != 0) ? q = minZ / minY : q = 0;
    intDomAct.removeAnyLesserThan(varY, minZ / minX);
    intDomAct.removeAnyLesserThan(varZ, minX * minY);
    
    // ---------------------------------------------------------
    // Fast truncations - maximums
    // ---------------------------------------------------------
    int r;
    (maxZ % maxY == 0) ? r=0 : r = 1;
    intDomAct.removeAnyGreaterThan(varX, maxZ / maxY + r);
    (maxZ % maxX == 0) ? r=0 : r = 1;
    intDomAct.removeAnyGreaterThan(varY, maxZ / maxX + r);
    intDomAct.removeAnyGreaterThan(varZ, maxX * maxY);
    
    // Check if any domain has been emptied
    if
    (
        intDomRepr.getApproximateCardinality(varX) < 1 or
        intDomRepr.getApproximateCardinality(varY) < 1 or
        intDomRepr.getApproximateCardinality(varZ) < 1
    )
    {
        // A domain has been emptied
        return;
    }
    
    // ---------------------------------------------------------
    // Verify support for the minimum of x
    // ---------------------------------------------------------
    bool supported {false};
    int valY {variables->domains.getMin(varY)};
    do
    {
        // Iterating over values of x
        do
        {
            // Iterating over values of y
            if( intDomRepr.contain(varZ, minX * valY) )
            {
                supported = true;
            }
        }while(!supported and intDomRepr.getNextValue(varY, valY, &valY));
    }while(!supported and intDomRepr.getNextValue(varX, minX, &minX));
    if(!supported)
    {
        // No support
        intDomRepr.removeAll(varX);
        return;
    }
    // Remove unsupported values
    intDomAct.removeAnyLesserThan(varX, minX);
    
    // ---------------------------------------------------------
    // Verify support for the minimum of y
    // ---------------------------------------------------------
    supported = false;
    int valX {variables->domains.getMin(varX)};
    do
    {
        // Iterating over values of y
        do
        {
            // Iterating over values of x
            if( intDomRepr.contain(varZ, valX * minY) )
            {
                supported = true;
            }
        }while(!supported and intDomRepr.getNextValue(varX, valX, &valX));
    }while(!supported and intDomRepr.getNextValue(varY, minY, &minY));
    if(!supported)
    {
        // No support
        intDomRepr.removeAll(varY);
        return;
    }
    // Remove unsupported values
    intDomAct.removeAnyLesserThan(varY, minY);
    
    // ---------------------------------------------------------
    // Verify support for the minimum of z
    // ---------------------------------------------------------
    supported = false;
    valX = variables->domains.getMin(varX);
    do
    {
        // Iterating over values of z
        do
        {
            // Iterating over values of x
            if( (minZ % valX == 0) and intDomRepr.contain(varY, minZ / valX) )
            {
                supported = true;
            }
        }while(!supported and intDomRepr.getNextValue(varX, valX, &valX));
    }while(!supported and intDomRepr.getNextValue(varZ, minZ, &minZ));
    if(!supported)
    {
        // No support
        intDomRepr.removeAll(varZ);
        return;
    }
    // Remove unsupported values
    intDomAct.removeAnyLesserThan(varZ, minZ);
    
    // ---------------------------------------------------------
    // Verify support for the maximum of x
    // ---------------------------------------------------------
    supported = false;
    valY = variables->domains.getMin(varY);
    do
    {
        // Iterating over values of x
        do
        {
            // Iterating over values of y
            if( intDomRepr.contain(varZ, maxX * valY) )
            {
                supported = true;
            }
        }while(!supported and intDomRepr.getNextValue(varY, valY, &valY));
    }while(!supported and intDomRepr.getPrevValue(varX, maxX, &maxX));
    if(!supported)
    {
        // No support
        intDomRepr.removeAll(varX);
        return;
    }
    // Remove unsupported values
    intDomAct.removeAnyGreaterThan(varX, maxX);
    
    // ---------------------------------------------------------
    // Verify support for the maximum of y
    // ---------------------------------------------------------
    supported = false;
    valX = variables->domains.getMin(varX);
    do
    {
        // Iterating over values of y
        do
        {
            // Iterating over values of x
            if( intDomRepr.contain(varZ, valX * maxY) )
            {
                supported = true;
            }
        }while(!supported and intDomRepr.getNextValue(varX, valX, &valX));
    }while(!supported and intDomRepr.getPrevValue(varY, maxY, &maxY));
    if(!supported)
    {
        // No support
        intDomRepr.removeAll(varY);
        return;
    }
    // Remove unsupported values
    intDomAct.removeAnyGreaterThan(varY, maxY);
    
    // ---------------------------------------------------------
    // Verify support for the maximum of z
    // ---------------------------------------------------------
    supported = false;
    valX = variables->domains.getMin(varX);
    do
    {
        // Iterating over values of z
        do
        {
            // Iterating over values of x
            if( (maxZ % valX == 0) and intDomRepr.contain(varY, maxZ / valX) )
            {
                supported = true;
            }
        }while(!supported and intDomRepr.getNextValue(varX, valX, &valX));
    }while(!supported and intDomRepr.getPrevValue(varZ, maxZ, &maxZ));
    if(!supported)
    {
        // No support
        intDomRepr.removeAll(varZ);
        return;
    }
    // Remove unsupported values
    intDomAct.removeAnyGreaterThan(varZ, maxZ);
    
    #ifdef NDEBUG
        // assert check
    #endif
}

cudaDevice bool IntTimes::satisfied(IntConstraints* constraints, int index, IntVariables* variables)
{
    Vector<int>* constraintVariables = &constraints->variables[index];
    // Indices of the variables
    int x {variables->domains.getMin(constraintVariables->at(0))};
    int y {variables->domains.getMin(constraintVariables->at(1))};
    int z {variables->domains.getMin(constraintVariables->at(2))};

    //Satisfaction check is performed only when all variables is ground
    for (int i = 0; i < constraintVariables->size; i += 1)
    {
        if (not variables->domains.isSingleton(constraintVariables->at(i)))
        {
            return true;
        }
    }

    return x * y == z;
}
