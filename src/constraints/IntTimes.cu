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
    
    bool xMaxIsPos {variables->domains.getMax(varX) >= 0};
    bool xMinIsNeg {variables->domains.getMin(varX) < 0};
    bool yMaxIsPos {variables->domains.getMax(varY) >= 0};
    bool yMinIsNeg {variables->domains.getMin(varY) < 0};
    bool zMaxIsPos {variables->domains.getMax(varZ) >= 0};
    bool zMinIsNeg {variables->domains.getMin(varZ) < 0};
    
    int posMinX {variables->domains.getMin(varX)};
    int posMinY {variables->domains.getMin(varY)};
    int posMinZ {variables->domains.getMin(varZ)};
    int posMaxX {variables->domains.getMax(varX)};
    int posMaxY {variables->domains.getMax(varY)};
    int posMaxZ {variables->domains.getMax(varZ)};
    int negMinX {variables->domains.getMin(varX)};
    int negMinY {variables->domains.getMin(varY)};
    int negMinZ {variables->domains.getMin(varZ)};
    int negMaxX {variables->domains.getMax(varX)};
    int negMaxY {variables->domains.getMax(varY)};
    int negMaxZ {variables->domains.getMax(varZ)};
    int minX {variables->domains.getMin(varX)};
    int minY {variables->domains.getMin(varY)};
    int minZ {variables->domains.getMin(varZ)};
    int maxX {variables->domains.getMax(varX)};
    int maxY {variables->domains.getMax(varY)};
    int maxZ {variables->domains.getMax(varZ)};
    
    // ---------------------------------------------------------
    // Check z lower bound
    // ---------------------------------------------------------
    {
        int minVal { minZ };
        // 100 * -100
        if(xMaxIsPos and yMinIsNeg)
        {
            int p {posMaxX * negMaxY};
            if( p < minVal)
            {
                minVal = p;
            }
        }
        // -100 * 100
        if(xMinIsNeg and yMaxIsPos)
        {
            int p { negMaxX * posMaxY };
            if(p < minVal)
            {
                minVal = p;
            }
        }
        // 1 * 1
        if(xMaxIsPos and yMaxIsPos)
        {
            int p {posMinX * posMinY};
            if( p < minVal)
            {
                minVal = p;
            }
        }
        // -1 * -1
        if(xMinIsNeg and yMinIsNeg)
        {
            int p { negMaxX * negMaxY };
            if(p < minVal)
            {
                minVal = p;
            }
        }
        
        intDomAct.removeAnyLesserThan(varZ, minVal);
    } //~
    // ---------------------------------------------------------
    // Check z upper bound
    // ---------------------------------------------------------
    {
        int maxVal { maxZ };
        // 100 * 100
        if(xMaxIsPos and yMaxIsPos)
        {
            int p {posMaxX * posMaxY};
            if( p > maxVal)
            {
                maxVal = p;
            }
        }
        // -100 * -100
        if(xMinIsNeg and yMinIsNeg)
        {
            int p { negMinX * negMinY };
            if(p > maxVal)
            {
                maxVal = p;
            }
        }
        // 100 * -1
        if(xMaxIsPos and yMinIsNeg)
        {
            int p {posMaxX * negMaxY};
            if( p > maxVal)
            {
                maxVal = p;
            }
        }
        // -1 * 100
        if(xMinIsNeg and yMaxIsPos)
        {
            int p { negMaxX * posMaxY };
            if(p > maxVal)
            {
                maxVal = p;
            }
        }
        
        intDomAct.removeAnyGreaterThan(varZ, maxVal);
    } //~
    
    // ---------------------------------------------------------
    // Check x lower bound
    // ---------------------------------------------------------
    {
        int minVal { minX };
        // 100 / -1
        if(zMaxIsPos and yMinIsNeg)
        {
            int q {posMaxZ / negMaxY};
            if(q < minVal)
            {
                minVal = q;
            }
        }
        // -100 / 1
        if(zMinIsNeg and yMaxIsPos and posMinY > 0)
        {
            int q { negMinZ / posMinY };
            if(q < minVal)
            {
                minVal = q;
            }
        }
        // 1 / 100
        // Beware this particlar case: 
        //--x = {0..n} y = {1..m} z = {1..k}
        //--posMinZ / posMaxY = 1 / m which can be < 1 (approximated to 0)
        //--The zero won't be removed from x! (it should, since x=0 is not supported)
        if(zMaxIsPos and yMaxIsPos and posMinZ > 0)
        {
            int q {posMinZ / posMaxY};
            (q < 1) ? q = 1 ; // Fix for the problem above
            if(q < minVal)
            {
                minVal = q;
            }
        }
        // -1 / -100
        // Watch out as above
        //   x = {0..n} y = {-100..m} z = {-1..k}
        if(zMinIsNeg and yMinIsNeg)
        {
            int q { negMaxZ * negMinY };
            if(q < minVal)
            {
                minVal = q;
            }
        }
        
        intDomAct.removeAnyLesserThan(varZ, minVal);
    } //~
    // ---------------------------------------------------------
    // Check z upper bound
    // ---------------------------------------------------------
    {
        int maxVal { maxZ };
        // 100 * 100
        if(xMaxIsPos and yMaxIsPos)
        {
            int p {posMaxX * posMaxY};
            if( p > maxVal)
            {
                maxVal = p;
            }
        }
        // -100 * -100
        if(xMinIsNeg and yMinIsNeg)
        {
            int p { negMinX * negMinY };
            if(p > maxVal)
            {
                maxVal = p;
            }
        }
        // 100 * -1
        if(xMaxIsPos and yMinIsNeg)
        {
            int p {posMaxX * negMaxY};
            if( p > maxVal)
            {
                maxVal = p;
            }
        }
        // -1 * 100
        if(xMinIsNeg and yMaxIsPos)
        {
            int p = negMaxX * posMaxY;
            if(p > maxVal)
            {
                maxVal = p;
            }
        }
        
        intDomAct.removeAnyGreaterThan(varZ, maxVal);
    } //~
    

    
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
