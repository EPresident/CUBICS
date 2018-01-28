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
    
    // ---> CLARIFICATION / REMINDER <---
    // Just to be 99.99999% clear!
    // In the following code this "convention" is adopted,
    // considering as an example the full range {-100..100}:
    // - lowest negative value: -100
    // - highest negative value: -1
    // - lowest positive value: 0
    // - highest positive value: 100
    //
    // --- Elia
    
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
    
    // ---------------------------------------------------------
    // Check z lower bound
    // ---------------------------------------------------------
    {
        int minVal { INT_MAX };
        // If x has negative values, get the lowest negative possible
        // multiplying by the greatest positive value in y
        // e.g. 100 * -100
        if(xMaxIsPos and yMinIsNeg and posMaxX > 0)
        {
            int p {posMaxX * negMinY};
            if(p < minVal)
            {
                minVal = p;
            }
        }
        // If x has negative values, get the lowest negative possible
        // multiplying by the greatest positive value in y
        // e.g. -100 * 100
        if(xMinIsNeg and yMaxIsPos and posMaxY > 0)
        {
            int p { negMinX * posMaxY };
            if(p < minVal)
            {
                minVal = p;
            }
        }
        // If x and y both have positive values, get the lowest positive 
        // value possible by multiplying the smallest positive values
        // e.g. 1 * 1
        if(xMaxIsPos and yMaxIsPos)
        {
            int p {posMinX * posMinY};
            if(p < minVal)
            {
                minVal = p;
            }
        }
        // If x and y both have negative values, get the lowest positive 
        // value possible by multiplying the smallest negative values
        // e.g. -1 * -1
        if(xMinIsNeg and yMinIsNeg)
        {
            int p { negMaxX * negMaxY };
            if(p < minVal)
            {
                minVal = p;
            }
        }
        
        #ifdef NDEBUG
            assert(minVal < INT_MAX);
        #endif
        if(minVal > minZ) intDomAct.removeAnyLesserThan(varZ, minVal);
    } //~
    // ---------------------------------------------------------
    // Check z upper bound
    // ---------------------------------------------------------
    {
        int maxVal { INT_MIN };
        // If x and y both have positive values, get the highest positive 
        // value possible by multiplying the greatest positive values
        // e.g. 100 * 100
        if(xMaxIsPos and yMaxIsPos)
        {
            int p {posMaxX * posMaxY};
            if( p > maxVal)
            {
                maxVal = p;
            }
        }
        // If x and y both have negative values, get the highest positive 
        // value possible by multiplying the lowest negative values
        // e.g. -100 * -100
        if(xMinIsNeg and yMinIsNeg)
        {
            int p { negMinX * negMinY };
            if(p > maxVal)
            {
                maxVal = p;
            }
        }
        // If y has negative values, get the highest negative possible
        // multiplying by the smallest positive value in x
        // e.g. 1 * -1
        if(xMaxIsPos and yMinIsNeg)
        {
            int p {posMinX * negMaxY};
            if( p > maxVal)
            {
                maxVal = p;
            }
        }
        // If x has negative values, get the highest negative possible
        // multiplying by the smallest positive value in y 
        // e.g. -1 * 1
        if(xMinIsNeg and yMaxIsPos)
        {
            int p { negMaxX * posMinY };
            if(p > maxVal)
            {
                maxVal = p;
            }
        }
        
        #ifdef NDEBUG
            assert(maxVal > INT_MIN);
        #endif
        if(maxVal < maxZ) intDomAct.removeAnyGreaterThan(varZ, maxVal);
    } //~
    
    // ---------------------------------------------------------
    // Check x lower bound
    // ---------------------------------------------------------
    {
        int minVal { INT_MAX };
        // If z has positive values,
        // and y has negative values,
        // get the smallest negative possible
        // Of course it makes sense only if z > y
        // e.g. 100 / -1
        if(zMaxIsPos and yMinIsNeg and posMaxZ > negMaxY)
        {
            int q {posMaxZ / negMaxY};
            if(q < minVal)
            {
                minVal = q;
            }
        }
        // If z has negative values,
        // and y has positive values,
        // get the smallest negative possible
        // Of course it makes sense only if |z| > y
        // e.g. -100 / 1
        if(zMinIsNeg and yMaxIsPos and posMinY > 0 and -negMinZ > posMinY)
        {
            int q { negMinZ / posMinY };
            if(q < minVal)
            {
                minVal = q;
            }
        }
        // If z and y have positive values,
        // get the smallest positive x possible
        // e.g. 1 / 100
        //--x = {0..n} y = {1..m} z = {1..k}
        //--posMinZ / posMaxY = 1 / m
        if(zMaxIsPos and yMaxIsPos and posMaxY != 0)
        {
            int d {posMaxY};
            if(posMaxY > posMinZ and posMinZ != 0) d = posMinZ;
            // This way q is one if posMinZ < posMaxY
            // but it stays zero if posMinZ==0
            int q {posMinZ / d};
            if(q < minVal)
            {
                minVal = q;
            }
        }
        // If z and y have negative values,
        // get the smallest positive x possible
        // e.g. -1 / -100
        //   x = {0..n} y = {-100..m} z = {-1..k}
        if(zMinIsNeg and yMinIsNeg)
        {
            int q { negMaxZ / negMinY };
            if(q < 1) q = 1; // This way q is one if negMaxZ < negMinY
            if(q < minVal)
            {
                minVal = q;
            }
        }
        
        #ifdef NDEBUG
            assert(minVal < INT_MAX);
        #endif
        if(minVal > minX) intDomAct.removeAnyLesserThan(varX, minVal);
    } //~
    // ---------------------------------------------------------
    // Check x upper bound
    // ---------------------------------------------------------
    {
        int maxVal { INT_MIN };
        
        // If z and y have positive values,
        // get the biggest positive x possible
        // e.g. 100 / 1
        if(zMaxIsPos and yMaxIsPos and posMaxY > 0)
        {
            int d {posMinY} ;
            if(d == 0) d=1 ; 
            // This way I don't divide by zero
            // Since posMaxY > 0 I can divide by 1 instead
            int q {posMaxZ / d};
            if(posMaxZ % d != 0) q += 1; // ceiling
            if( q > maxVal)
            {
                maxVal = q;
            }
        }
        
        // If z and y have negative values,
        // get the biggest positive x possible
        // e.g. -100 / -1
        if(zMinIsNeg and yMinIsNeg)
        {
            int q { negMinZ / negMaxY };
            if(negMinZ % negMaxY != 0) q += 1; // ceiling
            if(q > maxVal)
            {
                maxVal = q;
            }
        }
        // If z has positive values,
        // and y has negative values,
        // get the biggest negative x possible (close to zero)
        // e.g. 1 / -100
        if(zMaxIsPos and yMinIsNeg)
        {
            int q {posMinZ / negMinY};
            //(posMinZ % negMaxY != 0) ? q -= 1; // ceiling
            if(q >= 0) q = -1; // make sure to stay negative (posMinZ can be 0)
            if( q > maxVal)
            {
                maxVal = q;
            }
        }
        // If z has negative values,
        // and y has positive values,
        // get the biggest negative x possible (close to zero)
        // e.g. -1 / 100
        if(zMinIsNeg and yMaxIsPos and posMaxY > 0)
        {
            int q = negMaxZ / posMaxY;
            //(negMaxZ % posMaxY != 0) ? q -= 1; // ceiling
            if(q >= 0) q = -1; // make sure to stay negative
            if(q > maxVal)
            {
                maxVal = q;
            }
        }
        
        #ifdef NDEBUG
            assert(maxVal > INT_MIN);
        #endif
        if(maxVal < maxX) intDomAct.removeAnyGreaterThan(varX, maxVal);
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
