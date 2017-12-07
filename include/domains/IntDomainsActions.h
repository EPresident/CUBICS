#pragma once

#include <data_structures/Vector.h>
#include <data_structures/Lock.h>

/**
* This struct represents domain reduction actions for the domains
* of the (integer) variables.
*/
struct IntDomainsActions
{
    /// Single elements to be removed.
    Vector<Vector<int>> elementsToRemove;
    /// All elements smaller than the lower bounds are to be removed.
    Vector<int> lowerbounds;
    /// All elements greater than the upper bounds are to be removed.
    Vector<int> upperbounds;
#ifdef GPU
    Vector<Lock> locks;
#endif

    /// Initialize the struct for "count" variables. 
    void initialize(int count);
    void deinitialize();

    /** 
    * Add a new empty action, i.e. add room for a new element in 
    * the vectors of the struct. 
    */
    void push();

    /** 
    * Clear the "index"-th action.
    * This is typically called after the action is performed by IntDomains.
    */
    cudaDevice void clear(int index);

    /** 
    * Queue the action to remove the "val" value from the "index"-th domain.
    * "val" must be within the bounds, and the operation is mutually exclusive
    * on GPU.
    * This is done by adding "val" to elementsToRemove["index"].
    */
    cudaDevice void removeElement(int index, int val);
    /** 
    * Queue the action to remove the any value smaller than "val" from 
    * the "index"-th domain.
    * "val" must be within the bounds.
    * This is done by setting lowerbounds["index"] to "val".
    */
    cudaDevice void removeAnyLesserThan(int index, int val);
    /** 
    * Queue the action to remove the any value greater than "val" from 
    * the "index"-th domain.
    * "val" must be within the bounds.
    * This is done by setting upperbounds["index"] to "val".
    */
    cudaDevice void removeAnyGreaterThan(int index, int val);
};
