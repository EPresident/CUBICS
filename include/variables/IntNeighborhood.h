#pragma once

#include <data_structures/Vector.h>
#include <data_structures/BitsArray.h>
#include <domains/IntDomainsRepresentations.h>
#include <domains/IntDomainsActions.h>

/**
* This struct represents a neighborhood of integer variables, i.e.
* a subset of the model's variables and their domains.
*/
struct IntNeighborhood
{
    enum EventTypes
    {
        None,
        Changed ///< A domain has been changed
    };
    //-----------------------------------------------------------
    // Domains+variables stuff
    //-----------------------------------------------------------
    /// Number of integer variables.
    int count;
    #ifdef GPU
    /// Blocks required to launch the kernels with default block size
    int variablesBlocks;
    #endif
    /// Bitmask for each variable in the domain. 1 = is a neighbor.
    BitsArray neighMask;
    /// Representations of the domains of the neighbors
    IntDomainsRepresentations neighRepr;
    /// Domain actions of the neighbors
    IntDomainsActions neighActions;
    /**
     * Map from variable name (number) to its representation in \a neighRepr
     * If map[i]=j, then variable j has its neigh repr at index i.
     */ 
    Vector<int> map;
    /// A list of domain events (domain changed) in chronological order.
    Vector<int> events;
    //-----------------------------------------------------------
    // Propagation stuff
    //-----------------------------------------------------------
    /// Indicates if the i-th constraint has been propagated or not
    Vector<bool> constraintToPropagate;
    /// true if at least one domain has become empty
    bool someEmptyDomain;
    /// true if at least one constraint has to be (re)propagated
    bool someConstraintsToPropagate;
    bool allConstraintsSatisfied;

    cudaHostDevice void initialize(Vector<int>* neighbors, IntDomainsRepresentations* originalRepr);
    cudaHostDevice void deinitialize();

    /**
     * Add the neighbor variables to the neighborhood.
     * \param neighbors list of the variable composing the neighborhood
     * \param originalRepr the representation of the original domain of the variable
     */
    void pushNeighbors(Vector<int>* neighbors, IntDomainsRepresentations* originalRepr);
    
    /// Find which representation is for \a var and return it in \a repr (used for kernel)
    cudaDevice void getBinding(int var, int* repr);
    /// Find which representation is for \a var (used as a normal function)
    cudaDevice int getRepresentationIndex(int var);
    
    /// Return true if \a var is in the neighborhood
    cudaDevice inline bool isNeighbor(int var)
    {
        assert(var > 0 and var < count);
        return neighMask.get(var);
    }
};
