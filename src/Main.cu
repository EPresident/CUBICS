#include <cstdlib>
#include <iostream>
#include <sstream>
#include <chrono>
#include <random>

#include <utils/Utils.h>
#include <flatzinc/flatzinc.h>
#include <searchers/IntBacktrackSearcher.h>
#include <searchers/IntLNSSearcher.h>
#include <options/Options.h>
#include <wrappers/Wrappers.h>

using namespace std;

int main(int argc, char * argv[])
{
    //-------------------------------------------------------------------------------
    // Start timer
    //-------------------------------------------------------------------------------
    std::chrono::steady_clock::time_point startTime {std::chrono::steady_clock::now()};
    
    IntBacktrackSearcher* backtrackSearcher;
    IntLNSSearcher* LNSSearcher;
    IntSNBSearcher* SNBSearcher;
    //-------------------------------------------------------------------------------
    // Parse command line arguments
    //-------------------------------------------------------------------------------
    Options opts;
    opts.initialize();
    opts.parseOptions(argc, argv);
    //-------------------------------------------------------------------------------
    // Initialize FlatZinc model and printer
    //-------------------------------------------------------------------------------
    FlatZinc::Printer printer;
    FlatZinc::FlatZincModel* fzModel = FlatZinc::parse(opts.inputFile, printer);

    // Max elapsed time in ns
    long long timeout = opts.timeout * 1000000;
    cout << "Timeout: " << opts.timeout << " ms" << endl ;

    // Number of neighborhoods processed in parallel
    int neighborhoodsAmount = 4;

    //-------------------------------------------------------------------------------
    // Initialize searcher
    //-------------------------------------------------------------------------------
    switch(opts.mode)
    {
        case Options::SearchMode::Backtracking:
            MemUtils::malloc(&backtrackSearcher);
            backtrackSearcher->initialize(fzModel);
            break;

        case Options::SearchMode::LNS:
            MemUtils::malloc(&LNSSearcher);
            LNSSearcher->initialize(fzModel,opts.unassignRate, opts.iterations);
            break;
        
        case Options::SearchMode::SNBS:
            MemUtils::malloc(&SNBSearcher);
            SNBSearcher->initialize(fzModel,opts.unassignAmount, opts.iterations);
            break;
    }

    bool* satisfiableModel;
    MemUtils::malloc(&satisfiableModel); // Must be readable by GPU
    *satisfiableModel = true;
    
    //-------------------------------------------------------------------------------
    // Make sure the model is satisfiable, by propagating the constaints. (GPU/CPU)
    //-------------------------------------------------------------------------------
    #ifdef GPU
    LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSetLimit(cudaLimitMallocHeapSize, HEAP_SIZE));
    #endif
    switch(opts.mode)
    {
        case Options::SearchMode::Backtracking:
            #ifdef GPU
            Wrappers::propagateConstraints<<<1, 1>>>(&backtrackSearcher->
                propagator, satisfiableModel);
            #else
            *satisfiableModel = SNBSearcher->propagator.propagateConstraints();
            #endif
            break;

        case Options::SearchMode::LNS:
        case Options::SearchMode::SNBS:
            IntConstraintsPropagator tempProp;
            tempProp.initialize(fzModel->intVariables, fzModel->intConstraints);
            #ifdef GPU
            Wrappers::propagateConstraints<<<1, 1>>>(&tempProp, satisfiableModel);
            #else
            *satisfiableModel = tempProp.propagateConstraints();
            #endif
            tempProp.deinitialize();
            break;
    }
    #ifdef GPU
    LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSynchronize());
    #endif
    //-------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------
    /*
    * LNS & Co. only: backup original domains after the first propagation
    */
    //-------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------
    if(opts.mode == Options::SearchMode::LNS or opts.mode == Options::SearchMode::SNBS)
    {
        IntDomainsRepresentations* originalDomains;
        MemUtils::malloc(&originalDomains);
        int varCount {fzModel->intVariables->count};
        originalDomains->initialize(varCount);
        IntDomainsRepresentations* intDomRepr  = &fzModel->intVariables->domains.representations;
        for (int vi = 0; vi < varCount; vi += 1)
        {   
            int min = intDomRepr->minimums[vi];
            int max = intDomRepr->maximums[vi];
            int offset = intDomRepr->offsets[vi];
            int version = intDomRepr->versions[vi];
            Vector<unsigned int>* bitvector = &intDomRepr->bitvectors[vi];
            originalDomains->push(min, max, offset, version, bitvector);
        }
        if(opts.mode == Options::SearchMode::LNS)
        {
            LNSSearcher->originalDomains = originalDomains;
        }
        else
        {
            //SNBSearcher->originalDomains = originalDomains;
        }
    }
    
    //-------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------
    /*
    * LNS & Co. only: generate neighborhoods with Fisher-Yates
    */
    //-------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------
    Vector<Vector<int>> neighborhoods;
    if(opts.mode == Options::SearchMode::LNS or opts.mode == Options::SearchMode::SNBS)
    {
        long randSeed = 1273916546123835; // Arbitrary seed, FIXME
        std::mt19937 mt_rand = std::mt19937(randSeed);
        int optVariable = fzModel->optVar();
        
        neighborhoods.initialize(neighborhoodsAmount);
        for(int nbh = 0; nbh < neighborhoodsAmount; nbh += 1)
        {
            // Fill variables vector to be shuffled
            neighborhoods[nbh].initialize(fzModel->intVariables->count);
            for(int i = 0; i < fzModel->intVariables->count; i += 1)
            {
                neighborhoods[nbh].push_back(i);
            }
            
            // Shuffle (Fisher-Yates/Knuth)
            for(int i = 0; i < fzModel->intVariables->count-1; i += 1)
            {
                // We want a random variable index (bar the optVariable)
                std::uniform_int_distribution<int> rand_dist(i, fzModel->intVariables->count-2);
                int j{rand_dist(mt_rand)};
                int tmp{neighborhoods[nbh][i]};
                neighborhoods[nbh][i] = neighborhoods[nbh][j];
                neighborhoods[nbh][j] = tmp;
            }
            neighborhoods[nbh].push_back(optVariable);
        }
    }
    //-------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------
    
    long long elapsedTime { std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - startTime).count() };

    if (*satisfiableModel)
    {
        bool* solutionFound;
        MemUtils::malloc(&solutionFound);
        *solutionFound = true;

        unsigned int solutionCount = 0;

        // Check if only the best solution is required
        bool onlyBestSolution = false;
        switch(opts.mode)
        {
            case Options::SearchMode::Backtracking:
                onlyBestSolution = onlyBestSolution or backtrackSearcher->searchType ==
                    IntBacktrackSearcher::SearchType::Maximization;
                onlyBestSolution = onlyBestSolution or backtrackSearcher->searchType ==
                    IntBacktrackSearcher::SearchType::Minimization;
                break;

            case Options::SearchMode::LNS:
                onlyBestSolution = onlyBestSolution or LNSSearcher->searchType ==
                    IntLNSSearcher::SearchType::Maximization;
                onlyBestSolution = onlyBestSolution or LNSSearcher->searchType ==
                    IntLNSSearcher::SearchType::Minimization;
                break;
            
            case Options::SearchMode::SNBS:
                onlyBestSolution = onlyBestSolution or SNBSearcher->searchType ==
                    IntSNBSearcher::SearchType::Maximization;
                onlyBestSolution = onlyBestSolution or SNBSearcher->searchType ==
                    IntSNBSearcher::SearchType::Minimization;
                break;
        }
        onlyBestSolution = onlyBestSolution and opts.solutionsCount == 1;
        std::stringstream bestSolution;
        //-------------------------------------------------------------------------------
        /*
        * LNS & Co. only: find a first solution
        */
        //-------------------------------------------------------------------------------
        if(false and opts.mode == Options::SearchMode::LNS or opts.mode == Options::SearchMode::SNBS)
        {
            IntDomainsRepresentations* solDomRepr;
            MemUtils::malloc(&solDomRepr);
            int varCount {fzModel->intVariables->count};
            MemUtils::malloc(&backtrackSearcher);
            backtrackSearcher->initialize(fzModel);
            #ifdef GPU
            Wrappers::getNextSolution<<<1, 1>>>(backtrackSearcher, solutionFound, timeout - elapsedTime);
            LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSynchronize());
            #else
            solutionFound = backtrackSearcher->getNextSolution();
            #endif
            backtrackSearcher->deinitialize();
            // Print/store the found solution.
            if (not onlyBestSolution)
            {
                solutionCount += 1;

                printer.print(cout, *fzModel);
                cout << "----------" << endl;
            }
            else
            {
                solutionCount = 1;

                bestSolution.str("");
                printer.print(bestSolution, *fzModel);
            }
            solDomRepr->initialize(varCount);
            IntDomainsRepresentations* intDomRepr  = &fzModel->intVariables->domains.representations;
            for (int vi = 0; vi < varCount; vi += 1)
            {   
                int min = intDomRepr->minimums[vi];
                int max = intDomRepr->maximums[vi];
                int offset = intDomRepr->offsets[vi];
                int version = intDomRepr->versions[vi];
                Vector<unsigned int>* bitvector = &intDomRepr->bitvectors[vi];
                solDomRepr->push(min, max, offset, version, bitvector);
            }
            if(opts.mode == Options::SearchMode::LNS)
            {
                LNSSearcher->bestSolution = solDomRepr;
            }
            else
            {
                //SNBSearcher->bestSolution = solDomRepr;
            }
            elapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - startTime).count();
        }
        //-------------------------------------------------------------------------------
        /*
        * Find solutions until the search criteria are met.
        * That means finding one/n/all solutions, depending on the user
        * provided arguments.
        */
        //-------------------------------------------------------------------------------
        while (*solutionFound and 
               (solutionCount < opts.solutionsCount or onlyBestSolution) and 
               elapsedTime < timeout
              )
        {
            elapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - startTime).count();
            long searcherTimeout {timeout - elapsedTime};
            //-------------------------------------------------------------------------------
            // Get next solution (GPU/CPU)
            //-------------------------------------------------------------------------------
            switch(opts.mode)
            {
                case Options::SearchMode::Backtracking:
                    #ifdef GPU
                    Wrappers::getNextSolution<<<1, 1>>>(backtrackSearcher, solutionFound, searcherTimeout);
                    LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSynchronize());
                    #else
                    solutionFound = backtrackSearcher->getNextSolution(searcherTimeout);
                    #endif
                    break;

                case Options::SearchMode::LNS:
                    #ifdef GPU
                    Wrappers::getNextSolution<<<1, 1>>>(LNSSearcher, solutionFound, searcherTimeout);
                    LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSynchronize());
                    #else
                    *solutionFound = LNSSearcher->getNextSolution(searcherTimeout);
                    #endif
                    break;
                
                case Options::SearchMode::SNBS:
                    #ifdef GPU
                    Wrappers::getNextSolution<<<1, 1>>>(SNBSearcher, solutionFound, searcherTimeout);
                    LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSynchronize());
                    #else
                    *solutionFound = SNBSearcher->getNextSolution(searcherTimeout);
                    #endif
                    break;
            }
            
            //-------------------------------------------------------------------------------
            // Measure time elapsed
            //-------------------------------------------------------------------------------
            elapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - startTime).count();
            
            if (*solutionFound)
            {
                // Print/store the found solution.
                if (not onlyBestSolution)
                {
                    solutionCount += 1;

                    printer.print(cout, *fzModel);
                    cout << "----------" << endl;
                }
                else
                {
                    solutionCount = 1;

                    bestSolution.str("");
                    printer.print(bestSolution, *fzModel);
                }
            }
        }

        // Print best solution.
        if(onlyBestSolution)
        {
            cout << bestSolution.rdbuf();
            cout << "----------" << endl;
        }

        if (solutionCount > 0)
        {
            cout << "==========" << endl;
        }
        else
        {
            cout << "=====UNSATISFIABLE=====" << endl;
        }
    }
    else
    {
        cout << "=====UNSATISFIABLE=====" << endl;
    }

    elapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - startTime).count();
    cout << "Elapsed time: " << elapsedTime / 1000000000.0 << " s" << endl;
    //-------------------------------------------------------------------------------
    // Print timeout message
    //-------------------------------------------------------------------------------
    if(elapsedTime >= timeout)
    {
        cout << ">>> Timed out! <<<" << endl;
    }
    
    return EXIT_SUCCESS;
}
