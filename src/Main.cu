#include <cstdlib>
#include <iostream>
#include <sstream>
#include <chrono>

#include <utils/Utils.h>
#include <flatzinc/flatzinc.h>
#include <searchers/IntBacktrackSearcher.h>
#include <searchers/IntLNSSearcher.h>
#include <options/Options.h>
#include <wrappers/Wrappers.h>

using namespace std;

int main(int argc, char * argv[])
{
    IntBacktrackSearcher* backtrackSearcher;
    IntLNSSearcher* LNSSearcher;
    IntSNBSearcher* SNBSearcher;
    
    // Parse command line arguments
    Options opts;
    opts.initialize();
    opts.parseOptions(argc, argv);
    
    // Initialize FlatZinc model and printer
    FlatZinc::Printer printer;
    FlatZinc::FlatZincModel* fzModel = FlatZinc::parse(opts.inputFile, printer);

    // Initialize searcher
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
    
    // Max elapsed time in ns
    long long timeout = opts.timeout * 1000000;
    cout << "Timeout: " << opts.timeout << " ms" << endl ;
    std::chrono::steady_clock::time_point startTime {std::chrono::steady_clock::now()};
    
    // Make sure the model is satisfiable, by propagating the constaints. (GPU/CPU)
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
            #ifdef GPU
            Wrappers::propagateConstraints<<<1, 1>>>(&LNSSearcher->BTSearcher.
                propagator, satisfiableModel);
            #else
            *satisfiableModel = LNSSearcher->BTSearcher->propagator.propagateConstraints();
            #endif
            break;
        
        case Options::SearchMode::SNBS:
            #ifdef GPU
            Wrappers::propagateConstraints<<<1, 1>>>(&SNBSearcher->
                propagator, satisfiableModel);
            #else
            *satisfiableModel = SNBSearcher->propagator.propagateConstraints();
            #endif
            break;
    }
    #ifdef GPU
    LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSynchronize());
    #endif
    
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
        
        /*
        * Find solutions until the search criteria are met.
        * That means finding one/n/all solutions, depending on the user
        * provided arguments.
        */
        while (*solutionFound and 
               (solutionCount < opts.solutionsCount or onlyBestSolution) and 
               elapsedTime < timeout
              )
        {
            elapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - startTime).count();
            long searcherTimeout {timeout - elapsedTime};
            // Get next solution (GPU/CPU)
            switch(opts.mode)
            {
                case Options::SearchMode::Backtracking:
                    #ifdef GPU
                    Wrappers::getNextSolution<<<1, 1>>>(backtrackSearcher, solutionFound);
                    LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSynchronize());
                    #else
                    solutionFound = backtrackSearcher->getNextSolution();
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
            
            // Measure time
            elapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - startTime).count();
            //cout << "Solution: " << elapsedTime << endl;
            
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
    
    // Print timeout message
    if(elapsedTime >= timeout)
    {
        cout << ">>> Timed out! <<<" << endl;
    }
    
    return EXIT_SUCCESS;
}
