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

int launchBTS(Options* options);
int launchLNS(Options* options);
int launchSNBS(Options* options);

int main(int argc, char * argv[])
{
    // Parse command line arguments
    Options opts;
    opts.initialize();
    opts.parseOptions(argc, argv);

    switch(opts.mode)
    {
        case Options::SearchMode::Backtracking:
            return launchBTS(&opts);

        case Options::SearchMode::LNS:
            return launchLNS(&opts);
        
        case Options::SearchMode::SNBS:
            return launchSNBS(&opts);
    }
}

//-------------------------------------------------------------------
//-------------------------------------------------------------------
//-------------------------------------------------------------------
//-------------------------------------------------------------------
//-------------------------------------------------------------------

int launchSNBS(Options* options)
{
    Options& opts = *options;
    // Initialize FlatZinc model and printer
    FlatZinc::Printer printer;
    FlatZinc::FlatZincModel* fzModel = FlatZinc::parse(opts.inputFile, printer);

    // Initialize searcher
    IntSNBSearcher* SNBSearcher;
    MemUtils::malloc(&SNBSearcher);
    SNBSearcher->initialize(fzModel,opts.unassignAmount, opts.iterations);

    bool* satisfiableModel;
    MemUtils::malloc(&satisfiableModel); // Must be readable by GPU
    *satisfiableModel = true;
    
    // Max elapsed time in ms
    long timeout = opts.timeout;
    cout << "Timeout: " << timeout << endl ;
    std::chrono::steady_clock::time_point startTime {std::chrono::steady_clock::now()};
    
    // Make sure the model is satisfiable, by propagating the constaints. (GPU/CPU)
#ifdef GPU
    LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSetLimit(cudaLimitMallocHeapSize, HEAP_SIZE));

    Wrappers::propagateConstraints<<<1, 1>>>(&SNBSearcher->propagator,
                                             satisfiableModel);
    LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSynchronize());
#else
    *satisfiableModel = SNBSearcher->propagator.propagateConstraints();
#endif
    
    long elapsedTime { std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - startTime).count() };

    if (*satisfiableModel)
    {
        bool* solutionFound;
        MemUtils::malloc(&solutionFound);
        *solutionFound = true;

        unsigned int solutionCount = 0;

        // Check if only the best solution is required
        bool onlyBestSolution = false;
        onlyBestSolution = onlyBestSolution or SNBSearcher->searchType ==
            IntSNBSearcher::SearchType::Maximization;
        onlyBestSolution = onlyBestSolution or SNBSearcher->searchType ==
            IntSNBSearcher::SearchType::Minimization;
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
            // Get next solution (GPU/CPU)
            elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - startTime).count();
            long searcherTimeout {timeout - elapsedTime};
#ifdef GPU
            Wrappers::getNextSolution<<<1, 1>>>(SNBSearcher, solutionFound, searcherTimeout);
            LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSynchronize());
#else
            //*solutionFound = backtrackSearcher->getNextSolution();
            *solutionFound = SNBSearcher->getNextSolution(searcherTimeout);
#endif
            
            // Measure time
            elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
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

    elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - startTime).count();
    cout << "Elapsed time: " << elapsedTime << " ms" << endl;
    
    return EXIT_SUCCESS;
}

//-------------------------------------------------------------------
//-------------------------------------------------------------------
//-------------------------------------------------------------------
//-------------------------------------------------------------------
//-------------------------------------------------------------------

int launchLNS(Options* options)
{
    Options& opts = *options;
    // Initialize FlatZinc model and printer
    FlatZinc::Printer printer;
    FlatZinc::FlatZincModel* fzModel = FlatZinc::parse(opts.inputFile, printer);
    // Initialize searcher
    IntLNSSearcher* LNSSearcher;
    MemUtils::malloc(&LNSSearcher);
    LNSSearcher->initialize(fzModel,opts.unassignRate, opts.iterations);

    bool* satisfiableModel;
    MemUtils::malloc(&satisfiableModel); // Must be readable by GPU
    *satisfiableModel = true;
    
    // Max elapsed time in ms
    long timeout = opts.timeout;
    cout << "Timeout: " << timeout << endl ;
    std::chrono::steady_clock::time_point startTime {std::chrono::steady_clock::now()};
    
    // Make sure the model is satisfiable, by propagating the constaints. (GPU/CPU)
#ifdef GPU
    LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSetLimit(cudaLimitMallocHeapSize, HEAP_SIZE));

    Wrappers::propagateConstraints<<<1, 1>>>(&LNSSearcher->BTSearcher.propagator,
                                             satisfiableModel);
    LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSynchronize());
#else
    *satisfiableModel = LNSSearcher->BTSearcher.propagator.propagateConstraints();
#endif
    
    long elapsedTime { std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - startTime).count() };

    if (*satisfiableModel)
    {
        bool* solutionFound;
        MemUtils::malloc(&solutionFound);
        *solutionFound = true;

        unsigned int solutionCount = 0;

        // Check if only the best solution is required
        bool onlyBestSolution = false;
        onlyBestSolution = onlyBestSolution or LNSSearcher->searchType ==
            IntLNSSearcher::SearchType::Maximization;
        onlyBestSolution = onlyBestSolution or LNSSearcher->searchType ==
            IntLNSSearcher::SearchType::Minimization;
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
            // Get next solution (GPU/CPU)
            elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - startTime).count();
            long searcherTimeout {timeout - elapsedTime};
#ifdef GPU
            Wrappers::getNextSolution<<<1, 1>>>(LNSSearcher, solutionFound, searcherTimeout);
            LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSynchronize());
#else
            *solutionFound = LNSSearcher->getNextSolution(searcherTimeout);
#endif
            
            // Measure time
            elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
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

    elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - startTime).count();
    cout << "Elapsed time: " << elapsedTime << " ms" << endl;
    
    return EXIT_SUCCESS;
}

//-------------------------------------------------------------------
//-------------------------------------------------------------------
//-------------------------------------------------------------------
//-------------------------------------------------------------------
//-------------------------------------------------------------------

int launchBTS(Options* options)
{
    Options& opts = *options;
    // Initialize FlatZinc model and printer
    FlatZinc::Printer printer;
    FlatZinc::FlatZincModel* fzModel = FlatZinc::parse(opts.inputFile, printer);
    // Initialize backtrack searcher
    IntBacktrackSearcher* backtrackSearcher;
    MemUtils::malloc(&backtrackSearcher);
    backtrackSearcher->initialize(fzModel);

    bool* satisfiableModel;
    MemUtils::malloc(&satisfiableModel); // Must be readable by GPU
    *satisfiableModel = true;
    
    // Max elapsed time in ms
    long timeout = opts.timeout;
    cout << "Timeout: " << timeout << endl ;
    std::chrono::steady_clock::time_point startTime {std::chrono::steady_clock::now()};
    
    // Make sure the model is satisfiable, by propagating the constaints. (GPU/CPU)
#ifdef GPU
    LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSetLimit(cudaLimitMallocHeapSize, HEAP_SIZE));

    Wrappers::propagateConstraints<<<1, 1>>>(&backtrackSearcher->propagator, satisfiableModel);
    LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSynchronize());
#else
    *satisfiableModel = backtrackSearcher->propagator.propagateConstraints();
#endif
    
    long elapsedTime { std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - startTime).count() };

    if (*satisfiableModel)
    {
        bool* solutionFound;
        MemUtils::malloc(&solutionFound);
        *solutionFound = true;

        unsigned int solutionCount = 0;

        // Check if only the best solution is required
        bool onlyBestSolution = false;
        onlyBestSolution = onlyBestSolution or backtrackSearcher->searchType == IntBacktrackSearcher::SearchType::Maximization;
        onlyBestSolution = onlyBestSolution or backtrackSearcher->searchType == IntBacktrackSearcher::SearchType::Minimization;
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
            // Get next solution (GPU/CPU)
            elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - startTime).count();
            long searcherTimeout {timeout - elapsedTime};
#ifdef GPU
            Wrappers::getNextSolution<<<1, 1>>>(backtrackSearcher, solutionFound); // timeout
            LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSynchronize());
#else
            *solutionFound = backtrackSearcher->getNextSolution(); // timeout
#endif
            
            // Measure time
            elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
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

    elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - startTime).count();
    cout << "Elapsed time: " << elapsedTime << " ms" << endl;
    
    return EXIT_SUCCESS;
}
