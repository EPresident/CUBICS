#include <cstdlib>
#include <iostream>
#include <sstream>

#include <utils/Utils.h>
#include <flatzinc/flatzinc.h>
#include <searchers/IntBacktrackSearcher.h>
#include <searchers/IntLNSSearcher.h>
#include <options/Options.h>
#include <wrappers/Wrappers.h>

using namespace std;

int main(int argc, char * argv[])
{
    // Parse command line arguments
    Options opts;
    opts.initialize();
    opts.parseOptions(argc, argv);

    // Initialize FlatZinc model and printer
    FlatZinc::Printer printer;
    FlatZinc::FlatZincModel* fzModel = FlatZinc::parse(opts.inputFile, printer);

    // Initialize backtrack searcher
    /*IntBacktrackSearcher* backtrackSearcher;
    MemUtils::malloc(&backtrackSearcher);
    backtrackSearcher->initialize(fzModel);*/
    IntLNSSearcher* LNSSearcher;
    MemUtils::malloc(&LNSSearcher);
    LNSSearcher->initialize(fzModel,0.33333);

    bool* satisfiableModel;
    MemUtils::malloc(&satisfiableModel); // Must be readable by GPU
    *satisfiableModel = true;

    // Make sure the model is satisfiable, by propagating the constaints. (GPU/CPU)
#ifdef GPU
    LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSetLimit(cudaLimitMallocHeapSize, HEAP_SIZE));

    //Wrappers::propagateConstraints<<<1, 1>>>(&backtrackSearcher->propagator, satisfiableModel);
    Wrappers::propagateConstraints<<<1, 1>>>(&LNSSearcher->BTSearcher.propagator,
                                             satisfiableModel);
    LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSynchronize());
#else
    //*satisfiableModel = backtrackSearcher->propagator.propagateConstraints();
    *satisfiableModel = LNSSearcher->BTSearcher.propagator.propagateConstraints();
#endif

    if (*satisfiableModel)
    {
        bool* solutionFound;
        MemUtils::malloc(&solutionFound);
        *solutionFound = true;

        unsigned int solutionCount = 0;

        // Check if only the best solution is required
        bool onlyBestSolution = false;
        //onlyBestSolution = onlyBestSolution or backtrackSearcher->searchType == IntBacktrackSearcher::SearchType::Maximization;
        onlyBestSolution = onlyBestSolution or LNSSearcher->searchType ==
            IntLNSSearcher::SearchType::Maximization;
        //onlyBestSolution = onlyBestSolution or backtrackSearcher->searchType == IntBacktrackSearcher::SearchType::Minimization;
        onlyBestSolution = onlyBestSolution or LNSSearcher->searchType ==
            IntLNSSearcher::SearchType::Minimization;
        onlyBestSolution = onlyBestSolution and opts.solutionsCount == 1;
        std::stringstream bestSolution;
        
        /*
        * Find solutions until the search criteria are met.
        * That means finding one/n/all solutions, depending on the user
        * provided arguments.
        */
        while (*solutionFound and (solutionCount < opts.solutionsCount or onlyBestSolution))
        {
            // Get next solution (GPU/CPU)
#ifdef GPU
            //Wrappers::getNextSolution<<<1, 1>>>(backtrackSearcher, solutionFound);
            Wrappers::getNextSolution<<<1, 1>>>(LNSSearcher, solutionFound);
            LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSynchronize());
#else
            //*solutionFound = backtrackSearcher->getNextSolution();
            *solutionFound = LNSSearcher->getNextSolution();
#endif
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

    return EXIT_SUCCESS;
}
