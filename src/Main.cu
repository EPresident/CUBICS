#include <cstdlib>
#include <iostream>
#include <sstream>

#include <utils/Utils.h>
#include <flatzinc/flatzinc.h>
#include <searchers/IntBacktrackSearcher.h>
#include <options/Options.h>
#include <wrappers/Wrappers.h>
#include <statistics/Statistics.h>

using namespace std;

int main(int argc, char * argv[])
{
    Statistics* stats;
    MemUtils::malloc(&stats);
    stats->initialize();
    stats->setStartElaborationTime();

    Options opts;
    opts.initialize();
    opts.parseOptions(argc, argv);

    FlatZinc::Printer printer;
    FlatZinc::FlatZincModel* fzModel = FlatZinc::parse(opts.inputFile, printer);

    IntBacktrackSearcher* backtrackSearcher;
    MemUtils::malloc(&backtrackSearcher);
    backtrackSearcher->initialize(fzModel, stats);

    bool* satisfiableModel;
    MemUtils::malloc(&satisfiableModel);
    *satisfiableModel = true;

    stats->setStartSolveTime();

#ifdef GPU
    LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSetLimit(cudaLimitMallocHeapSize, HEAP_SIZE));

    Wrappers::propagateConstraints<<<1, 1>>>(&backtrackSearcher->propagator, satisfiableModel);
    LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSynchronize());
#else
    *satisfiableModel = backtrackSearcher->propagator.propagateConstraints();
#endif
    if (*satisfiableModel)
    {
        bool* solutionFound;
        MemUtils::malloc(&solutionFound);
        *solutionFound = true;

        bool onlyBestSolution = false;
        onlyBestSolution = onlyBestSolution or backtrackSearcher->searchType == IntBacktrackSearcher::SearchType::Maximization;
        onlyBestSolution = onlyBestSolution or backtrackSearcher->searchType == IntBacktrackSearcher::SearchType::Minimization;
        onlyBestSolution = onlyBestSolution and opts.solutionsCount == 1;
        std::stringstream bestSolution;
        while (*solutionFound and (stats->solutionsCount < opts.solutionsCount or onlyBestSolution))
        {
#ifdef GPU
            Wrappers::getNextSolution<<<1, 1>>>(backtrackSearcher, solutionFound);
            LogUtils::cudaAssert(__PRETTY_FUNCTION__, cudaDeviceSynchronize());
#else
            *solutionFound = backtrackSearcher->getNextSolution();
#endif
            if (*solutionFound)
            {

                if (not onlyBestSolution)
                {
                    stats->solutionsCount += 1;

                    printer.print(cout, *fzModel);
                    cout << "----------" << endl;
                }
                else
                {
                    stats->solutionsCount = 1;

                    bestSolution.str("");
                    printer.print(bestSolution, *fzModel);
                }
            }
        }

        if(onlyBestSolution)
        {
            cout << bestSolution.rdbuf();
            cout << "----------" << endl;
        }

        if (stats->solutionsCount > 0)
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

    stats->setEndSolveTime();
    stats->setEndElaborationTime();

    if(opts.printStats)
    {
        stats->print();
    }

    return EXIT_SUCCESS;
}
