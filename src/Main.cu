#include <cstdlib>
#include <iostream>

#include <utils/Utils.h>
#include <flatzinc/flatzinc.h>
#include <searchers/IntBacktrackSearcher.h>
#include <options/Options.h>

using namespace std;

int main(int argc, char * argv[])
{
    Options opts;
    opts.initialize();
    opts.parseOptions(argc, argv);

    FlatZinc::Printer printer;
    FlatZinc::FlatZincModel* fzModel = FlatZinc::parse(opts.inputFile, printer);

    IntBacktrackSearcher* backtrackSearcher;
    MemUtils::malloc(&backtrackSearcher);
    backtrackSearcher->initialize(fzModel->intVariables, fzModel->intConstraints);

    bool* satisfiableModel;
    MemUtils::malloc(&satisfiableModel);
    *satisfiableModel = true;

    *satisfiableModel = backtrackSearcher->propagator.propagateConstraints();
    if (*satisfiableModel)
    {
        bool* solutionFound;
        MemUtils::malloc(&solutionFound);
        *solutionFound = true;

        unsigned int solutionCount = 0;
        while (*solutionFound and solutionCount < opts.solutionsCount)
        {
            *solutionFound = backtrackSearcher->getNextSolution();
            if (*solutionFound)
            {
                solutionCount += 1;

                printer.print(cout, *fzModel);
                cout << "----------" << endl;
            }
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
