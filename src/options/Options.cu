#include <cstdlib>
#include <cstring>
#include <climits>

#include <options/Options.h>

using namespace std;

void Options::initialize()
{
    name = "CUBICS (CUDA BasIc Constrain Solver)";
    version = "1.0";
    solutionsCount = 1;
    inputFile = nullptr;
    opt = new AnyOption();
}

void Options::parseOptions(int argc, char * argv[])
{
    opt->addUsage("Usage: cubics [OPTIONS]... [FILE]");
    opt->addUsage("");
    opt->addUsage("Options:");
    opt->addUsage("");
    opt->addUsage(" -h  --help         Prints this help");
    opt->addUsage(" -n [NUMBER]        Print at most NUMBER solution");
    opt->addUsage(" -a                 Print all the solutions");
    opt->addUsage(" --version          Print solver version");

    opt->setFlag('a');
    opt->setOption('n');
    opt->setFlag("help", 'h');
    opt->setFlag("version");

    opt->processCommandArgs(argc, argv);

    if (!opt->hasOptions())
    {
        opt->printUsage();
        exit(EXIT_FAILURE);
    }

    if (opt->getFlag("help") || opt->getFlag('h'))
    {
        opt->printUsage();
        exit(EXIT_SUCCESS);
    }

    if (opt->getValue('a') != nullptr)
    {
        solutionsCount = UINT_MAX;
    }

    if (opt->getValue('n') != nullptr)
    {
        solutionsCount = std::stoul(string(opt->getValue('n')));
    }

    if (opt->getValue("version") != nullptr)
    {
        cout << name << " " << version << endl ;
        exit(EXIT_SUCCESS);
    }

    char* arg  = opt->getArgv(0);

    int argLength = strlen(arg);
    if (argLength > 4)
    {
        if (strcmp(arg + (argLength - 4), ".fzn") == 0)
        {
            inputFile = static_cast<char*>(malloc((argLength + 1) * sizeof(char)));
            strcpy(inputFile, arg);
        }
    }

}
