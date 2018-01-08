#include <cstdlib>
#include <cstring>
#include <climits>

#include <options/Options.h>

using namespace std;

/**
* Initialize the members of the Options struct.
* solverVersion is hardcoded as 1.0, solutionsCount is defaulted to 1,
* inputFile is set as a nullptr.
*/
void Options::initialize()
{
    name = "CUBICS (CUDA BasIc Constrain Solver)";
    version = "1.0";
    solutionsCount = 1;
    inputFile = nullptr;
    opt = new AnyOption();
    iterations = 100;
    timeout = 300 * 1000;
}

/// Parse the arguments given from the command line.
void Options::parseOptions(int argc, char * argv[])
{
    opt->addUsage("Usage: cubics [OPTIONS]... [FILE]");
    opt->addUsage("");
    opt->addUsage("Options:");
    opt->addUsage("");
    opt->addUsage(" -h  --help         Prints this help");
    opt->addUsage(" -n [NUMBER]        Print at most NUMBER solution");
    opt->addUsage(" -a                 Print all the solutions");
    opt->addUsage(" -t [NUMBER]        Timeout after NUMBER milliseconds");
    opt->addUsage(" -i [NUMBER]        Do NUMBER LNS iterations");
    opt->addUsage(" --version          Print solver version");

    opt->setFlag('a');
    opt->setOption('n');
    opt->setOption('t');
    opt->setOption('i');
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
    
    if (opt->getValue('t') != nullptr)
    {
        timeout = std::stoul(string(opt->getValue('t')));
    }
    
    if (opt->getValue('i') != nullptr)
    {
        iterations = std::stoul(string(opt->getValue('i')));
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
