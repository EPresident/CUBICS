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
    timeout = 300 * 1000; // 5 minutes
    mode = Backtracking;
}

/// Parse the arguments given from the command line.
void Options::parseOptions(int argc, char * argv[])
{
    opt->addUsage("Usage: cubics [OPTIONS]... [FILE]");
    opt->addUsage("");
    opt->addUsage("Options:");
    opt->addUsage("");
    opt->addUsage(" -h  --help         Prints this help");
    opt->addUsage(" -n [NUMBER]        Print at most NUMBER solutions (default 1)");
    opt->addUsage(" -a                 Print all the solutions");
    opt->addUsage(" -t [NUMBER]        Timeout after NUMBER milliseconds (default 5 minutes)");
    opt->addUsage(" --version          Print solver version");
    opt->addUsage(" --lns [NUMBER]     Do Large Neighborhood Search unassigning NUMBER percent of the variables");
    opt->addUsage(" --snbs [NUMBER]    Do Small Neighborhood Brute Search, with a neighborhood of size NUMBER");
    opt->addUsage("");
    opt->addUsage("With --lns or --snbs option");
    opt->addUsage(" -i [NUMBER]        Do NUMBER LNS/SNBS iterations (default 100)");

    opt->setFlag('a');
    opt->setOption('n');
    opt->setOption('t');
    opt->setOption('i');
    opt->setOption("lns");
    opt->setOption("snbs");
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

    if (opt->getValue("lns") != nullptr)
    {
        mode = LNS;
        unassignRate = std::stoul(string(opt->getValue("lns"))) / 100.0;
        if(unassignRate < 0.01 or unassignRate > 0.99) 
        {
            cout << "Percent of variables unassigned must be between 1 and 99!" << endl ;
            exit(EXIT_FAILURE);
        }
    }

    if (opt->getValue("snbs") != nullptr)
    {
        mode = SNBS;
        unassignAmount = std::stoul(string(opt->getValue("snbs")));
        if(unassignAmount < 1) 
        {
            cout << "Number of variables unassigned must be greater than zero!" << endl ;
            exit(EXIT_FAILURE);
        }
        
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
