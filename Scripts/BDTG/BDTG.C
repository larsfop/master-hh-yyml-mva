#include "HH_CV_Application.h"
#include "HH_CV_Classification.h"

#include <stdio.h>
#include <unistd.h>
#include <getopt.h>

int main(int argc, char** argv)
{

    // Default arguments
    TString channel = "1l0tau";
    TString path = "../../";
    bool isApplication = false;
    bool isClassification = false;
    Int_t n_threads = 1;
    Int_t nFolds = 4;

    int opt;

    // Parse command line arguments
    while ((opt = getopt(argc, argv, "c:bap:n:k:")) != -1)
    {
        switch (opt)
        {
            case 'c':
                channel = optarg;
                break;
            case 'b':
                isClassification = true;
                break;
            case 'a':
                isApplication = true;
                break;
            case 'p':
                path += optarg;
                break;
            case 'n':
                n_threads = atoi(optarg);
                break;
            case 'k':
                Int_t nFolds = atoi(optarg);
                break;
        }
    }

    printf("Processing channel: %s\n", channel.Data());

    // Enable multithreading
    printf("Running with %i threads\n", n_threads);
    ROOT::EnableImplicitMT(n_threads);

    printf("Running with %i folds\n", nFolds);
    
    std::cout << isClassification << std::endl;

    
    if (isClassification)
    {
        HH_CV_Classification(channel, path, nFolds);
    }

    if (isApplication)
    {
        HH_CV_Application(channel, path);
    }

    return 0;
}