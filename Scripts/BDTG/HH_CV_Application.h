
#include "TChain.h"
#include "TTree.h"
#include "TString.h"
#include "TROOT.h"

#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"
#include "TMVA/TMVAGui.h"
#include "TMVA/CrossValidation.h"
#include "TMVA/Reader.h"
#include "TMVA/MethodCuts.h"

int HH_CV_Application(TString channel, TString path);

void CV_Application(TString channel, TString path);