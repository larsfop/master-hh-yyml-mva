#pragma once

#include <string>

#include "TChain.h"
#include "TTree.h"
#include "TString.h"
#include "TFile.h"
#include "TSystem.h"
#include "TObjString.h"
#include "TROOT.h"
#include "TH1F.h"
#include "TCanvas.h"

#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"
#include "TMVA/TMVAGui.h"
#include "TMVA/CrossValidation.h"

int HH_CV_Classification(TString channel, TString path, Int_t nFolds);

void CV_Classification(TString channel, TString path, Int_t nFolds);