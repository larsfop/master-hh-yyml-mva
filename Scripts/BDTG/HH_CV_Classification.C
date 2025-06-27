#include "HH_CV_Classification.h"

void CV_Classification(TString channel, TString path, Int_t nFolds)
{
    // ------------------------------------------------------------
    // Load events
    // Events are loaded from root files and 
    // added into TCHains for the background and signal events
    TString signal_VBF = path + "Input_Files/signal_VBF_" + channel + ".root";
    TString signal_ggF = path + "Input_Files/signal_ggF_" + channel + ".root";
    
    TString Vyy = path + "Input_Files/Vyy_" + channel + ".root";
    TString SH = path + "Input_Files/SH_" + channel + ".root";
    TString Sherpa = path + "Input_Files/Sherpa_" + channel + ".root";

    TChain Signal("output", "output");
    TChain Bkg("output", "output");
    
    Signal.Add(signal_VBF);
    Signal.Add(signal_ggF);

    Bkg.Add(Vyy);
    Bkg.Add(SH);
    Bkg.Add(Sherpa);

    TString outfilename = (TString) path + "Output/" + channel + "/BDTG/TMVA_" + channel + ".root";
    TFile *outfile = TFile::Open(outfilename, "RECREATE");
    TMVA::Factory *factory = new TMVA::Factory("TMVAClassification", outfile, "V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;U;G,D:AnalysisType=Auto");

    TString dataname = path + "Output/" + channel + "/BDTG";
    TMVA::DataLoader *dataloader = new TMVA::DataLoader(dataname);

    // ------------------------------------------------------------
    // Select which variables that will be used for the training
    dataloader->AddVariable("pt_H",           "pt_H",           "",  'D');
    dataloader->AddVariable("lep_phi0_1",     "lep_phi0",       "",  'D');
    dataloader->AddVariable("met",            "met",            "",  'D');
    dataloader->AddVariable("lep_pt_1",       "lep_p1_1",       "",  'D');
    dataloader->AddVariable("N_j_central",    "N_j_central",    "",  'D');
    dataloader->AddVariable("Dphi_metyy",     "Dphi_metyy",     "",  'D');

    if (channel == "1l0tau")
    {
        dataloader->AddVariable("y1_phi0",        "y1_phi0",        "",  'D');
        dataloader->AddVariable("minDphi_metjl",  "minDphi_metjl",  "",  'D');
        dataloader->AddVariable("Dr_lv",          "Dr_lv",          "",  'D');
        dataloader->AddVariable("Dr_yyW",         "Dr_yyW",         "",  'D');
        dataloader->AddVariable("eta_W",          "eta_W",          "",  'D');
    }
    else if (channel == "0l1tau")
    {
        dataloader->AddVariable("y1_phi0",        "y1_phi0",        "",  'D');
        dataloader->AddVariable("y1_eta",         "y1_eta",         "",  'D');
    }
    else
    {
        dataloader->AddVariable("phi_H",          "phi_H",          "",  'D');
        dataloader->AddVariable("lep_pt_2",       "lep_p2_2",       "",  'D');
        dataloader->AddVariable("met_phi",        "met_phi",        "",  'D');
        dataloader->AddVariable("minDphi_metjl",  "minDphi_metjl",  "",  'D');
        dataloader->AddVariable("Dphi_metll",     "Dphi_metll",     "",  'D');
        dataloader->AddVariable("Dr_lv",          "Dr_lv",          "",  'D');
        dataloader->AddVariable("m_ll",           "m_ll",           "",  'D');
        dataloader->AddVariable("Dr_ll",          "Dr_ll",          "",  'D');
        dataloader->AddVariable("Dphi_ll",        "Dphi_ll",        "",  'D');
        dataloader->AddVariable("Dphi_yyll",      "Dphi_yyll",      "",  'D');
        dataloader->AddVariable("Jet_pt1",        "Jet_pt1",        "",  'D');
    }
    
    // Spectators are variables that the model will keep,
    // but not use for the training
    dataloader->AddSpectator("eventID");
    dataloader->AddSpectator("MCTypes");
    dataloader->AddSpectator("McNumber");

    dataloader->AddSignalTree((TTree *)& Signal);
    dataloader->AddBackgroundTree((TTree *)& Bkg);

    dataloader->SetWeightExpression("weight");

    // ------------------------------------------------------------
    // Start the training and evaluation of the data
    dataloader->PrepareTrainingAndTestTree(
        "",
        "nTest_Signal=0:"
        "nTest_Background=0:"
        "SplitMode=Random:"
        "NormMode=EqualNumEvents:"
        "V"
    );

    TString analysisType = "Classification";
    TString splitType = "Deterministic";
    TString splitExpr = "int([eventID])\%int([NumFolds])";
    TString opt = Form(
        "V"
        ":!Silent"
        ":ModelPersistence"
        ":AnalysisType=%s"
        ":SplitType=%s"
        ":NumFolds=%i"
        ":SplitExpr=%s"
        ":OutputEnsembling=Avg"
        ":FoldFileOutput=True",
        analysisType.Data(), splitType.Data(), nFolds,
        splitExpr.Data()
    );

    Int_t nTrees = 1000;
    TString method_opt = Form(
        "!H"
        ":!V"
        ":NTrees=%i"
        ":MinNodeSize=2.5%"
        ":BoostType=Grad"
        ":UseBaggedBoost"
        ":BaggedSampleFraction=0.5"
        ":NegWeightTreatment=Pray"
        ":Shrinkage=0.10"
        ":nCuts=20"
        ":MaxDepth=2",
        nTrees
    );

    TMVA::CrossValidation cv{"TMVACrossValidation", dataloader, outfile, opt};

    cv.BookMethod(TMVA::Types::kBDT, "BDTG", method_opt);

    cv.Evaluate();

    // ------------------------------------------------------------
    // Print out results from the training and evaluation
    for (auto results : cv.GetResults()) {
        results.GetROCValues();
        results.GetSigValues();
        results.GetEff01Values();
        results.GetEff10Values();
        results.GetEff30Values();
        results.Print();
    }

    outfile->Close();
}

int HH_CV_Classification(TString channel = "1l0tau", TString path = "../../", Int_t nFolds = 4)
{
    TMVA::Tools::Instance();

    CV_Classification(channel, path, nFolds);

    // ------------------------------------------------------------
    // Rename the fold outputs
    TString outpath = Form("%s/Output/%s/BDTG/", path.Data(), channel.Data());
    for (UInt_t ifold = 1; ifold < 5; ifold++) {
        TString tfold = Form(
            "mv %sBDTG_fold%i.root %s/BDTG_%s_fold_%i.root",
            outpath.Data(), ifold, outpath.Data(), channel.Data(), ifold
        );
        gSystem->Exec(tfold.Data());
    }

    return 0;
}