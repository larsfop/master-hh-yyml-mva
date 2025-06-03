#include "HH_CV_Application.h"

void CV_Application(TString channel, TString path)
{
    Long64_t eventID;
    Int_t MCTypes, MC16Types;
    Int_t N_j_central, N_j_removal, N_jrec, N_Cluster, N_j;
    Double_t BDT;
    Float_t f_eventID;
    Float_t f_MCTypes;
    Float_t f_MC16Types;
    Int_t McNumber;

    Double_t HT, y1_phi0, lep_phi_1, lep_eta_1, lep_phi0_1, m_yyll, MT_W1, Dr_yyW, Dr_yyll, Dy_Hlrec, Dy_bigyy, pt_jrec, mbig, minDphi_metjl, met, pt_H, lep_pt_1, Dphi_metll, Dphi_metyy, phi_H, met_phi, y1_eta, y2_eta, y2_phi, eta_W, rbig;                                                                                 // 2l0tau
    Double_t m_H, Dr_yyl1, M_XminusS, met_phi0, Dphi_yyW, Jet_pt1, Jet_pt2, phi_W, pt_W, y2_pt, sumet, y1_pt, y2_phi0, Dy_bigyy2, ptbig2, mbig2, rfr0, rfr1, msum, etsumtot, ptsum, ptbig, msum2, ptsum2, etsumleft, pt_lv, met_sig, lep_pt_2, pt_ll, MT, pt_yyll, m_ll, Dphi_ll, Dphi_yyll, Dr_lv, pt_W2, Dr_ll, Dr_jj, Dr_lj; // 2l0tau

    Float_t f_HT, f_y1_phi0, f_lep_phi_1, f_lep_eta_1, f_lep_phi0_1, f_m_yyll, f_MT_W1, f_Dr_yyW, f_Dr_yyll, f_Dy_Hlrec, f_Dy_bigyy, f_pt_jrec, f_mbig, f_minDphi_metjl, f_met, f_pt_H, f_lep_pt_1, f_Dphi_metll, f_Dphi_metyy, f_phi_H, f_met_phi, f_N_j_central, f_y1_eta, f_y2_eta, f_y2_phi, f_eta_W, f_rbig;
    Float_t f_m_H, f_Dr_yyl1, f_M_XminusS, f_met_phi0, f_Dphi_yyW, f_Jet_pt1, f_Jet_pt2, f_phi_W, f_pt_W, f_y2_pt, f_sumet, f_y1_pt, f_y2_phi0, f_mbig2, f_ptbig2, f_Dy_bigyy2, f_rfr0, f_rfr1, f_msum, f_ptsum, f_etsumtot, f_ptbig, f_msum2, f_ptsum2, f_etsumleft, f_pt_lv, f_N_Cluster, f_met_sig, f_N_j, f_N_jrec, f_lep_pt_2, f_pt_ll, f_pt_yyll, f_MT, f_m_ll, f_Dphi_ll, f_Dphi_yyll, f_Dr_lv, f_pt_W2, f_Dr_ll, f_Dr_jj, f_Dr_lj;
    Float_t f_BDT, f_McNumber;

    TMVA::Reader *reader = new TMVA::Reader("Color:Silent");
    reader->AddVariable("pt_H", &f_pt_H);
    reader->AddVariable("lep_phi0_1", &f_lep_phi0_1);
    reader->AddVariable("met", &f_met);
    reader->AddVariable("lep_pt_1", &f_lep_pt_1);
    reader->AddVariable("N_j_central", &f_N_j_central);
    reader->AddVariable("Dphi_metyy", &f_Dphi_metyy);

    if (channel == "1l0tau")
    {
        reader->AddVariable("y1_phi0", &f_y1_phi0);
        reader->AddVariable("minDphi_metjl", &f_minDphi_metjl);
        reader->AddVariable("Dr_lv", &f_Dr_lv);
        reader->AddVariable("Dr_yyW", &f_Dr_yyW);
        reader->AddVariable("eta_W", &f_eta_W);
    }
    else if (channel == "0l1tau")
    {
        reader->AddVariable("y1_phi0", &f_y1_phi0);
        reader->AddVariable("y1_eta", &f_y1_eta);
    }
    else
    {
        reader->AddVariable("phi_H", &f_phi_H);
        reader->AddVariable("lep_pt_2", &f_lep_pt_2);
        reader->AddVariable("met_phi", &f_met_phi);
        reader->AddVariable("minDphi_metjl", &f_minDphi_metjl);
        reader->AddVariable("Dphi_metll", &f_Dphi_metll);
        reader->AddVariable("Dr_lv", &f_Dr_lv);
        reader->AddVariable("m_ll", &f_m_ll);
        reader->AddVariable("Dr_ll", &f_Dr_ll);
        reader->AddVariable("Dphi_ll", &f_Dphi_ll);
        reader->AddVariable("Dphi_yyll", &f_Dphi_yyll);
        reader->AddVariable("Jet_pt1", &f_Jet_pt1);
    }

    // reader->AddVariable("HT",              &f_HT);
    // reader->AddVariable("MT_W1",           &f_MT_W1);
    // reader->AddVariable("mbig",            &f_mbig);
    // reader->AddVariable("ptbig",           &f_ptbig);
    // reader->AddVariable("Dy_bigyy",        &f_Dy_bigyy);
    // reader->AddVariable("Dy_bigyy2",       &f_Dy_bigyy2);
    // reader->AddVariable("Dr_lj",           &f_Dr_lj);

    reader->AddSpectator("eventID", &f_eventID);
    reader->AddSpectator("MCTypes", &f_MCTypes);
    reader->AddSpectator("McNumber", &f_McNumber);
    // reader->AddSpectator("MC16Types", &f_MC16Types);

    reader->BookMVA("BDT method", Form("%sOutput/%s/BDTG/weights/TMVACrossValidation_BDTG.weights.xml", path.Data(), channel.Data()));

    TString files[6]{
        (TString) path + "Input_Files/signal_VBF_" + channel + ".root",
        (TString) path + "Input_Files/signal_ggF_" + channel + ".root",
        (TString) path + "Input_Files/Vyy_" + channel + ".root",
        (TString) path + "Input_Files/SH_" + channel + ".root",
        (TString) path + "Input_Files/Sherpa_" + channel + ".root",
        (TString) path + "Input_Files/data_" + channel + ".root",
    };
    TString newpath = (TString) path + "Output/Files/";
    TString newfiles[6]{
        (TString) newpath + "signal_VBF_" + channel + ".root",
        (TString) newpath + "signal_ggF_" + channel + ".root",
        (TString) newpath + "Vyy_" + channel + ".root",
        (TString) newpath + "SH_" + channel + ".root",
        (TString) newpath + "Sherpa_" + channel + ".root",
        (TString) newpath + "data_" + channel + ".root",
    };

    for (int i = 0; i < 6; i++)
    {
        TFile file(files[i], "READ");
        TTree *oldtree = file.Get<TTree>("output");
        std::cout << " Now working on:" << file.GetName() << std::endl;
        std::cout << "--- Processing: " << oldtree->GetEntries() << " events" << std::endl;

        // oldtree->SetBranchAddress("phi_H",            &phi_H);
        oldtree->SetBranchAddress("pt_H", &pt_H);
        oldtree->SetBranchAddress("lep_phi_1", &lep_phi_1);
        oldtree->SetBranchAddress("lep_eta_1", &lep_eta_1);
        oldtree->SetBranchAddress("lep_phi0_1", &lep_phi0_1);
        oldtree->SetBranchAddress("lep_pt_1", &lep_pt_1);
        // oldtree->SetBranchAddress("met_phi",         &met_phi);
        oldtree->SetBranchAddress("met", &met);
        oldtree->SetBranchAddress("N_j_central", &N_j_central);
        // oldtree->SetBranchAddress("HT",              &HT);
        oldtree->SetBranchAddress("y1_eta", &y1_eta);
        oldtree->SetBranchAddress("y1_phi0", &y1_phi0);
        oldtree->SetBranchAddress("y2_eta", &y2_eta);
        oldtree->SetBranchAddress("y2_phi", &y2_phi);
        oldtree->SetBranchAddress("eta_W", &eta_W);
        // oldtree->SetBranchAddress("MT_W1",           &MT_W1);
        oldtree->SetBranchAddress("Dr_yyW", &Dr_yyW);
        oldtree->SetBranchAddress("Dr_lv", &Dr_lv);
        oldtree->SetBranchAddress("Dphi_metll", &Dphi_metll);
        oldtree->SetBranchAddress("Dphi_metyy", &Dphi_metyy);
        oldtree->SetBranchAddress("minDphi_metjl", &minDphi_metjl);
        // oldtree->SetBranchAddress("mbig",            &mbig);
        // oldtree->SetBranchAddress("ptbig",           &ptbig);
        // oldtree->SetBranchAddress("Dy_bigyy",        &Dy_bigyy);
        // oldtree->SetBranchAddress("Dy_bigyy2",       &Dy_bigyy2);
        // oldtree->SetBranchAddress("Dr_lj",           &Dr_lj);

        oldtree->SetBranchAddress("eventID", &eventID);
        oldtree->SetBranchAddress("MCTypes", &MCTypes);
        oldtree->SetBranchAddress("McNumber", &McNumber);
        oldtree->SetBranchAddress("MC16Types", &MC16Types);
        oldtree->SetBranchAddress("BDT_" + channel, &BDT);

        Double_t sample;

        TFile newfile(newfiles[i], "RECREATE");
        TTree *newtree = oldtree->CloneTree(0);

        auto b_1l0tau = newtree->Branch((TString) "BDTG", &sample);

        for (Long64_t j = 0; j < oldtree->GetEntries(); j++)
        {
            if (j % 10000 == 0)
                std::cout << "--- ... Processing event: " << j << std::endl;

            oldtree->GetEntry(j);

            sample = -10;

            f_eventID = (Float_t)eventID;
            f_MCTypes = (Float_t)MCTypes;
            f_McNumber = (Float_t)McNumber;
            f_MC16Types = (Float_t)MC16Types;

            f_HT = (Float_t)HT;
            f_y1_phi0 = (Float_t)y1_phi0;
            f_lep_phi_1 = (Float_t)lep_phi_1;
            f_lep_eta_1 = (Float_t)lep_eta_1;
            f_lep_phi0_1 = (Float_t)lep_phi0_1;
            f_m_yyll = (Float_t)m_yyll;
            f_MT_W1 = (Float_t)MT_W1;
            f_Dr_yyW = (Float_t)Dr_yyW;
            f_Dr_yyll = (Float_t)Dr_yyll;
            f_Dy_Hlrec = (Float_t)Dy_Hlrec;
            f_Dy_bigyy = (Float_t)Dy_bigyy;
            f_pt_jrec = (Float_t)pt_jrec;
            f_mbig = (Float_t)mbig;
            f_minDphi_metjl = (Float_t)minDphi_metjl;
            f_met = (Float_t)met;
            f_pt_H = (Float_t)pt_H;
            f_lep_pt_1 = (Float_t)lep_pt_1;
            f_Dphi_metll = (Float_t)Dphi_metll;
            f_Dphi_metyy = (Float_t)Dphi_metyy;
            f_phi_H = (Float_t)phi_H;
            f_met_phi = (Float_t)met_phi;
            f_N_j_central = (Float_t)N_j_central;
            f_y1_eta = (Float_t)y1_eta;
            f_y2_eta = (Float_t)y2_eta;
            f_y2_phi = (Float_t)y2_phi;
            f_eta_W = (Float_t)eta_W;
            f_rbig = (Float_t)rbig;

            // 0l0tau
            f_m_H = (Float_t)m_H;
            f_Dr_yyl1 = (Float_t)Dr_yyl1;
            f_M_XminusS = (Float_t)M_XminusS;
            f_met_phi0 = (Float_t)met_phi0;
            f_y1_eta = (Float_t)y1_eta;
            f_Dphi_yyW = (Float_t)Dphi_yyW;
            f_Jet_pt1 = (Float_t)Jet_pt1;
            f_Jet_pt2 = (Float_t)Jet_pt2;
            f_phi_W = (Float_t)phi_W;
            f_pt_W = (Float_t)pt_W;
            f_y2_pt = (Float_t)y2_pt;
            f_y1_pt = (Float_t)y1_pt;
            f_y2_phi0 = (Float_t)y2_phi0;
            f_sumet = (Float_t)sumet;
            f_mbig2 = (Float_t)mbig2;
            f_ptbig2 = (Float_t)ptbig2;
            f_ptbig = (Float_t)ptbig;
            f_Dy_bigyy2 = (Float_t)Dy_bigyy2;
            f_rfr0 = (Float_t)rfr0;
            f_rfr1 = (Float_t)rfr1;
            f_msum = (Float_t)msum;
            f_etsumtot = (Float_t)etsumtot;
            f_ptsum = (Float_t)ptsum;
            f_N_jrec = (Float_t)N_jrec;
            f_N_j = (Float_t)N_j;
            f_pt_lv = (Float_t)pt_lv;
            f_msum2 = (Float_t)msum2;
            f_ptsum2 = (Float_t)ptsum2;
            f_etsumleft = (Float_t)etsumleft;
            f_met_sig = (Float_t)met_sig;
            f_lep_pt_2 = (Float_t)lep_pt_2;
            f_pt_ll = (Float_t)pt_ll;
            f_MT = (Float_t)MT;
            f_pt_yyll = (Float_t)pt_yyll;
            f_Dr_jj = (Float_t)Dr_jj;
            f_Dr_lj = (Float_t)Dr_lj;

            f_BDT = (Float_t)BDT;

            // TMVA BDTG output is between -1 and 1, we shift it to 0 and 1 for consistency with the NN
            sample = (reader->EvaluateMVA("BDT method") + 1) / 2;
            newtree->Fill();
        }
        std::cout << "--- End of event loop: " << std::endl;
        newtree->AutoSave();
        newtree->Write("", TObject::kOverwrite);
        file.Close();
        newfile.Close();
    }
}

int HH_CV_Application(TString channel = "1l0tau", TString path = "../../")
{
    TMVA::Tools::Instance();

    CV_Application(channel, path);

    return 0;
}