{
// Create the points:
   const int n  = 7;
   double x[n]  = {139,1000,1500,2000,2500,3000,4000};
   double y[n]  = {0.19761,0.530032,0.649155,0.749579,0.838054,0.918042,1.06006};

   double x2[n] = {139,1000,1500,2000,2500,3000,4000};
   double y2[n] = {0.0529935,0.14214,0.174085,0.201016,0.224743,0.246193 ,0.284279};

   double x3[n] = {139,1000,1500,2000,2500,3000,4000};
   double y3[n] = {0.11012,0.295363,0.361746,0.417707,0.467011,0.511584,0.590727};

   double x4[n] = {139,1000,1500,2000,2500,3000,4000};
   double y4[n] = {0.0909029,0.24382,0.298618,0.344814,0.385514,0.422309,0.487641};

   double x5[n] = {139,1000,1500,2000,2500,3000,4000};
   double y5[n] = {0.0774965,0.207862,0.254578,0.293961,0.328658,0.360027,0.415724,};

   double x6[n] = {139,1000,1500,2000,2500,3000,4000};
   double y6[n] = {0.0992279,0.26615,0.325966,0.376393,0.42082,0.460985,0.5323};


// Create 6 graphs:
 TGraph *gr1 = new TGraph(n,x,y);
   gr1->SetMarkerColor(1);
   gr1->SetMarkerStyle(20);
   gr1->SetLineColor(1);
   //gr1->Draw("AP");

   TGraph *gr2 = new TGraph(n,x2,y2);
   gr2->SetMarkerColor(2);
   gr2->SetMarkerStyle(20);
   gr2->SetLineColor(2);
   //gr2->Draw("P SAME");

   TGraph *gr3 = new TGraph(n,x3,y3);
   gr3->SetMarkerColor(3);
   gr3->SetMarkerStyle(20);
   gr3->SetLineColor(3);
   //gr3->Draw("P SAME");

   TGraph *gr4 = new TGraph(n,x4,y4);
   gr4->SetMarkerColor(4);
   gr4->SetMarkerStyle(20);
   gr4->SetLineColor(4);
   //gr4->Draw("P SAME");


   TGraph *gr5 = new TGraph(n,x5,y5);
   gr5->SetMarkerColor(5);
   gr5->SetMarkerStyle(20);
   gr5->SetLineColor(5);
   //gr5->Draw("P SAME");

   TGraph *gr6 = new TGraph(n,x6,y6);
   gr6->SetMarkerColor(6);
   gr6->SetMarkerStyle(20);
   gr6->SetLineColor(6);
   //gr6->Draw("P SAME");

   // Create a TMultiGraph and draw it:
   TMultiGraph *mg = new TMultiGraph();
   mg->Add(gr1);
   mg->Add(gr2);
   mg->Add(gr3);
   mg->Add(gr4);
   mg->Add(gr5);
   mg->Add(gr6);
   //mg->Draw("ALP");
   mg->Draw("ALP");
   mg->SetTitle("integrated luminosity (13TeV to 14TeV); projection increase luminosity fb-1; Significance //sigma"); 
   //mg->GetXaxis()->CenterTitle();
   

    auto legend = new TLegend(0.1,0.7,0.5,0.9);
   legend->SetHeader("Decay channels","C"); // option "C" allows to center the header
   //legend->AddEntry("gr1","All channels","l");
   //TLegendEntry *entry=leg->AddEntry("gr1","All channels","l");
   //entry->SetLineColor(3);
   
   legend->AddEntry("gr1","yy + Multilepton","l");
   legend->AddEntry("gr2","yy + 0l1tau subchannel","l");
   legend->AddEntry("gr3","yy + 1l1tau subchannel","l");
   legend->AddEntry("gr4","yy + 0l2tau subchannel","l");
   legend->AddEntry("gr5","yy + 2l1tau subchannel","l");
   legend->AddEntry("gr6","yy + 1l0tau subchannel","l");
   legend->Draw();
 } 