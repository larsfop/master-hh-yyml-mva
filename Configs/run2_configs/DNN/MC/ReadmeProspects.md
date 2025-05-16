aProspects study readme file

step one: Clone the entire directory
```
git clone ssh:https://gitlab.cern.ch/qsha/hh-yyml

```

step two: choose what luminosity scaling you want in the config files by commenting out the rest of the alternatives for the seperate channels



To create signifcance plot for the seperate channels run Trex fitter wns on all the different channels

```
Trex fitter n source/file

where w creates the RooStats xmls and workspace
n read input ntuples (valid only if the proper option is specified in the config file)
and s calculates significance

To get the signifcance for the combined channels you need to run wns on all sub channels and then wms on the yyml.use.config

where m is multi-fit

Two different scalings are being currently used. On in regards to the luminosity increase which is done in the job section

and the other which comes from the cross section changes due to increase in center of mass energy from 13 TeV to 14 TeV. This is done in the sample section. 

one plot script is currently added to show the signifcance plot