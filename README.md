# Di-Higgs yyML machine learning analysis

This project uses different machine learning methods such as boosted decision trees (BDT) and neural networks (NN) to analyse generated Monte Carlo data based on the di-Higgs decay into yy + multilepton analysis with the ATLAS detector. The idea behind this analysis is to investigate the usage of neural networks, with either binary or multiclass classification compared to the usage of boosted decision trees.


### Scripts
Small description of the scripts used in this project, how to use can be found in the Workflow section.

- ##### Boosted decision trees
C++ files for training and application of the BDT models using [ROOT](https://root.cern/) and the [TMVA ROOT](https://root.cern.ch/doc/master/group__TMVA.html) framework.
The scripts can be ran as ROOT macros directly or see the workflow section.
Can be compiled, see the Setup section.

- ##### Neural networks
Python scipts for training and application of the NN methods using [PyTorch](https://pytorch.org/), hyperparameter optimisation using RayTune.
The scripts can be ran using shellscripts as descibed in the workflow section.
The NN options can be set in Configs/NN.

- ##### Plots
The produce_plots.py script plots the NN grid-searched hyperparameters, the model evaluation for the BDTs and NNs and the comparison between the applied models.

- ##### Statistical analysis
The statistical analysis is done using [TRExFitter](https://trexfitter-docs.web.cern.ch/trexfitter-docs/latest/) and creates the prefit plots, and find the significance and upper limits on the parameter of interest, which in this case is the di-Higgs production cross section over its Standard Model prediction.
The setup is found under Configs/TRExFitter.


### Setup
- Place ROOT files in Input_Files directory
- Create hd5f files (the root files must have the same internal structure)
```
python3 Scripts/utils/ReadData.py
```
- Required packages
    - ROOT: 26.26/06 (other version might also work)
    - TRExFitter: 0.4.0 (latest will probably work)
    - Python
    ```
    pip install requirements.txt
    ```
- The C++ ROOT scripts can be compiled using CMake
The method of compling may not work for all ROOT versions, however a complied executable is included or the scripts can be ran as ROOT macros.
```
./compile_bdtg.sh
```


### Workflow
How to use the scripts in the project.

#### Hyperparameter optimisation
Does a grid- or randomised-search for the NNs.
The BDT hyperparameters were not optimised for this analysis.
```
./param_tuner.sh <flags> <sub_channels>
```

#### Classification
Use this script to train the models for the different sub-channels, if no sub-channels are set it will train the model for each iteratively.
The BDTs are trained using cross-validation which are then combined later for the application.
The NNs uses cross-validation to evaluate the models and without cross-validation for the application.
Use -h for more information on the flags.
```
./classification.sh <flags> <sub_channels>
```
If you have problems training the BDTs the c++ file can be ran as a ROOT macro
```
root -q 'HH_CV_Classification.C("<sub_channel>")'
```

#### Application
Prepares the data and applies the trained models for the statistical analysis.
Applies the models for the BDTs and NNs respectively.
```
./application.sh <sub_channels>
```
The c++ files can also be ran as ROOT macro
```
root -q 'HH_CV_Application.C("<sub_channel>")'
```

#### Plots
After the previous scripts have been ran the plots can be created.
Generates plots for the grid-search, model evaluation and comparing the models after application.
```
./plots.sh <flags>
```

#### TRExFitter
The Statistical analysis done with the scripts.
The TRExFitter options can be set using the -o flag or ran without this flag will use the pre-defined options used in this analysis.
The sub_channels options now includes '2l' for combined 2L sub-channels or 'yyml' for the full combination multifit.
```
./tf.sh <flags> <sub_channels>
```