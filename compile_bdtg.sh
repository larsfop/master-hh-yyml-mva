#!/bin/bash

# Compile TMVA BDTG classification and application root c++ files

echo "Compiling c++ ROOT files"
(
    cd Scripts/BDTG

    mkdir build

    cd build

    cmake ../ 

    cmake --build .

    mv bdtg ../
    mv libbdtg_rdict.pcm ../
)