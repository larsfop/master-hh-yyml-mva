#!/bin/bash

# Trap SIGINT to kill all child processes
trap "kill 0" SIGINT

function help() {
    echo "Usage: $0 [-e epochs] [-w class_weight] [-k kfolds] [-j n_jobs] [-s] [-p] [-t] [-b] [-m] [channels]"
    echo "Options:"
    echo "  -e  Number of epochs (default: 100)"
    echo "  -w  Class weight (default: 0)"
    echo "  -k  Number of folds (default: 0)"
    echo "  -j  Number of cv jobs (default: nfolds)"
    echo "  -s  Early stopping (default: true)"
    echo "  -p  Progress bar (default: true)"
    echo "  -t  Use BDT"
    echo "  -b  Use NN binary classification (default)"
    echo "  -m  Use NN multiclass classification"
    echo "channels: Specify the channel (1l0tau, 0l1tau, 2l0tau, 1l1tau, 0l2tau)"
    echo "If no channels are specified, all channels will be processed."
    exit 1
}

# Parse command line arguments
epochs=100
kfolds=0
classification=binary
mva=dnn

# Parse command line arguments
while getopts e:k:j:hsptbmwg flag
do
    case "${flag}" in
        e) epochs=${OPTARG};; # Number of epochs
        k) kfolds=${OPTARG};; # Number of folds
        w) opts="$opts -bw ${OPTARG}";; # balanced weights, if true uses class weights by default
        g) opts="$opts -gw ${OPTARG}";; # generator weights, if true, not balanced by default
        j) n_jobs=${OPTARG};; # Number of jobs
        s) opts="$otps -es";; # Early stopping
        p) opts="$opts -pb";; # Progress bar
        t) mva="bdt";; # Use BDTG
        b) classification="binary";; # Use binary classification
        m) classification="multiclass";; # Use multiclass classification
        h) help;; # Show help message
    esac
done

# Remove the parsed arguments
shift $((OPTIND-1))

channels=$@

if [ -z $channels ]; then
    channels=( "1l0tau" "0l1tau" "2l0tau" "1l1tau" "0l2tau" )
fi

if [ -z $n_jobs ]; then
    n_jobs=$kfolds
fi

for channel in "${channels[@]}"; do
    # Check if all provided channels are valid
    if [ "$channel" != "1l0tau" ] && [ "$channel" != "0l1tau" ] && [ "$channel" != "2l0tau" ] && [ "$channel" != "1l1tau" ] && [ "$channel" != "0l2tau" ]; 
    then
        echo "Invalid channel: $channel ; Valid channels are: 1l0tau, 0l1tau, 2l0tau, 1l1tau, 0l2tau"
        exit 1
    fi

    if [ $mva == "bdt" ]; 
    then
        mkdir Output/$channel/BDTG
        mkdir Output/Files
    (
        cd Scripts/BDTG

        ./bdtg -c $channel -n $n_jobs -k $kfolds -c
    )
    elif [ $mva == "dnn" ]; 
    then
    (
        cd Scripts/NN

        python3 dnn_classification.py -c $channel -e $epochs -cl $classification -k $kfolds -j $n_jobs $opts
    )
    else
        echo "Invalid mva option: $mva ; Valid options are: dnn, bdt"
        exit 1
    fi

    wait
done