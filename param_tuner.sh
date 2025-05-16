#!/bin/bash

trap "kill 0" SIGINT

# Function to display help message
function help() {
    echo "Usage: $0 [-h] [channel]"
    echo "Options:"
    echo "  -e <epochs>        Number of epochs (default: 100)"
    echo "  -t <trials>       Number of trials (default: 1)"
    echo "  -k <kfolds>       Number of folds for cross-validation (default: 0)"
    echo "  -n <n_jobs>       Number of jobs (default: 10)"
    echo "  -o <options>      Options for classification (e.g., 'm' for multiclass, 'b' for binary, 'v' for verbose)"
    echo "  -g <grid_search>  Grid search (default: None)"
    echo "  -r <rand_search>  Random search (default: None)"
    echo "  -s                Scheduler (default: None)"
    echo "  -h                Show this help message"
    exit 1
}

channel=1l0tau
epochs=100
trials=1
folds=0
n_jobs=10
mc=0
scheduler=0

while getopts e:t:k:n:o:g:r:sh flag
do
    case "${flag}" in
        e) epochs=${OPTARG};; # Number of epochs
        t) trials=${OPTARG};; # Number of trials
        k) folds=${OPTARG};; # Number of folds for cross-validation
        n) n_jobs=${OPTARG};; # Number of jobs in parallel
        s) opts="$opts -s";; # Scheduler
        # Boolean flag options
        o) for i in $(seq 0 $((${#OPTARG}-1))); do
            case ${OPTARG:$i:1} in
                m) mc=1;; # Multiclass classification
                b) mc=0;; # Binary classification
                v) opts="$opts -v";; # Verbose
            esac
        done;;
        g) grid_search=${OPTARG};; # Grid search
        r) rand_search=${OPTARG};; # Random search
        h) help;; # Show help message
    esac
done

shift $((OPTIND-1))

if [ $mc -eq 1 ]; then
    classification="multiclass"
    echo "Multiclass classification"
elif [ $mc -eq 0 ]; then
    classification="binary"
    echo "Binary classification"
fi

if [ -n "$grid_search" ] && [ -n "$rand_search" ]; then
    echo "Cannot specify both grid search and random search"
    exit 1
elif [ -n "$grid_search" ]; then
    echo "Grid search: $grid_search"
    grid_search="-g $grid_search"
elif [ -n "$rand_search" ]; then
    echo "Random search: $rand_search"
    rand_search="-r $rand_search"
else
    echo "Need to specify grid search or random search"
    exit 1
fi

# Check if at least one channel is provided
if [ "$#" -lt 1 ]; then
    channels=(1l0tau 0l1tau 2l0tau 1l1tau 0l2tau)
elif [ "$1" == '2l' ]; then
    channels=(2l0tau 1l1tau 0l2tau)
elif [ "$1" == '1l' ]; then
    channels=(1l0tau 0l1tau)
else
    channels=("$@")
fi

# Check if all provided channels are valid
for channel in "${channels[@]}";
do
    if [ "$channel" != "1l0tau" ] && [ "$channel" != "0l1tau" ] && [ "$channel" != "2l0tau" ] && [ "$channel" != "1l1tau" ] && [ "$channel" != "0l2tau" ]; then
        echo "Invalid channel: $channel ; Valid channels are: 1l0tau, 0l1tau, 2l0tau, 1l1tau, 0l2tau"
        exit 1
    fi
done

echo "channels: ${channels[@]}"

for channel in "${channels[@]}"; do
    python3 Scripts/NN/HyperOptim.py $opts -c $channel -e $epochs -t $trials $grid_search $rand_search -cl $classification -n $n_jobs -k $folds &
done

wait