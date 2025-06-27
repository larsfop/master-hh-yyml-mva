#!/bin/bash

# # Trap SIGINT to kill all child processes
trap "kill 0" SIGINT

# Function to display help message
function help() {
    echo "Usage: $0 [-h] [-s suffix] [-o options] [channels]"
    echo
    echo "Options:"
    echo "  -h          Display this help message and exit."
    echo "  -s suffix   Specify the file format for output plots (default: pdf)."
    echo "  -o options  Specify additional options for plot generation:"
    echo "                h - Produce hyperparameter grid search plots."
    echo "                e - Produce MVA evaluation plots."
    echo "                c - Produce MVA comparison plots."
    echo
    echo "Channels:"
    echo "  Specify one or more channels to process. Valid channels are:"
    echo "    1l0tau, 0l1tau, 2l0tau, 1l1tau, 0l2tau."
    echo "  If no channels are specified, all channels will be processed."
    exit 1
}

suffix=pdf

# Parse command line arguments
while getopts hs:o: flag
do
    case "${flag}" in
        h) help;;
        s) suffix=${OPTARG};;
        o) for i in $(seq 0 $((${#OPTARG}-1))); do
            case ${OPTARG:$i:1} in
                h) opts="$opts -ho";;
                e) opts="$opts -me";;
                c) opts="$opts -mc";;
            esac
        done;;
    esac
done

# Remove the parsed arguments
shift $((OPTIND-1))

# Read remaining arguments as channels
channels=$@

# If no channels are provided, use default channels
if [ -z "$channels" ]; then
    channels=( "1l0tau" "0l1tau" "2l0tau" "1l1tau" "0l2tau" )
fi

# Check if all provided channels are valid
for channel in "${channels[@]}"; do
    if [ "$channel" != "1l0tau" ] && [ "$channel" != "0l1tau" ] && [ "$channel" != "2l0tau" ] && [ "$channel" != "1l1tau" ] && [ "$channel" != "0l2tau" ]; 
    then
        echo "Invalid channel: $channel ; Valid channels are: 1l0tau, 0l1tau, 2l0tau, 1l1tau, 0l2tau"
        exit 1
    fi
done

# Run entire script if options are not provided
if [ -z "$opts" ]; then
    opts="-ho -me -mc"
fi

for channel in "${channels[@]}"; do
    (
        cd Scripts

        python3 produce_plots.py -c $channel -s $suffix $opts

        wait
    )
done