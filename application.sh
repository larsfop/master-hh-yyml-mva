#!/bin/bash

# Trap SIGINT to kill all child processes
trap "kill 0" SIGINT

# Function to display help message
function help() {
    echo "Usage: $0 [-h] [channels]"
    echo "Options:"
    echo "  -h  Show this help message"
    echo "channels: Specify the channel (1l0tau, 0l1tau, 2l0tau, 1l1tau, 0l2tau)"
    echo "If no channels are specified, all channels will be processed."
    exit 1
}

# Parse command line arguments
while getopts h flag
do
    case "${flag}" in
        h) help;;
    esac
done

channels=$@

# Set default channels if none are provided
if [ -z $channels ]; then
    channels=( "1l0tau" "0l1tau" "2l0tau" "1l1tau" "0l2tau" )
fi

for channel in "${channels[@]}"; do
    # Check if all provided channels are valid
    if [ "$channel" != "1l0tau" ] && [ "$channel" != "0l1tau" ] && [ "$channel" != "2l0tau" ] && [ "$channel" != "1l1tau" ] && [ "$channel" != "0l2tau" ]; 
    then
        echo "Invalid channel: $channel ; Valid channels are: 1l0tau, 0l1tau, 2l0tau, 1l1tau, 0l2tau"
        exit 1
    fi

    # Run the BDTG application first
    # Must be ran first as it overwrites the output ROOT files with the input ROOT files
    (
        cd Scripts/BDTG

        ./bdtg -c $channel -a

        wait
    )

    for classification in "binary" "multiclass"; do
        (
            cd Scripts/NN

            python3 dnn_application.py -c $channel -cl $classification

            wait
        )
    done
done