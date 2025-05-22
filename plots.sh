#!/bin/bash

# # Trap SIGINT to kill all child processes
trap "kill 0" SIGINT

# Function to display help message
function help() {
    echo "Usage: $0 [-h] [channels]"
    echo "Options:"
    echo "  -h  Show this help message"
    echo "channels: Specify the channels (1l0tau, 0l1tau, 2l0tau, 1l1tau, 0l2tau)"
    echo "If no channels are specified, all channels will be processed."
    exit 1
}

suffix=pdf

# Parse command line arguments
while getopts hs: flag
do
    case "${flag}" in
        h) help;;
        s) suffix=${OPTARG};;
    esac
done

# Remove the parsed arguments
shift $((OPTIND-1))

channels=$@

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

for channel in "${channels[@]}"; do
    (
        cd Scripts

        python3 produce_plots.py -c $channel -s $suffix

        wait
    )
done