#!/bin/bash

# Trap SIGINT to kill all child processes
trap "kill 0" SIGINT

# Parse command line arguments
while getopts gw flag
do
    case "${flag}" in
        w) opts="$opts -bw ${OPTARG}";; # balanced weights, if true uses class weights by default
        g) opts="$opts -gw ${OPTARG}";; # generator weights, if true, not balanced by default
    esac
done

# Remove the parsed arguments
shift $((OPTIND-1))

channel=$1

if [ -z $channel ]; then
    channel=1l0tau
fi


(
    cd Scripts/NN

    python3 dnn_classification.py -e 100 -es $opts -c $channel

    python3 dnn_classification.py -e 100 -es $opts -cl multiclass -c $channel

    wait

    cd ..

    python3 produce_plots.py -c $channel

    wait
)

shopt -s globstar; tar cvf plots.tar **/*.pdf