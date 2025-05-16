#!/bin/bash

function help() {
    echo "Usage: $0 [-b] [-t] [-h] [channel]"
    echo "Options:"
    echo "  -b  Use DNN MVA"
    echo "  -t  Use BDTG MVA (default)"
    echo "  -h  Show this help message"
    echo "channel: Specify the channel (1l0tau, 0l1tau, 2l0tau, 1l1tau, 0l2tau)"
    exit 1
}

path=Configs/run2_configs/
mva=BDTG

while getopts bmtho: flag
do
    case "${flag}" in
        b) mva=DNN/BC;;
        m) mva=DNN/MC;;
        t) mva=BDTG;;
        o) opts=$OPTARG;;
        h) help;;
    esac
done

shift $((OPTIND-1))

channels=$@

if [ -z "$channels" ]; then
    channels=( "1l0tau" "0l1tau" "2l0tau" "1l1tau" "0l2tau" "2l" "yyml" )
fi

# Check if all provided channels are valid
for channel in "${channels[@]}"; do
    if [[ "$channel" != "1l0tau" && "$channel" != "0l1tau" && "$channel" != "2l0tau" && "$channel" != "1l1tau" && "$channel" != "0l2tau" && "$channel" != "2l" && "$channel" != "yyml" ]]; then
        echo "Invalid channel: $channel"
        echo "Valid channels are: 1l0tau, 0l1tau, 2l0tau, 1l1tau, 0l2tau, 2l, yyml"
        exit 1
    fi
done

for channel in "${channels[@]}"; do
    echo "Running TRexFitter for channel: $channel"
    # Check if the config file exists
    if [ ! -f Configs/run2_configs/$mva/$channel"_run2.config" ]; then
        echo "Config file not found: Configs/run2_configs/$mva/$channel"_run2.config""
        exit 1
    fi

    if [ $channel == "yyml" ]; then
        if [ -z $opts ]; then
            trex-fitter mnw Configs/run2_configs/$mva/$channel"_run2.config"

            wait

            trex-fitter mf Configs/run2_configs/$mva/$channel"_run2.config"

            wait

            trex-fitter ms Configs/run2_configs/$mva/$channel"_run2.config"

            wait

            trex-fitter ml Configs/run2_configs/$mva/$channel"_run2.config"

            wait
        else
            trex-fitter m$opts Configs/run2_configs/$mva/$channel"_run2.config"
        fi
    else
        if [ -z $opts ]; then
            trex-fitter nwdf Configs/run2_configs/$mva/$channel"_run2.config"

            wait

            trex-fitter sl Configs/run2_configs/$mva/$channel"_run2.config"

            wait
        else
            trex-fitter $opts Configs/run2_configs/$mva/$channel"_run2.config"
        fi
    fi
done
