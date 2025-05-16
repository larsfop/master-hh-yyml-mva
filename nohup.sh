#!/bin/bash

(
    ./param_tuner.sh -g lr -o b -k 4 0l2tau
    wait
    ./param_tuner.sh -g lb -o b -k 4 0l2tau
    wait
    ./param_tuner.sh -g h -o b -k 4 0l2tau
    wait
    ./param_tuner.sh -g lr -o m -k 4 0l2tau
    wait
    ./param_tuner.sh -g lb -o m -k 4 0l2tau
    wait
    ./param_tuner.sh -g h -o m -k 4 0l2tau
    wait
)