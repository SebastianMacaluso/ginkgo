#!/usr/bin/env bash

ROOTDIR=/Users/dpappadopulo/Projects

python3 generate_jets.py \
    --outdir "${ROOTDIR}/ToyJetsShower/data" \
    --filename "testjets" \
    --num_samples 10