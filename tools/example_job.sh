#!/bin/bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh

#python3 -m virtualenv myvenv
#source myvenv/bin/activate
#pip install .

python tools/training/TrainModelFromGraph.py --train