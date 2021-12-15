#!/bin/bash

eval "$(conda shell.bash hook)"
cd src
conda activate pyg_cuda102
python RunRandomBandlimited_Epochs.py
