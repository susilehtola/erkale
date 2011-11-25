#!/bin/bash

if [ ! -d emd ]; then
 mkdir emd
fi

cd emd
# Use HF calculation to calculate EMD
erkale_emd ../run.emd
