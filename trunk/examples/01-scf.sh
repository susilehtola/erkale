#!/bin/bash

if [ ! -d scf ]; then
 mkdir scf
fi

cd scf
# Perform initial LDA calculation
erkale ../run.lda
# Use LDA calculation to seed HF calculation
erkale ../run.hf
