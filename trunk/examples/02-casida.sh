#!/bin/bash

if [ ! -d cas ]; then
 mkdir cas
fi

cd cas
# Use HF calculation for the Casida calculation
erkale_casida ../run.cas
