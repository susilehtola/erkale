#!/bin/bash

if [ ! -d xrs ]; then
 mkdir xrs
fi

cd xrs
erkale_xrs ../run.xrs
