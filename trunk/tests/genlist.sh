#!/bin/bash

# Generate list of tests
f=$(mktemp)
for j in $(ls *.run | sed "s|.run||g"); do
    # Is the run a XRS run?
    xrs=0
    if(( $(echo $j | grep _xrs | wc -l) )); then
	let xrs++
    fi
    if(( $(echo $j | grep _fch | wc -l) )); then
	let xrs++
    fi
    if(( $(echo $j | grep _xch | wc -l) )); then
	let xrs++
    fi

    # Get the amount of basis functions in the run
    reflog="../refdata/$(basename $j .run).log"
    if [[ -f $reflog ]]; then
	Nbf=$(grep "Basis set contains" $reflog | awk '{print $4}')
    else
	Nbf=10000
    fi

    # Generate input
    if(( $xrs )); then
	# The ground-state calculation is
	gs=$(echo $j | sed "s|_xrs||g;s|_fch||g;s|_xch||g")
	echo "$Nbf run_xrs($j ${gs}_chk)" >> $f
    else
	echo "$Nbf run_test($j)" >> $f
    fi
done

# Header
echo "# **** $(date) ****" > TestList.txt
# Sort the list wrt amount of functions and get rid of the basis function index
sort -g $f | awk '{for(i=2;i<=NF;i++) {printf("%s ",$i)} printf("\n")}' >> TestList.txt
echo "# **** END ****" >> TestList.txt
# and delete the temp file
\rm $f

