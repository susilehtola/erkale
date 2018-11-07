#!/bin/bash

# 2018-09-22 Susi Lehtola

els=("" H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr);

if [[ -f genpot.m ]]; then
    \rm genpot.m
fi
for((Z=1;Z<${#els[@]};Z++)); do    
    el=${els[Z]}

    rin="../atpot/PBE-nr-sp/v_${el}.dat"
    if [[ ! -f ${rin} ]]; then
	continue
    fi
    cat ${rin} | awk '{print $1}' > r_${el}.dat
    cat >> genpot.m <<EOF
r_${el}=load("r_${el}.dat");
v_${el}=gszpot($Z,r_${el});
out=fopen("v_${el}.dat","w");
for i=1:length(r_${el})
  fprintf(out,"%.15e %.15e\n",r_${el}(i),v_${el}(i));
end
fclose(out);
EOF
done

