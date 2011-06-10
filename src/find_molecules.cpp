#include "find_molecules.h"
#include <cmath>

std::vector< std::vector<size_t> > find_molecules(const std::vector<atom_t> & atoms) {
  // List of molecules, atoms in molecules
  std::vector< std::vector<size_t> > ret;

  // Loop over atoms
  for(size_t i=0;i<atoms.size();i++) {

    // Check first, if atom should belong to a molecule.
    bool found=0;
    for(size_t imol=0;imol<ret.size();imol++) {
      // Loop over atoms in molecule
      for(size_t iat=0;iat<ret[imol].size();iat++) {
	// The index of the other atom is
	size_t j=ret[imol][iat];

	// Compute the distance between the atoms.
	double d=0.0;
	d+=(atoms[i].x-atoms[j].x)*(atoms[i].x-atoms[j].x);
	d+=(atoms[i].y-atoms[j].y)*(atoms[i].y-atoms[j].y);
	d+=(atoms[i].z-atoms[j].z)*(atoms[i].z-atoms[j].z);
	d=sqrt(d);
	
	// Check if bond exists
	if(check_bonds(d,atoms[i].el,atoms[j].el)) {
	  // Yes, add to molecule
	  ret[imol].push_back(i);
	  found=1;
	  // Stop loop
	  break;
	}
      }

      if(found)
	// Atom was already added to a molecule
	break;
    }

    if(!found) {
      // Need to add new molecule.
      std::vector<size_t> hlp;
      hlp.push_back(i);
      // Add it to the list
      ret.push_back(hlp);
    }
  }

  return ret;
}

bool check_bonds(double d, const std::string & at1, const std::string & at2, double tol) {
  // Check if bond exists between atoms of type at1 and at2.

  // Default - no bond.
  bool bond=0;

  // H-H
  bond=(bond || check_bond(at1,at2,d,"H","H",0.74+tol));
  // H-B
  bond=(bond || check_bond(at1,at2,d,"H","B",1.19+tol));
  // H-C
  bond=(bond || check_bond(at1,at2,d,"H","C",1.09+tol));
  // H-Si
  bond=(bond || check_bond(at1,at2,d,"H","Si",1.48+tol));
  // H-Ge
  bond=(bond || check_bond(at1,at2,d,"H","Ge",1.53+tol));
  // H-Sn
  bond=(bond || check_bond(at1,at2,d,"H","Sn",1.70+tol));
  // H-N
  bond=(bond || check_bond(at1,at2,d,"H","N",1.01+tol));
  // H-P
  bond=(bond || check_bond(at1,at2,d,"H","P",1.44+tol));
  // H-As
  bond=(bond || check_bond(at1,at2,d,"H","As",1.52+tol));
  // H-O
  bond=(bond || check_bond(at1,at2,d,"H","O",0.96+tol));
  // H-S
  bond=(bond || check_bond(at1,at2,d,"H","S",1.34+tol));
  // H-Se
  bond=(bond || check_bond(at1,at2,d,"H","Se",1.46+tol));
  // H-Te
  bond=(bond || check_bond(at1,at2,d,"H","Te",1.70+tol));
  // H-F
  bond=(bond || check_bond(at1,at2,d,"H","F",0.92+tol));
  // H-Cl
  bond=(bond || check_bond(at1,at2,d,"H","Cl",1.27+tol));
  // H-Br
  bond=(bond || check_bond(at1,at2,d,"H","Br",1.41+tol));
  // H-I
  bond=(bond || check_bond(at1,at2,d,"H","I",1.61+tol));

  // B-Cl
  bond=(bond || check_bond(at1,at2,d,"B","Cl",1.75+tol));

  // C-C
  bond=(bond || check_bond(at1,at2,d,"C","C",1.54+tol));
  // C-Si
  bond=(bond || check_bond(at1,at2,d,"C","Si",1.85+tol));
  // C-Ge
  bond=(bond || check_bond(at1,at2,d,"C","Ge",1.95+tol));
  // C-Sn
  bond=(bond || check_bond(at1,at2,d,"C","Sn",2.16+tol));
  // C-Pb
  bond=(bond || check_bond(at1,at2,d,"C","Pb",2.30+tol));
  // C-N
  bond=(bond || check_bond(at1,at2,d,"C","N",1.47+tol));
  // C-P
  bond=(bond || check_bond(at1,at2,d,"C","P",1.84+tol));
  // C-O
  bond=(bond || check_bond(at1,at2,d,"C","O",1.43+tol));
  // C-S
  bond=(bond || check_bond(at1,at2,d,"C","S",1.82+tol));
  // C-F
  bond=(bond || check_bond(at1,at2,d,"C","F",1.35+tol));
  // C-Cl
  bond=(bond || check_bond(at1,at2,d,"C","Cl",1.77+tol));
  // C-Br
  bond=(bond || check_bond(at1,at2,d,"C","Br",1.94+tol));
  // C-I
  bond=(bond || check_bond(at1,at2,d,"C","I",2.14+tol));

  // N-N
  bond=(bond || check_bond(at1,at2,d,"N","N",1.45+tol));
  // N-O
  bond=(bond || check_bond(at1,at2,d,"N","O",1.40+tol));
  // N-F
  bond=(bond || check_bond(at1,at2,d,"N","F",1.36+tol));
  // N-Cl
  bond=(bond || check_bond(at1,at2,d,"N","Cl",1.75+tol));

  // P-P
  bond=(bond || check_bond(at1,at2,d,"P","P",2.21+tol));
  // P-O
  bond=(bond || check_bond(at1,at2,d,"P","O",1.63+tol));
  // P=S
  bond=(bond || check_bond(at1,at2,d,"P","S",1.86+tol));
  // P-F
  bond=(bond || check_bond(at1,at2,d,"P","F",1.54+tol));
  // P-Cl
  bond=(bond || check_bond(at1,at2,d,"P","Cl",2.03+tol));

  // As-As
  bond=(bond || check_bond(at1,at2,d,"As","As",2.43+tol));
  // As-O
  bond=(bond || check_bond(at1,at2,d,"As","O",1.78+tol));
  // As-F
  bond=(bond || check_bond(at1,at2,d,"As","F",1.71+tol));
  // As-Cl
  bond=(bond || check_bond(at1,at2,d,"As","Cl",2.16+tol));
  // As-Br
  bond=(bond || check_bond(at1,at2,d,"As","Br",2.33+tol));
  // As-I
  bond=(bond || check_bond(at1,at2,d,"As","I",2.54+tol));

  // Sb-Cl
  bond=(bond || check_bond(at1,at2,d,"Sb","Cl",2.32+tol));

  // O-O
  bond=(bond || check_bond(at1,at2,d,"O","O",1.48+tol));
  // O-F
  bond=(bond || check_bond(at1,at2,d,"O","F",1.42+tol));
  // O=S
  bond=(bond || check_bond(at1,at2,d,"O","S",1.43+tol));

  // S-S
  bond=(bond || check_bond(at1,at2,d,"S","S",1.49+tol));
  // S-F
  bond=(bond || check_bond(at1,at2,d,"S","F",1.56+tol));
  // S-Cl
  bond=(bond || check_bond(at1,at2,d,"S","Cl",2.07+tol));

  // Se=Se
  bond=(bond || check_bond(at1,at2,d,"Se","Se",2.15+tol));

  // F-F
  bond=(bond || check_bond(at1,at2,d,"F","F",1.42+tol));
  // Cl-Cl
  bond=(bond || check_bond(at1,at2,d,"Cl","Cl",1.99+tol));
  // Br-Br
  bond=(bond || check_bond(at1,at2,d,"Br","Br",2.28+tol));
  // I-I
  bond=(bond || check_bond(at1,at2,d,"I","I",2.67+tol));
  // I-F
  bond=(bond || check_bond(at1,at2,d,"I","F",1.91+tol));
  // I-Cl
  bond=(bond || check_bond(at1,at2,d,"I","Cl",2.32+tol));

  // Kr-F
  bond=(bond || check_bond(at1,at2,d,"Kr","F",1.90+tol));
  // Xe-O
  bond=(bond || check_bond(at1,at2,d,"Xe","O",1.75+tol));
  // Xe-F
  bond=(bond || check_bond(at1,at2,d,"Xe","F",1.95+tol));

  // Si-Si
  bond=(bond || check_bond(at1,at2,d,"Si","Si",2.33+tol));
  // Si-O
  bond=(bond || check_bond(at1,at2,d,"Si","O",1.63+tol));
  // Si-S
  bond=(bond || check_bond(at1,at2,d,"Si","S",2.00+tol));
  // Si-F
  bond=(bond || check_bond(at1,at2,d,"Si","F",1.60+tol));
  // Si-Cl
  bond=(bond || check_bond(at1,at2,d,"Si","Cl",2.02+tol));
  // Si-Br
  bond=(bond || check_bond(at1,at2,d,"Si","Br",2.15+tol));
  // Si-I
  bond=(bond || check_bond(at1,at2,d,"Si","I",2.43+tol));

  // Ge-Ge
  bond=(bond || check_bond(at1,at2,d,"Ge","Ge",2.41+tol));
  // Ge-F
  bond=(bond || check_bond(at1,at2,d,"Ge","F",1.68+tol));
  // Ge-Cl
  bond=(bond || check_bond(at1,at2,d,"Ge","Cl",2.10+tol));
  // Ge-Br
  bond=(bond || check_bond(at1,at2,d,"Ge","Br",2.30+tol));

  // Sn-Cl
  bond=(bond || check_bond(at1,at2,d,"Sn","Cl",2.33+tol));
  // Sn-Br
  bond=(bond || check_bond(at1,at2,d,"Sn","Br",2.50+tol));
  // Sn-I
  bond=(bond || check_bond(at1,at2,d,"Sn","I",2.70+tol));

  // Pb-Cl
  bond=(bond || check_bond(at1,at2,d,"Pb","Cl",2.42+tol));
  // Pb-I
  bond=(bond || check_bond(at1,at2,d,"Pb","I",2.79+tol));


}

bool check_bond(const std::string & atom1, const std::string & atom2, double d,const std::string & test1, const std::string & test2, double maxlen) {

  // First, check if atoms 1 and 2 are of types test1 and test2.
  if( ((atom1==test1) && (atom2==test2)) || ((atom1==test2) && (atom2==test1))) {
    // Check if bond distance is within maximum allowed
    if(d<=maxlen)
      // Bond exists
      return 1;
    else
      // Bond does not exist
      return 0;
  } else
    // Atoms not of the given types.
    return 0;
}
