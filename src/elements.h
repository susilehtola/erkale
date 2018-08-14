/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */


#ifndef ERKALE_ELEMENTS
#define ERKALE_ELEMENTS

#include <string>
#include <vector>

/// Names of elements
const std::string element_names[]={
  "",
  "Hydrogen","Helium",
  "Lithium","Beryllium","Boron","Carbon","Nitrogen","Oxygen","Fluorine","Neon",
  "Sodium","Magnesium","Aluminium","Silicon","Phosphorus","Sulfur","Chlorine","Argon",
  "Potassium","Calcium","Scandium","Titanium","Vanadium","Chromium","Manganese","Iron","Cobalt","Nickel","Copper","Zinc","Gallium","Germanium","Arsenic","Selenium","Bromine","Krypton",
  "Rubidium","Strontium","Yttrium","Zirconium","Niobium","Molybdenum","Technetium","Ruthenium","Rhodium","Palladium","Silver","Cadmium","Indium","Tin","Antimony","Tellurium","Iodine","Xenon",
  "Caesium","Barium","Lanthanum","Cerium","Praseodymium","Neodynium","Promethium","Samarium","Europium","Gadolinium","Terbium","Dysprosium","Holmium","Erbium","Thulium","Ytterbium", "Lutetium", "Hafnium", "Tantalum", "Tungsten", "Rhenium", "Osmium", "Iridium", "Platinum", "Gold", "Mercury", "Thallium", "Lead", "Bismuth", "Polonium", "Astatine", "Radon",
  "Francium", "Radium", "Actinium", "Thorium", "Protactinium", "Uranium", "Neptunium", "Plutonium", "Americium", "Curium", "Berkelium", "Californium", "Einsteinium", "Fermium", "Mendelevium", "Nobelium", "Lawrencium", "Rutherfordium", "Dubnium", "Seaborgium", "Bohrium", "Hassium", "Meitnerium", "Darmstadtium", "Roentgenium", "Copernicium", "Nihonium", "Flevorium", "Moscovium", "Livermorium", "Tennessine", "Oganesson"
};

/// Symbols of elements
const std::string element_symbols[]={
  "",
  "H","He",
  "Li","Be","B","C","N","O","F","Ne",
  "Na","Mg","Al","Si","P","S","Cl","Ar",
  "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
  "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
  "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
  "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"
};

/// Maximum supported element
const int maxZ=112;

/// The row of the elements
const int element_row[]={
  0,
  1, 1,
  2, 2, 2, 2, 2, 2, 2, 2,
  3, 3, 3, 3, 3, 3, 3, 3,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};

/// Ordering of shells: 1s, 2s, 2p, 3s, 3p, 4s, 3d, 4p, 5s, 4d, 5p, 6s, 4f, 5d, 6p, 7s, 5f, 6d, 7p
const int shell_order[]={0, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1};

/// Get number of shells with am l
std::vector<int> shell_count(int Z);

/// Atomic masses from http://www.chem.qmul.ac.uk/iupac/AtWt/index.html
/// Technetium has no stable isotopes, set mass to -1e-10
const double atomic_masses[]={
  0.0,
  1.008000, 4.002602,
  6.940000, 9.012182, 10.810000, 12.011000, 14.007000, 15.999000, 18.998403, 20.179700,
  22.989769, 24.305000, 26.981539, 28.085000, 30.973762, 32.060000, 35.450000, 39.948000,
  39.098300, 40.078000, 44.955912, 47.867000, 50.941500, 51.996100, 54.938045, 55.845000, 58.933195, 58.693400, 63.546000, 65.380000, 69.723000, 72.630000, 74.921600, 78.960000, 79.904000, 83.798000,
  85.467800, 87.620000, 88.905850, 91.224000, 92.906380, 95.960000, -1e-10, 101.070000, 102.905500, 106.420000, 107.868200, 112.411000, 114.818000, 118.710000, 121.760000, 127.600000, 126.904470, 131.293000
};

/// Magic numbers - noble gases
const int magicno[]={0, 2, 10, 18, 36, 54, 86, 118};

/// Get nuclear charge of element
int get_Z(std::string el);

/// Get atomic angular momentum
int atom_am(int Z);

#endif
