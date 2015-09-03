## What is ERKALE? ##
ERKALE is a quantum chemistry program used to solve the electronic structure of atoms, molecules and molecular clusters. It was originally developed at the University of Helsinki, and is currently developed at UC Berkeley.

The original purpose of ERKALE is the computation of x-ray properties, such as
ground-state electron momentum densities and Compton profiles, and core (x-ray absorption and  x-ray Raman scattering) and valence electron excitation spectra of atoms and molecules. Subsequently, it has gained unique functionalities for basis set development as well as self-interaction corrected density-functional theory calculations.


## Why use ERKALE? ##
  * it's free
  * it's reasonably fast
  * it's easy to develop with
  * it has modern features
  * it's cleanly written in C++
  * it runs on any platform or operating system

Patches and new developers are warmly welcome. Please send mail to susi.lehtola (at) alumni.helsinki.fi.

If you have found ERKALE to be useful, please mail me as well.


## Citation ##
If you use ERKALE in a scientific publication, please cite the software as

> S. Lehtola, ERKALE — HF/DFT from Hel (2014), http://erkale.googlecode.com.

Furthermore, there is an article detailing how ERKALE operates and
what kinds of calculations it can be used for. Please cite this work
as well.

> J. Lehtola et al., ["ERKALE — A Flexible Program Package for X-ray Properties of Atoms and Molecules"](http://onlinelibrary.wiley.com/doi/10.1002/jcc.22987/abstract), J. Comp. Chem. 33, 1572 (2012).

## Documentation ##
There is a [Users' Guide](http://theory.physics.helsinki.fi/~jzlehtol/erkale/doc/userguide.pdf) and a [list of available functionals](http://theory.physics.helsinki.fi/~jzlehtol/erkale/doc/functionals.pdf).

## Performance ##
These are some typical figures on the speed of ERKALE on an 12-core system for water clusters of increasing size, using the 6-31G`**` basis and the PBE exchange-correlation functionals with density fitting.

| **Nmol** | **Nbf** | **t** (1 core) | **t** (12 cores) |
|:---------|:--------|:---------------|:-----------------|
| 5        | 125     | 2 min          | 15 s             |
| 14       | 350     | 22 min         | 2 min            |
| 21       | 525     | 1 h            | 7 min            |
| 29       | 725     | 2.5 h          | 15 min           |
| 40       | 1000    | 5 h            | 31 min           |
| 53       | 1325    | 11 h           | 1h               |
| 69       | 1725    | 24 h           | 2.5 h            |
| 88       | 2200    | 1d 17 h        | 4.5 h            |


## Current features ##
Features that have been currently implemented in ERKALE include
  * Hartree-Fock (HF)
    * restricted and unrestricted calculations
    * restricted open-shell calculations
    * resolution of the identity (RI-HF)
  * density-functional theory (DFT)
    * restricted and unrestricted calculations
    * wide variety of functionals (see below)
      * local-density approximation (LDA), _e.g._, S-VWN
      * generalized-gradient approximation (GGA), _e.g._, PBE
      * meta-GGA, _e.g._, TPSS
      * hybrid functionals, _e.g._, B3LYP, PBE0 and TPSSh
      * range separated hybrid functionals, _e.g._, HSE06 and wB97X
      * VV10 non-local correlation, _e.g._, wB97X-V and B97M-V
    * Perdew-Zunger self-interaction correction
      * variational formulation
      * complex optimal orbitals
      * stability analysis
  * in-core or on-the-fly calculations
    * also density fitting, a.k.a. resolution of the identity is supported
    * full Cholesky decomposition is supported, eliminating the need for auxiliary basis sets
  * support for spherical harmonics basis sets
  * OpenMP parallellization
  * modern convergence accelerators
    * Pulay's direct inversion in the iterative subspace (DIIS)
    * ADIIS (close to EDIIS)
    * Broyden accelerator
  * forces and geometry optimization
  * population analyses
    * Mulliken
    * Löwdin
    * Bader
    * Hirshfeld
    * intrinsic atomic orbital
    * iterative Hirshfeld
    * iterative Stockholder
    * Voronoi
  * orbital localization
    * Foster-Boys
    * fourth moment
    * Edmiston-Ruedenberg
    * Pipek-Mezey using any of the following charge methods
      * Mulliken
      * Löwdin
      * Bader
      * Becke
      * Hirshfeld
      * intrinsic atomic orbital
      * iterative Hirshfeld
      * iterative Stockholder
      * Voronoi
  * electron momentum density (EMD) calculations
    * radial momentum density and isotropic Compton profile
    * momentum density on a grid
  * x-ray absorption and x-ray Raman spectrum calculations
    * modeled using the transition-potential approximation
    * momentum transfer dependent transitions
    * double basis set method for improving the description of the continuum
  * time-dependent calculations (TD-DFT) in the Casida formalism
    * momentum transfer dependent transitions
  * portable checkpoint files
    * intermediary results are saved in platform independent HDF5 format

The evaluation of the DFT exchange-correlation functionals is done exclusively by [libxc](http://www.tddft.org/programs/octopus/wiki/index.php/Libxc). See the conclusive list of [available functionals](http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:manual#Available_functionals).

There are interfaces of ERKALE to other codes as well. Currently there is a tool for converting Gaussian(TM) checkpoints into ERKALE format, with which, e.g., EMD properties can be evaluated with ERKALE using a coupled-cluster reference computed with Gaussian(TM).

## Installation ##

You should be able to install ERKALE (and its prerequisites) on any modern `*`nix, MacOS or Windows computer.

### Completeness-optimization tools ###

You can use ERKALE's completeness-optimization routines without having to compile the whole of ERKALE itself. Get the completeness tools [here](http://theory.physics.helsinki.fi/~jzlehtol/erkale/erkale-bastool.tgz). To compile you need a C++ compiler, a LAPACK library and GSL.


### Linux, precompiled packages ###
If you are running Fedora or Red Hat Enterprise 6 (or one of its clones such as CentOS or Scientific Linux), you can use the precompiled packages.

First, configure your system. On Fedora, run (as root)
```
 # wget -O /etc/yum.repos.d/erkale.repo http://theory.physics.helsinki.fi/~jzlehtol/erkale/erkale-fedora.repo
```
On RHEL, you need to run
```
 # wget -O /etc/yum.repos.d/erkale.repo http://theory.physics.helsinki.fi/~jzlehtol/erkale/erkale-el.repo
```
instead.

Then, simply install ERKALE using yum:
```
 # yum install erkale
```
You can then run ERKALE with
```
 $ erkale input
```
for the sequential version, or
```
 $ erkale_omp input
```
for the parallellized version.


### Linux, from source ###
To compile ERKALE you need
  * a C++ compiler ([G++](http://gcc.gnu.org/) is recommended, any recent version should be fine)
  * [CMake](http://www.cmake.org/), at least version 2.8
  * [HDF5](http://www.hdfgroup.org/HDF5/), at least version 1.8
  * [The GNU Scientific Library](http://www.gnu.org/s/gsl/), at least version 1.4
  * [libint](http://sourceforge.net/p/libint/home)
  * [libxc](http://www.tddft.org/programs/octopus/wiki/index.php/Libxc), at least version 2.0.0
  * [Armadillo](http://arma.sourceforge.net/)
and
  * a LAPACK library such as [ATLAS](http://math-atlas.sourceforge.net/).
  * the [subversion](http://subversion.apache.org/) client


If you are running a Fedora system you can get all of these by running
```
 # yum -y install atlas-devel gsl-devel hdf5-devel libint-devel libxc-devel subversion
```
For Debian/Ubuntu you need to run
```
 # apt-get -y install libatlas-base-dev libgsl0-dev libhdf5-dev libint-dev libxc-dev subversion
```
You will then have to install the Armadillo template library by hand.

We have also supplied a script that can be used to compile ERKALE, and if wanted its prerequirements:
[http://erkale.googlecode.com/svn/trunk/compile.sh](http://erkale.googlecode.com/svn/trunk/compile.sh)

If you don't have all of the prerequirements installed (as with the instructions above), or you wish to use a newer version of some library (foo), you can toggle compiling of the prerequirements by toggling the system\_foo variable in the compile script.

### Windows, from binary ###

First of all, to use ERKALE in Windows, you need to install [Cygwin](http://www.cygwin.com/).

For ease of use, I have compiled a ready-to-use version of ERKALE to be installed in Cygwin. The tarballs are available [here](http://theory.physics.helsinki.fi/~jzlehtol/erkale/windows).

### Windows, from source ###

First of all, to use ERKALE in Windows, you need to install [Cygwin](http://www.cygwin.com/). You also need to install the following packages from the Cygwin installer:
  * gcc4
  * gcc4-fortran
  * gcc4-g++
  * gsl-devel
  * libhdf5-devel
  * liblapack-devel
  * wget
  * subversion
  * cmake
  * make
  * patch
  * pkg-config

After this, you should be able to compile ERKALE using the build script.