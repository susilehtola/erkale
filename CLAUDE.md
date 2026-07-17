# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ERKALE is a C++ quantum chemistry program for Hartree–Fock and density-functional calculations on molecules using Gaussian basis sets. The primary executable `erkale` reads a keyword input file and writes results to a binary HDF5 checkpoint (`.chk`). Auxiliary executables share `liberkale` and provide geometry optimization, orbital localization, density cubes, population analysis, Casida TDDFT, X-ray spectra (XRS), electron momentum density (EMD), atomic solvers, basis set tools, and completeness profiles.

## Build

Modern CMake (>= 3.24). **LAPACK/BLAS and GSL** are found on the system (`find_package`, required); **libxc, libcint (or the API-compatible qcint), HDF5, Armadillo, libwignernj, nlohmann/json, and Eigen3 (NEO only)** are found if installed and otherwise fetched and built from source automatically (`FetchContent` + `FIND_PACKAGE_ARGS`). No in-tree configure step beyond CMake; `compile.sh` is gone.

Standard out-of-source build (OpenMP on by default; executables get an `_omp` suffix):

```bash
cmake -B build
cmake --build build -j
```

For a serial build use a second build dir with `-DUSE_OPENMP=OFF`. Note the historical `openmp/` and `serial/` directories are still used as scratch build dirs on this dev machine; a fresh build can use any directory name.

Important options:
- `-DUSE_OPENMP=OFF` — serial build (no `_omp` suffix)
- `-DERKALE_SYSTEM_LIBRARY=/path/to/basis` — system-wide basis set repository
- `-DERKALE_BLAS_INT64=ON|OFF|AUTO` — 64-bit (ILP64) BLAS integers; `AUTO` (default) probes the found BLAS and defines `ARMA_BLAS_LONG_LONG` when 64-bit
- `-DBUILD_SHARED_LIBS=ON` — build shared `liberkale`

The Armadillo configuration macros (`ARMA_DONT_USE_WRAPPER`, `ARMA_USE_BLAS`, `ARMA_USE_LAPACK`, `ARMA_NO_DEBUG`, and `ARMA_BLAS_LONG_LONG`) live on the `erkale_arma` interface target in the top-level `CMakeLists.txt`, **not** in `src/global.h`. The git revision is embedded automatically in a git checkout (CMake generates `version.h` and defines `SVNRELEASE`); tarball builds skip it.

## Tests

Tests live in `tests/` and use CTest. Each `*.run` file is an ERKALE input; the harness runs `erkale` on it, then runs `chkcompare` to diff the produced `.chk` against `refdata/`.

```bash
cd openmp
ctest                         # run all tests
ctest -R H2O_tz_pbe            # run a single test by name regex
ctest --output-on-failure -V   # verbose output
make basictests && ./src/test/basictests   # in-process unit tests for integrals, etc.
```

CTest sets `ERKALE_LIBRARY=<source>/basis`, `ERKALE_SYSDIR=<source>/tests/xyz`, and `ERKALE_REFDIR=<source>/refdata` — match these when running tests manually. Adding a new regression test means adding a `*.run` and reference `.chk`/`.log` to `refdata/`, then regenerating `tests/TestList.txt` via `tests/genlist.sh`.

## Architecture

**Core library `liberkale`** (`src/`) is the engine. Key modules:

- `basis.{h,cpp}`, `basislibrary.*`, `tempered.*` — basis set construction and the `BasisSet` class
- `integrals.*`, `eriworker.*`, `eritable.*`, `eri_digest.*`, `eriscreen.*` — one- and two-electron integral evaluation (libcint wrappers, screening, in-memory tables); one-electron integrals go through `basis.cpp`'s libcint block helper
- `erichol.*`, `density_fitting.*`, `erifit.*` — Cholesky decomposition and density fitting (RI) of ERIs
- `dftgrid.*`, `dftfuncs.*`, `lebedev.*`, `lobatto.*`, `chebyshev.*` — DFT integration grids and libxc functional wrappers
- `scf-base.cpp` plus the **generated** `scf-fock.cpp`, `scf-solvers.cpp`, `scf-force.cpp` — SCF driver, Fock builds, energy gradients
- `diis.*`, `broyden.*`, `trrh.*`, `gdm.*`, `lbfgs.*` — convergence accelerators and optimizers
- `localization.*`, `unitary.*`, `pzstability.*` — orbital localization (Boys, FB, ER, Pipek–Mezey, generalized) and PZ-SIC stability
- `hirshfeld*.*`, `bader*.*`, `stockholder.*`, `population.cpp` — partitioning / population analysis
- `guess.*`, `sap.*` — initial guess construction
- `checkpoint.*` — HDF5 I/O of all SCF state
- `external/` — read/write Gaussian fchk files
- `atom/`, `slaterfit/` — radial atomic solver and Slater→Gaussian fitting (linked into `liberkale` directly)

**Subdirectory libraries / executables** (each has its own `CMakeLists.txt`):

- `emd/` — `liberkale_emd` plus `erkale_emd` (electron momentum density)
- `xrs/` — `liberkale_xrs` plus `erkale_xrs` (X-ray excitations)
- `casida/` — `erkale_casida` (linear-response TDDFT; depends on emd and xrs libs)
- `completeness/` — `liberkale_cmp` and `erkale_completeness` (basis set completeness profiles)
- `atom/`, `slaterfit/` — sources are folded into `liberkale`; `atom/` also builds `erkale_atom`
- `basistool/` — `erkale_basistool` (basis set manipulation, pivoted Cholesky basis reduction)
- `test/` — `basictests`, `chkcompare`, `integraltest`
- `external/` — interop with ADF and Gaussian fchk
- `contrib/` — research / experimental executables (`erkale_dumpxc`, `erkale_moints`, `erkale_formchk`, `erkale_guessbench`, `erkale_co-opt`, `erkale_energy-opt`, NEO, RIMP2, etc.); these are independent and not all are guaranteed to compile cleanly

**Generated sources.** Three SCF files (`scf-fock.cpp.in`, `scf-solvers.cpp.in`, `scf-force.cpp.in`) are concatenated through the C++ preprocessor multiple times with combinations of `-DRESTRICTED`/`-DUNRESTRICTED` and `-DHF`/`-DDFT`/`-D_ROHF` macros to produce the spin/method specializations. `scf-includes.cpp.in` is prepended once. `eriworker_gentransform` is built first and then run to emit `eriworker_routines.cpp` (spherical-harmonic transform tables) which is concatenated with `eriworker.cpp`. When editing any `*.cpp.in`, the generated file in the build directory will be regenerated; to debug the expansion, inspect e.g. `openmp/src/scf-fock.cpp`.

**Spin / method specialization pattern.** Inside `*.cpp.in` files, `#ifdef RESTRICTED` / `UNRESTRICTED` and `HF` / `DFT` / `_ROHF` guards switch between code paths that share most of the body. Edits should preserve all variants — if a change applies only to one variant, guard it.

**Settings.** Runtime keyword handling is centralized in `settings.{h,cpp}`. New SCF input keywords are registered there.

**Checkpoint format.** All persistent state — geometry, basis, density/Fock matrices, orbitals, occupations, energies — flows through `Checkpoint` (HDF5). Tests rely on bit-stable `.chk` output for comparison via `chkcompare`.

## Conventions

- `liberkale*` are the libraries; every standalone tool is named `erkale_<thing>` and gets `_omp` appended in OpenMP builds.
- Linear algebra uses **Armadillo** (`arma::mat`, `arma::vec`, `arma::cx_mat`) throughout — prefer Armadillo over raw LAPACK calls.
- Headers in `src/*.h` are installed to `include/erkale/`; subdirectory headers go to `include/erkale/<subdir>/`.
- `*~` and `#*#` files in the tree are editor backup/autosave artifacts — ignore them.
