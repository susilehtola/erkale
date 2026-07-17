ERKALE -- HF/DFT from Hel
-----------------------------------

* [**Binary packages for Fedora / Red Hat**](https://copr.fedoraproject.org/coprs/jussilehtola/erkale/)

* [**Wiki page**](https://github.com/susilehtola/erkale/wiki/ERKALE)

* [**Users' guide**](doc/userguide.pdf)

* **List of functionals** [on libxc page](http://www.tddft.org/programs/libxc/functionals/)

* **Building:** see [INSTALL](INSTALL). In short, with the prerequisites
  installed from your distribution:

  ```
  cmake -B build
  cmake --build build -j
  ```

  CMake finds the system libraries and automatically fetches and builds
  any of the CMake-based dependencies (libxc, libcint, HDF5, Armadillo,
  libwignernj, nlohmann/json) that are missing.
