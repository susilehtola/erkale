/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "checkpoint.h"
#include <istream>

// Helper macros
#define CHECK_OPEN() {if(!opend) {ERROR_INFO(); throw std::runtime_error("Cannot access checkpoint file that has not been opened!\n");}}
#define CHECK_WRITE() {if(!writemode) {throw std::runtime_error("Cannot write to checkpoint file that was opened for reading only!\n");}}
#define CHECK_EXIST() {if(!exist(name)) { std::ostringstream oss; oss << "The entry " << name << " does not exist in the checkpoint file!\n"; throw std::runtime_error(oss.str()); } }

Checkpoint::Checkpoint(const std::string & fname, bool writem, bool trunc) {
  writemode=writem;
  filename=fname;
  opend=false;

  if(writemode && (trunc || !file_exists(fname))) {
    // Truncate existing file, using default creation and access properties.
    file=H5Fcreate(fname.c_str(),H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
    opend=true;

    // Save checkpoint version
    int ver=ERKALE_CHKVER;
    write("chkver",ver);
    // Close the file
    close();
  } else {
    // Open file
    open();

    // Check version
    if(!exist("chkver"))
      throw std::runtime_error("Incompatible version of checkpoint file.\n");
    // Read version
    int ver;
    read("chkver",ver);
    if(ver!=ERKALE_CHKVER) {
      std::ostringstream oss;
      oss << "Tried to open checkpoint file version " << ver << " but only version " << ERKALE_CHKVER << " is supported.\n";
      throw std::runtime_error(oss.str());
    }
  }
}

Checkpoint::~Checkpoint() {
  if(opend)
    close();
}

void Checkpoint::open() {
  // Check that file exists
  if(!file_exists(filename)) {
    throw std::runtime_error("Trying to open nonexistent checkpoint file \"" + filename + "\"!\n");
  }

  if(!opend) {
    if(writemode)
      // Open in read-write mode
      file=H5Fopen(filename.c_str(),H5F_ACC_RDWR  ,H5P_DEFAULT);
    else
      // Open in read-only mode
      file=H5Fopen(filename.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);

    // File has been opened
    opend=true;
  } else
    throw std::runtime_error("Trying to open checkpoint file that has already been opened!\n");
}

void Checkpoint::close() {
  if(opend) {
    H5Fclose(file);
    opend=false;
  } else
    throw std::runtime_error("Trying to close file that has already been closed!\n");
}

void Checkpoint::flush() {
  if(opend && writemode)
      H5Fflush(file,H5F_SCOPE_GLOBAL);
}

bool Checkpoint::is_open() const {
  return opend;
}

bool Checkpoint::exist(const std::string & name) {
  bool cl=false;
  if(!opend) {
    cl=true;
    open();
  }

  bool ret=H5Lexists(file, name.c_str(), H5P_DEFAULT);

  if(cl) close();

  return ret;
}

void Checkpoint::remove(const std::string & name) {
  CHECK_WRITE();

  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  if(exist(name)) {
    // Remove the entry from the file.
    H5Ldelete(file, name.c_str(), H5P_DEFAULT);
  }

  if(cl) close();
}

void Checkpoint::write(const std::string & name, const arma::mat & m) {
  CHECK_WRITE();

  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Remove possible existing entry
  remove(name);

  // Dimensions of the matrix
  hsize_t dims[2];
  dims[0]=m.n_rows;
  dims[1]=m.n_cols;

  // Create a dataspace.
  hid_t dataspace=H5Screate_simple(2,dims,NULL);

  // Create a datatype.
  hid_t datatype=H5Tcopy(H5T_NATIVE_DOUBLE);

  // Create the dataset using the defined dataspace and datatype, and
  // default dataset creation properties.
  hid_t dataset=H5Dcreate(file,name.c_str(),datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Write the data to the file.
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, m.memptr());

  // Close everything.
  H5Dclose(dataset);
  H5Tclose(datatype);
  H5Sclose(dataspace);
  if(cl) close();
}

void Checkpoint::read(const std::string & name, arma::mat & m) {
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }
  CHECK_EXIST();

  // Open the dataset.
  hid_t dataset = H5Dopen (file, name.c_str(), H5P_DEFAULT);

  // Get the data type
  hid_t datatype = H5Dget_type(dataset);

  // Get the class info
  hid_t hclass=H5Tget_class(datatype);

  if(hclass!=H5T_FLOAT) {
    std::ostringstream oss;
    oss << "Error - " << name << " is not a floating point value!\n";
    ERROR_INFO();
    throw std::runtime_error(oss.str());
  }

  // Get dataspace
  hid_t dataspace = H5Dget_space(dataset);
  // Get number of dimensions
  int ndim = H5Sget_simple_extent_ndims(dataspace);
  if(ndim!=2) {
    std::ostringstream oss;
    oss << "Error - " << name << " should have dimension 2, instead dimension is " << ndim << "!\n";
    ERROR_INFO();
    throw std::runtime_error(oss.str());
  }

  // Get the size of the matrix
  hsize_t dims[ndim];
  H5Sget_simple_extent_dims(dataspace,dims,NULL);

  // Allocate memory
  m.zeros(dims[0],dims[1]);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, m.memptr());

  // Close dataspace
  H5Sclose(dataspace);
  // Close datatype
  H5Tclose(datatype);
  // Close dataset
  H5Dclose(dataset);

  if(cl) close();
}


void Checkpoint::cwrite(const std::string & name, const arma::cx_mat & m) {
  arma::mat mreal=arma::real(m);
  arma::mat mim=arma::imag(m);

  write(name+".re",mreal);
  write(name+".im",mim);
}

void Checkpoint::cread(const std::string & name, arma::cx_mat & m) {
  arma::mat mreal, mim;
  read(name+".re",mreal);
  read(name+".im",mim);
  m=mreal*COMPLEX1+mim*COMPLEXI;
}

void Checkpoint::write(const std::string & name, const std::vector<double> & v) {
  CHECK_WRITE();
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Remove possible existing entry
  remove(name);

  // Dimensions of the vector
  hsize_t dims[1];
  dims[0]=v.size();

  // Create a dataspace.
  hid_t dataspace=H5Screate_simple(1,dims,NULL);

  // Create a datatype.
  hid_t datatype=H5Tcopy(H5T_NATIVE_DOUBLE);

  // Create the dataset using the defined dataspace and datatype, and
  // default dataset creation properties.
  hid_t dataset=H5Dcreate(file,name.c_str(),datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Write the data to the file.
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(v[0]));

  // Close everything.
  H5Dclose(dataset);
  H5Tclose(datatype);
  H5Sclose(dataspace);
  if(cl) close();
}

void Checkpoint::read(const std::string & name, std::vector<double> & v) {
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }
  CHECK_EXIST();

  // Open the dataset.
  hid_t dataset = H5Dopen (file, name.c_str(), H5P_DEFAULT);

  // Get the data type
  hid_t datatype  = H5Dget_type(dataset);

  // Get the class info
  hid_t hclass=H5Tget_class(datatype);

  if(hclass!=H5T_FLOAT) {
    std::ostringstream oss;
    oss << "Error - " << name << " is not a floating point value!\n";
    ERROR_INFO();
    throw std::runtime_error(oss.str());
  }

  // Get dataspace
  hid_t dataspace = H5Dget_space(dataset);
  // Get number of dimensions
  int ndim = H5Sget_simple_extent_ndims(dataspace);
  if(ndim!=1) {
    std::ostringstream oss;
    oss << "Error - " << name << " should have dimension 1, instead dimension is " << ndim << "!\n";
    ERROR_INFO();
    throw std::runtime_error(oss.str());
  }

  // Get the size of the matrix
  hsize_t dims[ndim];
  H5Sget_simple_extent_dims(dataspace,dims,NULL);

  // Allocate memory
  v.resize(dims[0]);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(v[0]));

  // Close dataspace
  H5Sclose(dataspace);
  // Close datatype
  H5Tclose(datatype);
  // Close dataset
  H5Dclose(dataset);
  if(cl) close();
}

void Checkpoint::write(const std::string & name, const std::vector<hsize_t> & v) {
  CHECK_WRITE();
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Remove possible existing entry
  remove(name);

  // Dimensions of the vector
  hsize_t dims[1];
  dims[0]=v.size();

  // Create a dataspace.
  hid_t dataspace=H5Screate_simple(1,dims,NULL);

  // Create a datatype.
  hid_t datatype=H5Tcopy(H5T_NATIVE_HSIZE);

  // Create the dataset using the defined dataspace and datatype, and
  // default dataset creation properties.
  hid_t dataset=H5Dcreate(file,name.c_str(),datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Write the data to the file.
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(v[0]));

  // Close everything.
  H5Dclose(dataset);
  H5Tclose(datatype);
  H5Sclose(dataspace);
  if(cl) close();
}

void Checkpoint::read(const std::string & name, std::vector<hsize_t> & v) {
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }
  CHECK_EXIST();

  // Open the dataset.
  hid_t dataset = H5Dopen (file, name.c_str(), H5P_DEFAULT);

  // Get the data type
  hid_t datatype  = H5Dget_type(dataset);

  // Get the class info
  hid_t hclass=H5Tget_class(datatype);

  if(hclass!=H5T_INTEGER) {
    std::ostringstream oss;
    oss << "Error - " << name << " is not an integer value!\n";
    ERROR_INFO();
    throw std::runtime_error(oss.str());
  }

  // Get dataspace
  hid_t dataspace = H5Dget_space(dataset);
  // Get number of dimensions
  int ndim = H5Sget_simple_extent_ndims(dataspace);
  if(ndim!=1) {
    std::ostringstream oss;
    oss << "Error - " << name << " should have dimension 1, instead dimension is " << ndim << "!\n";
    ERROR_INFO();
    throw std::runtime_error(oss.str());
  }

  // Get the size of the matrix
  hsize_t dims[ndim];
  H5Sget_simple_extent_dims(dataspace,dims,NULL);

  // Allocate memory
  v.resize(dims[0]);
  H5Dread(dataset, H5T_NATIVE_HSIZE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(v[0]));

  // Close dataspace
  H5Sclose(dataspace);
  // Close datatype
  H5Tclose(datatype);
  // Close dataset
  H5Dclose(dataset);
  if(cl) close();
}

void Checkpoint::write(const BasisSet & basis) {
  CHECK_WRITE();
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Remove possible existing entries
  remove("basis.nucs");
  remove("basis.contr");
  remove("basis.data");

  // Get number of shells
  size_t Nsh=basis.get_Nshells();
  // Get number of nuclei
  size_t Nnuc=basis.get_Nnuc();

  // Initialize dataspace
  hsize_t dimsf[1];
  hid_t dataspace;
  hid_t datatype;
  hid_t dataset;

  /* First, write out nuclei. */
  nuc_t nucs[Nnuc];
  // Silence valgrind warning
  memset(nucs,0,Nnuc*sizeof(nuc_t));
  for(size_t i=0;i<Nnuc;i++) {
    // Get nucleus
    nucleus_t n=basis.get_nucleus(i);

    // Store data
    nucs[i].ind=n.ind;
    nucs[i].rx=n.r.x;
    nucs[i].ry=n.r.y;
    nucs[i].rz=n.r.z;
    nucs[i].Z=n.Z;
    nucs[i].bsse=n.bsse;
    strncpy(nucs[i].sym,n.symbol.c_str(),SYMLEN);
  }

  datatype = H5Tcreate(H5T_COMPOUND, sizeof(nuc_t));
  H5Tinsert(datatype, "ind", HOFFSET(nuc_t, ind), H5T_NATIVE_HSIZE);
  H5Tinsert(datatype, "rx", HOFFSET(nuc_t, rx), H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "ry", HOFFSET(nuc_t, ry), H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "rz", HOFFSET(nuc_t, rz), H5T_NATIVE_DOUBLE);

  H5Tinsert(datatype, "bsse", HOFFSET(nuc_t, bsse), H5T_NATIVE_HBOOL);
  H5Tinsert(datatype, "Z", HOFFSET(nuc_t, Z), H5T_NATIVE_INT);

  // Symbol
  hid_t symtype=H5Tcopy(H5T_C_S1);
  H5Tset_size(symtype,SYMLEN);
  H5Tinsert(datatype, "sym", HOFFSET(nuc_t, sym), symtype);

  dimsf[0]=Nnuc;
  // Create dataspace
  dataspace = H5Screate_simple(1,dimsf,NULL);
  // Create dataset
  dataset=H5Dcreate(file,"basis.nucs",datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  // Write out the data
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, nucs);

  // Free dataset
  H5Dclose(dataset);
  H5Sclose(dataspace);
  H5Tclose(datatype);
  H5Tclose(symtype);

  /* Write shell data, contractions first */

  dimsf[0]=Nsh;
  dataspace = H5Screate_simple(1,dimsf,NULL);

  // Create array holding exponents and contraction coefficients of shells.
  hvl_t contrs[Nsh];

  // Initialize data.
  for(size_t i=0;i<Nsh;i++) {
    // Get contraction on shell
    std::vector<contr_t> cntr=basis.get_contr(i);

    // Allocate memory
    contrs[i].p=malloc(cntr.size()*sizeof(contr_t));
    contrs[i].len=cntr.size();

    // Store contractions
    for(size_t j=0;j<cntr.size();j++)
      ((contr_t *) contrs[i].p)[j]=cntr[j];
  }

  // Create compound datatype
  hid_t contrdata = H5Tcreate (H5T_COMPOUND, sizeof(contr_t));
  H5Tinsert(contrdata, "c", HOFFSET(contr_t, c), H5T_NATIVE_DOUBLE);
  H5Tinsert(contrdata, "z", HOFFSET(contr_t, z), H5T_NATIVE_DOUBLE);
  // Create datatype
  datatype = H5Tvlen_create(contrdata);

  // and the dataspace is

  // Create the dataset using the defined dataspace and datatype, and
  // default dataset creation properties
  dataset=H5Dcreate(file,"basis.contr",datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  // Write the data.
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, contrs);

  // Free memory.
  H5Dclose(dataset);
  H5Tclose(datatype);
  H5Tclose(contrdata);
  for(size_t i=0;i<Nsh;i++)
    free(contrs[i].p);

  /* Done with contractions, write other (fixed-length) data. */

  shell_data_t shdata[Nsh];
  for(size_t i=0;i<Nsh;i++) {
    shdata[i].indstart=basis.get_first_ind(i);
    shdata[i].am=basis.get_am(i);
    shdata[i].uselm=basis.lm_in_use(i);
    shdata[i].cenind=basis.get_shell_center_ind(i);
  }

  // Create dataset for shared data
  datatype = H5Tcreate(H5T_COMPOUND, sizeof(shell_data_t));

  H5Tinsert(datatype, "indstart", HOFFSET(shell_data_t, indstart), H5T_NATIVE_HSIZE);
  H5Tinsert(datatype, "am", HOFFSET(shell_data_t, am), H5T_NATIVE_INT);
  H5Tinsert(datatype, "uselm", HOFFSET(shell_data_t, uselm), H5T_NATIVE_HBOOL);

  H5Tinsert(datatype, "cenind", HOFFSET(shell_data_t, cenind), H5T_NATIVE_HSIZE);

  // Write the data.
  dataset=H5Dcreate(file,"basis.data",datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, shdata);

  H5Dclose(dataset);
  H5Tclose(datatype);
  H5Sclose(dataspace);
  if(cl) close();
}


void Checkpoint::read(BasisSet & basis) {
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }
  if(!exist("basis.nucs"))
    throw std::runtime_error("Checkpoint does not have nuclei!\n");
  if(!exist("basis.contr"))
    throw std::runtime_error("Checkpoint does not have contractions!\n");
  if(!exist("basis.data"))
    throw std::runtime_error("Checkpoint does not have shell data!\n");

  hid_t dataset, datatype, dataspace;
  hsize_t dims[1];

  // ***** First, read in the nuclei *****

  // Open the dataset.
  dataset = H5Dopen (file, "basis.nucs", H5P_DEFAULT);

  // Create the datatype
  datatype = H5Tcreate(H5T_COMPOUND, sizeof(nuc_t));
  H5Tinsert(datatype, "ind", HOFFSET(nuc_t, ind), H5T_NATIVE_HSIZE);
  H5Tinsert(datatype, "rx", HOFFSET(nuc_t, rx), H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "ry", HOFFSET(nuc_t, ry), H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "rz", HOFFSET(nuc_t, rz), H5T_NATIVE_DOUBLE);

  H5Tinsert(datatype, "bsse", HOFFSET(nuc_t, bsse), H5T_NATIVE_HBOOL);
  H5Tinsert(datatype, "Z", HOFFSET(nuc_t, Z), H5T_NATIVE_INT);

  hid_t symtype=H5Tcopy(H5T_C_S1);
  H5Tset_size(symtype,SYMLEN);
  H5Tinsert(datatype, "sym", HOFFSET(nuc_t, sym), symtype);

  // Get dataspace
  dataspace = H5Dget_space(dataset);
  // Get the number of nuclei
  H5Sget_simple_extent_dims(dataspace,dims,NULL);
  size_t Nnuc=dims[0];

  // Read the data.
  nuc_t nucs[Nnuc];
  H5Dread(dataset,datatype,H5S_ALL,H5S_ALL,H5P_DEFAULT,nucs);

  // Free memory.
  H5Sclose(dataspace);
  H5Tclose(datatype);
  H5Tclose(symtype);
  H5Dclose(dataset);

  // ***** Then, read the contrations *****

  // Open the dataset.
  dataset = H5Dopen (file, "basis.contr", H5P_DEFAULT);

  // Create datatype
  hid_t contrdata = H5Tcreate (H5T_COMPOUND, sizeof(contr_t));
  H5Tinsert(contrdata, "c", HOFFSET(contr_t, c), H5T_NATIVE_DOUBLE);
  H5Tinsert(contrdata, "z", HOFFSET(contr_t, z), H5T_NATIVE_DOUBLE);
  datatype = H5Tvlen_create(contrdata);

  // Get dataspace
  dataspace = H5Dget_space(dataset);
  // Get the number of shells
  H5Sget_simple_extent_dims(dataspace,dims,NULL);
  size_t Nsh=dims[0];

  // Read the data.
  hvl_t cntrs[Nsh];
  H5Dread(dataset,datatype,H5S_ALL,H5S_ALL,H5P_DEFAULT,cntrs);

  // Free memory.
  H5Sclose(dataspace);
  H5Tclose(datatype);
  H5Tclose(contrdata);
  H5Dclose(dataset);

  // ***** and the shell info *****

  // Open the dataset.
  dataset = H5Dopen (file, "basis.data", H5P_DEFAULT);

  // and the data type
  datatype = H5Tcreate(H5T_COMPOUND, sizeof(shell_data_t));

  H5Tinsert(datatype, "indstart", HOFFSET(shell_data_t, indstart), H5T_NATIVE_HSIZE);
  H5Tinsert(datatype, "am", HOFFSET(shell_data_t, am), H5T_NATIVE_INT);
  H5Tinsert(datatype, "uselm", HOFFSET(shell_data_t, uselm), H5T_NATIVE_HBOOL);
  H5Tinsert(datatype, "cenind", HOFFSET(shell_data_t, cenind), H5T_NATIVE_HSIZE);

  // Get dataspace
  dataspace = H5Dget_space(dataset);
  // Get the number of shells
  H5Sget_simple_extent_dims(dataspace,dims,NULL);
  if(Nsh!=dims[0])
    throw std::runtime_error("Number of shells does not equal amount of contractions!\n");

  shell_data_t shdata[Nsh];
  H5Dread(dataset,datatype,H5S_ALL,H5S_ALL,H5P_DEFAULT,shdata);

  // Free memory.
  H5Sclose(dataspace);
  H5Tclose(datatype);
  H5Dclose(dataset);

  // ***** Store the data *****

  std::vector< std::vector<contr_t> > contrs(Nsh);
  // Store data
  for(size_t i=0;i<Nsh;i++) {
    // Pointer to entry
    contr_t *p=(contr_t *) cntrs[i].p;
    for(size_t j=0;j<cntrs[i].len;j++) {
      contr_t hlp;
      hlp.z=p[j].z;
      hlp.c=p[j].c;
      contrs[i].push_back(hlp);
    }
  }
  for(size_t i=0;i<Nsh;i++)
    free(cntrs[i].p);

  // Initialize the basis set
  basis=BasisSet();
  // Add the nuclei
  for(size_t i=0;i<Nnuc;i++) {
    nucleus_t nuc;

    nuc.ind=nucs[i].ind;
    nuc.r.x=nucs[i].rx;
    nuc.r.y=nucs[i].ry;
    nuc.r.z=nucs[i].rz;
    nuc.Z=nucs[i].Z;
    nuc.bsse=nucs[i].bsse;
    nuc.symbol=nucs[i].sym;

    basis.add_nucleus(nuc);
  }

  // Add the shells
  for(size_t i=0;i<Nsh;i++) {
    basis.add_shell(shdata[i].cenind,shdata[i].am,shdata[i].uselm,contrs[i],false);
  }

  // Finalize the basis, no conversion of coefficients or normalization
  basis.finalize(false,false);

  if(cl) close();
}

void Checkpoint::write(const energy_t & en) {
  CHECK_WRITE();
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Remove possible existing entry
  remove("Energy");

  // Create a dataspace.
  hid_t dataspace=H5Screate(H5S_SCALAR);

  // Create a datatype.
  hid_t datatype;
  datatype = H5Tcreate(H5T_COMPOUND, sizeof(energy_t));
  H5Tinsert(datatype, "Ecoul", HOFFSET(energy_t, Ecoul), H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "Ekin" , HOFFSET(energy_t, Ekin) , H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "Enuca", HOFFSET(energy_t, Enuca), H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "Exc"  , HOFFSET(energy_t, Exc)  , H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "Eone" , HOFFSET(energy_t, Eone) , H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "Eel"  , HOFFSET(energy_t, Eel)  , H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "Enucr", HOFFSET(energy_t, Enucr), H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "Enl",   HOFFSET(energy_t, Enl),   H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "Esic",   HOFFSET(energy_t, Esic),   H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "E"    , HOFFSET(energy_t, E)    , H5T_NATIVE_DOUBLE);

  // Create the dataset using the defined dataspace and datatype, and
  // default dataset creation properties.
  hid_t dataset=H5Dcreate(file,"Energy",datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Write the data to the file.
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &en);

  // Close everything.
  H5Dclose(dataset);
  H5Tclose(datatype);
  H5Sclose(dataspace);
  if(cl) close();
}

void Checkpoint::read(energy_t & en) {
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }
  std::string name="Energy";
  CHECK_EXIST();

  // Open the dataset.
  hid_t dataset = H5Dopen (file, name.c_str(), H5P_DEFAULT);

  // Get the data type
  hid_t datatype;
  datatype = H5Tcreate(H5T_COMPOUND, sizeof(energy_t));
  H5Tinsert(datatype, "Ecoul", HOFFSET(energy_t, Ecoul), H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "Ekin" , HOFFSET(energy_t, Ekin) , H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "Enuca", HOFFSET(energy_t, Enuca), H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "Exc"  , HOFFSET(energy_t, Exc)  , H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "Eone" , HOFFSET(energy_t, Eone) , H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "Eel"  , HOFFSET(energy_t, Eel)  , H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "Enucr", HOFFSET(energy_t, Enucr), H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "Enl",   HOFFSET(energy_t, Enl),   H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "Esic",  HOFFSET(energy_t, Esic) , H5T_NATIVE_DOUBLE);
  H5Tinsert(datatype, "E"    , HOFFSET(energy_t, E)    , H5T_NATIVE_DOUBLE);

  // Get dataspace
  hid_t dataspace = H5Dget_space(dataset);

  // Get type
  H5S_class_t type = H5Sget_simple_extent_type(dataspace);
  if(type!=H5S_SCALAR)
    throw std::runtime_error("Error - dataspace is not of scalar type!\n");

  // Read
  H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &en);

  // Close dataspace
  H5Sclose(dataspace);
  // Close datatype
  H5Tclose(datatype);
  // Close dataset
  H5Dclose(dataset);
  if(cl) close();
}


void Checkpoint::write(const std::string & name, double val) {
  CHECK_WRITE();
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Remove possible existing entry
  remove(name);

  // Create a dataspace.
  hid_t dataspace=H5Screate(H5S_SCALAR);

  // Create a datatype.
  hid_t datatype=H5Tcopy(H5T_NATIVE_DOUBLE);

  // Create the dataset using the defined dataspace and datatype, and
  // default dataset creation properties.
  hid_t dataset=H5Dcreate(file,name.c_str(),datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Write the data to the file.
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &val);

  // Close everything.
  H5Dclose(dataset);
  H5Tclose(datatype);
  H5Sclose(dataspace);
  if(cl) close();
}

void Checkpoint::read(const std::string & name, double & v) {
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }
  CHECK_EXIST();

  // Open the dataset.
  hid_t dataset = H5Dopen (file, name.c_str(), H5P_DEFAULT);

  // Get the data type
  hid_t datatype  = H5Dget_type(dataset);

  // Get the class info
  hid_t hclass=H5Tget_class(datatype);

  if(hclass!=H5T_FLOAT) {
    std::ostringstream oss;
    oss << "Error - " << name << " is not a floating point value!\n";
    ERROR_INFO();
    throw std::runtime_error(oss.str());
  }

  // Get dataspace
  hid_t dataspace = H5Dget_space(dataset);
  // Get type
  H5S_class_t type = H5Sget_simple_extent_type(dataspace);
  if(type!=H5S_SCALAR)
    throw std::runtime_error("Error - dataspace is not of scalar type!\n");

  // Read
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);

  // Close dataspace
  H5Sclose(dataspace);
  // Close datatype
  H5Tclose(datatype);
  // Close dataset
  H5Dclose(dataset);
  if(cl) close();
}


void Checkpoint::write(const std::string & name, int val) {
  CHECK_WRITE();
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Remove possible existing entry
  remove(name);

  // Create a dataspace.
  hid_t dataspace=H5Screate(H5S_SCALAR);

  // Create a datatype.
  hid_t datatype=H5Tcopy(H5T_NATIVE_INT);

  // Create the dataset using the defined dataspace and datatype, and
  // default dataset creation properties.
  hid_t dataset=H5Dcreate(file,name.c_str(),datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Write the data to the file.
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &val);

  // Close everything.
  H5Dclose(dataset);
  H5Tclose(datatype);
  H5Sclose(dataspace);
  if(cl) close();
}

void Checkpoint::read(const std::string & name, int & v) {
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }
  CHECK_EXIST();

  // Open the dataset.
  hid_t dataset = H5Dopen (file, name.c_str(), H5P_DEFAULT);

  // Get the data type
  hid_t datatype  = H5Dget_type(dataset);

  // Get dataspace
  hid_t dataspace = H5Dget_space(dataset);

  // Get the class info
  hid_t hclass=H5Tget_class(datatype);
  if(hclass!=H5T_INTEGER)
    throw std::runtime_error("Error - datatype is not integer!\n");

  // Get type
  H5S_class_t type = H5Sget_simple_extent_type(dataspace);
  if(type!=H5S_SCALAR)
    throw std::runtime_error("Error - dataspace is not of scalar type!\n");

  // Read
  H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);

  // Close dataspace
  H5Sclose(dataspace);
  // Close datatype
  H5Tclose(datatype);
  // Close dataset
  H5Dclose(dataset);
  if(cl) close();
}

void Checkpoint::write(const std::string & name, hsize_t val) {
  CHECK_WRITE();
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Remove possible existing entry
  remove(name);

  // Create a dataspace.
  hid_t dataspace=H5Screate(H5S_SCALAR);

  // Create a datatype.
  hid_t datatype=H5Tcopy(H5T_NATIVE_HSIZE);

  // Create the dataset using the defined dataspace and datatype, and
  // default dataset creation properties.
  hid_t dataset=H5Dcreate(file,name.c_str(),datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Write the data to the file.
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &val);

  // Close everything.
  H5Dclose(dataset);
  H5Tclose(datatype);
  H5Sclose(dataspace);
  if(cl) close();
}

void Checkpoint::read(const std::string & name, hsize_t & v) {
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }
  CHECK_EXIST();

  // Open the dataset.
  hid_t dataset = H5Dopen (file, name.c_str(), H5P_DEFAULT);

  // Get the data type
  hid_t datatype  = H5Dget_type(dataset);

  // Get dataspace
  hid_t dataspace = H5Dget_space(dataset);

  // Get the class info
  hid_t hclass=H5Tget_class(datatype);
  if(hclass!=H5T_INTEGER)
    throw std::runtime_error("Error - datatype is not integer!\n");

  // Get type
  H5S_class_t type = H5Sget_simple_extent_type(dataspace);
  if(type!=H5S_SCALAR)
    throw std::runtime_error("Error - dataspace is not of scalar type!\n");

  // Read
  H5Dread(dataset, H5T_NATIVE_HSIZE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);

  // Close dataspace
  H5Sclose(dataspace);
  // Close datatype
  H5Tclose(datatype);
  // Close dataset
  H5Dclose(dataset);
  if(cl) close();
}

void Checkpoint::write(const std::string & name, bool val) {
  hbool_t tmp;
  tmp=val;
  write_hbool(name,tmp);
}

void Checkpoint::read(const std::string & name, bool & v) {
  hbool_t tmp;
  read_hbool(name,tmp);
  v=tmp;
}

void Checkpoint::write_hbool(const std::string & name, hbool_t val) {
  CHECK_WRITE();
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Remove possibly existing entry
  remove(name);

  // Create a dataspace.
  hid_t dataspace=H5Screate(H5S_SCALAR);

  // Create a datatype.
  hid_t datatype=H5Tcopy(H5T_NATIVE_HBOOL);

  // Create the dataset using the defined dataspace and datatype, and
  // default dataset creation properties.
  hid_t dataset=H5Dcreate(file,name.c_str(),datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Write the data to the file.
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &val);

  // Close everything.
  H5Dclose(dataset);
  H5Tclose(datatype);
  H5Sclose(dataspace);
  if(cl) close();
}

void Checkpoint::read_hbool(const std::string & name, hbool_t & v) {
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }
  CHECK_EXIST();

  // Open the dataset.
  hid_t dataset = H5Dopen (file, name.c_str(), H5P_DEFAULT);

  // Get the data type
  hid_t datatype  = H5Dget_type(dataset);

  // Get dataspace
  hid_t dataspace = H5Dget_space(dataset);

  // Get type
  H5S_class_t type = H5Sget_simple_extent_type(dataspace);
  if(type!=H5S_SCALAR)
    throw std::runtime_error("Error - dataspace is not of scalar type!\n");

  // Read
  H5Dread(dataset, H5T_NATIVE_HBOOL, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);

  // Close dataspace
  H5Sclose(dataspace);
  // Close datatype
  H5Tclose(datatype);
  // Close dataset
  H5Dclose(dataset);

  if(cl) close();
}

void Checkpoint::write(const std::string & name, const std::string & val) {
  CHECK_WRITE();
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }

  // Remove possibly existing entry
  remove(name);

  // Dimensions of the vector
  hsize_t dims[1];
  dims[0]=val.size()+1;

  // Create a dataspace.
  hid_t dataspace=H5Screate_simple(1,dims,NULL);

  // Create a datatype.
  hid_t datatype=H5Tcopy(H5T_NATIVE_CHAR);

  // Create the dataset using the defined dataspace and datatype, and
  // default dataset creation properties.
  hid_t dataset=H5Dcreate(file,name.c_str(),datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Write the data to the file.
  H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, val.c_str());

  // Close everything.
  H5Dclose(dataset);
  H5Tclose(datatype);
  H5Sclose(dataspace);
  if(cl) close();
}

void Checkpoint::read(const std::string & name, std::string & val) {
  bool cl=false;
  if(!opend) {
    open();
    cl=true;
  }
  CHECK_EXIST();

  // Open the dataset.
  hid_t dataset = H5Dopen (file, name.c_str(), H5P_DEFAULT);

  // Get the data type
  hid_t datatype  = H5Dget_type(dataset);

  // Get the class info
  hid_t hclass=H5Tget_class(datatype);

  if(hclass!=H5T_INTEGER) {
    std::ostringstream oss;
    oss << "Error - " << name << " does not consist of characters!\n";
    ERROR_INFO();
    throw std::runtime_error(oss.str());
  }

  // Get dataspace
  hid_t dataspace = H5Dget_space(dataset);
  // Get number of dimensions
  int ndim = H5Sget_simple_extent_ndims(dataspace);
  if(ndim!=1) {
    std::ostringstream oss;
    oss << "Error - " << name << " should have dimension 1, instead dimension is " << ndim << "!\n";
    ERROR_INFO();
    throw std::runtime_error(oss.str());
  }

  // Get the size of the matrix
  hsize_t dims[ndim];
  H5Sget_simple_extent_dims(dataspace,dims,NULL);

  // Allocate work memory
  char *wrk=(char *)malloc(dims[0]*sizeof(char));
  H5Dread(dataset, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, wrk);
  val=std::string(wrk);
  free(wrk);

  // Close dataspace
  H5Sclose(dataspace);
  // Close datatype
  H5Tclose(datatype);
  // Close dataset
  H5Dclose(dataset);
  if(cl) close();
}

bool file_exists(const std::string & name) {
  std::ifstream file(name.c_str());
  return file.good();
}

std::string get_cwd() {
  // Initial array size
  size_t m=1024;
  char *p=(char *) malloc(m);
  char *r=NULL;

  while(true) {
    // Get cwd in array p
    r=getcwd(p,m);
    // Success?
    if(r==p)
      break;
    
    // Failed, increase m
    m*=2;
    p=(char *) realloc(p,m);
  }

  std::string cwd(p);
  free(p);
  return cwd;
}

void change_dir(std::string dir, bool create) {
  if(create) {
    std::string cmd="mkdir -p "+dir;
    int err=system(cmd.c_str());
    if(err) {
      std::ostringstream oss;
      oss << "Could not create directory \"" << dir << "\".\n";
      throw std::runtime_error(oss.str());
    }
  }

  // Go to directory
  int direrr=chdir(dir.c_str());
  if(direrr) {
    std::ostringstream oss;
    oss << "Could not change to directory \"" << dir << "\".\n";
    throw std::runtime_error(oss.str());
  }
}

std::string tempname() {
  // Get random file name
  char *tmpname=tempnam("./",".chk");
  std::string name(tmpname);
  free(tmpname);

  return name;
}
