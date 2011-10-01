/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "checkpoint.h"

Checkpoint::Checkpoint(const std::string & fname, bool write) {
  if(write)
    // Truncate existing file, using default creation and access properties.
    file=H5Fcreate(fname.c_str(),H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
  else
    file=H5Fopen(fname.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);
}

Checkpoint::~Checkpoint() {
}

void Checkpoint::write(const std::string & name, const arma::mat & m) {
  // Dimensions of the matrix
  hsize_t dims[2];
  dims[0]=m.n_rows;
  dims[1]=m.n_cols;

  // Create a dataspace.
  hid_t dataspace=H5Screate_simple(2,dims,NULL);

  // Create a datatype.
  hid_t datatype=H5Tcopy(H5T_NATIVE_DOUBLE);
  // Set little endian data order
  //  herr_t status=H5Tset_order(datatype, H5T_ORDER_LE);
  H5Tset_order(datatype, H5T_ORDER_LE);

  // Create the dataset using the defined dataspace and datatype, and
  // default dataset creation properties.
  hid_t dataset=H5Dcreate(file,name.c_str(),datatype,dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Write the data to the file.
  //  status=H5Dwrite(dataset,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,m.memptr());
  H5Dwrite(dataset,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,m.memptr());

  // Close everything.
  H5Dclose(dataset);
  H5Tclose(datatype);
  H5Sclose(dataspace);
}

void Checkpoint::read(const std::string & name, arma::mat & m) const {
  // Open the dataset.
  hid_t dataset = H5Dopen (file, name.c_str(), H5P_DEFAULT);

  // Get the data type
  hid_t datatype  = H5Dget_type(dataset);

  // Get the class info
  hid_t hclass=H5Tget_class(datatype);

  if(hclass!=H5T_FLOAT)
    throw std::runtime_error("Error - dataspace is not floating point!\n");

  // Get dataspace
  hid_t dataspace = H5Dget_space(dataset);
  // Get number of dimensions
  int ndim = H5Sget_simple_extent_ndims(dataspace);
  if(ndim!=2)
    throw std::runtime_error("Error - dataspace does not have dimension 2!\n");

  // Get the size of the matrix
  hsize_t dims[ndim];
  H5Sget_simple_extent_dims(dataspace,dims,NULL);

  // Allocate memory
  m.zeros(dims[0],dims[1]);
  //  herr_t status=H5Dread(dataset,H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, m.memptr());
  H5Dread(dataset,H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, m.memptr());

  // Close dataspace
  H5Sclose(dataspace);
  // Close datatype
  H5Tclose(datatype);
  // Close dataset
  H5Dclose(dataset);
}

void Checkpoint::write(const BasisSet & basis) {

}
