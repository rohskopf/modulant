/*
 input.h

 Copyright (c) 2018 Andrew Rohskopf

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <vector>
#include <string>
#include "mpi.h"

#include <iostream>
#include <new>
#include <cstdlib>
#include "pointers.h"
#include "config.h"

using namespace std;

namespace MODULANT_NS
{
  class Input: protected Pointers
  {
  public:
    Input(class MLT *);
    ~Input();

    FILE * fh_debug; // Debug file handle
	int rank; // MPI rank
	int cpp; // configs per proc

    // Declare member functions

    void readInput(); // function to read INPUT file
    void readconfigs(); // function to read CONFIGS file
    void readparams(); // function to read PARAMS file
    void neighbors(); // generate neighbor list for all configs
    void calcDescriptors(); // calculate descriptors for all configs
    double calc2body(int,int,int,double*); // calculate 2-body descriptor
    double calc3body(int,int,int,int,double*); // calculate 3-body descriptor

    // readinput() variables

    // Descriptor variables

    int nconfigs; // total number of configs
    int nrad; // number of radial descriptors in network
    int nang; // number of angular descriptors in network
	int nd; // number of descriptors per atom
    int ntypes; // number of atom types in system
    double rc; // cutoff
    int neighmax; // maximum number of neighbors

    // Neural network variables

    int nhl; // number of hidden layers
    int *structure; // structure of network, structure[k] = number of units in kth layer
    char *activations; // activation function of each layer
	int nsamples; // number of training samples
    double *tin; // input training data
    double *tout; // output training data
	double eta; // learning rate
	int nepochs; // number of epochs for training

    // readconfigs() variables

    int *natoms_all; // number of atoms in all configurations
    double ***positions_all; // positions of every atom in all configurations
    double ***forces_all; // forces on every atom in all configurations
	double *e0; // reference energies
    Config *configs; // array of configuration objects
	int *types; // atom types array for all atoms in all configs
	int natomsTot; // Total number of atoms in all configurations
	double wf; // force weight

	/*
	struct descriptor{
		double value;
		//int nbody;
		//int types[3];
		int m;
		int n;
		int s;
		int nbody;
		int types[3];
		//int n;
		//int s;
	};
	descriptor *descriptors; // 1D descriptors array for MPI
	*/

    // readparams() variables

    double **params2; // 2-body params
    double **params3; // 3-body params
    

  };
}

