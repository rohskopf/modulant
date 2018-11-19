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
#include "ann.h"
#include "config.h"


using namespace std;

namespace MODULANT_NS
{
  class Systems: protected Pointers
  {
  public:
    Systems(class MLT *);
    ~Systems();

    FILE * fh_debug; // Debug file handle

	void readConfigs(); // read and store configuration data
	void calcNeighborLists(); // calculate neighborlists for all configs
	void calcDescriptors(); // calculate descriptors for all atoms in all configs
	void readWeights(); // read the input weights
	void buildNets(); // build all total networks in all configs
	void zeroGrads(); // zero the total error grads
	void feedForwardAll(); // Feed forward all networks for all systems
	void calcError(); 
	void backpropAll(); // backpropagate all networks for all systems
	void updateWeights(); // update weights for all atoms in all systems
	void writeWeights(); // write the final weights

	int rank;
	int nconfigs; // number of configs on this proc
	int nconfigsTot; // number of configs on all procs
	int nd; // number of descriptors per atom
	double error; // total error
	double energy_error; // energy error
	double force_error; // force error
	
	Ann *weights; // ANN object storing the weights for each atom type

	Config *configs; // array of configs

	Ann *grads; // gradients of object function involving all systems 

  };

}
