/*
 ann.h

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
#include "layer.h"

using namespace std;


class Ann
{
public:

	Ann();
	~Ann();

	FILE * fh_debug; // Debug file handle

	void openDebug();
	void initialize();
	void zero();
	void feedForward();
	void feedForwardDerivatives();
	void backprop();
	void backpropDerivatives();

	Memory * memory;

	int n; // atom indx
	int m; // config indx
	int a; // coordinate indx for df_dw
	int rank; // proc id
	int type; // atom type
	int annType; // type of ANN
				 // 0 - normal ANN describing an atom
				 // 1 - sum of ANNs, simply used to store the sum of all gradients in a config
				 // 2 - total error gradients, used to store total gradients
				 // 3 - stores the main weights, which all other weights (in type 0 ANNs) are pointers to
				 // 4 - stores derivatives of forces wrt weights
				 // 5 - stores sum of force derivatives term, summed over directions

    int nhl; // number of hidden layers
    int *structure; // structure of network, structure[k] = number of units in kth layer
	char *activations; // activation functions per layer
	int nweights; // number of weights
	double error; // objective function
	double energy; // energy of this atom
	double *d2E_dwdg; // 2nd derivative of ANN output wrt weights and ANN inputs
					  // d2E_dwg[s] - derivative wrt sth ANN input
	
	Layer *layers; // Stores in array of neuron objects for each layer


};


