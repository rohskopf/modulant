/*
 layer.h

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
#include "memory.h"

using namespace std;

using namespace MODULANT_NS;

class Neuron
{
public:
    Neuron();
    ~Neuron();

    Memory * memory;

    // Declare member functions

    void print();
    void initialize(); // Allocate output weights, and other stuff based on ANN type
	void allocateDodi(int); // allocate derivatives of outputs wrt inputs
	void allocateD2E(int);  // allocate network output derivatives wrt inputs
    void finalize(); // Deallocate output weights
	void activate(); // Activation function
	void dActivate(); // Derivative of activation function
	void d2Activate(); // 2nd derivative of activation function

    int nnl=0; // number of neurons in next layer
	int annType; // ANN type that this neuron is associated with
	int neuronType; // 0 - normal activation neuron
					// 1 - bias neuron
	int type; // atom type that this neuron is associated with

    double *weights; // weights going out of neuron to next layer
	double *nGradients; // numerical gradients
	double *gradients; // backprop gradients
	double *gradientsp; // backprop gradients per proc
    double *dodi; // derivatives of output wrt inputs
	double *dxdi; // derivatives of unactivated intput wrt network inputs
	double *dSigma_dG; // derivatives of sigma wrt network inputs
	double **d2e_dwdg; // derivatives of gradients wrt network inputs
					   // d2e_dwdg[k][s] - derivative wrt kth weight and sth input
	
	double **test; // 2d test array

	double output; // neuron output
	double dOutput; // derivative of output
	double d2Output; // 2nd derivative of output
	double input; // sum of weighted inputs into neuron
    char activation; // activation function
	double sigma; // error term

	int n; // atom index
	int l; // layer index
	int j; // neuron index
	int m; // config index

};


