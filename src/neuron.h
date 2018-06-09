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

using namespace NN_NS;

class Neuron
{
public:
    Neuron();
    ~Neuron();

    Memory * memory;

    // Declare member functions

    void print();
    void initialize(); // Allocate output weights
    void finalize(); // Deallocate output weights
	void activate(); // Activation function
	void dActivate(); // Derivative of activation function

    int nnl=0; // number of neurons in next layer
    double *weights; // weights going out of neuron to next layer
	double *dWeights; // incremental derivative of output weights
    double output; // neuron output
	double dOutput; // derivative of output
	double input; // sum of weighted inputs into neuron
    char activation; // activation function
	double sigma; // error term


};


