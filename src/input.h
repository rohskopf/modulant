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

using namespace std;

namespace NN_NS
{
  class Input: protected Pointers
  {
  public:
    Input(class NN *);
    ~Input();

    FILE * fh_debug; // Debug file handle

    // Declare member functions

    void readinput(); // function to read INPUT file
    void readData(); // function to read training data

    // readinput() variables

    int nhl; // number of hidden layers
    int *structure; // structure of network, structure[k] = number of units in kth layer
    char *activations; // activation function of each layer
	int nsamples; // number of training samples
    double **din; // input training data
    double *tin; // input training data
    double **dout; // output training data
    double *tout; // output training data


  };
}

