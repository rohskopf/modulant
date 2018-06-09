/*
 net.h

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

namespace NN_NS
{
  class Net: protected Pointers
  {
  public:
    Net(class NN *);
    ~Net();

    FILE * fh_debug; // Debug file handle

    // Declare member functions

    void build(); // Builds the network
    void feedForward(int);
    void backprop(int);
    void zeroGrad(); // zero the weight derivatives on all neurons

    int nhl; // number of hidden layers
    int *structure; // structure of network, structure[k] = number of units in kth layer

    Layer *layers; // Stores in array of neuron objects for each layer


  };
}

