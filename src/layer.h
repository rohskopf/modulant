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
#include "neuron.h"
#include "memory.h"

using namespace std;

using namespace NN_NS;


class Layer
{

public:
    Layer();
    ~Layer();

    Memory * memory;

    FILE * fh_debug; // Debug file handle

    // Declare member functions

    void initialize();

    int nunits; // number of units (neurons) in this layer
    Neuron *neurons; // Neurons in this layer


};


