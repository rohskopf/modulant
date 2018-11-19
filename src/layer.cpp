/*
 net.cpp

 Copyright (c) 2018 Andrew Rohskopf

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

/* ----------------------------------------------------------------------
    The Layer class does the following:

    Stores information of the layer (number of units).
    Stores all neuron objects for that layer.
------------------------------------------------------------------------- */

#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include "mpi.h"
#include <math.h>       /* sqrt */
#include <random>

#include "layer.h"
#include "input.h"
#include "memory.h"
#include "neuron.h"

using namespace std;

using namespace MODULANT_NS;

Layer::Layer()
{


}

Layer::~Layer() 
{

    memory->deallocate(neurons);


};

void Layer::initialize()
{

    //printf("  Initializing layer with %d neurons.\n", nunits);

    memory->allocate(neurons, nunits);
    
}
