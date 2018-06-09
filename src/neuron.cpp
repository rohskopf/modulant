/*
 net.cpp

 Copyright (c) 2018 Andrew Rohskopf

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

/* ----------------------------------------------------------------------
    The Neuron class encodes a single neuron in a network.

	Variables
		nnl: number of neurons in next layer, determines number of output weights
		weights: an array of output weights, of size nnl
		output: neuron output after activation function
		activation: type of activation function

   ------------------------------------------------------------------------- */

#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include "mpi.h"
#include <math.h>       /* sqrt */
#include <random>

#include "neuron.h"
#include "input.h"
#include "memory.h"

using namespace std;

using namespace NN_NS;

Neuron::Neuron()
{

    //printf("    Neuron allocated!\n");

}

Neuron::~Neuron() 
{
    //memory->deallocate(weights);


};

void Neuron::print()
{

	printf("  Activation: %c\n", activation);
    printf("  Output Weights: ");
    for (int w=0; w<nnl; w++){
    printf("%f ", weights[w]);
    }
    printf("\n");

    
}

void Neuron::initialize()
{
    memory->allocate(weights, nnl);
	memory->allocate(dWeights, nnl);

}

void Neuron::finalize()
{
    memory->deallocate(weights);
	memory->deallocate(dWeights);

}


/* ----------------------------------------------------------------------
	Activation function takes an input an sets the neuron output.
   ------------------------------------------------------------------------- */


void Neuron::activate()
{

    if (activation == 'l'){
		output=input;
	}
	else if (activation == 't'){
		output = atan(input);
	}

}


/* ----------------------------------------------------------------------
	Derivative of activation function
   ------------------------------------------------------------------------- */


void Neuron::dActivate()
{

	//printf("Input: %f\n", input);

    if (activation == 'l'){
		dOutput = 1.0;
	}
	else if (activation == 't'){
		dOutput = (1.0/cos(input))*(1.0/cos(input)); 
	}

}
