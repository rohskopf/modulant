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

using namespace MODULANT_NS;

Neuron::Neuron()
{

    //printf("    Neuron allocated!\n");

}

Neuron::~Neuron() 
{

	

	
	if (annType==0){
		memory->deallocate(gradientsp);
		//printf("m,n,l,j: %d,%d,%d,%d\n",m,n,l,j);
		memory->deallocate(dodi);
		memory->deallocate(dxdi);
		memory->deallocate(dSigma_dG);
		memory->deallocate(d2e_dwdg);
		memory->deallocate(test);
	}
	if (annType==1){
		memory->deallocate(gradients);
	}
	if (annType==2){
		memory->deallocate(gradientsp);
		memory->deallocate(gradients);
	}
	if (annType==3){
		memory->deallocate(weights);
	}
	if (annType==4){
		memory->deallocate(gradients);
	}
	if (annType==5){
		//printf("Deallocating!\n");
		memory->deallocate(gradients);
	}


};

void Neuron::print()
{

	/*
	printf("  Activation: %c\n", activation);
    printf("  Output Weights: ");
    for (int w=0; w<nnl; w++){
	double testWeight;
    printf("%f ", weights[w]);
    }
    printf("\n");
	*/
	/*
	if (annType==0){
		printf("m,n,l,j=%d,%d,%d,%d\n",m,n,l,j);
		memory->deallocate(gradients);
	}
	*/

	printf("     m,n,l,j=%d,%d,%d,%d\n", m,n,l,j);

	for (int k=0; k<nnl; k++){
		for (int s=0; s<13; s++){
			printf("     d2e_dwdg[%d][%d]: %f\n", k,s,d2e_dwdg[k][s]);
		}
	}
    
}

void Neuron::initialize()
{
	//printf("nnl: %d\n", nnl);

	
	if (annType==0){
		memory->allocate(gradientsp,nnl);
	}
	if (annType == 1){
		memory->allocate(gradients,nnl);
	}
	if (annType ==2){
		memory->allocate(gradientsp,nnl);
		memory->allocate(gradients,nnl);
	}
	if (annType == 3){
    	memory->allocate(weights, nnl);

	}
	if (annType==4){
		memory->allocate(gradients,nnl);
	}
	if (annType==5){
		memory->allocate(gradients,nnl);
		for (int k=0; k<nnl; k++){
			gradients[k]=0.0;
		}
	}



}

/* ----------------------------------------------------------------------
    Allocate derivatives of neuron outputs and inputs wrt total network 
	inputs.

	numInputs - number of inputs neurons for the total neural network
------------------------------------------------------------------------- */

void Neuron::allocateDodi(int numInputs){

	//printf("m,n,l,j: %d,%d,%d,%d\n", m,n,l,j);
	//printf("  numInputs: %d\n", numInputs);
	memory->allocate(dodi,numInputs);
	memory->allocate(dxdi,numInputs);
	memory->allocate(dSigma_dG, numInputs);
}

/* ----------------------------------------------------------------------
    Allocate derivatives of gradients wrt network inputs

	numInputs - number of inputs neurons for the total neural network
------------------------------------------------------------------------- */

void Neuron::allocateD2E(int numInputs){
	//printf("nnl,numInputs: %d,%d\n", nnl,numInputs);
	memory->allocate(d2e_dwdg,nnl,numInputs);
	memory->allocate(test,nnl,numInputs);
	/*
	for (int k=0;k<nnl;k++){
		for (int s=0; s<numInputs; s++){
			d2e_dwdg[k][s]=1.0;
			//printf("m,l,j,k,s: %d,%d,%d,%d,%d\n",m,l,j,k,s);
		}
	}
	*/
}

void Neuron::finalize()
{

	printf("deallocating neuron %d of layer %d -----\n", j,l);
	printf("annType %d, atom type %d\n", annType, type);

	memory->deallocate(gradients);

	/*
	if (annType == 0){
		printf("m,n = %d,%d\n", m,n);
		printf("neuronType = %d\n", neuronType);
    	//memory->deallocate(weights);
		printf("deallcoating nGradients-----------\n");
		//memory->deallocate(nGradients);
		printf("deallocating gradients-------------\n");
		//memory->deallocate(gradients);
		printf("deallocating gradientsp------------\n");
		memory->deallocate(gradientsp);
		printf("deallocated\n");
	}
	else if (annType == 1){

		memory->deallocate(gradients);

	}
	else if (annType == 2){

		memory->deallocate(gradients);
		memory->deallocate(gradientsp);
	}
	else if (annType == 3){
		printf("Atom type %d\n", type);
    	memory->deallocate(weights);
		printf("successful deallocation\n");

	}
	*/

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
		output = tanh(input);
	}

}


/* ----------------------------------------------------------------------
	Derivative of activation function
	Note that output must already be calculated (saves time).
   ------------------------------------------------------------------------- */


void Neuron::dActivate()
{


    if (activation == 'l'){
		dOutput = 1.0;
	}
	else if (activation == 't'){
		dOutput = 1.0 - output*output;
	}

}

/* ----------------------------------------------------------------------
	2nd derivative of activation function
	Note that dOutput must already be calculated (saves time).
   ------------------------------------------------------------------------- */


void Neuron::d2Activate()
{


    if (activation == 'l'){
		d2Output = 0.0;
	}
	else if (activation == 't'){
		d2Output = -2.0*output*(dOutput);
	}

}
