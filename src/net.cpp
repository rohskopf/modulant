/*
 net.cpp

 Copyright (c) 2018 Andrew Rohskopf

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

/* ----------------------------------------------------------------------
    The Net class does the following:

    Stores information about the entire network, and neuron objects.
    Requires a wrapper (Input class) to provide info for allocation.
------------------------------------------------------------------------- */

#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include "mpi.h"
#include <math.h>       /* sqrt */
#include <random>

#include "net.h"
#include "input.h"
#include "memory.h"

using namespace std;

using namespace NN_NS;

Net::Net(NN *nn) : Pointers(nn) {
    fh_debug = fopen("DEBUG_net", "w");
}

Net::~Net() 
{

    fclose(fh_debug);
    memory->deallocate(layers);

};

/* ----------------------------------------------------------------------
    Build the network according to the inputs.
------------------------------------------------------------------------- */

void Net::build()
{
    //printf(" Building the network...\n");

    nhl = input->nhl;
    structure = input->structure;

    //printf(" %d input neurons.\n", structure[0]);
    //printf(" %d total layers.\n", nhl+2);

    memory->allocate(layers,nhl+2);

    // Open WEIGHTS file
    ifstream fh("WEIGHTS");
    string line;
    double weight;
	char activation;
    for (int k=0; k<nhl+2; k++){

        if (k!=nhl+1){
            layers[k].nunits = structure[k]+1; // Non-output layers have bias neuron
            //std::cout << line << std::endl;
        }

        else if (k=nhl+1) layers[k].nunits = structure[k]; // No bias neuron for output layer

        layers[k].initialize(); // Allocate the neurons in this layer

		activation = input->activations[k];
		
		//std::cout << activation << std::endl;

		//printf("activation: %c\n", activation);

        for (int n=0; n<structure[k]+1; n++){
    
            getline(fh,line);
            //std::cout << line << std::endl;
            stringstream ss(line);

            if (k != nhl+1){ // If we aren't on the last layer

                layers[k].neurons[n].nnl = structure[k+1]+1;
                layers[k].neurons[n].initialize(); // Allocate the output weights for this neuron

                for (int w=0; w<structure[k+1]+1; w++){
                    ss >> weight;
                    //printf("weight: %f\n", weight);
                    layers[k].neurons[n].weights[w] = weight;
                }

            }

            /*
            else if (k==nhl+1){
                layers[k].neurons[n].nnl=0;
            }
            */

			// Set activation functions

			for (int n=0; n<structure[k]+1; n++){

				if (k < nhl+1){ // if we are not on last layer
					if (n < structure[k]){
						layers[k].neurons[n].activation = activation;
					}
				    else if (n==structure[k]){ // bias neuron
						layers[k].neurons[n].activation = 'b';
					}
				}

				else if (k == nhl+1){
					if (n < structure[k]){
						layers[k].neurons[n].activation = activation;
					}
					
				}

		
			}

        }
    }

    fh.close();

    // Set the bias neurons to have an output of 1.0

    for (int k=0; k<nhl+1; k++){ // bias neurons only exist in input and hidden layers
        layers[k].neurons[structure[k]].output = 1.0;
    }
    
}

/* ----------------------------------------------------------------------
    Zero the weight derivatives for all neurons
------------------------------------------------------------------------- */

void Net::zeroGrad()
{

	for (int l=0; l<input->nhl+1; l++){
		//printf("l: %d\n", l);
		for (int j=0; j<input->structure[l]+1; j++){
			//printf("j: %d\n", j);
			for (int k=0; k<input->structure[l+1]; k++){
				layers[l].neurons[j].dWeights[k] = 0.0;
			}
		}
	}
}

/* ----------------------------------------------------------------------
    Feed forward starting from the inputs.
    Inputs:
        d: Index of training sample to feed forward with
------------------------------------------------------------------------- */

void Net::feedForward(int d)
{

    printf("Feeding forward!\n");

    // Set inputs
	for (int i=0; i<input->structure[0]; i++){
		//printf("%f\n", input->tin[d*input->structure[0] + i]);
        layers[0].neurons[i].output = nn->din[d*input->structure[0] + i];
    }

    // Feed forward

    double sumInputs;
    for (int k=1; k<nhl+2; k++){

        for (int n=0; n<input->structure[k]; n++){
            sumInputs=0.0;
            //printf("Layer %d, neuron %d\n", k, n+1);

            // Loop through previous neurons (and bias), and their output weights
    
            for (int p=0; p<structure[k-1]+1; p++){
                //printf("Previous output: %f\n", layers[k-1].neurons[p].output);
                //printf("Previous weight: %f\n", layers[k-1].neurons[p].weights[n]);
                sumInputs += layers[k-1].neurons[p].weights[n]*layers[k-1].neurons[p].output;
            }
			layers[k].neurons[n].input = sumInputs;

			// Activate the neuron (apply activation function)

            layers[k].neurons[n].activate();

            //printf("New output: %f\n", layers[k].neurons[n].output);
            
        }

    }

}

/* ----------------------------------------------------------------------
	Backpropogate
   ------------------------------------------------------------------------- */

void Net::backprop(int d)
{

	int numLayers = input->nhl+1; // Actually number of layers-1
	printf("Backpropagating through %d layers.\n", numLayers);

	// Calculate error term for output layer neurons

	double output;
	double diff;
	double sigma;
	double x;
	for (int n=0; n<structure[input->nhl+1]; n++){
		//printf("Neuron %d\n", n);
		output = layers[numLayers].neurons[n].output;
		//diff = input->dout[n][d] - output;
		diff = nn->dout[d*structure[input->nhl+1] + n] - output;
		//printf("Output: %f\n", output);
		//printf("diff: %f\n", diff);
		// Calculate neuron output derivative
		layers[numLayers].neurons[n].dActivate();
		// Calculate error term
		layers[numLayers].neurons[n].sigma = layers[numLayers].neurons[n].dOutput * diff; 
		//printf("sigma: %f\n", layers[numLayers].neurons[n].sigma);
		//Calculate gradient
		for (int m=0; m<structure[input->nhl]+1; m++){
			//printf("m: %d\n", m);
			x = layers[numLayers-1].neurons[m].weights[n]*layers[numLayers-1].neurons[m].output;
			layers[numLayers-1].neurons[m].dWeights[n] += layers[numLayers].neurons[n].sigma * x;
			//printf("dWeight: %f\n", layers[numLayers-1].neurons[m].dWeights[n]);
		}
	}

	// Calculate error terms for hidden layer neurons
		
	double sumSigma;
	for (int k=input->nhl; k > 0; k--){
		//printf("l: %d -----\n", k);
		for (int n=0; n<input->structure[k]; n++){
			//printf("j %d\n", n);
			// Calculate neuron output derivative
			layers[k].neurons[n].dActivate();
			// Calculate sum of weighted sigmas for downstream neurons
			sumSigma=0.0;
			for (int m=0; m<input->structure[k+1]; m++){
				//printf("m: %d\n", m);
				//printf("weight: %f\n", layers[k].neurons[n].weights[m]);
				//printf("sigma: %f\n", layers[k+1].neurons[m].sigma);
				sumSigma += layers[k].neurons[n].weights[m] * layers[k+1].neurons[m].sigma;
			}
			//printf("sumSigma: %f\n", sumSigma);
			//printf("dOutput: %f\n", layers[k].neurons[n].dOutput);
			layers[k].neurons[n].sigma = layers[k].neurons[n].dOutput * sumSigma;
			//printf("sigma: %f\n", layers[k].neurons[n].sigma);
			//Calculate gradient
			for (int m=0; m<structure[k-1]+1; m++){
				//printf("i: %d\n", m);
				x = layers[k-1].neurons[m].weights[n]*layers[k-1].neurons[m].output;
				//printf("x: %f\n", x);
				layers[k-1].neurons[m].dWeights[n] += layers[k].neurons[n].sigma * x;
				//printf("dWeight: %f\n", layers[k-1].neurons[m].dWeights[n]);
			}
		}
	}
	

}

