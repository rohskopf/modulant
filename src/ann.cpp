/*
 ann.cpp

 Copyright (c) 2018 Andrew Rohskopf

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

/* ----------------------------------------------------------------------
    The Ann class represents a single atomic NN.
------------------------------------------------------------------------- */

#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include "mpi.h"
#include <math.h>       /* sqrt */
#include <random>

#include "input.h"
#include "memory.h"

using namespace std;

using namespace MODULANT_NS;

Ann::Ann()
{

}

Ann::~Ann() 
{
    //fclose(fh_debug);
	
	//printf("annType: %d\n", annType);
	/*
	if (annType==3){
		printf("YUP!\n");
	}
	*/
	
	memory->deallocate(layers);
	//memory->deallocate(inputDerivatives);
	/*
	if (fh_debug!=0){
		fclose(fh_debug);
	}
	*/


};

/* ----------------------------------------------------------------------
    Open a debug file depending on type of ANN
------------------------------------------------------------------------- */

void Ann::openDebug()
{

	if (annType==0){
		char debug[64];
		sprintf (debug, "debug/systems/config/ann/ANN%d_NET%d_PROC%d", n+1,m+1,rank);
		//fh_debug = fopen(debug, "w");
	}
	else if (annType== 1){
		char debug[64];
		sprintf (debug, "debug/systems/config/ann/ANNSUM_NET%d_TYPE%d_PROC%d",m+1,type,rank);
		//fh_debug = fopen(debug, "w");

	}
	else if (annType==2){
		char debug[64];
		sprintf (debug, "debug/systems/config/ann/ERRGRADS_TYPE%d_PROC%d",type,rank);
		//fh_debug = fopen(debug, "w");
	}
	else if (annType==3){
		char debug[64];
		sprintf (debug, "debug/systems/config/ann/WEIGHTS_TYPE%d_PROC%d",type,rank);
		//fh_debug = fopen(debug, "w");
	}
	else if (annType==4){
		char debug[128];
		sprintf (debug, "debug/systems/config/ann/dfdw/m%da%dt%dn%dp%d",m,a,type,n,rank);
		// Couldn't get debug file to open... will try later
		//printf("m,t,n,a: %d,%d,%d,%d\n",m,type,n,a);
		//cout << debug << endl;
		//fh_debug = fopen(debug, "w");
		/*
		if (fh_debug==0){
			printf("File not opened!\n");
		}
		*/
		//fclose(fh_debug);
		//printf("after-----\n");
	}
	else if (annType==5){

	}
}

/* ----------------------------------------------------------------------
    Build the network according to input settings.
	Sets bias output values = 1.0
	Set network weights according to WEIGHTS file.
------------------------------------------------------------------------- */

void Ann::initialize()
{

	//fprintf(fh_debug, "Atom type: %d\n", type);
	
	//fprintf(fh_debug,"Building ANN\n");
    //printf(" Building the network...\n");


    //printf(" %d input neurons.\n", structure[0]);
    //printf(" %d total layers.\n", nhl+2);

    memory->allocate(layers,nhl+2);


	char activation;
	nweights = 0;
    for (int l=0; l<nhl+2; l++){

        //if (l!=nhl+1){
            layers[l].nunits = structure[l]+1; // Input and hidden layers have bias neuron
            //std::cout << line << std::endl;
        //}
        //else if (l=nhl+1) layers[l].nunits = structure[l]; // No bias neuron for output layer

        layers[l].initialize(); // Allocate the neurons in this layer

		activation = activations[l];
		
		//std::cout << activation << std::endl;

		//printf("activation: %c\n", activation);

        for (int j=0; j<structure[l]+1; j++){
    


			/*
			for (int a=0; a<5; a++){
				for (int b=0; b<5; b++){
					printf("%f\n", layers[l].neurons[j].test[a][b]);
				}
			}
			*/

            if (l != nhl+1){ // If we aren't on the last layer

                layers[l].neurons[j].nnl = structure[l+1]; // Number of next layer neurons (not including bias)
				layers[l].neurons[j].annType = annType;
				layers[l].neurons[j].l = l;
				layers[l].neurons[j].j = j;
				layers[l].neurons[j].type = type;
				layers[l].neurons[j].n = n;
				layers[l].neurons[j].m = m;

                layers[l].neurons[j].initialize(); // Allocate the output weights for this neuron
				//layers[l].neurons[j].print();

				//layers[l].neurons[j].allocateD2E(structure[0]);
                for (int k=0; k<structure[l+1]; k++){

					nweights++;
                }

            }

			if (l==nhl+1){
				layers[l].neurons[j].nnl = 0;
				layers[l].neurons[j].annType=annType;
				layers[l].neurons[j].l=l;
				layers[l].neurons[j].j=j;
				layers[l].neurons[j].type=type;
				layers[l].neurons[j].n=n;
				layers[l].neurons[j].m=m;
				layers[l].neurons[j].initialize();
			}
			// d2e needs to be allocated for output neuron as well since it will be deallocated later

			//printf("before\n");
			layers[l].neurons[j].allocateDodi(structure[0]); // allocate output derivatives wrt inputs
			layers[l].neurons[j].allocateD2E(structure[0]);
			//printf("after\n");

            /*
            else if (k==nhl+1){
                layers[k].neurons[n].nnl=0;
            }
            */

			// Set activation functions
			
			//for (int j=0; j<structure[l]+1; j++){

				if (l < nhl+1){ // if we are not on last layer
					if (j < structure[l]){
						layers[l].neurons[j].activation = activation;
					}
				    else if (j==structure[l]){ // bias neuron
						layers[l].neurons[j].activation = 'b';
					}
				}

				else if (l == nhl+1){
					if (j < structure[l]){
						layers[l].neurons[j].activation = activation;
					}
					
				}

		
			//}

        }
    }

    // Set the bias neurons to have an output of 1.0

    for (int l=0; l<nhl+1; l++){ // bias neurons only exist in input and hidden layers
        layers[l].neurons[structure[l]].output = 1.0;
    }



    
}

/*-----------------------------------------------------------------------
    Zero the weight derivatives for all neurons, for the next epoch.
------------------------------------------------------------------------- */

void Ann::zero()
{

	for (int l=0; l<nhl+1; l++){
		//printf("l: %d\n", l);
		for (int j=0; j<structure[l]+1; j++){
			//printf("j: %d\n", j);
			for (int k=0; k<structure[l+1]; k++){
				//layers[l].neurons[j].dWeightsp[k] = 0.0;
				layers[l].neurons[j].gradientsp[k] = 0.0;
			}
		}
	}
}

/* ----------------------------------------------------------------------
    Feed forward starting from the inputs.
    Inputs:
        d: Index of training sample to feed forward with
------------------------------------------------------------------------- */

void Ann::feedForward()
{

    //printf("Feeding forward!\n");
	//fprintf(fh_debug, "Feeding forward.\n");

    // Feed forward

    double sumInputs;
    for (int l=1; l<nhl+2; l++){

        for (int j=0; j<structure[l]; j++){
            sumInputs=0.0;
            //fprintf(fh_debug," Layer %d, neuron %d\n", l, j+1);

            // Loop through previous neurons (and bias), and their output weights
    
            for (int i=0; i<structure[l-1]+1; i++){
				//fprintf(fh_debug," Previous neuron (l,i): %d,%i\n", l,i);
                //fprintf(fh_debug," Previous weight: %f\n", layers[l-1].neurons[i].weights[j]);
                //fprintf(fh_debug," Previous output: %f\n", layers[l-1].neurons[i].output);
                sumInputs += layers[l-1].neurons[i].weights[j]*layers[l-1].neurons[i].output;
            }
			layers[l].neurons[j].input = sumInputs;

			// Activate the neuron (apply activation function)

            layers[l].neurons[j].activate();

            //fprintf(fh_debug," New output: %f\n", layers[l].neurons[j].output);
            
        }

    }

	energy = layers[nhl+1].neurons[0].output;
	//fprintf(fh_debug, "Energy: %f\n", layers[nhl+1].neurons[0].output);

}

/* ----------------------------------------------------------------------
    Feed forward starting from the inputs.
    Inputs:
------------------------------------------------------------------------- */

void Ann::feedForwardDerivatives()
{

	// First hidden layer
	double wij;
	double dodi;
	double dxdi;
	for (int j=0; j<structure[1]; j++){
		layers[1].neurons[j].dActivate();
		//fprintf(fh_debug, "Layer 1, neuron %d\n",j);
		// Loop through previous layer neurons
		for (int i=0; i<structure[0]; i++){
			wij = layers[0].neurons[i].weights[j];
			dodi = wij*layers[1].neurons[j].dOutput;
			dxdi = wij;
			layers[1].neurons[j].dodi[i] = dodi;
			layers[1].neurons[j].dxdi[i] = dxdi;
			//fprintf(fh_debug, " dodi: %f\n", dodi);
			//fprintf(fh_debug, " dxdi: %f\n", dxdi);
		}
	}

	// Next layers

	double xi; // for the neuron outputs
	double xi_x; // for the neuron inputs
	for (int s=0; s<structure[0]; s++){	
		for (int l=2; l<= nhl+1; l++){
			//fprintf(fh_debug,"l,s: %d,%d\n", l,s);
			for (int j=0; j<structure[l]; j++){
				//fprintf(fh_debug, " neuron %d\n", j);
				layers[l].neurons[j].dActivate();

				xi=0.0;
				xi_x=0.0;
				for (int i=0; i<structure[l-1]; i++){
					wij = layers[l-1].neurons[i].weights[j];
					xi += wij*layers[l-1].neurons[i].dodi[s];
					xi_x += wij*layers[l-1].neurons[i].dxdi[s];
				}
				layers[l].neurons[j].dodi[s] = layers[l].neurons[j].dOutput*xi;
				layers[l].neurons[j].dxdi[s] = xi_x;
				//fprintf(fh_debug, "  layers[%d].neurons[%d].dodi[%d]: %f\n", l,j,s,layers[l].neurons[j].dodi[s]);
				//fprintf(fh_debug, "  layers[%d].neurons[%d].dxdi[%d]: %f\n", l,j,s,layers[l].neurons[j].dxdi[s]);
			}
		}
	}

}


/* ----------------------------------------------------------------------
	Backpropogate to calculate gradient of ANN output.
	This is the derivative of the ANN output (energy) wrt weights.
   ------------------------------------------------------------------------- */

void Ann::backprop()
{

	//fprintf(fh_debug,"---Backpropagating for d=%d.\n", d);

	// Calculate error term for output layer neurons
	
	double output;
	double diff;
	double sigma;
	double x;
	for (int j=0; j<structure[nhl+1]; j++){
	    //fprintf(fh_debug, "Layer %d, neuron %d.\n", nhl+2, j+1);
		//printf("Neuron %d\n", n);
		output = layers[nhl+1].neurons[j].output;
		//fprintf(fh_debug, "output: %f\n", output);
		//diff = input->dout[n][d] - output;
		//diff = nn->dout[d*structure[nhl+1] + j] - output;
		//fprintf(fh_debug, "diff: %f\n", diff);
		//error += 0.5*diff*diff/input->nsamples;
		//fprintf(fh_debug, "error: %E\n", error);
		//printf("  Output: %f\n", output);
		//printf("  diff: %f\n", diff);
		// Calculate neuron output derivative
		layers[nhl+1].neurons[j].dActivate();
		// Calculate error term
		layers[nhl+1].neurons[j].sigma = layers[nhl+1].neurons[j].dOutput;	
		//fprintf(fh_debug,"sigma: %f\n", layers[nhl+1].neurons[j].dOutput);
		//Calculate gradient for last hidden layer
		for (int i=0; i<structure[nhl]+1; i++){
			//printf("m: %d\n", m);
			//x = layers[numLayers-1].neurons[m].weights[m]*layers[numLayers-1].neurons[m].output;
			x = layers[nhl].neurons[i].output;
			//printf("x: %f\n", x);
			//printf("layers[nhl+1].neurons[j].sigma: %f\n",layers[nhl+1].neurons[j].sigma);
			//printf("grad: %f\n", layers[nhl].neurons[i].gradientsp[j]);
			//fprintf(fh_debug,"  x: %f\n", x);
			//layers[nhl].neurons[i].dWeightsp[j] += (layers[nhl+1].neurons[j].sigma * x)/input->nsamples;
			layers[nhl].neurons[i].gradientsp[j] = layers[nhl+1].neurons[j].sigma * x;
			//fprintf(fh_debug,"  layers[nhl+1].neurons[j].sigma: %f\n", layers[nhl+1].neurons[j].sigma);
			//fprintf(fh_debug,"  gradient: %f\n", layers[nhl].neurons[i].gradientsp[j]);
		}
	}

	// Calculate error terms for other hidden layer neurons
		
	double sumSigma;
	for (int l=nhl; l > 0; l--){
		//printf("l: %d -----\n", l);
		for (int j=0; j<structure[l]; j++){
	    	//fprintf(fh_debug, "Layer %d, neuron %d.\n", l+1, j+1);
			//printf("j %d\n", n);
			// Calculate neuron output derivative
			layers[l].neurons[j].dActivate();
			// Calculate sum of weighted sigmas for downstream neurons
			sumSigma=0.0;
			for (int k=0; k<structure[l+1]; k++){
				//fprintf(fh_debug, " k: %d\n",k);
				//printf("m: %d\n", m);
				//fprintf(fh_debug," wjk: %f\n", layers[l].neurons[j].weights[k]);
				//fprintf(fh_debug," sigma_k: %f\n", layers[l+1].neurons[k].sigma);
				sumSigma += layers[l].neurons[j].weights[k] * layers[l+1].neurons[k].sigma;
			}
			//printf("sumSigma: %f\n", sumSigma);
			//printf("dOutput: %f\n", layers[k].neurons[n].dOutput);
			layers[l].neurons[j].sigma = layers[l].neurons[j].dOutput * sumSigma;
			//printf("sigma: %f\n", layers[k].neurons[n].sigma);
			//Calculate gradient
			for (int i=0; i<structure[l-1]+1; i++){
				//fprintf(fh_debug, " i: %d\n",i);
				//x = layers[l-1].neurons[i].weights[j]*layers[l-1].neurons[i].output;
				x = layers[l-1].neurons[i].output;
				//fprintf(fh_debug, " x: %f\n", x);
				//layers[l-1].neurons[i].dWeightsp[j] += (layers[l].neurons[j].sigma * x)/input->nsamples;
				layers[l-1].neurons[i].gradientsp[j] = layers[l].neurons[j].sigma * x;
				//fprintf(fh_debug,"  layers[l].neurons[j].sigma: %f\n", layers[l].neurons[j].sigma);
				//fprintf(fh_debug,"  gradient: %f\n", layers[l-1].neurons[i].gradientsp[j]);
			}
		}
	}
	
	

}

/* ----------------------------------------------------------------------
	Backpropogate to calculate gradient of network derivatives.
	This is the derivative of the derivative of ANN output (energy)
	wrt weights, wrt ANN inputs.
   ------------------------------------------------------------------------- */

void Ann::backpropDerivatives()
{

	double dxdg;
	double d1Output;
	double d2Output;
	double sumSigma;
	double sumdSigma;

	// First calculate dSigma_dG for each neuron
	
	for (int s=0; s<structure[0]; s++){

		// Start with output layer
		for (int j=0; j<structure[nhl+1]; j++){
			//fprintf(fh_debug, "Layer %d, neuron %d.\n", nhl+1, j);
			//printf("Neuron %d\n", n);
			layers[nhl+1].neurons[j].d2Activate();
			d2Output = layers[nhl+1].neurons[j].d2Output;
			dxdg = layers[nhl+1].neurons[j].dxdi[s];
			layers[nhl+1].neurons[j].dSigma_dG[s] = dxdg*d2Output;
			//fprintf(fh_debug, " output: %.10f\n", layers[nhl+1].neurons[j].output);
			//fprintf(fh_debug, " dOutput: %.10f\n", layers[nhl+1].neurons[j].dOutput);
			//fprintf(fh_debug, " d2Output: %.10f\n", d2Output);
			//fprintf(fh_debug, " dxdg: %f\n", dxdg);
		}
		// Back propagate through hidden layers
		for (int l=nhl; l > 0; l--){
			//fprintf(fh_debug,"l,s: %d,%d -----\n", l,s);
			for (int j=0; j<structure[l]; j++){
				//fprintf(fh_debug, " Neuron %d.\n", j);
				//printf("j %d\n", n);
				// Calculate neuron output 2nd derivative
				layers[l].neurons[j].d2Activate();
				d2Output = layers[l].neurons[j].d2Output;
				dxdg = layers[l].neurons[j].dxdi[s];
				d1Output = layers[l].neurons[j].dOutput;
				// Calculate sum of weighted sigmas and sigma derivative from downstream neurons
				sumSigma=0.0;
				sumdSigma=0.0;
				for (int k=0; k<structure[l+1]; k++){
					//fprintf(fh_debug, " k: %d\n",k);
					//printf("m: %d\n", m);
					//fprintf(fh_debug," wjk: %f\n", layers[l].neurons[j].weights[k]);
					//fprintf(fh_debug," sigma_k: %f\n", layers[l+1].neurons[k].sigma);
					sumSigma += layers[l].neurons[j].weights[k] * layers[l+1].neurons[k].sigma;
					sumdSigma += layers[l].neurons[j].weights[k] * layers[l+1].neurons[k].dSigma_dG[s];
				}

				layers[l].neurons[j].dSigma_dG[s] = dxdg*d2Output*sumSigma + d1Output*sumdSigma;
				//fprintf(fh_debug, "  sumSigma: %f\n", sumSigma);
				//fprintf(fh_debug, "  sumdSigma: %f\n", sumdSigma);
				// Calculate sum of weighted sigma derivatives from downstream n	

			}
		}
	}

	// Now calculate d2e_dwdg for each weight and network input
		
	double dSigma_dG;
	double oj;
	double sigmak;
	double dodi;
	for (int l=0; l<nhl+1; l++){
		//fprintf(fh_debug,"Layer %d.\n", l);
		for (int j=0; j<structure[l]+1; j++){ // include bias neuron as well
			//fprintf(fh_debug," j: %d\n", j);
			//layers[l].neurons[j].print();

			for (int k=0; k<structure[l+1]; k++){
				//printf("  k: %d\n", k);
				for (int s=0; s<structure[0]; s++){

					//fprintf(fh_debug, "l,j,k,s: %d,%d,%d,%d\n", l,j,k,s);
					dSigma_dG = layers[l+1].neurons[k].dSigma_dG[s];
					oj = layers[l].neurons[j].output;
					sigmak = layers[l+1].neurons[k].sigma;
					dodi = layers[l].neurons[j].dodi[s];
					layers[l].neurons[j].d2e_dwdg[k][s] = \
						(dSigma_dG*oj) + (sigmak*dodi);
						
				}
			}
		}
	}
	

}
