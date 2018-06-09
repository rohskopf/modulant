/*
 nn.cpp

 Copyright (c) 2018 Andrew Rohskopf

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include <iostream>
#include <iomanip>
#include <vector>
#include "mpi.h"

#include "memory.h"
#include "timer.h"
#include "input.h"
#include "net.h"
//#include "neighbor.h"

using namespace std;

using namespace NN_NS;

NN::NN(int narg, char **arg)
{


    /************************** Set up MPI settings **************************/

    int color,key,global,local;
    MPI_Comm comm;

    // Split the communicators so that multiple instances can be run
    MPI_Comm_rank(MPI_COMM_WORLD, &global);
    color = global / 1; // Change "1" to 2 in order to use 2 procs per instance, etc..
    key = global; 
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &comm);
    MPI_Comm_rank(comm,&local);

    //  Get the number of processes.
    procs = MPI::COMM_WORLD.Get_size ( ); //Get_size gets number of processes (np) in communicator group
    //  Get the individual process ID.
    rank = MPI::COMM_WORLD.Get_rank ( ); // Get_rank gets the rank of the calling process in the communicator

    /************************** Initial Screen Output **************************/
    if (rank == 0)
    {
        std::cout << " +-----------------------------------------------------------------+" << std::endl;
        std::cout << " +                            NN 0.0                               +" << std::endl;
        std::cout << " +-----------------------------------------------------------------+" << std::endl;
        std::cout << " Running on " << procs << " procs" << std::endl;
    }
  
    timer = new Timer(this);

    //if (rank == 0) std::cout << " Job started at " << timer->DateAndTime() << std::endl;

    /************************** Proceed with Program ***************************/

    char debug[64];
	sprintf (debug, "PROC%d", rank);
	fh_debug = fopen(debug, "w");
	fprintf(fh_debug, "This is process %d!\n", rank);

    // Dynamically allocated pointers

    create();

    // Initialize with input

    input = new Input(this);
    
    initialize();

    // Check the neurons
	/*
    for (int k=0; k<input->nhl+2; k++){
        printf("Layer: %d\n", k);
        if (k<input->nhl+1){
            for (int n=0; n<input->structure[k]+1; n++){
                printf(" Neuron: %d\n", n);
                net->layers[k].neurons[n].print();
            }
        }
        else if (k==input->nhl+1){
            for (int n=0; n<input->structure[k]; n++){
                printf(" Neuron: %d\n", n);
                net->layers[k].neurons[n].print();
            }
        }
    }
	*/

	// Split training data

	int spp = input->nsamples/procs; // number of samples per proc
	int sendCountIn = input->structure[0]*spp;
	int sendCountOut = input->structure[input->nhl+1]*spp;
	memory->allocate(din, sendCountIn);
	memory->allocate(dout, sendCountOut);

	MPI_Scatter(input->tin, sendCountIn, MPI::DOUBLE, din, sendCountIn, MPI::DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(input->tout, sendCountOut, MPI::DOUBLE, dout, sendCountOut, MPI::DOUBLE, 0, MPI_COMM_WORLD);

	// Check training sample scattering
	fprintf(fh_debug, "Inputs: ");	
	for (int d=0; d<input->structure[0]*spp; d++){
		fprintf(fh_debug, "%f ", din[d]);
	}
	fprintf(fh_debug, "\n");
	fprintf(fh_debug, "Outputs: ");	
	for (int d=0; d<input->structure[input->nhl+1]*spp; d++){
		fprintf(fh_debug, "%f ", dout[d]);
	}
	fprintf(fh_debug, "\n");;


	// Perform optimization

	net->zeroGrad();
    //for (int d=0; d<input->nsamples; d++){
    for (int d=0; d<1; d++){
        net->feedForward(d);
		net->backprop(d);
    }



    // Delete dynamically allocated pointers

    finalize();

    //if (rank == 0) std::cout << std::endl << " Job finished at " 
    //    << timer->DateAndTime() << std::endl;

    if (rank == 0) timer->print_elapsed();

    fclose(fh_debug);
}

void NN::create()
{
    memory = new Memory(this);
    net = new Net(this);

}

void NN::initialize()
{
    input->readinput();
    input->readData();
    net->build();

}

NN::~NN()
{
    delete timer;
    delete input;

	memory->deallocate(din);
	memory->deallocate(dout);

    //fclose(fh_debug);

}

void NN::finalize()
{

    // Deallocate neuron weights

    for (int k=0; k<input->nhl+2; k++){

        for (int n=0; n<input->structure[k]+1; n++){
    
            if (k != input->nhl+1){ // If we aren't on the last layer
                net->layers[k].neurons[n].finalize(); // Deallocate the output weights for this neuron
            }

        }
    }

    delete net;
    delete memory;
    //delete neighbor;


}

