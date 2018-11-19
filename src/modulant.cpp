/*
 nnp.cpp

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
#include "error.h"
#include "timer.h"
#include "input.h"
#include "systems.h"
#include "descriptor.h"

using namespace std;

using namespace MODULANT_NS;

MLT::MLT(int narg, char **arg)
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
        std::cout << " +                         Modulant 0.0                            +" << std::endl;
        std::cout << " +-----------------------------------------------------------------+" << std::endl;
        std::cout << " Running on " << procs << " procs" << std::endl;
    }
  
    timer = new Timer(this);

    //if (rank == 0) std::cout << " Job started at " << timer->DateAndTime() << std::endl;
	
    char debug[64];
	sprintf (debug, "debug/D_MLT%d", rank);
	fh_debug = fopen(debug, "w");

    /************************** Proceed with Program **************************/

    input = new Input(this);
   	
    // Dynamically allocated pointers and initialization

    create();

	initialize();

	if (rank==0) printf(" Epoch | Error\n");
	for (int e=0; e<input->nepochs; e++){

		// Zero grads and errors
	
		//printf("zero grads\n");
		systems->zeroGrads();
		systems->error=0.0;

		// Feedforward every system on this proc
		//printf("feedforward\n");
		systems->feedForwardAll();

		// Calculate error
		//printf("calcerror\n");
		systems->calcError();

		// Backprop every system on this proc and calculate gradients
		//printf("backprop\n");
		systems->backpropAll();

		// Sum error and gradients across all procs
		//printf("sumprocs\n");
		sumProcs();

		if (rank ==0) printf("%d %f %f %f\n", e+1, trainError, trainEnergyError, trainForceError);
	
		// Update the weights

		systems->updateWeights();	


	}

	systems->writeWeights();
	
    // Delete dynamically allocated pointers

    finalize();

    //if (rank == 0) std::cout << std::endl << " Job finished at " 
    //    << timer->DateAndTime() << std::endl;

    if (rank == 0) timer->print_elapsed();

}

void MLT::create()
{
    memory = new Memory(this);
	error = new Error(this);
    systems = new Systems(this);
	descriptor = new Descriptor(this);

}

void MLT::initialize()
{
	
	//fprintf(fh_debug, "test\n");
	// Read inputs settings, configs, weights, descriptor parameters. 

	printf("Reading input...\n");	
    input->readInput();
	systems->readConfigs();
	printf("Reading weights...\n");
	systems->readWeights();
	descriptor->readParams();
	// Preliminary routines that only need to be done once.
	printf("Calcularing neighborlists...\n");
	systems->calcNeighborLists();
	printf("Calcularing descriptors...\n");
	systems->calcDescriptors();
	//scatter();
	printf("Building nets...\n");
	systems->buildNets();

	/*
	for (int m=0; m<systems->nconfigs; m++){
		for (int n=0; n<systems->configs[m].natoms; n++){
			for (int l=0; l<systems->configs[m].ann[n].nhl+1; l++){
				for (int j=0; j<systems->configs[m].ann[n].layers[l].nunits;j++){
					//printf("m,n,l,j: %d,%d,%d,%d\n", m,n,l,j);
					systems->configs[m].ann[n].layers[l].neurons[j].print();
				}
			}
		}
	}
	*/	

}

/* ----------------------------------------------------------------------
	Sum total error and gradients across all procs.
------------------------------------------------------------------------- */

void MLT::sumProcs()
{

	// Sum the error across all networks on all procs
	MPI_Allreduce(&systems->error, \
				  &trainError, \
				  1, \
				  MPI::DOUBLE, MPI_SUM, MPI_COMM_WORLD);	

	// Sum the error across all networks on all procs
	MPI_Allreduce(&systems->energy_error, \
				  &trainEnergyError, \
				  1, \
				  MPI::DOUBLE, MPI_SUM, MPI_COMM_WORLD);	

	// Sum the error across all networks on all procs
	MPI_Allreduce(&systems->force_error, \
				  &trainForceError, \
				  1, \
				  MPI::DOUBLE, MPI_SUM, MPI_COMM_WORLD);	

	// Sum the gradients on all procs
	for (int t=0; t<input->ntypes; t++){
		for (int l=0; l<input->nhl+1; l++){
			//fprintf(fh_debug, "Layer %d\n", l);
			for (int j=0; j<input->structure[l]+1; j++){
				//fprintf(fh_debug, "Neuron %d\n", j);
				MPI_Allreduce(systems->grads[t].layers[l].neurons[j].gradientsp, \
							systems->grads[t].layers[l].neurons[j].gradients, \
							input->structure[l+1], \
							MPI::DOUBLE, MPI_SUM, MPI_COMM_WORLD);	
			}	
		}
	}


}

MLT::~MLT()
{

	//memory->deallocate(app);
	//memory->deallocate(types);

	//memory->deallocate(dpp);
	//memory->deallocate(natoms_all);
	//memory->deallocate(descriptors);
	//memory->deallocate(e0);

    delete timer;
    delete input;
	//delete systems;


    fclose(fh_debug);
	

}

void MLT::finalize()
{

	/*
	for (int n=0; n<systems->configs[0].natoms; n++){
		for (int l=0; l<systems->configs[0].ann[n].nhl+1; l++){
			for (int j=0; j<systems->configs[0].ann[n].structure[l]+1; j++){
				printf("m,n,l,j = %d,%d,%d,%d\n",1,n,l,j);
				systems->configs[0].ann[n].layers[l].neurons[j].finalize();

			}
		}
	}
	*/

	delete descriptor;
	delete systems;
	delete error;
	delete memory;

}

