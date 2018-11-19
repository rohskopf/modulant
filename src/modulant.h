/*
 nnp.h

 Copyright (c) 2018 Andrew Rohskopf

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

/* Declaration of pointers used in the whole program. */

#pragma once
#include <string>
#include <vector>
#include "mpi.h"

namespace MODULANT_NS
{
    class MLT
    {
    public:

        class Memory *memory;
		class Error *error;
        class Timer *timer;
        class Input *input;
		class Systems *systems;
		class Descriptor *descriptor;
        MLT(int, char **);
        ~MLT();

        FILE * fh_debug; // Debug file handle        

        void create(); // Function to create class instances
        void initialize(); // Initializes the program, reads input
		void sumProcs(); // function to sum gradients and error over procs
        void finalize(); // Deallocates class instances
        int procs; // number of processes
        int rank; // rank of process
        int t; // timestep
	
		int *natoms_all; // number of atoms in all configs on this proc
		int natomsTot; // total number of atoms on this proc
		int ndTot; // total number of descriptors on this proc
		int cpp; // number of configs per proc
		int *dpp; // number of descriptors for each proc
		int *app; // total number of atoms for each proc
		int *types; // types of all atoms on this proc

		/*
		struct descriptor{
			double value;
			//int nbody;
			//int types[3];
			int m;
			int n;
			int s;
			int nbody;
			int types[3];
		};
		descriptor *descriptors; // 1D descriptors array for MPI, split over procs
		*/
        double *e0; // reference energies, split over procs

		double trainError; // summed error over all procs
		double trainEnergyError;
		double trainForceError;


    };
}

