#pragma once

#include <vector>
#include <string>
#include "mpi.h"

#include <iostream>
#include <new>
#include <cstdlib>
#include "pointers.h"
#include "memory.h"
#include "ann.h"

using namespace std;

using namespace MODULANT_NS;

class Config
{
public:

    Config();
    //Config(class POP *);
    ~Config();

    Memory * memory;
	FILE * fh_debug; // Debug file handle

    void initialize(int,int); // set number of atoms and allocate arrays
    void allocate_descriptors(int,int); // allocate descriptor array
	void initializeAtoms();
	void feedForward();
	void calcError();
	void backprop();

	int mp; // config index on this proc
	int m; // total config index
	int rank; // proc id
	
	// Atomic system info
	
    int natoms; // number of atoms
	int natomsTotal; // number of atoms including neighbors and neighbors of neighbors
	int ntypes; // number of atom types
    int nd; // number of descriptors
    double **x; // positions of all atoms (including neighbors)
				// x[n][a] is ath component of atom n position
				// n can range from 0 to natoms*neighmax*neighmax since we store neighbors of neighbors
				// the first 0 <= n < natoms components are the positions in the box
    double **f0; // reference forces on all atoms 
	double **f; // calculated forces on all atoms
    double pe0; // reference potential energy
    double *box; // box dimensions
    int *tags; // tags of all atoms ranging from 0 to N-1
			   // tags[i] gives the tag of atom i, 0 <= i < natoms*neighmax*neighmax
			   // this array is a mapping between the real ID and atom tag in the box
    int *types; // types of all atoms
    int **neighlist; // neighbor list
					 // j=neighlist[n][jj] is the ID of neighbor jj (0 <= jj < numneigh[n])
					 // jtag=tags[j] is the tag of neighbor jj
					 // includes neighbors of neighbors
					 // l=neighlist[j][ll] is ID neighbor ll (0 <= ll <= numneigh[j])
					 // ltag=tags[l] is tag of neighbor ll, which is a neighbor of j
					 // after nei
    int *numneigh; // number of neighbors for each atom

	// Descriptor info
	
	struct descriptor{ // descriptor object stores descriptor info
		double value;
		int nbody;
		int types[3];
	};
	descriptor **descriptors; // descriptors[n][s] is sth descriptor of atom n
	struct atom{
		double x[3]; // position
		int i[3]; // periodic box position
	    int tag; // tag of atom in original box (starts at zero)
		int id; // unique ID of atom (starts at zero)
		int type;
		int numneigh;
	};
	atom *atoms; // atoms[n] contains info on nth atom	

	double ****dgdr; // spatial first derivative of descriptors 
				     // dgdr[n][j][s][a] is derivative of neighbor "j" descriptor "s" wrt atom "n" coordinate "a"
				     // there are numneigh[n] valid elements in dgdr[n], although it is allocated to neighmax
					 // Also need to include a self-term... Derivative of atom "n" descriptor "s" wrt atom "n" coordinate "a"
	double ***dgdrSelf; // spatial first self-derivatives of descriptors
						// dgdrSelf[n][s][a] - derivative of atom "n" descriptor "s" wrt atom "n" coordinate "a"


	// System network info
	
	double pe; // potential energy of config calculated by network
	double errsq; // squared error for this config
	double errdiff; // absolute difference error for this config
	double sum_fsqerr; // sum of squared force errors


	Ann *sumGrads; // summed gradients for all atom types
	Ann ***df_dw; // Derivatives of forces wrt weights
			      // df_dw[t][n][a] - "t" type gradients of atom "n" force in "a" direction 
	Ann *sumfGrads; // Force sum of gradients term
					 // sumfGrads[t] - Force sum of gradients term for atom type t
	Ann *ann; // atomistic neural networks
};
