#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include "mpi.h"

#include "config.h"
#include "memory.h"
#include "input.h"

using namespace std;

using namespace MODULANT_NS;

Config::Config()
{
    
}

Config::~Config() 
{
	
	/*
	for (int n=0; n<natoms; n++){
		for (int l=0; l<ann[n].nhl+1; l++){
			for (int j=0; j<ann[n].structure[l]+1; j++){
				printf("m,n,l,j = %d,%d,%d,%d\n",m,n,l,j);
				printf("neuronType = %d\n", ann[n].layers[l].neurons[j].neuronType);
				ann[n].layers[l].neurons[j].finalize();

			}
		}
	}
	*/

	
	memory->deallocate(ann);
	memory->deallocate(sumGrads);
    memory->deallocate(x);
	memory->deallocate(f0);
    memory->deallocate(f);
    memory->deallocate(tags);
    memory->deallocate(types);
    memory->deallocate(box);
    memory->deallocate(neighlist);
    memory->deallocate(numneigh);
    memory->deallocate(descriptors);
	memory->deallocate(dgdr);
	memory->deallocate(dgdrSelf);
	
	memory->deallocate(df_dw);
	memory->deallocate(sumfGrads);

	memory->deallocate(atoms);

	fclose(fh_debug);
};


/* ----------------------------------------------------------------------
    Set number of atoms in this configuration.
    Also allocate arrays.
	Many quantities are allocated as natoms*neighmax since we store 
	neighbors of neighbors.
------------------------------------------------------------------------- */

void Config::initialize(int natoms_in, int neighmax)
{

	char debug[64];
	sprintf (debug, "debug/systems/config/D_CONFIG%d_PROC%d", m,rank);
	fh_debug = fopen(debug, "w");

    natoms = natoms_in;
	
	fprintf(fh_debug, "%d atoms.\n", natoms);
	
	memory->allocate(ann,natoms);
	memory->allocate(sumGrads,ntypes);
	memory->allocate(df_dw,ntypes,natoms,3);
	memory->allocate(sumfGrads,ntypes);
	

    memory->allocate(x,natoms*neighmax*neighmax,3); // since we are storing neighbors of neighbors
    for (int n=0; n<natoms*neighmax; n++){
        for (int j=0; j<3; j++){
            x[n][j] = 0.0;
        }
    }

	memory->allocate(f0,natoms,3);
    for (int n=0; n<natoms; n++){
        for (int j=0; j<3; j++){
            f0[n][j] = 0.0;
        }
    }

    memory->allocate(f,natoms,3);
    for (int n=0; n<natoms; n++){
        for (int j=0; j<3; j++){
            f[n][j] = 0.0;
        }
    }

    memory->allocate(tags,natoms*neighmax*neighmax); // store tags for neighbors of neighbors as well
    for (int n=0; n<natoms; n++){
        tags[n]=0;
    }

    memory->allocate(types,natoms);
    for (int n=0; n<natoms; n++){
        types[n]=0;
    }

    memory->allocate(box, 3);
    for (int i=0; i<3; i++){
        box[i]=0.0;
    }

    memory->allocate(neighlist, natoms*neighmax, neighmax); // store neighlist for neighbors too
    for (int i=0; i<natoms; i++){
        for (int j=0; j<neighmax; j++){
            neighlist[i][j] = 0;
        }
    }

    memory->allocate(numneigh, natoms*neighmax); // store numneigh[j] for neighbors j too
    for (int i=0; i<natoms; i++){
        numneigh[i] = 0;
    }


	//printf("asdf\n");
}

/* ----------------------------------------------------------------------
    Allocate atoms array
------------------------------------------------------------------------- */

void Config::initializeAtoms()
{
	memory->allocate(atoms, natomsTotal);
}

/* ----------------------------------------------------------------------
    Allocate descriptor array
------------------------------------------------------------------------- */

void Config::allocate_descriptors(int nd_in,int neighmax)
{
    nd = nd_in;

    memory->allocate(descriptors, natoms, nd);

	memory->allocate(dgdr, natoms, neighmax, nd, 3);
	for (int n=0; n<natoms; n++){
		for (int j=0; j<neighmax; j++){
			for (int s=0; s<nd; s++){
				for (int a=0; a<3; a++){
					dgdr[n][j][s][a] = 0.0;
				}
			}
		}
	}

	memory->allocate(dgdrSelf, natoms, nd, 3);
	for (int n=0; n<natoms; n++){
		for (int s=0; s<nd; s++){
			for (int a=0; a<3; a++){
				dgdrSelf[n][s][a] = 0.0;
			}
		}
	}

}

/* ----------------------------------------------------------------------
    Feed forward starting from the inputs.
    Inputs:
        d: Index of training sample to feed forward with
------------------------------------------------------------------------- */

void Config::feedForward()
{

	//fprintf(fh_debug, "Feeding forward\n");
	
	// Loop over all atoms and set the inputs to each ANN, then feed forward
	
	
	pe = 0.0;
	for (int n=0; n<natoms; n++){
		ann[n].feedForward();
		ann[n].feedForwardDerivatives();
		//fprintf(fh_debug, "Atom %d PE = %f\n", n,ann[n].energy);
		pe += ann[n].energy;
	}
	

	// Check outputs
	/*	
	for (int n=0; n<natoms; n++){
		fprintf(fh_debug, " E_%d = %f\n", n,ann[n].energy);
	}
	fprintf(fh_debug, "etot = %f\n", etot);
	*/

	// Calculate forces
	
	for (int n=0; n<natoms; n++){
		for (int a=0; a<3; a++){
			f[n][a] = 0.0;
		}
	}

	int j,jtag;
	double dEj_dGjs;
	double dGjs_dAn;	
	for (int n=0; n<natoms; n++){
		//fprintf(fh_debug, "Atom %d\n", n);
		for (int jj=0; jj<atoms[n].numneigh; jj++){
			//fprintf(fh_debug, " jj,jtag %d %d\n", jj,jtag);
			j = neighlist[n][jj];
			jtag = atoms[j].tag;
			for (int s=0; s<nd; s++){
				dEj_dGjs=ann[jtag].layers[ann[jtag].nhl+1].neurons[0].dodi[s];
				//fprintf(fh_debug, "  dEj_dGj%d: %f\n",s,dEj_dGjs);
				for (int a=0; a<3; a++){
					dGjs_dAn = dgdr[n][jj][s][a];
					f[n][a] -= dEj_dGjs*dGjs_dAn;
					//fprintf(fh_debug, "   %f\n", dGjs_dAn);
					//fprintf(fh_debug, "     dEdG*dgdr: %f\n", dEj_dGjs*dgdr[n][jj][s][a]);
				}
				//fprintf(fh_debug,"  f[%d]: %f %f %f\n", n,f[n][1],f[n][2],f[n][3]);
			}
		}
		// Also include the self-term
		for (int s=0; s<nd; s++){
			dEj_dGjs = ann[n].layers[ann[n].nhl+1].neurons[0].dodi[s];
			for (int a=0; a<3; a++){
				dGjs_dAn = dgdrSelf[n][s][a];
				f[n][a] -= dEj_dGjs*dGjs_dAn;
				//f[n][a] = 
				//fprintf(fh_debug, "dGjs_dAn: %f\n", dGjs_dAn);
			}
		}	
		fprintf(fh_debug, "    f%d: %f %f %f\n", n,f[n][0],f[n][1],f[n][2]);
	}

}

/* ----------------------------------------------------------------------
	Calculate error for this config   
------------------------------------------------------------------------- */

void Config::calcError()
{
	errdiff=(pe/natoms)-(pe0/natoms);
	errsq = errdiff*errdiff;

	sum_fsqerr = 0.0;
	for (int n=0; n<natoms; n++){
		for (int a=0; a<3; a++){
			sum_fsqerr += (f[n][a]-f0[n][a])*(f[n][a]-f0[n][a]);
			//fprintf(fh_debug, "%f %f\n", f0[n][a],f[n][a]);
		}
		//fprintf(fh_debug, "\n");
	}
	//printf("sum_fsqerr: %f\n", sum_fsqerr);

}

/* ----------------------------------------------------------------------
	Backpropogate
	Calculate quantitities used for error gradient calculation.
	This function, for a particular config, calculates quantities
	that each config will contribute to the total error gradient.
	This includes:
		sumGrads - sum of all ANN output gradients
   ------------------------------------------------------------------------- */

void Config::backprop()
{

	//fprintf(fh_debug," Backpropagating for d=%d.\n", d);
	
	// Calculate error

	//printf("  etot: %f\n", etot);	
	//errdiff = etot-e0;
	//errsq = errdiff*errdiff;

	// Calculate output gradients for all ANNs
	
	for (int n=0; n<natoms; n++){
		ann[n].backprop();
		ann[n].backpropDerivatives();
	}

	// Sum all gradients across atoms

	// First zero the summed terms
	
	for (int t=0; t<ntypes; t++){
		for (int n=0; n<natoms; n++){
			for (int l=0; l<ann[n].nhl+1; l++){
				for (int j=0; j<ann[n].structure[l]+1; j++){
					for (int k=0; k<ann[n].structure[l+1]; k++){
						sumGrads[t].layers[l].neurons[j].gradients[k]=0.0;
						sumfGrads[t].layers[l].neurons[j].gradients[k]=0.0;
						for (int a=0; a<3; a++){
							df_dw[t][n][a].layers[l].neurons[j].gradients[k]=0.0;
						}
					}
				}
			}
		}
	}
	
	double grad;
	int type;
	for (int t=0; t<ntypes; t++){
		for (int n=0; n<natoms; n++){

			// Add to sumGrads depending on the type of atom involved
			
			type = ann[n].type;

			if (type==t+1){
				for (int l=0; l<ann[n].nhl+1; l++){
					for (int j=0; j<ann[n].structure[l]+1; j++){
						for (int k=0; k<ann[n].structure[l+1]; k++){

							grad = ann[n].layers[l].neurons[j].gradientsp[k];
							//fprintf(fh_debug,"Atom %d, layer %d, unit %d, grad %f: %f\n", n,l,j,k,grad);
							sumGrads[t].layers[l].neurons[j].gradients[k] += grad;
						}
					}
				}
			}
		}
	}

	// Calculate force gradients
	
	int j,jtag;
	//int jn; // neuron index "j"
	for (int t=0; t<ntypes; t++){
		for (int n=0; n<natoms; n++){

			for (int jj=0; jj<atoms[n].numneigh; jj++){
				j = neighlist[n][jj];
				//jtag=tags[j];
				jtag = atoms[j].tag;
				// Loop over all gradients (layers and neurons)
				for (int l=0; l<ann[n].nhl+1; l++){
					for (int jn=0; jn<ann[n].structure[l]+1; jn++){
						for (int k=0; k<ann[n].structure[l+1]; k++){
							for (int s=0; s<nd; s++){
								for (int a=0; a<3;a++){
									df_dw[t][n][a].layers[l].neurons[jn].gradients[k] -= \
										ann[jtag].layers[l].neurons[jn].d2e_dwdg[k][s] *\
										dgdr[n][jj][s][a];
								}
							}
						}
					}
				}
			}
			// Get the self-term contributions
			for (int l=0; l<ann[n].nhl+1; l++){
				for (int jn=0; jn<ann[n].structure[l]+1; jn++){
					for (int k=0; k<ann[n].structure[l+1]; k++){
						for (int s=0; s<nd; s++){
							for (int a=0; a<3;a++){
								df_dw[t][n][a].layers[l].neurons[jn].gradients[k] -= \
									ann[n].layers[l].neurons[jn].d2e_dwdg[k][s] *\
									dgdrSelf[n][s][a];
							}
						}
					}
				}
			}

		
		}
	}

	// Calculate force gradient sums
	double fdiff;
	for (int t=0; t<ntypes; t++){
		for (int n=0; n<natoms; n++){
			for (int a=0; a<3; a++){
				fdiff = f[n][a] - f0[n][a];
				for (int l=0; l<ann[n].nhl+1; l++){
					for (int j=0; j<ann[n].structure[l]+1; j++){
						for (int k=0; k<ann[n].structure[l+1]; k++){
							sumfGrads[t].layers[l].neurons[j].gradients[k] += \
							fdiff*df_dw[t][n][a].layers[l].neurons[j].gradients[k];
						}
					}
				}
			}	

		}	
	}

	// Check gradients summed across atoms
	/*
	for (int t=0; t<ntypes; t++){
		
		for (int l=0; l<sumGrads[t].nhl+1; l++){
			for (int j=0; j<sumGrads[t].structure[l]+1; j++){
				for (int k=0; k<sumGrads[t].structure[l+1]; k++){

					fprintf(fh_debug,"T%d, Layer %d, unit %d, grad %d:\n", t,l,j,k);
					fprintf(fh_debug, " %f\n",sumfGrads[t].layers[l].neurons[j].gradients[k]);
				}
			}
		}
			
		
	}
	*/

	



}
