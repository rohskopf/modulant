/*
 systems.cpp

 Copyright (c) 2018 Andrew Rohskopf

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

/* ----------------------------------------------------------------------
    The System class stores quantities and objects associated with all 
	configs (systems).
	
------------------------------------------------------------------------- */

#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include "mpi.h"
#include <math.h>       /* sqrt */
#include <random>
#include "dirent.h"

#include "systems.h"
#include "input.h"
#include "memory.h"
#include "error.h"
#include "descriptor.h"

using namespace std;

using namespace MODULANT_NS;

Systems::Systems(MLT *mlt) : Pointers(mlt) {

    rank = MPI::COMM_WORLD.Get_rank ( ); // Get_rank gets the rank of the calling process in the communicator
	char debug[64];
	sprintf (debug, "debug/systems/D_SYSTEMS%d", rank);
	fh_debug = fopen(debug, "w");
}

Systems::~Systems() 
{

	/*
	double wt;
	for (int t=0; t<input->ntypes; t++){
		for (int l=0; l<input->nhl+1; l++){
			for (int j=0; j<input->structure[l]+1; j++){
				for (int k=0; k<input->structure[l+1]; k++){
					wt = weights[t].layers[l].neurons[j].weights[k];
					wt = printf("wt: %f\n", wt);
				}
			}
		}

	}
	*/

	memory->deallocate(configs);
	memory->deallocate(grads);
	memory->deallocate(weights);
	//printf("asdf\n");
	
	fclose(fh_debug);

};

/* ----------------------------------------------------------------------
    Read the CONFIG files.

	Allocate the configs array.
------------------------------------------------------------------------- */

void Systems::readConfigs()
{ 

	// Get list of directories in /configs
		
	string dir = "configs/";
    DIR *dp;
	struct dirent *dirp;
	if((dp  = opendir(dir.c_str())) == NULL) {
		mlt->error->exit("systems.cpp", "No configs/ directory.");
	}	
	//mlt->systems->initConfigs();

	nconfigsTot=-2; // subtract 2 since ./ and ../ directories are included
	while ((dirp = readdir(dp)) != NULL) {
		nconfigsTot++;
		//cout << dirp->d_name << endl;
		//files.push_back(string(dirp->d_name));
	}
	if (rank==0) printf(" %d total configs.\n", nconfigsTot);
	if (nconfigsTot % mlt->procs !=0) mlt->error->exit("input.cpp", "Number of configs not divisible by procs.");
		
	nconfigs = nconfigsTot/mlt->procs;
	memory->allocate(configs, nconfigs);
	//if (rank==0) printf(" %d total configs.\n", nfiles);

	// We want to split the number of configs read per proc
	string line;
	int type;
    double x,y,z,fx,fy,fz;
    double lx, ly, lz;
	int natoms;
	int f; // file index
	for (int m=0; m<nconfigs; m++){
		f = m+(nconfigs*rank)+1;
		//printf("filenum: %d\n", f);
		char filename[64];
		sprintf (filename, "configs/%d/CONFIG", f);
		ifstream fh_config(filename);
		getline(fh_config, line);
        stringstream ss(line);
        ss >> natoms;
        //natoms_all[m] = natoms;
		configs[m].mp = m;
		configs[m].m = f;
		configs[m].rank = rank;
		configs[m].ntypes = input->ntypes;
        configs[m].initialize(natoms, input->neighmax);
        //printf("natoms: %d\n", configs[m].natoms);
		
        for (int n=0; n<natoms; n++){

            configs[m].tags[n] = n;

            getline(fh_config,line);
            stringstream ss2(line);
            ss2 >> type >> x >> y >> z >> fx >> fy >> fz;
            configs[m].types[n] = type;
            configs[m].x[n][0] = x;
            configs[m].x[n][1] = y;
            configs[m].x[n][2] = z;
            configs[m].f0[n][0] = fx;
            configs[m].f0[n][1] = fy;
            configs[m].f0[n][2] = fz;
            //printf("%d %f %f %f %f %f %f\n", type,x,y,z,fx,fy,fz);
        }

        getline(fh_config,line);
        stringstream ss3(line);
        ss3 >> configs[m].pe0;
		//e0[m]=pe;

        getline(fh_config,line);
        stringstream ss4(line);
        ss4 >> lx >> ly >> lz;
        configs[m].box[0] = lx;
        configs[m].box[1] = ly;
        configs[m].box[2] = lz;
	
		//configs[m].calcNeighborList();

		fh_config.close();
	}


	printf(" %d configs on  proc %d.\n", nconfigs,rank);
}

/* ----------------------------------------------------------------------
    Loop through all configurations and calculate a full neighborlist for each
------------------------------------------------------------------------- */

void Systems::calcNeighborLists()
{
    //printf(" Calculating neighborlists...\n");

    double cutsq = input->rc*input->rc;
	double cutsq2 = (2*input->rc)*(2*input->rc); // double of cutoff squared
    //double boxinvx = 1.0/lx;
    //double boxinvy = 1.0/ly;
    //double boxinvz = 1.0/lz;

    int neighcount;
    //int imagecounter; // used to store new images in the list of positions
    int nearintx, nearinty, nearintz;
    double xi,yi,zi,xj,yj,zj;
    double xjp, yjp, zjp; // Image positions
    double xij,yij,zij, rsq, rij;

    double xtmp, ytmp, ztmp;
	double xn,yn,zn;
	double xp,yp,zp;

    int natoms;
    double lx,ly,lz;
    int types;

	double rc2 = 2.0*input->rc;
	double rc = input->rc;

	// Allocate atoms array for each config by counting total number of atoms
	// (includes periodic images out to twice the cutoff)	
	for (int m=0; m<nconfigs; m++){

		natoms = configs[m].natoms;
        lx = configs[m].box[0];
        ly = configs[m].box[1];
        lz = configs[m].box[2];
        //imagecounter = natoms;
		configs[m].natomsTotal = natoms;
        
		for (int n=0; n<natoms; n++){

			neighcount=0;
            xn = configs[m].x[n][0];
            yn = configs[m].x[n][1];
            zn = configs[m].x[n][2];

			//fprintf(fh_debug,"n: %d\n",n);
			
			// Apply periodic operations for 26 periodic boxes

			// 1
			if (xn+lx < lx+rc2){
				configs[m].natomsTotal++;

			}
			//2
			if (xn-lx > 0.0-rc2){
				configs[m].natomsTotal++;


			}
			//3
			if (xn+lx < lx+rc2 && yn+ly < ly+rc2){
				configs[m].natomsTotal++;


			}
			//4
			if (xn+lx < lx+rc2 && zn+lz < lz+rc2){
				configs[m].natomsTotal++;


			}
			//5
			if (xn+lx < lx+rc2 && yn-ly > 0.0-rc2){
				configs[m].natomsTotal++;


			}
			//6
			if (xn+lx < lx+rc2 && zn-ly > 0.0-rc2){
				configs[m].natomsTotal++;


			}
			//7
			if (xn-lx > 0.0-rc2 && yn+ly < ly+rc2){
				configs[m].natomsTotal++;


			}
			//8
			if (xn-lx > 0.0-rc2 && zn+lz < lz+rc2){
				configs[m].natomsTotal++;


			}
			//9
			if (xn-lx > 0.0-rc2 && yn-ly > 0.0-rc2){
				configs[m].natomsTotal++;


			}
			//10
			if (xn-lx > 0.0-rc2 && zn-lz > 0.0-rc2){
				configs[m].natomsTotal++;


			}
			//11
			if (xn+lx < lx+rc2 && yn+ly < ly+rc2 && zn+lz < lz+rc2){
				configs[m].natomsTotal++;


			}
			//12
			if (xn+lx < lx+rc2 && yn-ly > 0.0-rc2 && zn+lz < lz+rc2){
				configs[m].natomsTotal++;


			}
			//13
			if (xn+lx < lx+rc2 && yn+ly < ly+rc2 && zn-lz > 0.0-rc2){
				configs[m].natomsTotal++;


			}
			//14
			if (xn+lx < lx+rc2 && yn-ly > 0.0-rc2 && zn-lz > 0.0-rc2){
				configs[m].natomsTotal++;


			}
			//15
			if (xn-lx > 0.0-rc2 && yn+ly < ly+rc2 && zn+lz < lz+rc2){
				configs[m].natomsTotal++;


			}
			//16
			if (xn-lx > 0.0-rc2 && yn-ly > 0.0-rc2 && zn-lz > 0.0-rc2){
				configs[m].natomsTotal++;


			}
			//17
			if (xn-lx > 0.0-rc2 && yn+ly < ly+rc2 && zn-lz > 0.0-rc2){
				configs[m].natomsTotal++;


			}
			//18
			if (xn-lx > 0.0-rc2 && yn-ly > 0.0-rc2 && zn+lz < lz+rc2){
				configs[m].natomsTotal++;


			}
			//19
			if (yn+ly < ly+rc2){
				configs[m].natomsTotal++;


			}
			//20
			if (yn+ly < ly+rc2 && zn-lz > 0.0-rc2){
				configs[m].natomsTotal++;


			}
			//21
			if (yn+ly < ly+rc2 && zn+lz < lz+rc2){
				configs[m].natomsTotal++;


			}
			//22
			if (yn-ly > 0.0-rc2){
				configs[m].natomsTotal++;


			}
			//23
			if (yn-lz > 0.0-rc2 && zn+lz < lz+rc2){
				configs[m].natomsTotal++;


			}
			//24
			if (yn-lz > 0.0-rc2 && zn-lz > 0.0-rc2){
				configs[m].natomsTotal++;


			}
			//25
			if (zn+lz < lz+rc2){
				configs[m].natomsTotal++;


			}
			//26
			if (zn-lz > 0.0-rc2){
				configs[m].natomsTotal++;


			}
			

		}
		fprintf(fh_debug, "Found %d total atoms in config %d.\n", configs[m].natomsTotal, m);
		
		// Allocate atoms array

		configs[m].initializeAtoms();

	}

	// Store all atoms 
	// The first 0 to (N-1) atoms will be atoms in the original box
	 
	for (int m=0; m<nconfigs; m++){

		natoms = configs[m].natoms;
        lx = configs[m].box[0];
        ly = configs[m].box[1];
        lz = configs[m].box[2];
        //imagecounter = natoms;

        for (int n=0; n<configs[m].natoms; n++){

            xn = configs[m].x[n][0];
            yn = configs[m].x[n][1];
            zn = configs[m].x[n][2];
			//fprintf(fh_debug,"%f,%f,%f\n",xtmp,ytmp,ztmp);

			configs[m].atoms[n].x[0]=xn;
			configs[m].atoms[n].x[1]=yn;
			configs[m].atoms[n].x[2]=zn;

			configs[m].atoms[n].i[0]=0;
			configs[m].atoms[n].i[1]=0;
			configs[m].atoms[n].i[2]=0;

			configs[m].atoms[n].tag = n;
			configs[m].atoms[n].id = n;
			configs[m].atoms[n].type = configs[m].types[n];
		}

		// Store the periodic images, which have IDs after the original box atoms

		int id = configs[m].natoms;
		for (int n=0; n<configs[m].natoms; n++){

			xn = configs[m].x[n][0];
            yn = configs[m].x[n][1];
            zn = configs[m].x[n][2];

			// Apply periodic operations for 26 periodic boxes

			// 1
			if (xn+lx < lx+rc2){
				configs[m].atoms[id].x[0] = xn+lx;
				configs[m].atoms[id].x[1] = yn;
				configs[m].atoms[id].x[2] = zn;
				configs[m].atoms[id].i[0] = 1;
				configs[m].atoms[id].i[1] = 0;
				configs[m].atoms[id].i[2] = 0;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//2
			if (xn-lx > 0.0-rc2){
				configs[m].atoms[id].x[0] = xn-lx;
				configs[m].atoms[id].x[1] = yn;
				configs[m].atoms[id].x[2] = zn;
				configs[m].atoms[id].i[0] = -1;
				configs[m].atoms[id].i[1] = 0;
				configs[m].atoms[id].i[2] = 0;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//3
			if (xn+lx < lx+rc2 && yn+ly < ly+rc2){
				configs[m].atoms[id].x[0] = xn+lx;
				configs[m].atoms[id].x[1] = yn+ly;
				configs[m].atoms[id].x[2] = zn;
				configs[m].atoms[id].i[0] = 1;
				configs[m].atoms[id].i[1] = 1;
				configs[m].atoms[id].i[2] = 0;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//4
			if (xn+lx < lx+rc2 && zn+lz < lz+rc2){
				configs[m].atoms[id].x[0] = xn+lx;
				configs[m].atoms[id].x[1] = yn;
				configs[m].atoms[id].x[2] = zn+lz;
				configs[m].atoms[id].i[0] = 1;
				configs[m].atoms[id].i[1] = 0;
				configs[m].atoms[id].i[2] = 1;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//5
			if (xn+lx < lx+rc2 && yn-ly > 0.0-rc2){
				configs[m].atoms[id].x[0] = xn+lx;
				configs[m].atoms[id].x[1] = yn-ly;
				configs[m].atoms[id].x[2] = zn;
				configs[m].atoms[id].i[0] = 1;
				configs[m].atoms[id].i[1] = -1;
				configs[m].atoms[id].i[2] = 0;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;		
			}
			//6
			if (xn+lx < lx+rc2 && zn-lz > 0.0-rc2){
				configs[m].atoms[id].x[0] = xn+lx;
				configs[m].atoms[id].x[1] = yn;
				configs[m].atoms[id].x[2] = zn-lz;
				configs[m].atoms[id].i[0] = 1;
				configs[m].atoms[id].i[1] = 0;
				configs[m].atoms[id].i[2] = -1;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//7
			if (xn-lx > 0.0-rc2 && yn+ly < ly+rc2){
				configs[m].atoms[id].x[0] = xn-lx;
				configs[m].atoms[id].x[1] = yn+ly;
				configs[m].atoms[id].x[2] = zn;
				configs[m].atoms[id].i[0] = -1;
				configs[m].atoms[id].i[1] = 1;
				configs[m].atoms[id].i[2] = 0;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//8
			if (xn-lx > 0.0-rc2 && zn+lz < lz+rc2){
				configs[m].atoms[id].x[0] = xn-lx;
				configs[m].atoms[id].x[1] = yn;
				configs[m].atoms[id].x[2] = zn+lz;
				configs[m].atoms[id].i[0] = -1;
				configs[m].atoms[id].i[1] = 0;
				configs[m].atoms[id].i[2] = 1;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//9
			if (xn-lx > 0.0-rc2 && yn-ly > 0.0-rc2){
				configs[m].atoms[id].x[0] = xn-lx;
				configs[m].atoms[id].x[1] = yn-ly;
				configs[m].atoms[id].x[2] = zn;
				configs[m].atoms[id].i[0] = -1;
				configs[m].atoms[id].i[1] = -1;
				configs[m].atoms[id].i[2] = 0;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//10
			if (xn-lx > 0.0-rc2 && zn-lz > 0.0-rc2){
				configs[m].atoms[id].x[0] = xn-lx;
				configs[m].atoms[id].x[1] = yn;
				configs[m].atoms[id].x[2] = zn-lz;
				configs[m].atoms[id].i[0] = -1;
				configs[m].atoms[id].i[1] = 0;
				configs[m].atoms[id].i[2] = -1;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//11
			if (xn+lx < lx+rc2 && yn+ly < ly+rc2 && zn+lz < lz+rc2){
				configs[m].atoms[id].x[0] = xn+lx;
				configs[m].atoms[id].x[1] = yn+ly;
				configs[m].atoms[id].x[2] = zn+lz;
				configs[m].atoms[id].i[0] = 1;
				configs[m].atoms[id].i[1] = 1;
				configs[m].atoms[id].i[2] = 1;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//12
			if (xn+lx < lx+rc2 && yn-ly > 0.0-rc2 && zn+lz < lz+rc2){
				configs[m].atoms[id].x[0] = xn+lx;
				configs[m].atoms[id].x[1] = yn-ly;
				configs[m].atoms[id].x[2] = zn+lz;
				configs[m].atoms[id].i[0] = 1;
				configs[m].atoms[id].i[1] = -1;
				configs[m].atoms[id].i[2] = 1;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//13
			if (xn+lx < lx+rc2 && yn+ly < ly+rc2 && zn-lz > 0.0-rc2){
				configs[m].atoms[id].x[0] = xn+lx;
				configs[m].atoms[id].x[1] = yn+ly;
				configs[m].atoms[id].x[2] = zn-lz;
				configs[m].atoms[id].i[0] = 1;
				configs[m].atoms[id].i[1] = 1;
				configs[m].atoms[id].i[2] = -1;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//14
			if (xn+lx < lx+rc2 && yn-ly > 0.0-rc2 && zn-lz > 0.0-rc2){
				configs[m].atoms[id].x[0] = xn+lx;
				configs[m].atoms[id].x[1] = yn-ly;
				configs[m].atoms[id].x[2] = zn-lz;
				configs[m].atoms[id].i[0] = 1;
				configs[m].atoms[id].i[1] = -1;
				configs[m].atoms[id].i[2] = -1;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//15
			if (xn-lx > 0.0-rc2 && yn+ly < ly+rc2 && zn+lz < lz+rc2){
				configs[m].atoms[id].x[0] = xn-lx;
				configs[m].atoms[id].x[1] = yn+ly;
				configs[m].atoms[id].x[2] = zn+lz;
				configs[m].atoms[id].i[0] = -1;
				configs[m].atoms[id].i[1] = 1;
				configs[m].atoms[id].i[2] = 1;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//16
			if (xn-lx > 0.0-rc2 && yn-ly > 0.0-rc2 && zn-lz > 0.0-rc2){
				configs[m].atoms[id].x[0] = xn-lx;
				configs[m].atoms[id].x[1] = yn-ly;
				configs[m].atoms[id].x[2] = zn-lz;
				configs[m].atoms[id].i[0] = -1;
				configs[m].atoms[id].i[1] = -1;
				configs[m].atoms[id].i[2] = -1;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//17
			if (xn-lx > 0.0-rc2 && yn+ly < ly+rc2 && zn-lz > 0.0-rc2){
				configs[m].atoms[id].x[0] = xn-lx;
				configs[m].atoms[id].x[1] = yn+ly;
				configs[m].atoms[id].x[2] = zn-lz;
				configs[m].atoms[id].i[0] = -1;
				configs[m].atoms[id].i[1] = 1;
				configs[m].atoms[id].i[2] = -1;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//18
			if (xn-lx > 0.0-rc2 && yn-ly > 0.0-rc2 && zn+lz < lz+rc2){
				configs[m].atoms[id].x[0] = xn-lx;
				configs[m].atoms[id].x[1] = yn-ly;
				configs[m].atoms[id].x[2] = zn+lz;
				configs[m].atoms[id].i[0] = -1;
				configs[m].atoms[id].i[1] = -1;
				configs[m].atoms[id].i[2] = 1;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//19
			if (yn+ly < ly+rc2){
				configs[m].atoms[id].x[0] = xn;
				configs[m].atoms[id].x[1] = yn+ly;
				configs[m].atoms[id].x[2] = zn;
				configs[m].atoms[id].i[0] = 0;
				configs[m].atoms[id].i[1] = 1;
				configs[m].atoms[id].i[2] = 0;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//20
			if (yn+ly < ly+rc2 && zn-lz > 0.0-rc2){
				configs[m].atoms[id].x[0] = xn;
				configs[m].atoms[id].x[1] = yn+ly;
				configs[m].atoms[id].x[2] = zn-lz;
				configs[m].atoms[id].i[0] = 0;
				configs[m].atoms[id].i[1] = 1;
				configs[m].atoms[id].i[2] = -1;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//21
			if (yn+ly < ly+rc2 && zn+lz < lz+rc2){
				configs[m].atoms[id].x[0] = xn;
				configs[m].atoms[id].x[1] = yn+ly;
				configs[m].atoms[id].x[2] = zn+lz;
				configs[m].atoms[id].i[0] = 0;
				configs[m].atoms[id].i[1] = 1;
				configs[m].atoms[id].i[2] = 1;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//22
			if (yn-ly > 0.0-rc2){
				configs[m].atoms[id].x[0] = xn;
				configs[m].atoms[id].x[1] = yn-ly;
				configs[m].atoms[id].x[2] = zn;
				configs[m].atoms[id].i[0] = 0;
				configs[m].atoms[id].i[1] = -1;
				configs[m].atoms[id].i[2] = 0;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//23
			if (yn-ly > 0.0-rc2 && zn+lz < lz+rc2){
				configs[m].atoms[id].x[0] = xn;
				configs[m].atoms[id].x[1] = yn-ly;
				configs[m].atoms[id].x[2] = zn+lz;
				configs[m].atoms[id].i[0] = 0;
				configs[m].atoms[id].i[1] = -1;
				configs[m].atoms[id].i[2] = 1;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//24
			if (yn-ly > 0.0-rc2 && zn-lz > 0.0-rc2){
				configs[m].atoms[id].x[0] = xn;
				configs[m].atoms[id].x[1] = yn-ly;
				configs[m].atoms[id].x[2] = zn-lz;
				configs[m].atoms[id].i[0] = 0;
				configs[m].atoms[id].i[1] = -1;
				configs[m].atoms[id].i[2] = -1;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//25
			if (zn+lz < lz+rc2){
				configs[m].atoms[id].x[0] = xn;
				configs[m].atoms[id].x[1] = yn;
				configs[m].atoms[id].x[2] = zn+lz;
				configs[m].atoms[id].i[0] = 0;
				configs[m].atoms[id].i[1] = 0;
				configs[m].atoms[id].i[2] = 1;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
			//26
			if (zn-lz > 0.0-rc2){
				configs[m].atoms[id].x[0] = xn;
				configs[m].atoms[id].x[1] = yn;
				configs[m].atoms[id].x[2] = zn-lz;
				configs[m].atoms[id].i[0] = 0;
				configs[m].atoms[id].i[1] = 0;
				configs[m].atoms[id].i[2] = -1;
				configs[m].atoms[id].id = id;
				configs[m].atoms[id].tag = configs[m].atoms[n].tag;
				configs[m].atoms[id].type = configs[m].types[n];
				id++;
			}
		}

		// Loop through all atoms, and get their neighbors
	
		int jtag;
		int numneigh;
		for (int n=0; n<configs[m].natomsTotal; n++){
			xi=configs[m].atoms[n].x[0];
			yi=configs[m].atoms[n].x[1];
			zi=configs[m].atoms[n].x[2];
			//fprintf(fh_debug,"n: %d\n", n);
			numneigh = 0;
			for (int j=0; j<configs[m].natomsTotal; j++){
				if (j!=n){
					xj=configs[m].atoms[j].x[0];
					yj=configs[m].atoms[j].x[1];
					zj=configs[m].atoms[j].x[2];

					xij=xj-xi;
					yij=yj-yi;
					zij=zj-zi;

					rsq = xij*xij + yij*yij + zij*zij;

					if (rsq<input->rc*input->rc){
						//fprintf(fh_debug,"  neighbor\n");
						rij = sqrt(rsq);
						jtag = configs[m].atoms[j].tag;
						configs[m].neighlist[n][numneigh] = j;	
						numneigh++;
						//fprintf(fh_debug, "  rij: %f\n", rij);
						//fprintf(fh_debug, "    di: %f, %f, %f\n", xi,yi,zi);
						//fprintf(fh_debug, "    dj: %f, %f, %f\n", xj,yj,zj);
						//fprintf(fh_debug, "    dsq: %f, %f, %f\n", xij*xij,yij*yij,zij*zij);
						//fprintf(fh_debug, "    rsq: %f\n", rsq);
						//fprintf(fh_debug, "    j: %d\n", j);
						//fprintf(fh_debug, "    jtag: %d\n", jtag);
						//fprintf(fh_debug, "    ji: %d, %d, %d\n", configs[m].atoms[j].i[0],configs[m].atoms[j].i[1],configs[m].atoms[j].i[2]);
					    //fprintf(fh_debug, "    djtag: %f, %f, %f\n", configs[m].atoms[jtag].x[0], configs[m].atoms[jtag].x[1], configs[m].atoms[jtag].x[2]);	
						//fprintf(fh_debug, "    xij*xij: %f\n", xij*xij);
						//fprintf(fh_debug, "    yij*yij: %f\n", yij*yij);
						//fprintf(fh_debug, "    zij*zij: %f\n", zij*zij);
					
						/*	
						if (n==543){
							fprintf(fh_debug, "  rij: %f\n", rij);
							fprintf(fh_debug, "    j: %d\n", j);
							fprintf(fh_debug, "    jtag: %d\n", jtag);
						    fprintf(fh_debug, "    ji: %d, %d, %d\n", configs[m].atoms[j].i[0],configs[m].atoms[j].i[1],configs[m].atoms[j].i[2]);
							fprintf(fh_debug, "    dj: %f, %f, %f\n", xj,yj,zj);
					    	fprintf(fh_debug, "    djtag: %f, %f, %f\n", configs[m].atoms[jtag].x[0], configs[m].atoms[jtag].x[1], configs[m].atoms[jtag].x[2]);	
						}
						*/
						
					}
				}
			}

			configs[m].atoms[n].numneigh = numneigh;
			/*
			if (numneigh > 4){
				fprintf(fh_debug, "n: %d\n", n);
			}
			*/
		}
	}

}

/* ----------------------------------------------------------------------
    Calculate descriptors for all atoms in all configurations.
------------------------------------------------------------------------- */

void Systems::calcDescriptors()
{
    //printf(" Calculating descriptors...\n");
    //printf("\n");

    //printf("  %d radial descriptors for each interaction\n", nrad);
    //printf("  %d angular descriptors for each interaction\n", nang);
    //printf("\n");

    // First we need to calculate the number of descriptors

    // Radial descriptors
    int nsrad = 0;
    for (int t=0; t<input->ntypes; t++){
        nsrad = nsrad+input->nrad;
    }
    //printf("%d\n", ns_rad);
    int nsang = 0;
    for (int t1=0; t1<input->ntypes; t1++){
            for (int t2=0; t2<input->ntypes; t2++){
                nsang = nsang+input->nang;
            }
    }

    //printf("  %d radial descriptors for each atom\n", nsrad);
    //printf("  %d angular descriptors for each atom\n", nsang);
    //printf("\n");

    nd = input->structure[0]; // number of descriptors per atom
	
    //printf("  %d total descriptors per atom\n", nd);
    //printf("\n");

    // Allocate descriptor arrays in all configs

    for (int m=0; m<nconfigs; m++){
        configs[m].allocate_descriptors(nd, input->neighmax);
		//mlt->systems->configs[m].allocate_descriptors(nd);
    }

	// Allocate contiguous descriptors array
	
	//memory->allocate(descriptors, natomsTot*nd);

    // Calculate descriptors for all atoms in all configs
    int natoms, ti; //,tj,tk;
    int s; // full descriptor array index
    int s2; // 2-body descriptor index
    int s3; // 3-body descriptor index
    int types2[2];
    int types3[3];
	int indx;
    for (int m=0; m<nconfigs; m++){
    
        natoms = configs[m].natoms;

        for (int n=0; n<natoms; n++){

            s=0;
            s2=0;
            s3=0;
            ti = configs[m].types[n];
            //fprintf(fh_debug,"Atom %d.\n", n);

            // Radial descriptors

            for (int tj=0; tj<input->ntypes; tj++){
                
                for (int d=0; d<input->nrad; d++){

                    //fprintf(fh_debug, " Descriptor %d.\n", s);

                    //double value = descriptor->calc2Body(m,n,t1,d);
					configs[m].descriptors[n][s].value = descriptor->calc2Body(m,n,tj,d);
					configs[m].descriptors[n][s].nbody = 2;
					configs[m].descriptors[n][s].types[0] = ti;
					configs[m].descriptors[n][s].types[1] = tj+1;
					configs[m].descriptors[n][s].types[2] = 0;

					//fprintf(fh_debug, "  %f\n", configs[m].descriptors[n][s].value);
					
                    s++;
                    s2++;
                }
            }

            // Angular descriptors

            for (int tj=0; tj<input->ntypes; tj++){
                for (int tk=0; tk<input->ntypes; tk++){
					if (tk >= tj){ // Don't double count ti-tj-tk and ti-tk-tj interactions!
						for (int d=0; d<input->nang; d++){
						
							configs[m].descriptors[n][s].value = descriptor->calc3Body(m,n,tj,tk,d);
							//fprintf(fh_debug, "%f\n", configs[m].descriptors[n][s].value);
							configs[m].descriptors[n][s].nbody = 3;
							configs[m].descriptors[n][s].types[0] = ti;
							configs[m].descriptors[n][s].types[1] = tj+1;
							configs[m].descriptors[n][s].types[2] = tk+1;
							//fprintf(fh_debug, "  %f\n", configs[m].descriptors[n][s].value);
							s++;
							s3++;
						}
					}
                }
            }

        }
    }

	// Calculate descriptor derivatives (neighbor descriptor derivatives wrt atom n)
	int nbody;
	for (int m=0; m<nconfigs; m++){
		for (int n=0; n<configs[m].natoms; n++){
			//fprintf(fh_debug, "Atom %d.\n", n);
			for (int j=0; j<configs[m].atoms[n].numneigh; j++){
				//fprintf(fh_debug, " jj: %d\n", j);
				for (int s=0; s<nd; s++){
					for (int a=0; a<3; a++){
						//fprintf(fh_debug, "Config %d, atom %d, neighbor %d, descriptor %d, coordinate %d\n", m,n,j,s,a);
						if (configs[m].descriptors[n][s].nbody == 2){
							configs[m].dgdr[n][j][s][a] = descriptor->calcd2Body(m,n,j,s,a,false);
						}
						if (configs[m].descriptors[n][s].nbody == 3){
							configs[m].dgdr[n][j][s][a] = descriptor->calcd3Body(m,n,j,s,a,false);
							//fprintf(fh_debug, "%f\n", configs[m].dgdr[n][j][s][a]);
						}
						//fprintf(fh_debug, "   %f\n", configs[m].dgdr[n][j][s][a]);
					}
				}
			}
		}
	}

	// Calculate descriptor self-derivatives (atom n descriptor derivative wrt atom n)
	for (int m=0; m<nconfigs; m++){
		for (int n=0; n<configs[m].natoms; n++){
			for (int s=0; s<nd; s++){
				for (int a=0; a<3; a++){

					if (configs[m].descriptors[n][s].nbody==2){
						configs[m].dgdrSelf[n][s][a] = descriptor->calcd2Body(m,n,0,s,a,true);	
					}
					if (configs[m].descriptors[n][s].nbody==3){
						configs[m].dgdrSelf[n][s][a] = descriptor->calcd3Body(m,n,0,s,a,true);
					}

				}
			}
		}
	}
	
    /*
    // Check descriptors

    for (int m=0; m<nconfigs; m++){
    
        natoms = configs[m].natoms;

        for (int n=0; n<natoms; n++){

            s=0;
            type = configs[m].types[n];
            //printf("Atom %d.\n", n);

            // Radial descriptors

            for (int t1=0; t1<ntypes; t1++){
                
                configs[m].descriptors[n][s].print_info();
                s++;
            }

            // Angular descriptors

            for (int t1=0; t1<ntypes; t1++){
                for (int t2=0; t2<ntypes; t2++){
                
                    configs[m].descriptors[n][s].print_info();
                    s++;
                }
            }
        }
    }
    */
}



/* ----------------------------------------------------------------------
    Read the weights and store in the "weights" array.
	weights: An array of ANN objects that stores weights for the system.
			 It's useful to put it here since we don't have to loop through
			 every atom of every config when updating weights.
------------------------------------------------------------------------- */

void Systems::readWeights()
{


	memory->allocate(weights,input->ntypes);

	for (int t=0; t<input->ntypes; t++){
		weights[t].type = t+1;
		//printf("%d\n", weights[t].type);
		weights[t].rank = rank;

		weights[t].nhl = input->nhl;
		weights[t].structure = input->structure;
		weights[t].activations = input->activations;
		//fprintf(fh_debug,"m,t = %d,%d\n",m,nets[m].sumGrads[t].type);
		weights[t].annType = 3; // Type 3 is a main weight storage ANN
		weights[t].openDebug();
		weights[t].initialize();
	}


    // Read weights for each atom type
	char weightfile[64];
	double nweights, weight,activation;
	string line;
	for (int t=0; t<input->ntypes; t++){
		sprintf (weightfile, "WEIGHTS%d", t+1);
		ifstream fh(weightfile);
		
		for (int l=0; l<input->nhl+2; l++){

			for (int j=0; j<input->structure[l]+1; j++){

				getline(fh,line);
				stringstream ss(line);
				if (l!=input->nhl+1){ // if we aren't on last layer
					for (int k=0; k<input->structure[l+1]; k++){
						ss >> weight;
						//fprintf(fh_debug,"weight: %f\n", weight);
						weights[t].layers[l].neurons[j].weights[k] = weight;
						//fprintf(fh_debug, "Layer %d, neuron %d, weight %d = %f\n", l,j+1,k+1,weight);
						nweights++;
					}
				}
			}
			
		}

		fh.close();
	}

	/*
	// Check the weights
	for (int t=0; t<input->ntypes; t++){
		for (int l=0; l<input->nhl+1; l++){
			//printf("l: %d\n", l);
			for (int j=0; j<input->structure[l]+1; j++){
				//printf("j: %d\n", j);
				for (int k=0; k<input->structure[l+1]; k++){
					fprintf(fh_debug, "%f\n", weights[t].layers[l].neurons[j].weights[k]);	
				}
			}
		}
	}
	*/

}

/* ----------------------------------------------------------------------
    Build the networks of ANNs for all configs (systems)
------------------------------------------------------------------------- */

void Systems::buildNets()
{

	//fprintf(fh_debug,"Building system networks for %d configs.\n", nconfigs);

	// Build all individual ANNs for all configs
	
	//printf("Building system networks for %d configs.\n", nconfigs);
	
	for (int m=0; m<nconfigs; m++){
		
		// Set settings for all ANNs

		for (int n=0; n<configs[m].natoms; n++){

			configs[m].ann[n].n = n;
			configs[m].ann[n].m = m;
			configs[m].ann[n].rank = rank;

			configs[m].ann[n].nhl = input->nhl;
			configs[m].ann[n].structure = input->structure;
			configs[m].ann[n].activations = input->activations;

			configs[m].ann[n].type = configs[m].types[n];
			configs[m].ann[n].annType = 0; // These are normal ANNs, describing atoms
			configs[m].ann[n].openDebug();
			configs[m].ann[n].initialize(); // Build

			//fprintf(fh_debug, "n: %d\n", n);


			// Set the weights

			for (int l=0; l<input->nhl+1; l++){
			
				for (int j=0; j<input->structure[l]+1; j++){
					configs[m].ann[n].layers[l].neurons[j].weights = \
					weights[configs[m].ann[n].type-1].layers[l].neurons[j].weights;
				}
			}

			// Set the inputs for the ANN

			for (int s=0; s<nd; s++){

				configs[m].ann[n].layers[0].neurons[s].output = configs[m].descriptors[n][s].value;
				//fprintf(fh_debug, " %f\n", configs[m].descriptors[n][s].value);
			}

		}	

		// Build the sumGrads

		for (int t=0; t<input->ntypes; t++){
			configs[m].sumGrads[t].type = t+1;
			configs[m].sumGrads[t].rank = rank;
			configs[m].sumGrads[t].m = m;

			configs[m].sumGrads[t].nhl = input->nhl;
			configs[m].sumGrads[t].structure = input->structure;
			configs[m].sumGrads[t].activations = input->activations;
			//fprintf(fh_debug,"m,t = %d,%d\n",m,nets[m].sumGrads[t].type);
			configs[m].sumGrads[t].annType = 1; // Type 1 ANN stores summed gradients
			configs[m].sumGrads[t].openDebug();
			configs[m].sumGrads[t].initialize();
		}

		// Build the df_dw arrays
		
		for (int t=0; t<input->ntypes; t++){
			//printf("t: %d\n", t);
			for (int n=0; n<configs[m].natoms; n++){
				//printf("n: %d\n",n);
				for (int a=0; a<3; a++){
					//printf("m,t,n,a: %d,%d,%d,%d\n",m,t,n,a);
					configs[m].df_dw[t][n][a].type=t+1;
					configs[m].df_dw[t][n][a].rank = rank;
					configs[m].df_dw[t][n][a].m = m;
					configs[m].df_dw[t][n][a].n = n;

					configs[m].df_dw[t][n][a].nhl = input->nhl;
					configs[m].df_dw[t][n][a].structure = input->structure;
					configs[m].df_dw[t][n][a].activations = input->activations;
					configs[m].df_dw[t][n][a].a = a;
					configs[m].df_dw[t][n][a].annType = 4; // Type 4 ANN stores gradients of forces
					configs[m].df_dw[t][n][a].openDebug();
					//printf("done with loop...\n");
					configs[m].df_dw[t][n][a].initialize();
				}
			}

		}
	
		
		
		// Build the sumfGrads
		for (int t=0; t<input->ntypes; t++){
			configs[m].sumfGrads[t].type = t+1;
			configs[m].sumfGrads[t].rank = rank;
			configs[m].sumfGrads[t].m = m;

			configs[m].sumfGrads[t].nhl = input->nhl;
			configs[m].sumfGrads[t].structure = input->structure;
			configs[m].sumfGrads[t].activations = input->activations;
			//fprintf(fh_debug,"m,t = %d,%d\n",m,nets[m].sumGrads[t].type);
			configs[m].sumfGrads[t].annType = 5; // Type 5 ANN stores summed force gradients
			configs[m].sumfGrads[t].openDebug();
			configs[m].sumfGrads[t].initialize();
		}
		
	}
	
	// Build the ANNs for storing total error grads
	// grads[t] = grads associated with atom type t
	
	memory->allocate(grads,input->ntypes);
	for (int t=0; t<input->ntypes; t++){
		grads[t].type = t+1;
		grads[t].rank = rank;

		grads[t].nhl = input->nhl;
		grads[t].structure = input->structure;
		grads[t].activations = input->activations;
		grads[t].annType = 2; // Type 2 ANN stores total error gradients
		grads[t].openDebug();
		grads[t].initialize();
	}
	
}

/* ----------------------------------------------------------------------
    Zero the gradients for total network.
------------------------------------------------------------------------- */

void Systems::zeroGrads()
{

	// Zero the error gradients
	
	for (int t=0; t<input->ntypes; t++){
		for (int l=0; l<input->nhl+1; l++){
			//printf("l: %d\n", l);
			for (int j=0; j<input->structure[l]+1; j++){
				//printf("j: %d\n", j);
				for (int k=0; k<input->structure[l+1]; k++){
					//layers[l].neurons[j].dWeightsp[k] = 0.0;
					grads[t].layers[l].neurons[j].gradientsp[k] = 0.0;
				}
			}
		}
	}

	// Zero the summed grads in each config
	for (int m=0; m<nconfigs; m++){
		for (int t=0; t<input->ntypes; t++){
			for (int l=0; l<input->nhl+1; l++){
				//printf("l: %d\n", l);
				for (int j=0; j<input->structure[l]+1; j++){
					//printf("j: %d\n", j);
					for (int k=0; k<input->structure[l+1]; k++){
						//layers[l].neurons[j].dWeightsp[k] = 0.0;
						configs[m].sumGrads[t].layers[l].neurons[j].gradients[k] = 0.0;
					}
				}
			}
		}
	}

	// Zero the error
	//error = 0.0;

}

/* ----------------------------------------------------------------------
    Feedforward all the ANNs in every config
------------------------------------------------------------------------- */

void Systems::feedForwardAll()
{

	for (int m=0; m<nconfigs; m++){

		configs[m].feedForward();
	}
}

/* ----------------------------------------------------------------------
    Calculate error
------------------------------------------------------------------------- */

void Systems::calcError()
{


	for (int m=0; m<nconfigs; m++){
		configs[m].calcError();
		energy_error = configs[m].errsq/(2.0*nconfigsTot);
		force_error = (input->wf/(3.0*configs[m].natoms*2.0*nconfigsTot))*configs[m].sum_fsqerr;
		error += energy_error + force_error;
		//error += configs[m].errsq/(2.0*nconfigsTot);
		//printf("wf: %f\n", input->wf);
		//printf("%f\n", (input->wf/(3.0*configs[m].natoms))*configs[m].sum_fsqerr/(2.0*nconfigsTot));
		//error+=(input->wf/(3.0*configs[m].natoms))*configs[m].sum_fsqerr/(2.0*nconfigsTot);
	}

}


/* ----------------------------------------------------------------------
    Backprop all the ANNs in every config
	Also calculate the error while we're at it
------------------------------------------------------------------------- */

void Systems::backpropAll()
{

	// Calculate gradients for all systems
	
	double energy_term;
	double sumfGrads;
	double force_term;
	
	for (int m=0; m<nconfigs; m++){

		configs[m].backprop();
	
		//error += nets[m].errsq/(2.0*input->nconfigs);

		for (int t=0; t<input->ntypes; t++){
			for (int l=0; l<input->nhl+1; l++){
				//printf("l: %d\n", l);
				for (int j=0; j<input->structure[l]+1; j++){
					//printf("j: %d\n", j);
					for (int k=0; k<input->structure[l+1]; k++){
						energy_term = configs[m].errdiff*configs[m].sumGrads[t].layers[l].neurons[j].gradients[k]/nconfigsTot;
						//energy_term = 0.0;
						sumfGrads = configs[m].sumfGrads[t].layers[l].neurons[j].gradients[k];
						//printf("wf: %d\n", input->wf);
						force_term = (input->wf/(3.0*configs[m].natoms))*sumfGrads/nconfigsTot;
						//layers[l].neurons[j].dWeightsp[k] = 0.0;
						//printf("m,t,l,j,k: %d,%d,%d,%d,%d\n", m,t,l,j,k);
						grads[t].layers[l].neurons[j].gradientsp[k] += \
							//configs[m].errdiff*configs[m].sumGrads[t].layers[l].neurons[j].gradients[k]/nconfigsTot;
							//energy_term + force_term;
							force_term;
							//fprintf(fh_debug, "force_term: %f\n", force_term);
							//wf*configs[m].sumfGrads[t].layers[l].neurons[j].gradients[k]/(nconfigsTot*128*3);
					}
				}
			}
		}
	}


}

/* ----------------------------------------------------------------------
    Update weights based on gradients.
------------------------------------------------------------------------- */

void Systems::updateWeights()
{


	// Update weights - will automatically update all ANN weights
		
	for (int t=0; t<input->ntypes; t++){
		for (int l=0; l<input->nhl+1; l++){
			//printf("l: %d\n", l);
			for (int j=0; j<input->structure[l]+1; j++){
				//printf("j: %d\n", j);
				for (int k=0; k<input->structure[l+1]; k++){
					weights[t].layers[l].neurons[j].weights[k] -= \
					input->eta*grads[t].layers[l].neurons[j].gradients[k];  
				}
			}
		}
	}

	

}

/* ----------------------------------------------------------------------
    Write the weights to output files
------------------------------------------------------------------------- */

void Systems::writeWeights()
{

	FILE * fh_w;
	double wt;	
	char filename[64];
	for (int t=0; t<input->ntypes; t++){
		sprintf (filename, "FINAL_WEIGHTS%d", t+1);
		fh_w = fopen(filename,"w");	
		for (int l=0; l<input->nhl+2; l++){

			for (int j=0; j<input->structure[l]+1; j++){

				if (l!=input->nhl+1){ // if we aren't on last layer
					for (int k=0; k<input->structure[l+1]; k++){
						//fprintf(fh_debug,"weight: %f\n", weight);
						wt = weights[t].layers[l].neurons[j].weights[k];
						fprintf(fh_w, "%f ",wt);
					}
					fprintf(fh_w, "\n");
				}
			}
			
		}

		fclose(fh_w);
	}



	

}
