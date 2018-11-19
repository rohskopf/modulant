/*
 input.cpp

 Copyright (c) 2018 Andrew Rohskopf

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

/* ----------------------------------------------------------------------
    The INPUT class does the following:

    Reads input settings from the INPUT file.
    Reads configurations from the CONFIG file.
    Generates a neighborlist for all configurations.
    Calculates descriptors for all atoms in all configurations.
------------------------------------------------------------------------- */

#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include "mpi.h"
#include <math.h>       /* sqrt */
#include <random>
#include "dirent.h"

#include "input.h"
#include "memory.h"
#include "error.h"
#include "systems.h"

using namespace std;

using namespace MODULANT_NS;

Input::Input(MLT *mlt) : Pointers(mlt) {
    //fh_debug = fopen("debug/input/D_INPUT", "w");

	rank = MPI::COMM_WORLD.Get_rank ( ); // Get_rank gets the rank of the calling process in the communicator
	char debug[64];
	sprintf (debug, "debug/input/D_INPUT_PROC%d", rank);
	fh_debug = fopen(debug, "w");
	
}

Input::~Input() 
{

	//memory->deallocate(types);
	//memory->deallocate(e0);
    memory->deallocate(natoms_all);
    //memory->deallocate(configs);
    //memory->deallocate(params2);
    //memory->deallocate(params3);
	//memory->deallocate(descriptors);

    fclose(fh_debug);

};

void Input::readInput()
{
    /* Read INPUT file */

    //int rank = MPI::COMM_WORLD.Get_rank ( ); // Get_rank gets the rank of the calling process in the communicator
    //printf("Reading input on proc %d.\n", rank);

    string line;

    // Declare scalar inputs
    double value;

    // Open INPUT file
    ifstream INPUT("INPUT");
    // Ignore the first line
    getline(INPUT, line); 
    string characters;
    // Get input variables
    for (int i=1; i<=13; i++)
    {
        getline(INPUT, line);
        switch (i)
        {
            case 1:{ 
                stringstream ss(line);
                ss >> characters >> nconfigs;
            }
            case 2:{ 
                stringstream ss(line);
                ss >> characters >> nrad;
            }
            case 3:{ 
                stringstream ss(line);
                ss >> characters >> nang;
            }
            case 4:{ 
                stringstream ss(line);
                ss >> characters >> ntypes;
            }
            case 5:{ 
                stringstream ss(line);
                ss >> characters >> rc;
            }
            case 6:{ 
                stringstream ss(line);
                ss >> characters >> neighmax;
            }

            case 7:{ 
                stringstream ss(line);
                ss >> characters >> nhl;
                memory->allocate(structure, nhl+2);
				memory->allocate(activations, nhl+2);
            }
            case 8:{ 
                stringstream ss(line);
                ss >> characters;
                for (int k=0; k<nhl+2; k++){
                    ss >> structure[k];
                }

            }
            case 9:{ 
                stringstream ss(line);
                ss >> characters;
                for (int k=0; k<nhl+2; k++){
                    ss >> activations[k];
					//printf("%c\n", test);
                }

            }
            case 10:{ 
                stringstream ss(line);
                ss >> characters >> nsamples;
            }
			case 11:{ 
                stringstream ss(line);
                ss >> characters >> nepochs;
            }
			case 12:{ 
                stringstream ss(line);
                ss >> characters >> eta;
            }
			case 13:{ 
                stringstream ss(line);
                ss >> characters >> wf;
				//printf("wf: %f\n", wf);
            }
  
        } // switch (i)

    } // for (int i=1..)

    INPUT.close();

    // Allocate some arrays

    memory->allocate(natoms_all, nconfigs);
    //memory->allocate(configs, nconfigs);

}

/* ----------------------------------------------------------------------
    Read the CONFIGS file and set basic input quantities for each config.
    Set number of atoms.
    Set positions and forces.
    Set potential energy and box size.
------------------------------------------------------------------------- */

void Input::readconfigs()
{
    // Read CONFIGS file
	/*
    ifstream fh_configs("CONFIGS");
    string line;

    int type;
    double x,y,z,fx,fy,fz;
    double pe; 
    double lx, ly, lz;

    memory->allocate(e0,nconfigs);

    int natoms;
    for (int m=0; m<nconfigs; m++){

        getline(fh_configs,line);
        stringstream ss(line);
        ss >> natoms;
        natoms_all[m] = natoms;
        configs[m].set_natoms(natoms, neighmax);
        //printf("natoms: %d\n", configs[m].natoms);

        for (int n=0; n<natoms; n++){

            configs[m].tags[n] = n;

            getline(fh_configs,line);
            stringstream ss2(line);
            ss2 >> type >> x >> y >> z >> fx >> fy >> fz;
            configs[m].types[n] = type;
            configs[m].x[n][0] = x;
            configs[m].x[n][1] = y;
            configs[m].x[n][2] = z;
            configs[m].f[n][0] = fx;
            configs[m].f[n][1] = fy;
            configs[m].f[n][2] = fz;
            //printf("%d %f %f %f %f %f %f\n", type,x,y,z,fx,fy,fz);
        }

        getline(fh_configs,line);
        stringstream ss3(line);
        ss3 >> pe;
        configs[m].pe = pe;
		e0[m]=pe;

        getline(fh_configs,line);
        stringstream ss4(line);
        ss4 >> lx >> ly >> lz;
        configs[m].box[0] = lx;
        configs[m].box[1] = ly;
        configs[m].box[2] = lz;
        
    }

    fh_configs.close();

	// Calculate total number of atoms in all configs

	natomsTot=0;
	for (int m=0; m<nconfigs; m++){
		natomsTot += natoms_all[m];
	}

	// Make a 1D atom types array for MPI
	
	memory->allocate(types,natomsTot);

	int indx;
	for (int m=0; m<nconfigs; m++){
		for (int n=0; n<natoms_all[m]; n++){
			indx = m*(natoms_all[m]) + n;
			types[indx] = configs[m].types[n];
		}
	}

	*/

	
	// Get list of directories in /configs
		
	string dir = "configs/";
    DIR *dp;
	struct dirent *dirp;
	if((dp  = opendir(dir.c_str())) == NULL) {
		error->exit("input.cpp", "No configs/ directory.");
	}	
	//mlt->systems->initConfigs();

	nconfigs=-2; // subtract 2 since ./ and ../ directories are included
	while ((dirp = readdir(dp)) != NULL) {
		nconfigs++;
		//cout << dirp->d_name << endl;
		//files.push_back(string(dirp->d_name));
	}
	if (nconfigs % mlt->procs !=0) error->exit("input.cpp", "Number of configs not divisible by procs.");
	//printf("Found %d configs.\n", configCount);


}

/* ----------------------------------------------------------------------
    Read the PARAMS file and store descriptor parameters
------------------------------------------------------------------------- */

void Input::readparams()
{

    // Read PARAMS file

    //printf(" Reading PARAMS...\n");

    ifstream fh_params("PARAMS");
    string line;

    memory->allocate(params2, nrad*ntypes*ntypes, 2);
    memory->allocate(params3, nang*ntypes*ntypes*ntypes, 3);

    // Get 2-body params

    double eta, rs;
    int s2=0;
    for (int t=0; t<ntypes; t++){
        //printf("%d\n", t);
        for (int t1=0; t1<ntypes; t1++){
            for (int s=0; s<nrad; s++){
                getline(fh_params,line);
                stringstream ss(line);
                ss >> eta >> rs;
                params2[s2][0]=eta;
                params2[s2][1]=rs;
                s2++;
            }
        }
    }


    // Get 3-body params

    double zeta,lambda;
    int s3=0;
    for (int t=0; t<ntypes; t++){
        //printf("%d\n", t);
        for (int t1=0; t1<ntypes; t1++){
            for (int t2=0; t2<ntypes; t2++){
                for (int s=0; s<nrad; s++){
                    getline(fh_params,line);
                    stringstream ss(line);
                    ss >> zeta >> eta >> lambda;
                    params3[s3][0]=zeta;
                    params3[s3][1]=eta;
                    params3[s3][2]=lambda;
                    s3++;
                }
            }
        }
    }


    fh_params.close();

}

/* ----------------------------------------------------------------------
    Loop through all configurations and calculate a full neighborlist for each
------------------------------------------------------------------------- */

void Input::neighbors()
{
    //printf(" Calculating neighborlists...\n");

    double cutsq = rc*rc;
    //double boxinvx = 1.0/lx;
    //double boxinvy = 1.0/ly;
    //double boxinvz = 1.0/lz;

    int neighcount;
    int imagecounter; // used to store new images in the list of positions
    int nearintx, nearinty, nearintz;
    double xi,yi,zi,xj,yj,zj;
    double xjp, yjp, zjp; // Image positions
    double xij,yij,zij, rsq, rij;

    double xtmp, ytmp, ztmp;

    int natoms;
    double lx,ly,lz;
    int types;

    for (int m=0; m<nconfigs; m++){

        natoms = configs[m].natoms;
        lx = configs[m].box[0];
        ly = configs[m].box[1];
        lz = configs[m].box[2];
        imagecounter = natoms;
        neighcount = 0;

        for (int i=0; i<natoms; i++){

            neighcount = 0;
            xtmp = configs[m].x[i][0];
            ytmp = configs[m].x[i][1];
            ztmp = configs[m].x[i][2];

            //fprintf(fh_debug, "Atom %d.\n", i);

            for (int j=0; j < natoms; j++){
                if (j != i)
                {

                    xij = xtmp - configs[m].x[j][0];
                    yij = ytmp - configs[m].x[j][1];
                    zij = ztmp - configs[m].x[j][2];

                    nearintx = std::round(xij*(1.0/lx));
                    nearinty = std::round(yij*(1.0/ly));
                    nearintz = std::round(zij*(1.0/lz));

                    xij = xij - lx*nearintx;
                    yij = yij - ly*nearinty;
                    zij = zij - lz*nearintz;

                    rsq = xij*xij + yij*yij + zij*zij;

                    if (rsq < cutsq){

                        rij = sqrt(rsq);

                        //fprintf(fh_debug, "rij: %f\n", rij);

                        /*
                        Need to add neighbor index to neighlist and position to x.
                        If neighbor is periodic, a new index is created along with a new position.
                        */

                        //neightags[i][neighcount] = j;

                        if (nearintx == 0.0 && nearinty == 0.0 && nearintz == 0.0){
                            configs[m].neighlist[i][neighcount] = j;
                        }
                        
                        if (nearintx != 0.0 || nearinty != 0.0 || nearintz != 0.0){

                            // In this case, we must add this image of j to our positions array

                            xjp = configs[m].x[i][0] - xij;
                            yjp = configs[m].x[i][1] - yij;
                            zjp = configs[m].x[i][2] - zij;

                            configs[m].x[imagecounter][0] = xjp;
                            configs[m].x[imagecounter][1] = yjp;
                            configs[m].x[imagecounter][2] = zjp;

                            // Likewise to our tags array.

                            //printf("%d\n", j);
                            configs[m].tags[imagecounter] = j;

                            // And then add the imagecount to the neighbor list for indexing later.

                            configs[m].neighlist[i][neighcount] = imagecounter;

                            imagecounter++;

                        } // if (nearintx == 0 && nearinty == 0 && nearintz == 0){

                        neighcount++;
                
                    } // if (rsq < cutsq){

                } // if (j!=i)

            } // for (int j=0; j<natoms; j++){

            configs[m].numneigh[i] = neighcount;

        } // for (int i=0; i<natoms; i++){

    }


    /*
    // Check the tags array
    int j;
    int jtag;
    for (int m=0; m<nconfigs; m++){
        natoms = configs[m].natoms;
        for (int n=0; n<natoms; n++){
            printf("n: %d\n", n);
            for (int jj=0; jj<configs[m].numneigh[n]; jj++){
                //printf("%d\n", jj);
                j = configs[m].neighlist[n][jj];
                jtag=configs[m].tags[j];
                printf("jtag: %d\n", jtag);
            }
        }
    }
    */



}

/* ----------------------------------------------------------------------
    Allocate descriptor array for every configuration.
    Calculate descriptors for all atoms in all configurations.
------------------------------------------------------------------------- */

void Input::calcDescriptors()
{
    //printf(" Calculating descriptors...\n");
    //printf("\n");

    //printf("  %d radial descriptors for each interaction\n", nrad);
    //printf("  %d angular descriptors for each interaction\n", nang);
    //printf("\n");

    // First we need to calculate the number of descriptors

    // Radial descriptors
    int nsrad = 0;
    for (int t=0; t<ntypes; t++){
        nsrad = nsrad+nrad;
    }
    //printf("%d\n", ns_rad);
    int nsang = 0;
    for (int t1=0; t1<ntypes; t1++){
            for (int t2=0; t2<ntypes; t2++){
                nsang = nsang+nang;
            }
    }

    //printf("  %d radial descriptors for each atom\n", nsrad);
    //printf("  %d angular descriptors for each atom\n", nsang);
    //printf("\n");

    nd = nsrad + nsang; // number of descriptors per atom
	
    //printf("  %d total descriptors per atom\n", nd);
    //printf("\n");

    // Allocate descriptor arrays in all configs

    for (int m=0; m<nconfigs; m++){
        //configs[m].allocate_descriptors(nd);
		//mlt->systems->configs[m].allocate_descriptors(nd);
    }
	for (int m=0; m<cpp; m++){
		//mlt->systems->configs[m].allocate_descriptors(nd);
	}

	// Allocate contiguous descriptors array
	
	//memory->allocate(descriptors, natomsTot*nd);

    // Calculate descriptors for all atoms in all configs

    int natoms, type;
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
            type = configs[m].types[n];
            //printf("Atom %d.\n", n);

            // Radial descriptors

            for (int t1=0; t1<ntypes; t1++){
                
                for (int d=0; d<nrad; d++){
					/*
                    //printf("Descriptor %d.\n", s);
                    configs[m].descriptors[n][s].set_info2(2,type,t1+1, \
                    calc2body(m,n,t1,params2[s2]));
                    //double value = calc2body(m,n,t1,params2[s2]);
                    //configs[m].descriptors[n][s].print_info();
					indx = m*natoms*nd + n*nd + s;
					//fprintf(fh_debug, "%d\n", indx);
					descriptors[indx].value=configs[m].descriptors[n][s].value;
					//descriptors[indx].nbody=configs[m].descriptors[n][s].nbody;
					//descriptors[indx].types[0]=configs[m].descriptors[n][s].type;
					//descriptors[indx].types[1]=configs[m].descriptors[n][s].t1;
					//descriptors[indx].types[2]=0;
					descriptors[indx].m=m;
					descriptors[indx].n=n;
					descriptors[indx].s=s;
					descriptors[indx].nbody=configs[m].descriptors[n][s].nbody;
					descriptors[indx].types[0]=configs[m].descriptors[n][s].type;
					descriptors[indx].types[1]=configs[m].descriptors[n][s].t1;
					descriptors[indx].types[2]=0;
					*/
                    s++;
                    s2++;
                }
            }

            // Angular descriptors

            for (int t1=0; t1<ntypes; t1++){
                for (int t2=0; t2<ntypes; t2++){
                    for (int d=0; d<nang; d++){
						/*
                        //printf("Descriptor %d.\n", s);
                        configs[m].descriptors[n][s].set_info3(3,type,t1+1,t2+1,\
                        calc3body(m,n,t1,t2,params3[s3]));
                        //configs[m].descriptors[n][s].print_info();
                        //double value = calc3body(m,n,t1,t2,params3[s3]);
						indx = m*natoms*nd + n*nd + s;
						//fprintf(fh_debug, "%d\n", indx);
						descriptors[indx].value=configs[m].descriptors[n][s].value;
						//descriptors[indx].nbody=configs[m].descriptors[n][s].nbody;
						//descriptors[indx].types[0]=configs[m].descriptors[n][s].type;
						//descriptors[indx].types[1]=configs[m].descriptors[n][s].t1;
						//descriptors[indx].types[2]=configs[m].descriptors[n][s].t2;
						descriptors[indx].m=m;
						descriptors[indx].n=n;
						descriptors[indx].s=s;
						descriptors[indx].nbody=configs[m].descriptors[n][s].nbody;
						descriptors[indx].types[0]=configs[m].descriptors[n][s].type;
						descriptors[indx].types[1]=configs[m].descriptors[n][s].t1;
						descriptors[indx].types[2]=configs[m].descriptors[n][s].t2;
						*/
                        s++;
                        s3++;
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
    Calculate 2-body descriptors from 
    http://amp.readthedocs.io/en/latest/theory.html.

    Inputs:
        m: Configuration index
        n: Atom index
        t1: Type of atom this descriptor describes an interaction with
        p: Parameter values.
            p[0]: eta
            p[1]: rs
------------------------------------------------------------------------- */


double Input::calc2body(int m, int n, int t1, double *p)
{

    //printf("Calculating 2-body X-%d descriptor for atom %d in config %d!\n", t1+1, n, m);

    double eta=p[0];
    double rs=p[1];

    int natoms = configs[m].natoms;
    int **neighlist = configs[m].neighlist;
    int *numneigh = configs[m].numneigh;
    int *types = configs[m].types;
    int *tags = configs[m].tags;
    double **x = configs[m].x;

    double xi = x[n][0];
    double yi = x[n][1];
    double zi = x[n][2];
    double xj,yj,zj;
    double xij,yij,zij,rij;
    int j; // neighbor index
    int jtag; // index of neighbor image in the original box
    double value = 0.0; // descriptor value
    double fc, rdiffsq;
    
    //printf("Atom %d has %d neighs.\n", n, numneigh[n]);
    //printf("ri: %f %f %f\n", xi,yi,zi);
    //printf("n: %d\n", n);
    for (int jj=0; jj<numneigh[n]; jj++)
    {

        j = neighlist[n][jj];
        jtag = tags[j];
        
        if (types[jtag]-1==t1){
        
            //printf("jtag: %d\n", jtag);

            xj = x[j][0];
            yj = x[j][1];
            zj = x[j][2];

            xij = xi-xj;
            yij = yi-yj;
            zij = zi-zj;

            rij = sqrt(xij*xij + yij*yij + zij*zij);

            fc = 0.5*(1.0 + cos(M_PI*(rij/rc)));

            rdiffsq = (rij-rs)*(rij-rs);

            //printf("fc: %f\n", fc);

            value += exp(-1.0*eta*rdiffsq/(rc*rc))*fc;

            //printf("term: %f\n", exp(-1.0*eta*rdiffsq/(rc*rc))*fc); 

            //printf("rij: %f\n", rij);

        }

    }

    //printf("2-body X-%d descriptor for atom %d in config %d = %f.\n", t1+1, n, m, value);
    //printf("params: %f %f\n", eta,rs);

    return value;
    

}

/* ----------------------------------------------------------------------
    Calculate 3-body descriptors from 
    http://amp.readthedocs.io/en/latest/theory.html.

    Inputs:
        m: Configuration index
        n: Atom index
        t1: First type of atom this descriptor describes an interaction with
        t1: Second type of atom this descriptor describes an interaction with
        p: Parameter values.
            p[0]: zeta
            p[1]: eta
            p[2]: lambda
------------------------------------------------------------------------- */


double Input::calc3body(int m, int n, int t1, int t2, double *p)
{

    //printf("Calculating %d-%d-%d descriptor for atom %d in config %d!\n", t1+1, t2+1,n, m);

    double zeta=p[0];
    double eta=p[1];
    double lambda=p[2];

    int natoms = configs[m].natoms;
    int **neighlist = configs[m].neighlist;
    int *numneigh = configs[m].numneigh;
    int *types = configs[m].types;
    int *tags = configs[m].tags;
    double **x = configs[m].x;

    int itag = tags[n];

    //fprintf(fh_debug,"Calculating %d-%d-%d descriptor for atom %d in config %d!\n", \
    //                  types[itag], t1+1, t2+1,n, m);

    double xi = x[n][0];
    double yi = x[n][1];
    double zi = x[n][2];
    double xj,yj,zj,xk,yk,zk;
    double xij,yij,zij,rij;
    double xik,yik,zik,rik;
    double xjk,yjk,zjk,rjk;
    double costheta,theta;
    int j,k; // neighbor index
    int jtag,ktag; // index of neighbor image in the original box
    double value = 0.0; // descriptor value
    double fcij,fcik,fcjk,rallsq,rdiffsq;
    
    //printf("Atom %d has %d neighs.\n", n, numneigh[n]);
    //printf("ri: %f %f %f\n", xi,yi,zi);
    //printf("n: %d\n", n);
    for (int jj=0; jj<numneigh[n]; jj++)
    {

        j = neighlist[n][jj];
        jtag = tags[j];
        
        if (types[jtag]-1==t1){

            for (int kk=0; kk<numneigh[n]; kk++){

                k = neighlist[n][kk];
                ktag = tags[k];

                if (k != j){

                    if (types[ktag]-1==t2){

                        //fprintf(fh_debug, "%d %d %d\n", types[itag],types[jtag],types[ktag]);
                        //fprintf(fh_debug, "%d %d %d\n", itag,jtag,ktag);
                        
                        xj=x[j][0];
                        yj=x[j][1];
                        zj=x[j][2];
                        xk=x[k][0];
                        yk=x[k][1];
                        zk=x[k][2];

                        xij=x[n][0]-xj;
                        yij=x[n][1]-yj;
                        zij=x[n][2]-zj;
                        xik=x[n][0]-xk;
                        yik=x[n][1]-yk;
                        zik=x[n][2]-zk;
                        xjk=xj-xk;
                        yjk=yj-yk;
                        zjk=zj-zk;

                        rij = sqrt(xij*xij+yij*yij+zij*zij);
                        rik = sqrt(xik*xik+yik*yik+zik*zik);
                        rjk = sqrt(xjk*xjk+yjk*yjk+zjk*zjk);

                        if (rij<rc) fcij = 0.5*(1.0 + cos(M_PI*(rij/rc)));
                        else fcij = 0.0;
                        if (rik<rc) fcik = 0.5*(1.0 + cos(M_PI*(rik/rc)));
                        else fcik = 0.0;
                        if (rjk<rc) fcjk = 0.5*(1.0 + cos(M_PI*(rjk/rc)));
                        else fcjk = 0.0;
    
                        rallsq = rij*rij + rik*rik + rjk*rjk;

                        costheta = (xij*xik + yij*yik + zij*zik)/(rij*rik);
                        theta = acos(costheta);

                        value += pow((1+lambda*costheta),zeta)*exp(-1.0*eta*rallsq/(rc*rc)) \
                                 *fcij*fcik*fcjk;

                        //fprintf(fh_debug, "theta: %f\n", theta*(180/3.141592653589));
                        //fprintf(fh_debug,"%f %f %f\n", rij,rik,rjk);
                        //fprintf(fh_debug, "rij: %f\n", rij);
                        //fprintf(fh_debug, "rik: %f\n", rik);
                        //fprintf(fh_debug, "rjk: %f\n", rjk);
                        /* 
                        fprintf(fh_debug, "pow((1+lambda*costheta),zeta): %f\n", pow((1+lambda*costheta),zeta));
                        fprintf(fh_debug, "exp(-1.0*eta*rallsq/(rc*rc)): %f\n", exp(-1.0*eta*rallsq/(rc*rc)));
                        fprintf(fh_debug, "fcij: %f\n", fcij);
                        fprintf(fh_debug, "fcik: %f\n", fcik);
                        fprintf(fh_debug, "fcjk: %f\n", fcjk);
                        fprintf(fh_debug, "value: %f\n", value);
                        */


                    }

                }

            }

        }

    }

    value *= pow(2,1-zeta);

    //fprintf(fh_debug, "%d-%d-%d descriptor for atom %d in config %d = %f.\n", types[itag],t1+1,\
    //                   t2+1,n, m, value);
    //printf("params: %f %f\n", eta,rs);

    return value;
    

}
