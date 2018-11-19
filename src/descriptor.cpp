/*
The descriptor class takes in a neighborlist and:
1) Filters it so that it contains relevant atom types
2) Calculates descriptors
3) Calculates derivatives?

*/

#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include <cmath>
#include <stdlib.h>     /* exit, EXIT_FAILURE */
#include "mpi.h"

#include "memory.h"
#include "error.h"
#include "input.h"
#include "systems.h"
#include "descriptor.h"

using namespace std;

using namespace MODULANT_NS;

Descriptor::Descriptor(MLT *mlt) : Pointers(mlt) {

	int rank = MPI::COMM_WORLD.Get_rank ( ); // Get_rank gets the rank of the calling process in the communicator
	char debug[64];
	sprintf (debug, "debug/descriptor/D_DESCRIPTOR_PROC%d", rank);
	fh_debug = fopen(debug, "w");
    
}

Descriptor::~Descriptor() 
{

	memory->deallocate(params2);
	memory->deallocate(params3);

	fclose(fh_debug);
};

/* ----------------------------------------------------------------------
   Simple factorial function for combinatorics 
------------------------------------------------------------------------- */


int Descriptor::factorial(int num)
{

	int value = num;
	for (int i=1; i<num; i++){
		value*=(num-i);
	}
	return value;
}

/* ----------------------------------------------------------------------
    Read params files for each atom type
	params2[t][g][p] = gth 2-body parameter p atom type t
	params3[t][g][p] = gth 3-body parameter p atom type t
	g runs from 0 to nrad*ntypes for 2-body
	g runs from 0 to nang*ntypes*ntypes for 3-body
------------------------------------------------------------------------- */


void Descriptor::readParams()
{

	// Read PARAMS file
	
	int nrad = mlt->input->nrad;
	int nang = mlt->input->nang;
	int ntypes = mlt->input->ntypes;

    //printf(" Reading PARAMS...\n");

	// Allocate 2body descriptors
	
	int n2body = ntypes; // Number of 2body interactions
    memory->allocate(params2, ntypes, nrad*ntypes, 2);

	// Allocate 3body descriptors

	int n3body = factorial(ntypes)/(factorial(2)-factorial(ntypes-2)) + ntypes; // number of 3body interactions
    memory->allocate(params3, ntypes, nang*n3body, 3);

	int nd = n2body*nrad + n3body*nang;

	//printf(" n2body: %d\n", n2body);
	//printf(" n3body: %d\n", n3body);
	//printf(" nd: %d\n", nd);

	//if (rank==0) printf(" %d descriptors per atom.\n",nd);

	if (nd != input->structure[0]) mlt->error->exit("descriptor.cpp", "Number of descriptors not equal to number of NN inputs");

	//fprintf(fh_debug, "n3body: %d\n", n3body);

	char paramfile[64];
	string line;
	
	for (int t1=0; t1<ntypes; t1++){

		sprintf (paramfile, "PARAMS%d", t1+1);
		ifstream fh(paramfile);

		// Get 2-body params

		int s2=0;
		for (int t2=0; t2<ntypes; t2++){
			for (int d=0; d<nrad; d++){
				getline(fh,line);
				stringstream ss(line);
				ss >> params2[t1][s2][0] >> params2[t1][s2][1];
				s2++;
			}

		}
		
		// Get 3-body params

		int s3=0;
		for (int t2=0; t2<ntypes; t2++){
			for (int t3=0; t3<ntypes; t3++){
				if (t3>=t2){
					for (int d=0; d<nang; d++){
						getline(fh,line);
						stringstream ss(line);
						ss >> params3[t1][s3][0] >> params3[t1][s3][1] >> params3[t1][s3][2];
						s3++;
						//cout << line << endl;
					}
				}
			}

		}
		
		fh.close();
	}

}

/* ----------------------------------------------------------------------
    Calculate 2-body descriptor from 
    http://amp.readthedocs.io/en/latest/theory.html.

    Inputs:
        m: Configuration index
        n: Atom index
        tj: Type of atom this descriptor describes an interaction with
			(starts at 0)
		d: Radial descriptor index to get the proper parameters  
  		   d runs from [0,nrad-1]		
------------------------------------------------------------------------- */


double Descriptor::calc2Body(int m, int n, int tj, int d)
{

	int ti = mlt->systems->configs[m].types[n];
	tj++;

	//fprintf(fh_debug, "Calculating %dth 2-body %d-%d descriptor for atom %d in config %d.\n", d+1,ti,tj,n+1,m+1);
	
	// Find which type of interaction this is
	
	int interactionIndex = input->nrad*(tj - 1)+d;
    double eta=params2[ti-1][interactionIndex][0];
    double rs=params2[ti-1][interactionIndex][1];
	double rc = input->rc;

	//fprintf(fh_debug," params: %f %f\n", eta, rs);

	double value = 0.0; // descriptor value
	
    int natoms = mlt->systems->configs[m].natoms;
    int **neighlist = mlt->systems->configs[m].neighlist;
    int *numneigh = mlt->systems->configs[m].numneigh;
    int *types = mlt->systems->configs[m].types;
    int *tags = mlt->systems->configs[m].tags;
    double **x = mlt->systems->configs[m].x;

    //double xi = x[n][0];
    //double yi = x[n][1];
    //double zi = x[n][2];
	double xi = mlt->systems->configs[m].atoms[n].x[0];
	double yi = mlt->systems->configs[m].atoms[n].x[1];
	double zi = mlt->systems->configs[m].atoms[n].x[2];
    double xj,yj,zj;
    double xij,yij,zij,rij;
    int j; // neighbor index
    int jtag; // index of neighbor image in the original box
    double fc, rdiffsq;
    
	
    //fprintf(fh_debug,"Atom %d has %d neighs.\n", n, numneigh[n]);
    //printf("ri: %f %f %f\n", xi,yi,zi);
    //printf("n: %d\n", n);
    for (int jj=0; jj<mlt->systems->configs[m].atoms[n].numneigh; jj++)
    {

        j = neighlist[n][jj];
        jtag = mlt->systems->configs[m].atoms[j].tag;

		//fprintf(fh_debug," j: %d\n", j);
		//fprintf(fh_debug," jtag: %d\n", jtag);
		//fprintf(fh_debug," types[jtag]: %d\n", types[jtag]);
		//fprintf(fh_debug," jtype: %d\n", mlt->systems->configs[m].atoms[j].type);
        
        if (mlt->systems->configs[m].atoms[j].type==tj){
        
            //xj = x[j][0];
            //yj = x[j][1];
            //zj = x[j][2];
			xj = mlt->systems->configs[m].atoms[j].x[0];
			yj = mlt->systems->configs[m].atoms[j].x[1];
			zj = mlt->systems->configs[m].atoms[j].x[2];

            xij = xi-xj;
            yij = yi-yj;
            zij = zi-zj;

            rij = sqrt(xij*xij + yij*yij + zij*zij);

            fc = 0.5*(1.0 + cos(M_PI*(rij/rc)));

            rdiffsq = (rij-rs)*(rij-rs);

            //fprintf(fh_debug, " fc: %f\n", fc);

            value += exp(-1.0*eta*rdiffsq)*fc;

            //printf("term: %f\n", exp(-1.0*eta*rdiffsq/(rc*rc))*fc); 

            //fprintf(fh_debug,"  rij: %f\n", rij);

        }

    }

    //printf("2-body X-%d descriptor for atom %d in config %d = %f.\n", t1+1, n, m, value);
    //printf("params: %f %f\n", eta,rs);
	
	//fprintf(fh_debug, "   value: %f\n", value);

	
    return value;

}

/* ----------------------------------------------------------------------
    Calculate derivative of 2-body descriptor from
    http://amp.readthedocs.io/en/latest/theory.html.
	wrt atom coordinate "a".

	Derivative of 2-body descriptor of neighbor "jj" wrt atom "n" coordinate "a".

    Inputs:
        m: Configuration index
        n: Atom index for which atom this derivative is taken respect to.
		jj: Neighbor of atom n index (irrelevant if self=true).
			This is the atom with which the descriptor derivative is being taken.
		s: Descriptor index
		a: Cartesian coordinate (1,2, or 3)
		self: True if self-derivative (itag == jtag)
			  False if neighbor derivative (itag != jtag)	
------------------------------------------------------------------------- */


double Descriptor::calcd2Body(int m, int n, int jj, int s, int a, bool self)
{

	int nbody;
	nbody = mlt->systems->configs[m].descriptors[n][s].nbody;
	if (nbody != 2) mlt->error->exit("descriptor.cpp", "Caught bad descriptor body type in calcd2Body");

	int **neighlist,*numneigh,*types,*tags;
	neighlist = mlt->systems->configs[m].neighlist;
	numneigh = mlt->systems->configs[m].numneigh;
	types = mlt->systems->configs[m].types;
	tags = mlt->systems->configs[m].tags;
	int ti,tj;
	int itag,j,jtag;
	int dtypes[2]; // descriptor types

	//ti = types[n];
	ti=mlt->systems->configs[m].atoms[n].type;
	//itag = tags[n];
	itag = mlt->systems->configs[m].atoms[n].tag;

	if (self){
		j = n;
	}
	else{
		j = neighlist[n][jj];
	}
	//jtag = tags[j];
	//tj = types[jtag];
	jtag = mlt->systems->configs[m].atoms[j].tag;
	tj = mlt->systems->configs[m].atoms[n].type;

	dtypes[0] = mlt->systems->configs[m].descriptors[jtag][s].types[0];
	dtypes[1] = mlt->systems->configs[m].descriptors[jtag][s].types[1];

	//if (self){
	//fprintf(fh_debug, "Config %d, atom %d, neighbor %d, descriptor %d, coordinate %d\n", m,n,jj,s,a);
	//fprintf(fh_debug, "n,jj,s,a: %d,%d,%d,%d\n", n,jj,s,a);
	//fprintf(fh_debug, "jtag: %d\n", jtag);
	//}
	
	//fprintf(fh_debug, " j: %d\n", j);
	//fprintf(fh_debug, " jtag: %d\n", jtag);
	//fprintf(fh_debug, " tj: %d\n", tj);
	//fprintf(fh_debug, " numneigh[j]: %d\n", numneigh[j]);

	double **x;
	x = mlt->systems->configs[m].x;

	//fprintf(fh_debug, " dti,dtj: %d,%d\n",dtypes[0],dtypes[1]);
	//fprintf(fh_debug, " tn,tj: %d,%d\n",ti,tj);
	
    double value=0.0;	
	
	if (tj != dtypes[0]) mlt->error->exit("descriptor.cpp", "Calculating descriptor derivative for wrong atom type");

	if (ti == dtypes[1]){ // if ti != dtypes[1], then this derivative is zero
	
		//fprintf(fh_debug,"Descriptor type %d,%d\n", ti,tj);

		int interactionIndex = s; // interaction index for params is simply the total descriptor index here

		double eta = params2[ti-1][interactionIndex][0];
		double rs = params2[ti-1][interactionIndex][1];
		double rc=input->rc;

		//fprintf(fh_debug, "interactionIndex: %d\n", interactionIndex);
		//fprintf(fh_debug, "params: %f %f \n", eta,rs);
	
		// Calculate the derivative
		//
		// While looping and summer value over neighbors l of atom j!!!

		double xj,yj,zj,xl,yl,zl;
		double xjl,yjl,zjl,rjl;
		double ajl; // displacement of this particular coordinate
		int l,ltag,tl;
		double fc, dfcdr, drda;

		xj = mlt->systems->configs[m].atoms[j].x[0];
		yj = mlt->systems->configs[m].atoms[j].x[1];
		zj = mlt->systems->configs[m].atoms[j].x[2];

		//fprintf(fh_debug, "dj: %f,%f,%f\n", xj,yj,zj);

		for (int ll=0; ll<mlt->systems->configs[m].atoms[j].numneigh; ll++){
			l = neighlist[j][ll];
			ltag = mlt->systems->configs[m].atoms[l].tag;
			tl = mlt->systems->configs[m].atoms[l].type;

			xl = mlt->systems->configs[m].atoms[l].x[0];
			yl = mlt->systems->configs[m].atoms[l].x[1];
			zl = mlt->systems->configs[m].atoms[l].x[2];

			xjl = xl-xj;
			yjl = yl-yj;
			zjl = zl-zj;
			rjl = sqrt(xjl*xjl+yjl*yjl+zjl*zjl);
			//ajl = x[l][a]-x[j][a];
			ajl = mlt->systems->configs[m].atoms[l].x[a] - mlt->systems->configs[m].atoms[j].x[a];
			//fprintf(fh_debug, "  l: %d\n", l);
			//fprintf(fh_debug, "   ltag: %d\n", ltag);
			//fprintf(fh_debug, "   rjl: %f\n", rjl);
			//fprintf(fh_debug, "  tj,tl: %d,%d\n", tj,tl);
			//fprintf(fh_debug, "  dl: %f,%f,%f\n", xl,yl,zl);
			//fprintf(fh_debug, "  ajl: %f\n", ajl);

			//dfcdr = calcdFc(rjl);
			//drda = calcdR(itag,jtag,ltag,ajl,rjl);

			value+=calcd2BodyTerm(interactionIndex,ti,itag,jtag,ltag,ajl,rjl);
			//fprintf(fh_debug, "   value: %f\n", value);
		}

	}
	else {
		value = 0.0;
	}

	//if (self){
	//fprintf(fh_debug, " value = %f\n", value);
	//}	
	
	return value;

}

/* ----------------------------------------------------------------------
    Calculate 3-body descriptor from 
    http://amp.readthedocs.io/en/latest/theory.html.

    Inputs:
        m: Configuration index
        n: Atom index
        tj: First type of atom this descriptor describes an interaction with.
			Starts at 0.
        tk: Second type of atom this descriptor describes an interaction with
			Starts at 0.
        d: Angular descriptor index, so that we can get proper parameters
		   d runs from [0,nang-1]
		   Can be used to index params for a particular interaction type by 
		   nang*(tj+tk)+d, tj and tk starts at 0.
------------------------------------------------------------------------- */


double Descriptor::calc3Body(int m, int n, int tj, int tk, int d)
{

	int ti = mlt->systems->configs[m].types[n];
	tj++;
	tk++;

	//fprintf(fh_debug, "Calculating %dth 3-body %d-%d-%d descriptor for atom %d in config %d.\n", d+1,ti,tj,tk,n+1,m+1);

	// Find which type of interaction this is
	
	int interactionIndex = input->nang*(tj+tk - 2)+d;
    double zeta=params3[ti-1][interactionIndex][0];
    double eta=params3[ti-1][interactionIndex][1];
    double lambda=params3[ti-1][interactionIndex][2];
	double rc = input->rc;

	//fprintf(fh_debug, " params: %f %f %f\n", zeta, eta, lambda);
	
	double value = 0.0; // descriptor value
    
	int natoms = mlt->systems->configs[m].natoms;
    int **neighlist = mlt->systems->configs[m].neighlist;
    int *numneigh = mlt->systems->configs[m].numneigh;
    int *types = mlt->systems->configs[m].types;
    int *tags = mlt->systems->configs[m].tags;
    double **x = mlt->systems->configs[m].x;

    double xi = mlt->systems->configs[m].atoms[n].x[0];
    double yi = mlt->systems->configs[m].atoms[n].x[1];
    double zi = mlt->systems->configs[m].atoms[n].x[2];
    double xj,yj,zj,xk,yk,zk;
    double xij,yij,zij,rij;
    double xik,yik,zik,rik;
    double xjk,yjk,zjk,rjk;
    double costheta,theta;
    int j,k; // neighbor index
    int jtag,ktag; // index of neighbor image in the original box
	int jtype,ktype;
    double fcij,fcik,fcjk,rallsq,rdiffsq;

	//fprintf(fh_debug, "numneigh: %d\n", mlt->systems->configs[m].atoms[n].numneigh);
    for (int jj=0; jj<mlt->systems->configs[m].atoms[n].numneigh; jj++)
    {

        j = neighlist[n][jj];
        jtag = mlt->systems->configs[m].atoms[j].tag;
		jtype = mlt->systems->configs[m].atoms[j].type;
		//fprintf(fh_debug," jtype: %d\n",jtype);
        
        if (jtype==tj){

            for (int kk=0; kk<mlt->systems->configs[m].atoms[n].numneigh; kk++){

                k = neighlist[n][kk];
                ktag = mlt->systems->configs[m].atoms[k].tag;
				ktype = mlt->systems->configs[m].atoms[k].type;
				//fprintf(fh_debug,"  ktype: %d\n",ktype);

                if (k != j){

                    if (ktype==tk){

                        //fprintf(fh_debug, "%d %d %d\n", types[itag],types[jtag],types[ktag]);
                        //fprintf(fh_debug, "%d %d %d\n", itag,jtag,ktag);
						//fprintf(fh_debug, "%d,%d,%d\n", n,j,k);
                       
						xj=mlt->systems->configs[m].atoms[j].x[0];	
						yj=mlt->systems->configs[m].atoms[j].x[1];	
						zj=mlt->systems->configs[m].atoms[j].x[2];	
						xk=mlt->systems->configs[m].atoms[k].x[0];	
						yk=mlt->systems->configs[m].atoms[k].x[1];	
						zk=mlt->systems->configs[m].atoms[k].x[2];	


                        xij=xi-xj;
                        yij=yi-yj;
                        zij=zi-zj;
                        xik=xi-xk;
                        yik=yi-yk;
                        zik=zi-zk;
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

                        value += pow((1+lambda*costheta),zeta)*exp(-1.0*eta*rallsq) \
                                 *fcij*fcik*fcjk;

                        //fprintf(fh_debug, "   theta: %f\n", theta*(180/3.141592653589));
						//fprintf(fh_debug, "   costheta: %f\n", costheta);
                        //fprintf(fh_debug,"%f %f %f\n", rij,rik,rjk);
                        //fprintf(fh_debug, "rij: %f\n", rij);
                        //fprintf(fh_debug, "rik: %f\n", rik);
                        //fprintf(fh_debug, "rjk: %f\n", rjk);
                         
                        //fprintf(fh_debug, "   pow((1+lambda*costheta),zeta): %f\n", pow((1+lambda*costheta),zeta));
                        //fprintf(fh_debug, "   exp(-1.0*eta*rallsq/(rc*rc)): %f\n", exp(-1.0*eta*rallsq/(rc*rc)));
                        //fprintf(fh_debug, "   fcij: %f\n", fcij);
                        //fprintf(fh_debug, "   fcik: %f\n", fcik);
                        //fprintf(fh_debug, "   fcjk: %f\n", fcjk);
                        //fprintf(fh_debug, "   value: %f\n", value);
                        


                    }

                }

            }

        }

    }

    value *= pow(2,1-zeta);

    //fprintf(fh_debug, "%d-%d-%d descriptor for atom %d in config %d = %f.\n", ti,tj,\
    //                  tk,n, m, value);
    //printf("params: %f %f\n", eta,rs);

	//fprintf(fh_debug," value = %f\n", value);
    return value;

}

/* ----------------------------------------------------------------------
    Calculate derivative of 3-body descriptor from
    http://amp.readthedocs.io/en/latest/theory.html.
	wrt atom coordinate "a".

    Inputs:
        m: Configuration index
        n: Atom index
		j: Neighbor of atom n index
		s: Descriptor index
		a: Cartesian coordinate (1,2, or 3)
		self: true for self-derivative, false otherwise	
------------------------------------------------------------------------- */

double Descriptor::calcd3Body(int m, int n, int jj, int s, int a, bool self)
{

	int nbody;
	nbody = mlt->systems->configs[m].descriptors[n][s].nbody;
	if (nbody != 3) mlt->error->exit("descriptor.cpp", "Caught bad descriptor body type in calcd3Body");

	int **neighlist,*numneigh,*types,*tags;
	neighlist = mlt->systems->configs[m].neighlist;
	numneigh = mlt->systems->configs[m].numneigh;
	types = mlt->systems->configs[m].types;
	tags = mlt->systems->configs[m].tags;
	int ti,tj;
	int itag,j,jtag;
	int dtypes[3]; // descriptor types

	//ti = types[n];
	//itag = tags[n];
	ti = mlt->systems->configs[m].atoms[n].type;
	itag = mlt->systems->configs[m].atoms[n].tag;
	if (self){
		j=n;
	}
	else{
		j = neighlist[n][jj];
	}
	//jtag = tags[j];
	//tj = types[jtag];
	jtag = mlt->systems->configs[m].atoms[j].tag;
	tj = mlt->systems->configs[m].atoms[j].type;

	dtypes[0] = mlt->systems->configs[m].descriptors[jtag][s].types[0];
	dtypes[1] = mlt->systems->configs[m].descriptors[jtag][s].types[1];
	dtypes[2] = mlt->systems->configs[m].descriptors[jtag][s].types[2];

	/*
	if (self){
		fprintf(fh_debug, "Config %d, atom %d, neighbor SELF, descriptor %d, coordinate %d\n", m,n,s,a);
	}
	else{
		fprintf(fh_debug, "Config %d, atom %d, neighbor %d, descriptor %d, coordinate %d\n", m,n,jj,s,a);
	}
	*/

	double **x;
	x = mlt->systems->configs[m].x;

	//fprintf(fh_debug, " dtypes[0],dtypes[1],dtypes[2]: %d,%d,%d\n",dtypes[0],dtypes[1],dtypes[2]);
	

		
    double value=0.0;

	if (tj != dtypes[0]) mlt->error->exit("descriptor.cpp", "Calculating descriptor derivative for wrong atom type");

	if (ti == dtypes[1] || ti==dtypes[2]){ // if ti is not dtypes[1] or dtypes[2], this derivative is zero
	
		//fprintf(fh_debug,"Descriptor type %d,%d,%d\n", dtypes[0],dtypes[1],dtypes[2]);

		int interactionIndex = s-input->nrad*input->ntypes;
		//fprintf(fh_debug, " interactionIndex: %d\n", interactionIndex);
		double zeta=params3[ti-1][interactionIndex][0];
		double eta=params3[ti-1][interactionIndex][1];
		double lambda=params3[ti-1][interactionIndex][2];
		double rc = input->rc;

		//double xj,yj,zj,xl,yl,zl;
		double xj,yj,zj;
		double xp,yp,zp,xq,yq,zq;
		double xjp,yjp,zjp,rjp;
		double ajp; // displacement of this particular coordinate
		int p,ptag,tp;
		double xjq,yjq,zjq,rjq;
		double ajq; // displacement of this particular coordinate
		int q,qtag,tq;
		double xpq,ypq,zpq,rpq,apq;
		double fc, dfcdr, drda;
		double costheta,fcjp,fcjq,fcpq,dCos,dRjp,dRjq,dRpq;
		double dFcjp,dFcjq,dFcpq;
		double djp[3],djq[3],dpq[3]; // displacements
		double dDjp[3],dDjq[3],dDpq[3]; // derivatives of displacements
		double dDjp_dot_djq;
		double dDjq_dot_djp;
		double djp_dot_djq;

		xj = mlt->systems->configs[m].atoms[j].x[0];
		yj = mlt->systems->configs[m].atoms[j].x[1];
		zj = mlt->systems->configs[m].atoms[j].x[2];
		double numneighj = mlt->systems->configs[m].atoms[j].numneigh;
		for (int pp=0; pp<numneighj; pp++){

			p = neighlist[j][pp];
			ptag = mlt->systems->configs[m].atoms[p].tag;
			tp = mlt->systems->configs[m].atoms[p].type;
			xp = mlt->systems->configs[m].atoms[p].x[0];
			yp = mlt->systems->configs[m].atoms[p].x[1];
			zp = mlt->systems->configs[m].atoms[p].x[2];
			xjp = xp-xj;
			yjp = yp-yj;
			zjp = zp-zj;
			rjp = sqrt(xjp*xjp+yjp*yjp+zjp*zjp);
			ajp = mlt->systems->configs[m].atoms[p].x[a] - mlt->systems->configs[m].atoms[j].x[a];
			//fprintf(fh_debug, "  l: %d\n", l);
			//fprintf(fh_debug, " rjp: %f\n", rjp);
			//fprintf(fh_debug, "  ltag: %d\n", ltag);
			//fprintf(fh_debug, " tj,tp: %d,%d\n", tj,tp);
			
			if (tp==dtypes[1]){

				for (int qq=0; qq<numneighj; qq++){

					q=neighlist[j][qq];

					//printf("q loop------------------------------------\n");
					//fprintf(fh_debug," p,q: %d,%d\n",p,q);	
					if (q != p){
						
						//printf(" q!=p------------------------------------\n");	
						qtag=mlt->systems->configs[m].atoms[q].tag;
						//printf(" ptag: %d\n", qtag);
						tq=mlt->systems->configs[m].atoms[q].type;
						xq = mlt->systems->configs[m].atoms[q].x[0];
						yq = mlt->systems->configs[m].atoms[q].x[1];
						zq = mlt->systems->configs[m].atoms[q].x[2];

						if (tq==dtypes[2] && (itag==jtag || itag==qtag || itag==ptag)){

							xjq=xq-xj;
							yjq=yq-yj;
							zjq=zq-zj;
							rjq=sqrt(xjq*xjq+yjq*yjq+zjq*zjq);
							ajq = mlt->systems->configs[m].atoms[q].x[a] - mlt->systems->configs[m].atoms[j].x[a];

							xpq=xq-xp;
							ypq=yq-yp;
							zpq=zq-zp;
							rpq=sqrt(xpq*xpq+ypq*ypq+zpq*zpq);
							apq = mlt->systems->configs[m].atoms[q].x[a] - mlt->systems->configs[m].atoms[p].x[a];
                        	
							costheta = (xjp*xjq + yjp*yjq + zjp*zjq)/(rjp*rjq);

							// Cutoff functions and their derivatives
							fcjp = calcFc(rjp);
							fcjq = calcFc(rjq);
							fcpq = calcFc(rpq);
							dFcjp = calcdFc(rjp);
							dFcjq = calcdFc(rjq);
							dFcpq = calcdFc(rpq);

							// Radial distance derivatives
							/*
							if (rpq==0){
								fprintf(fh_debug,"j: %d\n", j);
								fprintf(fh_debug,"p: %d\n", p);
								fprintf(fh_debug,"q: %d\n", q);
								fprintf(fh_debug,"xp: %f %f %f\n",x[p][0],x[p][1],x[p][2]);
								fprintf(fh_debug,"xq: %f %f %f\n",x[q][0],x[q][1],x[q][2]);
							}
							*/
							dRjp = calcdR(itag,jtag,ptag,ajp,rjp);
							dRjq = calcdR(itag,jtag,qtag,ajq,rjq);
							dRpq = calcdR(itag,ptag,qtag,apq,rpq);

							// Displacements

							for (int c=0; c<3; c++) djp[c]=mlt->systems->configs[m].atoms[p].x[c] - mlt->systems->configs[m].atoms[j].x[c];
							for (int c=0; c<3; c++) djq[c]=mlt->systems->configs[m].atoms[q].x[c] - mlt->systems->configs[m].atoms[j].x[c];
							for (int c=0; c<3; c++) dpq[c]=mlt->systems->configs[m].atoms[q].x[c] - mlt->systems->configs[m].atoms[p].x[c];

							// Displacement derivatives
							for (int c=0; c<3; c++) dDjp[c]=calcdD(itag,jtag,ptag,a,c);
							for (int c=0; c<3; c++) dDjq[c]=calcdD(itag,jtag,qtag,a,c);
							for (int c=0; c<3; c++) dDpq[c]=calcdD(itag,ptag,qtag,a,c);

							// Dot products for derivative of cos(theta)
							dDjp_dot_djq = dDjp[0]*djq[0]+dDjp[1]*djq[1]+dDjp[2]*djq[2];
							dDjq_dot_djp = dDjq[0]*djp[0]+dDjq[1]*djp[1]+dDjq[2]*djp[2];
							djp_dot_djq = djp[0]*djq[0]+djp[1]*djq[1]+djp[2]*djq[2];

							// Final dCos calculation

							dCos = (1.0/(rjp*rjq))*dDjp_dot_djq + (1.0/(rjp*rjq))*dDjq_dot_djp \
								   - (djp_dot_djq/(rjp*rjp*rjq))*dRjp \
								   - (djp_dot_djq/(rjp*rjq*rjq))*dRjq;

							// Calculate descriptor term

							value += pow(1.0+lambda*costheta,zeta-1.0)*exp(-1.0*eta*(rjp*rjp+rjq*rjq+rpq*rpq))  \
									 *( fcjp*fcjq*fcpq* ( lambda*zeta*dCos - 2.0*eta*(1.0+lambda*costheta) \
														 * (rjp*dRjp+rjq*dRjq+rpq*dRpq)/(rc*rc)) \
									+ (1.0+costheta)*(dFcjp*dRjp*fcjq*fcpq+fcjp*dFcjq*dRjq*fcpq+fcjp*fcjq*dFcpq*dRpq));
						
							/*	
							if (value==0.0 && rpq<input->rc){
								fprintf(fh_debug,"rpq: %f\n", rpq);
								fprintf(fh_debug,"%d,%d,%d,%d\n", itag,jtag,ptag,qtag);
							}
							*/
							//fprintf(fh_debug,"     costheta: %f\n", costheta);
					
							/*	
							fprintf(fh_debug,"     rjp: %f\n", rjp);
							fprintf(fh_debug,"     rjq: %f\n", rjq);
							fprintf(fh_debug,"     rpq: %f\n", rpq);
							fprintf(fh_debug,"     dRjp: %f\n", dRjp);
							fprintf(fh_debug,"     dRjq: %f\n", dRjq);
							fprintf(fh_debug,"     dRpq: %f\n", dRpq);
							*/
							//fprintf(fh_debug,"     fcpq: %f\n", fcpq);
							/*
							fprintf(fh_debug,"     dDjp: %f %f %f\n", dDjp[0],dDjp[1],dDjp[2]);
							fprintf(fh_debug,"     dDjq: %f %f %f\n", dDjq[0],dDjq[1],dDjq[2]);
							fprintf(fh_debug,"     dDpq: %f %f %f\n", dDpq[0],dDpq[1],dDpq[2]);
							*/
							/*
							fprintf(fh_debug,"     dDjp_dot_djq: %f\n", dDjp_dot_djq);
							fprintf(fh_debug,"     dDjq_dot_djp: %f\n", dDjq_dot_djp);
							fprintf(fh_debug,"     djp_dot_djq: %f\n", djp_dot_djq);
							*/
							//fprintf(fh_debug,"     dCos: %f\n", dCos);
							//fprintf(fh_debug,"     %f\n", pow(1.0+lambda*costheta,zeta-1.0));
							/*	
							fprintf(fh_debug,"     %f\n", fcjp*fcjq*dFcpq*dRpq);
							fprintf(fh_debug,"     %f\n", fcjp*dFcjq*dRjq*fcpq);
							fprintf(fh_debug,"     %f\n", dFcjp*dRjp*fcjq*fcpq);
							fprintf(fh_debug,"     %f\n",dFcjp*dRjp*fcjq*fcpq+fcjp*dFcjq*dRjq*fcpq+fcjp*fcjq*dFcpq*dRpq);
							*/

							//if (value != value){
								//fprintf(fh_debug, "%f %f %f %f %f %f %f %f %f\n",dFcjp,dRjp,fcjq,fcpq,fcjp,dFcjq,dRjq,dFcpq,dRpq);
								//fprintf(fh_debug, "     %f\n", dFcjp*dRjp*fcjq*fcpq+fcjp*dFcjq*dRjq*fcpq+fcjp*fcjq*dFcpq*dRpq);
								//fprintf(fh_debug, "     value: %f\n", value);

							//}
							
							//fprintf(fh_debug,"     value: %f\n", value);

							//fprintf(fh_debug,"  rjq: %f\n",rjq);
							//fprintf(fh_debug,"  tj,tp,tq: %d,%d,%d\n", tj,tp,tq);
							//fprintf(fh_debug,"  rpq: %f\n", rpq);
						}
					}	
				}

			}

		
		}

		value *= pow(2,1.0-zeta);

	}			

	else {
		//fprintf(fh_debug,"Wrong ti\n");
		value = 0.0;
	}

	//fprintf(fh_debug, "value: %f\n", value);
	return value;
}

/* ----------------------------------------------------------------------
    Calculate cutoff function
------------------------------------------------------------------------- */

double Descriptor::calcFc(double rij)
{

	double value;
	if (rij < input->rc) value = 0.5*(1+cos(M_PI*(rij/input->rc)));
	else value=0.0;
	return value;
}

/* ----------------------------------------------------------------------
    Calculate derivative of cutoff function wrt rij
------------------------------------------------------------------------- */

double Descriptor::calcdFc(double rij)
{
	double value;
	if (rij<input->rc) value = -0.5*(M_PI/input->rc)*sin(M_PI*rij/input->rc);
	else value=0.0;
	return value;
}

/* ----------------------------------------------------------------------
    Calculate derivative of radial distance wrt atomic coordinate

	Inputs

	itag,jtag,ltag - tags of atoms i, j and l (a neighbor of j). Used to 
					 calculate Dirac deltas
	ajl - displacement from from j to atom l
	rjl - radial distance between atom j and atom l
------------------------------------------------------------------------- */

double Descriptor::calcdR(int itag, int jtag, int ltag, double ajl, double rjl)
{
	
	double value;
	double sil,sij; // Dirac deltas

	//fprintf(fh_debug, "  itag,jtag,ltag: %d,%d,%d\n",itag,jtag,ltag);
	if (itag == ltag){
	   	sil = 1.0;
	}
	else{
	   	sil = 0.0;
	}
	if (itag == jtag){
		sij = 1.0;
	}
	else{
	   	sij = 0.0;
	}

	//fprintf(fh_debug, "        sil,sij: %d, %d\n", sil,sij);
	//fprintf(fh_debug, "        ajl: %f\n", ajl);
	//fprintf(fh_debug, "        rjl: %f\n", rjl);

	value = (sil-sij)*ajl/rjl;

	if (value != value){
		/*
		fprintf(fh_debug, "   %f\n", (sil-sij)*ajl/rjl);
		fprintf(fh_debug, "   %f %f %f %f\n",sil,sij,ajl,rjl);
		fprintf(fh_debug, " %d %d %d\n", itag,jtag,ltag);
		*/
	}

	return value;

}

/* ----------------------------------------------------------------------
    Calculate derivative of atomic displacement wrt atomic coordinate
	Derivative of displacement between atoms i and j wrt coordinate of atom n.

	This function only calculates a single component!!!

	Inputs

	ntag,itag,jtag - tags of atoms
	a - atom n coordinate (0,1,2) for (x,y,z)
	coord - coordinate to calculate (0,1,2)
------------------------------------------------------------------------- */

double Descriptor::calcdD(int ntag, int itag, int jtag, int a, int coord)
{

	double value;
	int sni,snj,sca; // Dirac deltas

	if (ntag==jtag) snj=1;
	else snj=0;
	if (ntag==itag) sni=1;
	else sni=0;
	if (a==coord) sca=1;
	else sca=0;

	value = (snj-sni)*sca;

	return value;
}

/* ----------------------------------------------------------------------
    Calculate derivative of cos theta wrt atomic coordinate.
	Derivative of cos(theta_ijk) wrt atomic coordinate of atom n

	Inputs

	ntag,itag,jtag,ktag - tags of atoms 
	ajl - displacement from from j to atom l
	rjl - radial distance between atom j and atom l
------------------------------------------------------------------------- */

double Descriptor::calcdCos()
{
	
	double value;
	int sil,sij; // Dirac deltas

}

/* ----------------------------------------------------------------------
    Calculate single term in the sum for descriptor derivative wrt
	coordinate of an atom

	Inputs

	interactionIndex - depends on type of descriptor, used to get descriptor
					   parameters.
	ti - Type of atom i, also used to get descriptor parameters.
	itag,jtag,ltag - tags of atoms i, j, and l (neighbor of j). Used to
					 calculate Dirac delta for the radial distance derivative.
	ajl - displacement from atom j to atom l (a neighbor of j).
	rjl - radial distance between atom j and atom l.
------------------------------------------------------------------------- */

double Descriptor::calcd2BodyTerm(int interactionIndex, int ti, int itag, int jtag, int ltag, double ajl, double rjl)
{
	double value;
	double eta = params2[ti-1][interactionIndex][0];
	double rs = params2[ti-1][interactionIndex][1];
	double rc=input->rc;

	double fc,dfcdr,drda;
	fc = calcFc(rjl);
	dfcdr = calcdFc(rjl);
	drda = calcdR(itag,jtag,ltag,ajl,rjl);

	double expArg = -1.0*eta*(rjl-rs)*(rjl-rs);

	value = ((-2.0*eta*(rjl-rs)*fc) + dfcdr)*exp(expArg)*drda;

	//fprintf(fh_debug, "   interactionIndex: %d\n", interactionIndex);
	//fprintf(fh_debug, "   expArg: %f\n", expArg);
	//fprintf(fh_debug, "   value: %f\n", value);
	//fprintf(fh_debug, "   drda: %f\n", drda);
	//fprintf(fh_debug, "   fc: %f\n", fc);
	//fprintf(fh_debug, "   dfcdr: %f\n", dfcdr);
	return value;

}
