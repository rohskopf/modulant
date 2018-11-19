#pragma once

#include <vector>
#include <string>
#include "mpi.h"

#include <iostream>
#include <new>
#include <cstdlib>
#include "pointers.h"

#include "memory.h"

using namespace std;

namespace MODULANT_NS
{
  class Descriptor: protected Pointers
  {
  public:
    Descriptor(class MLT *);
    ~Descriptor();

    FILE * fh_debug; // Debug file handle

	int factorial(int); // calculate factorial
	void readParams();

    double calc2Body(int,int,int,int);
	double calc3Body(int,int,int,int,int);

	// Descriptor derivatives
	double calcd2Body(int,int,int,int,int,bool);
	double calcd2BodyTerm(int,int,int,int,int,double,double);
	double calcd3Body(int,int,int,int,int,bool);

	double calcFc(double); // cutoff function
	double calcdFc(double); // derivative of cutoff function wrt radial distance
	double calcdR(int,int,int,double,double); // derivative of radial distance wrt atomic coordinate (x,y, or z)
	double calcdD(int,int,int,int,int);
	double calcdCos();

	double ***params2; // 2body params
	double ***params3; // 3body params


  };
}
