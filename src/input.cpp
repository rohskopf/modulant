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

#include "input.h"
#include "memory.h"
#include "nn.h"

using namespace std;

using namespace NN_NS;

Input::Input(NN *nn) : Pointers(nn) {
    fh_debug = fopen("DEBUG_input", "w");
}

Input::~Input() 
{

    memory->deallocate(structure);
	memory->deallocate(activations);
    memory->deallocate(din);
    memory->deallocate(dout);
	memory->deallocate(tin);
    memory->deallocate(tout);

    fclose(fh_debug);

};

void Input::readinput()
{
    /* Read INPUT file */

    string line;

    // Declare scalar inputs
    double value;

    // Open INPUT file
    ifstream INPUT("INPUT");
    // Ignore the first line
    getline(INPUT, line); 
    string characters;
	char test;
    // Get input variables
    for (int i=1; i<=4; i++)
    {
        getline(INPUT, line);
        switch (i)
        {
            case 1:{ 
                stringstream ss(line);
                ss >> characters >> nhl;
                memory->allocate(structure, nhl+2);
				memory->allocate(activations, nhl+2);
            }
            case 2:{ 
                stringstream ss(line);
                ss >> characters;
                for (int k=0; k<nhl+2; k++){
                    ss >> structure[k];
                }

            }
            case 3:{ 
                stringstream ss(line);
                ss >> characters;
                for (int k=0; k<nhl+2; k++){
                    ss >> activations[k];
					//printf("%c\n", test);
                }

            }
            case 4:{ 
                stringstream ss(line);
                ss >> characters >> nsamples;
            }
  
        } // switch (i)

    } // for (int i=1..)

    // Check the structure array
    /*
    for (int k=0; k<nhl+2; k++){
        printf("%d\n", structure[k]);
    }
    */
    INPUT.close();

}

/* ----------------------------------------------------------------------
    Reads training data.
------------------------------------------------------------------------- */

void Input::readData()
{

    printf("Reading %d training samples\n", nsamples);

    // Allocate the input and output training arrays
    
    memory->allocate(din, nsamples, structure[0]);
    memory->allocate(dout, nsamples, structure[nhl+1]);
	memory->allocate(tin, nsamples*structure[0]);
    memory->allocate(tout, nsamples*structure[nhl+1]);

    // Open TIN and TOUT training data files

    ifstream fh_tin("TIN");
    ifstream fh_tout("TOUT");

    string line;
	int numInputs = structure[0];
    for (int d=0; d<nsamples; d++){
    
        // Store the input training data

        getline(fh_tin, line);
        stringstream ss(line);
        for (int i=0; i<structure[0]; i++){
            //ss >> din[d][i];
			ss >> tin[d*structure[0]+i];
        }

        // Store the output training data

        getline(fh_tout, line);
        stringstream ss2(line);
        for (int o=0; o<structure[nhl+1]; o++){
            //ss2 >> dout[d][o];
			ss2 >> tout[d*structure[nhl+1]+o];
        }

    }

    // Check the input and output training data
    /*
    for (int d=0; d<nsamples; d++){
    
        printf("Inputs: ");
        for (int i=0; i<structure[0]; i++){
            printf("%f ", din[d][i]);
        }
        printf("\n Outputs: ");
        for (int o=0; o<structure[nhl+1]; o++){
            printf("%f ", dout[d][o]);
        }
        printf("\n");

    }
    */

    fh_tin.close();
    fh_tout.close();
}
