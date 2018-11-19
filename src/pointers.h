/*
 pointers.h

 Copyright (c) 2018 Andrew Rohskopf

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include "mpi.h"

#include "modulant.h"

namespace MODULANT_NS
{
    class Pointers
    {
    public:
        Pointers(MLT *ptr) :
            mlt(ptr),
            memory(ptr->memory),
			error(ptr->error),
            timer(ptr->timer),
            input(ptr->input),
            systems(ptr->systems),
			descriptor(ptr->descriptor)
            {}

        virtual ~Pointers() {}

    protected:
        MLT *mlt;
        Memory *&memory;
		Error *&error;
        Timer *&timer;
        Input *&input;
        Systems *&systems;
		Descriptor *&descriptor;
    };
}

