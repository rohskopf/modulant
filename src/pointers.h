/*
 pointers.h

 Copyright (c) 2018 Andrew Rohskopf

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include "mpi.h"

#include "nn.h"

namespace NN_NS
{
    class Pointers
    {
    public:
        Pointers(NN *ptr) :
            nn(ptr),
            memory(ptr->memory),
            timer(ptr->timer),
            input(ptr->input),
            net(ptr->net)
            {}

        virtual ~Pointers() {}

    protected:
        NN *nn;
        Memory *&memory;
        Timer *&timer;
        Input *&input;
        Net *&net;
    };
}

