# Modulant
***MODUL***ar ***A***tomistic ***N***etwork ***T***rainer

There are many deep learning packages for making potentials, but Modulant differs in that it is made purely from high-performance C++ code without any dependencies (only GCC and OpenMPI!). This serves a few purposes:

1. Full modular control over all aspects of the training/evaluation process.
2. Modularity allows tweaking each stage of the training/evaluation process to improve performance.
3. Educational - this program shows from start to finish how an atomistic neural network is built using object-oriented C++.

### Installing Modulant

In a Linux envirinoment, install the program by going into the `src` directory and doing:

    make clean
    make
    
### Running Modulant

The examples/ folder has applications of Modulant to simple training data.
The INPUT file is all you need - it has neural network parameters that can be changed.
