.SUFFIXES: .h .cpp
#-----------------------------------------------
# Makefile for MD program
# Please modify the variables properly.
# Use gcc >= 4.8 with openMPI >= 1.8
#-----------------------------------------------

# Enter the path to include directories
DIR =

# Enter path to any shared library (.so) files
SOLIB =

#-----------------------------------------------

CXX = mpiCC
CXXFLAGS = #-std=c++11 -Wno-write-strings

INCLUDE = -I${DIR}

CXXL =
LDFLAGS = 

LAPACK =
LIBS =

#-----------------------------------------------
# General rules 
#-----------------------------------------------

.cpp.o:
	${CXX} ${CXXFLAGS} -c $<

PROG = mlt

CXXSRC= modulant.cpp main.cpp memory.cpp error.cpp timer.cpp input.cpp systems.cpp config.cpp descriptor.cpp \
        layer.cpp neuron.cpp ann.cpp

OBJS= ${CXXSRC:.cpp=.o}

default: mlt

all: ${PROG}

mlt: ${OBJS}
	${CXX} ${CXXFLAGS} -o $@ ${OBJS} ${LIBS} ${SOLIB}

clean:
	rm -f ${OBJS}

.PHONY: clean

depend:
	gcc -MM *.cpp > .depend

-include .depend
