# TRLIB

This is a library that provides various methods related to the [trust region subproblem](https://de.wikipedia.org/wiki/Trust-Region-Verfahren#Bemerkung_zur_L.C3.B6sung_des_quadratischen_Problems).

## Dependencies
You have to make sure that the following requirements are provided:
- BLAS
- LAPACK
- CMake
- Doxygen (only needed if you want to read the documentation)
- Graphviz (only needed if you want to read the documentation)
- Check (only needed if you want to run the unittests)
- Protobuf C (only needed if you want to run the unittests)
- ZMQ (only needed if you want to run the unittests)
- Python Header, TRLIB works with python 2 and 3 and compiles for the versions it finds (only needed if you want to compile the python interface)
- Cython (only needed if you want to compile the python interface)
- NumPy (only needed if you want to use the python interface, version >= 1.10)
- SciPy (only needed if you want to run the rosenbrock example)
- IPython Notebook (only needed to have a look at some of the examples)

### Ubuntu/Debian Packages
Using Ubuntu/Debian, you can install all the requirements by executing:

    sudo apt-get install cmake doxygen graphviz check build-essential python3-dev python3-numpy python3-scipy cython3 ipython3-notebook liblapack-dev libblas-dev libprotobuf-dev protobuf-compiler libprotoc-dev protobuf-c-compiler libprotobuf-c-dev libzmq-dev

or for python2

    sudo apt-get install cmake doxygen graphviz check cython build-essential python-dev python-numpy python-scipy cython ipython-notebook liblapack-dev libblas-dev libprotobuf-dev protobuf-compiler libprotoc-dev protobuf-c-compiler libprotobuf-c-dev libzmq-dev


## Compilation
TRLIB is set up to create out of source builds using CMake. First create a build directory and change to that:

    mkdir build
    cd build

Set up CMake in this directory:

    cmake -DCMAKE_BUILD_TYPE=Release ..

Instead of `Release` you may also choose `Debug` which disables compiler optimization and enables debugging.

You may want change settings as described [CMakeOptions](#cmakeoptions) in CMake, you can execute

    ccmake .
Press `c` to reconfigure with changes and `q` to exit.

You can now compile TRLIB, generate the documentation and run the tests by executing

    make
    make install
    make doc
    make test

### CMakeOptions

Option                  | default | Description
------------------------|---------|------------
`TRLIB_MEASURE_TIME`    | `OFF`   | measure time for trlib function calls
`TRLIB_MEASURE_SUBTIME` | `OFF`   | measure time for blas and lapack function calls
`TRLIB_TEST_OUTPUT`     | `OFF`   | show output on stdout of trlib functions when individual tests are executed
`TRLIB_TEST_PLOT`       | `ON`    | use socket communication to send matrix data for plotting (only for debug purposes)

## Getting started
- [C API](@ref src/trlib.c)
- Python Library that shows how to use the low level reverse communication interface and the higher level driver interface
  as well how to make use of these interfaces in a standard trust region algorithm: `examples/umin.py`
- IPython Notebbok with standard trust region algorithm to minimize the Rosenbrock function: `examples/rosenbrock.ipynb`:
    
    cd examples
    ipython notebook rosenbrock
