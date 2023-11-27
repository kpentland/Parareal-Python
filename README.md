# pyParareal

This repository contains a basic implementation of the Parareal algorithm - wrtitten in
Python. 

Contained within are two test problems that [Parareal](https://en.wikipedia.org/wiki/Parareal) will solve
using user-specified coarse and fine solvers (here they are implemented as explicit 
Runge-Kutta methods). This version of Parareal has the option to run in parallel 
(using the 'multiprocessing' package) but is written mostly for educational purposes and
to provide an idea of how it works. 

Please feel free to use and adapt the code as you wish but please cite this GitHub 
page (see side panel) if used in published works. Please also get in contact if 
there are any issues with running the code. 

We make use of the following Python packages: numpy, multiprocessing, time, os, matplotlib.


## Files
* *Parareal.py*: script that contains the 'Parareal', 'RungeKutta', 'CoarseSolver', and 'FineSolver' functions used in the test scripts below.

* *FHN_Test_Serial.py*: test script that solves the FitzHugh-Nagumo model using Parareal (serially). 
* *Lorenz63_Test_Serial.py*: test script that solves the Lorenz63 using Parareal (serially). 

* *FHN_Test_Parallel.py*: test script that solves the FitzHugh-Nagumo model using Parareal (in parallel - must run from terminal). 
* *Lorenz63_Test_Parallel.py*: test script that solves the Lorenz63 using Parareal (in parallel - must run from terminal).  

* *probnum.mplstyle*: a style-file for plotting (adapted from the original - which is from [ProbNum](https://github.com/probabilistic-numerics/probnum ) ).

## Authors

* Kamran Pentland - Culham Centre for Fusion Energy, UKAEA
