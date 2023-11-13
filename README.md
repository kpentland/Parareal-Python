# Parareal

This repository contains a basic implementation of the Parareal algorithm - wrtitten in
Python. 

Contained within are two test problems that [Parareal](https://en.wikipedia.org/wiki/Parareal) will solve (serially)
using user-specified coarse and fine solvers (here they are implemented as explicit 
Runge-Kutta methods). This version of Parareal does not run in parallel and was
written only to provide an idea of how the algorithm works for educational purposes. 

Soon I hope to be able to provide a parallel version of this code but please bear with me. Please feel free to use 
and adapt the code as you wish but please cite this GitHub page (see side panel) if used in published works. Please 
also get in contact if there are any issues with running the code. The only dependencies 
should be numpy and matplotlib. 




## Files
* *Parareal.py*: script that contains the 'Parareal' and 'RungeKutta' functions used in the test scripts below.

* *FHN_Test.py*: test script that solves the FitzHugh-Nagumo model using Parareal. 
* *Lorenz63_Test.py*: test script that solves the Lorenz63 using Parareal. 

* *probnum.mplstyle*: a style-file for plotting (adapted from the original - which is from [ProbNum](https://github.com/probabilistic-numerics/probnum) ).

## Authors

* Kamran Pentland - Culham Centre for Fusion Energy, UKAEA
