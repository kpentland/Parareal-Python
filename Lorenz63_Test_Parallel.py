#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Here we use Parareal (and the RungeKutta function) to solve the Lorenz63
model (see below for ODE system and parameters).

Note that here we run Parareal in parallel and so we need to run the script
from the command line - the parallel processing is unlikely to run in and IDE 
like Spyder (Visual Studio remains untested). To run this script, open a terminal
and navigate to the directory where this script is located. To run, type: 
'python3 Lorenz63_Test_Parallel.py'

Note: the python command may be slightly different depending on which verison you 
have installed. 


Please note that Parareal requires N (i.e. the no. of time slices) processors to run 
efficiently in parallel and so if your computer does not have this many cores
do not expect to see speedup over the fine solver! To run properly, use a machine
with N processors, else Parareal will automatically use the no. of cores minus one
from your machine.


"""


# import relevant packages/modules
import numpy as np
from Parareal import Parareal, FineSolver
import time
import pickle



# Define ODE system to be solved
def lorenz63(t, u):
    dx = 10.0*(u[1] - u[0])
    dy = u[0]*(28.0 - u[2]) - u[1]
    dz = u[0]*u[1] - (8.0/3.0)*u[2]
    return np.array([dx, dy, dz])



# the main body of code will run here
if __name__ == "__main__":
    
    # Now let us define our inputs to Parareal:
    f = lorenz63
    tspan = [0,10]
    u0 = np.transpose(np.array([5.0,-5.0,20.0]))
    dim = len(u0)
    N = 40
    Ng = 400
    Nf = 4000
    epsilon = 1e-6


    # call serial fine solver
    t_f = np.linspace(tspan[0], tspan[1], Nf+1)
    dt = t_f[1] - t_f[0]
    
    serial_start = time.time()
    u_f = FineSolver(tspan = tspan, dt = dt, u0 = np.copy(u0), f = f, dense = True)  # runs fine solver with dense output
    serial_end = time.time()
    serial_time = serial_end - serial_start 
    print('Serial F solver time = %.4e seconds' % serial_time)



    # call Parareal in parallel
    t, u, diagnostics = Parareal(f,tspan,u0,N,Ng,Nf,epsilon,parallel=True)
    k = diagnostics['Iterations']
    print('Parallel time = %.4e seconds' % diagnostics['Parareal Timing'])


    # calculate error between serial fine solver solution and parareal solution (at iteration k)
    errors = np.linalg.norm( u_f[0:Nf+1:int(Nf/N),:] - u[:,dim*(k-1):dim*k], ord=np.inf, axis=1)
    
    print('Error between serial fine solution and parareal solution = %.4e' % np.max(errors))
    

    # to manipulate solution/diagnostic data after simulation you can save the data
    # in whichever format suits you. An example, using pickle files is given below:
        
    # Save variables to a pickle file
    saved_data = {
        'serial_solution': u_f,
        'serial_runtime': serial_time,
        'parareal_solution': u,
        'time_vector': t,
        'parallel_runtime': diagnostics['Parareal Timing'],
        'other_diagnostics': diagnostics
    }
    
    # save the data
    with open('Lorenz63_data.pkl', 'wb') as file:
        pickle.dump(saved_data, file)

    # # Load variables from the file using the command below in a Jupyter notebook for example
    # with open('FHN_data.pkl', 'rb') as file:
    #     loaded_data = pickle.load(file)
        