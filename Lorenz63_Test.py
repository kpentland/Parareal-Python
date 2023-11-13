#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Here we use Parareal (and the RungeKutta function) to solve the Lorenz63
model (see below for ODE system and parameters).


@author: kpentland
"""

# clear workspace
%clear -f 
%reset -f 


# import relevant packages
import numpy as np
import matplotlib.pyplot as plt
from Parareal import parareal, RungeKutta

# plotting style (credit to the 'probnum' package developers)
plt.style.use("probnum.mplstyle")



# Define ODE system to be solved
def lorenz63(t, u):
    dx = 10.0*(u[1] - u[0])
    dy = u[0]*(28.0 - u[2]) - u[1]
    dz = u[0]*u[1] - (8.0/3.0)*u[2]
    return np.array([dx, dy, dz])


# Define the coarse solver that you wish to use. 
def CoarseSolver(tspan, dt, u0, f, dense):
    u_end = RungeKutta(t = tspan, dt = dt, u0 = u0, f = f, method = 'RK4', dense = dense)
    return u_end

# Define the fine solver that you wish to use
def FineSolver(tspan, dt, u0, f, dense):
    u_end = RungeKutta(t = tspan, dt = dt, u0 = u0, f = f, method = 'RK4', dense = dense)
    return u_end

# Here we chose our 'RungeKutta' function defined in Parareal.py, however, this 
# can easily be modifed to use other solvers (e.g. Scipy built-in solvers). It 
# may require adapting the lines in the 'Parareal' function so do so with care! 


# Now let us define our inputs to Parareal:
f = lorenz63
tspan = [0,10]
u0 = np.transpose(np.array([5.0,-5.0,20.0]))
dim = len(u0)
N = 40
Ng = 400
Nf = 4000
epsilon = 1e-6
G = CoarseSolver
F = FineSolver



# call Parareal
t, u, err, k = parareal(f,tspan,u0,N,Ng,Nf,epsilon,F,G)


# call serial fine solver (to measure Parareal soluton accuracy later)
t_f = np.linspace(tspan[0], tspan[1], Nf+1)
dt = t_f[1] - t_f[0]
u_f = F(tspan = tspan, dt = dt, u0 = np.copy(u0), f = f, dense = True)  # runs fine solver with dense output
            






# Plot the solutions
plt.figure(1)
fig, axs = plt.subplots(3)

axs[0].margins(0)
axs[0].grid()
axs[0].plot(t_f, u_f[:,0], color='black', label = r"Fine", zorder=10, clip_on=False)
axs[0].scatter(t, u[:,(k-1)*dim], label=r"Parareal", color='red', marker='o', zorder=10, clip_on=False)
axs[0].set_xticklabels([])
axs[0].set(ylabel = r"State ($x$)");
axs[0].axis((tspan[0], tspan[1], -20, 20));


axs[1].margins(0)
axs[1].grid()
axs[1].plot(t_f, u_f[:,1], color='black', label = r"Fine", zorder=10, clip_on=False)
axs[1].scatter(t, u[:,(k-1)*dim + 1], label=r"Parareal", color='red', marker='o', zorder=10, clip_on=False)
axs[1].set_xticklabels([])
axs[1].set(ylabel = r"State ($y$)");
axs[1].axis((tspan[0], tspan[1], -25, 25));

axs[2].margins(0)
axs[2].grid()
axs[2].plot(t_f, u_f[:,2], color='black', zorder=10, clip_on=False, label = r'Fine')
axs[2].scatter(t, u[:,(k-1)*dim + 2], color='red', marker='o', zorder=10, clip_on=False, label = r'Parareal')
axs[2].set(xlabel = r"Time ($t$)", ylabel = r"State ($z$)");
axs[2].axis((tspan[0], tspan[1], 0, 50));
plt.legend(ncol = 2,  bbox_to_anchor=(0.8, 4.0), framealpha=1.0)




# plot the convergence rate (Parareal stops at tolerance)
plt.figure(3)
plt.margins(0)
plt.grid()
plt.xlabel(r"Iteration ($k$)");
plt.ylabel(r"Error at succesive iterations");
plt.yscale('log')
plt.axhline(y=epsilon, color='k', linestyle='--')
plt.plot(range(1,k+1),np.max(err, axis=0), color='red', marker = 'o', zorder=10, clip_on=False)
plt.axis((1, min(k+1,N), 1e-8, 100))





# plot errors between parareal and fine solutions on mesh t at each iteration k

plt.figure(4)
plt.margins(0)
plt.grid()
plt.xlabel(r"Time ($t$)");
plt.ylabel(r"Error vs. Fine solver");
plt.axis((tspan[0], tspan[1], 1e-10, 1))
plt.yscale('log')

for j in np.arange(1,k+1,1):
    errors = np.linalg.norm( u_f[0:Nf+1:int(Nf/N),:] - u[:,dim*(j-1):dim*j], ord=np.inf, axis=1)
    errors[errors == 0] = 1e-16
    plt.plot(t, errors, marker = 'o', zorder=1, clip_on=True, label = r'$k = %d$' % j)
plt.legend(loc = 'lower left', framealpha=1.0)


