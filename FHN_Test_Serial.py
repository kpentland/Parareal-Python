#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Here we use Parareal (and the RungeKutta function) to solve the FitzHugh-Nagumo
model (see below for ODE system and parameters). Note that we run Parareal
serially (see the corresponding parallel test script to run in parallel).

The whole script can be run in an IDE such as Spyder or Visual studio. Or it
could be run in a Jupyter notebook - thoughI did not try this.

"""

# clear workspace
%clear -f 
%reset -f 


# import relevant packages/modules
import numpy as np
import matplotlib.pyplot as plt
from Parareal import Parareal, FineSolver

# plotting style (credit to the 'probnum' package developers)
plt.style.use("probnum.mplstyle")



# Define ODE system to be solved
def FHN(t, u):
    a = 0.2
    b = 0.2
    c = 3

    dx = c*(u[0] - (np.power(u[0],3)/3) + u[1]) 
    dy = -(1/c)*(u[0] - a + b*u[1])
    return np.array([dx, dy])


# Now let us define our inputs to Parareal:
f = FHN
tspan = [0,40]
u0 = np.transpose(np.array([-1.0, 1.0]))
dim = len(u0)
N = 40
Ng = 160
Nf = 1600
epsilon = 1e-6
parallel = False

# call Parareal
t, u, diagnostics = Parareal(f,tspan,u0,N,Ng,Nf,epsilon,parallel)
k = diagnostics['Iterations']
err = diagnostics['Errors']

# call serial fine solver (to measure Parareal soluton accuracy later)
t_f = np.linspace(tspan[0], tspan[1], Nf+1)
dt = t_f[1] - t_f[0]
u_f = FineSolver(tspan = tspan, dt = dt, u0 = np.copy(u0), f = f, dense = True)  # runs fine solver with dense output
            




# Plot the solutions
plt.figure(1)
fig, axs = plt.subplots(2)

axs[0].margins(0)
axs[0].grid()
axs[0].plot(t_f, u_f[:,0], color='black', label = r"Fine", zorder=10, clip_on=False)
axs[0].scatter(t, u[:,(k-1)*dim], label=r"Parareal", color='red', marker='o', zorder=10, clip_on=False)
axs[0].set_xticklabels([])
axs[0].set(ylabel = r"State ($x$)");
axs[0].axis((tspan[0], tspan[1], -3, 3));


axs[1].margins(0)
axs[1].grid()
axs[1].plot(t_f, u_f[:,1], color='black', zorder=10, clip_on=False, label = r'Fine')
axs[1].scatter(t, u[:,(k-1)*dim + 1], color='red', marker='o', zorder=10, clip_on=False, label = r'Parareal')
axs[1].set(xlabel = r"Time ($t$)", ylabel = r"State ($y$)");
axs[1].axis((tspan[0], tspan[1], -2, 2));
plt.legend(ncol = 2,  bbox_to_anchor=(0.4, 0.83), framealpha=1.0)



# plot the convergence rate (Parareal stops at tolerance)
plt.figure(3)
plt.margins(0)
plt.grid()
plt.xlabel(r"Iteration ($k$)");
plt.ylabel(r"Error at succesive iterations");
plt.yscale('log')
plt.axhline(y=epsilon, color='k', linestyle='--')
plt.plot(range(1,k+1),np.max(err, axis=0), color='red', marker = 'o', zorder=10, clip_on=False)
plt.axis((1, min(k+1,N), 1e-8, 1))





# plot errors between parareal and fine solutions on mesh t at each iteration k
plt.figure(4)
plt.margins(0)
plt.grid()
plt.xlabel(r"Time ($t$)");
plt.ylabel(r"Error vs. Fine solver");
plt.axis((tspan[0], tspan[1], 1e-10, 1))
plt.yscale('log')

for j in np.arange(1,k+1,2):
    errors = np.linalg.norm( u_f[0:Nf+1:int(Nf/N),:] - u[:,dim*(j-1):dim*j], ord=np.inf, axis=1)
    errors[errors == 0] = 1e-16
    plt.plot(t, errors, marker = 'o', zorder=1, clip_on=True, label = r'$k = %d$' % j)
plt.legend(ncol = 2, loc = 'lower left', framealpha=1.0)



