# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 07:31:54 2020

@author: Enrique

Example of simulation of a prescribed indentation history over a Generalized Voigt model containing an arbitrary number of characteristic times
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time as tm

import sys
lib_path = r'C:\Users\Enrique\Documents\Github-repositories\afmviscoelastic'    #replace here with the path where the repo is in your pc
sys.path.append(lib_path)
from lib.afmsim import differential_constants,uq_prescribed
from lib.viscoelasticity import conv



#######################################material and aparatus properties###########################
nu = 0.5
R= 10.0e-9
Jg = 2.0e-10  # Glassy compliance
J = np.array([1.0e-8])
tau = np.array([0.5e-5])
u,q = differential_constants('Gen. Kelvin-Voigt', Jg,J,tau)
#######################################material and aparatus properties###########################

#########################################defining indentation history############################
simultime = 4.26e-4   #total time in seconds
z_dot = 2.0e-6  #rate of indentation
dt = 4.0e-9  #timestep
t = np.arange(0.0,simultime,dt)  #time array needed that is used in this example to define indentation history
z_indent = np.zeros(len(t))   #initialization of indentation history
z_indent[:] =  z_dot*t[:] # prescribed indentation, in this example is linear
#########################################defining indentation history############################


#####################################SIMULATION##################################################
uq_jit = jit()(uq_prescribed)  #using numba to run code faster
t0 = tm.time()
t,z,fts = uq_jit(simultime,u,q,z_indent,R)   #main line of the simulation, here calling the function that returns Fts history when given certain indentation history
t1 = tm.time()
print('Total simulation time: %2.3f'%(t1-t0))
#####################################SIMULATION##################################################

##################################COMPARING SIMULATION WITH ANALYTICAL SOLUTION##############################
t2 = tm.time()
cUF = conv('Gen. Kelvin-Voigt', t,fts,Jg,J,tau)
t3 = tm.time()
print('Total time for calculation of the convolution integral: %2.3f'%(t3-t2))   #around 400 s to run this simulation
ht_an = 3.0/16.0/np.sqrt(R)*cUF  #Analytical solution for h^1.5 according to Eq. 11 in López‐Guerra, et.al. JPolSciB, 55(10) 
fig,ax=plt.subplots()
ax.plot(t,ht_an,label='Analytical h^1.5',lw=5,alpha=0.5,ls='--')
ax.plot(t,z**1.5,label='Simulation h^1.5')
ax.legend(loc='best')
ax.set_xlabel('time, s', fontsize=15)
ax.set_ylabel('h^1.5, m^1.5', fontsize=15)
#ax.set_xscale('log')
#ax.set_yscale('log')
##################################COMPARING SIMULATION WITH ANALYTICAL SOLUTION##############################

        
        
    