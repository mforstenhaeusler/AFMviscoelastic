# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 04:13:07 2020

@author: Enrique
Example of using the contact_mode function to simulate a force spectroscopy experiment performed over a viscoelastic material
"""

import numpy as np
from numba import jit
import time as tm

import sys
lib_path = r'C:\Users\Enrique\Documents\Github-repositories\afmviscoelastic'    #replace here with the path where the repo is in your pc
sys.path.append(lib_path)
from lib.afmsim import contact_mode, differential_constants


######################################################SIMULATION PARAMETERS###############################################################
vb = 1000e-9  # velocity of approaching sample
zb = 100e-9  # initial position of cantilever
Q1 = 100  # Quality Factor
k_m1 = 0.1  # equilibrium cantilever mass
fo1 = 10e3  # natural frequency
period1 = 1.0 / fo1  # fundamental period
R = 20e-9  # radius tip
nu = 0.5  # poissons ratio of material

vdw = 2  # vdw constant, IF 0 van der Waals forces will be included if 2 vdw not included
H = 2.0e-19  # Hammaker constant 
a = 0.2e-9  # intermolecular distance

start = 0  # defines start of the simulation
landingtime = zb / vb
final_time = 1  # indentation time
stop = landingtime + final_time  # defines end of the simulation
timestep = period1 / 1.0e4
printstep = period1 / 10  # printstep of Force distance curve

time = np.arange(start, stop, timestep)  # time array 

arms = 5
Jg = 2.0e-10  # Glassy compliance
J_5 = np.zeros(arms)  # compliance array - 5 arm
tau_5 = np.zeros(arms)  # characteristic time array - 5 arm
J_5[0], tau_5[0] = 5.0e-9, 0.5e-4
J_5[1], tau_5[1] = 7.0e-9, 0.5e-3
J_5[2], tau_5[2] = 1.0e-10, 0.5e-2
J_5[3], tau_5[3] = 3.0e-6, 0.5e-1
J_5[4], tau_5[4] = 4.0e-6, 0.5e-0

u_5, q_5 = differential_constants('Gen. Kelvin-Voigt', Jg, J_5, tau_5)  # 5 Arm
######################################################SIMULATION PARAMETERS###############################################################

#################################################RUNNING THE SIMULATION###################################################################
AFM_jit = jit()(contact_mode)  # calls AFM simul function and initializes numba procedure to increase iter. speed
t0 = tm.time()
Fts_5, tip_5, base_5 = AFM_jit(time, timestep, zb, vb, u_5, q_5, k_m1, fo1, Q1, vdw, R, nu)
t1 = tm.time()
print('Total time to simulate dynamic spectroscopy at 350Khz: %2.3f'%(t1-t0))   #around 400 s to run this simulation
#################################################RUNNING THE SIMULATION###################################################################