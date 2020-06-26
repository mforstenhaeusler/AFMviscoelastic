"""
Created on Wed Jun 10th 2020
@author: Maximilian Forstenhaeusler
Description: This library contains core algorithm to simulate different AFM measuring techniques
"""

import numpy as np
import sympy as sym


def differential_constants(model, p1, p2, p3):
    """
    Description:
    This function calculates the differential constants u and q necessary to calculate the tip sample interaction
    force.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Parameters:
    :param model:   string
                    Model name, e.g. 'Gen. Kelvin Voigt', 'Gen. Maxwell'
    :param p1:      float
                    Either Ge or Jg
    :param p2:      array of floats
                    Either G or J
    :param pe:      array of floats
                    tau - charactersitic times
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Return, function output:
    retrun:         u:  array of floats
                        differential constants u0, u1, ... un
                    q:  array of floats
                        differential constants q0, q1, ... qn
    """
    s = sym.symbols('s')

    if model == 'Gen. Kelvin-Voigt':
        if len(p2) == len(p3):  # checks J and tau are same lenght
            U = p1 + sum(p2[:] / (1.0 + p3[:] * s))
            U_n = (U).normal()  # writes retardance with common denominator

            u_n = sym.numer(U_n)  # selects the numerator
            u_n = sym.expand(u_n)  # expands each term
            u_n = sym.collect(u_n, s)  # collect the terms with the same exponent of s

            q_n = sym.denom(U_n)  # selects the denominator
            q_n = sym.expand(q_n)  # expands each term
            q_n = sym.collect(q_n, s)  # collect the terms with the same exponent of s

            q_arr = np.zeros(len(p2) + 1)  # initializes array with lenght J, tau + 1 to store q
            u_arr = np.zeros(len(p2) + 1)  # initializes array with lenght J, tau + 1 to store u

            for i in range(len(p2) + 1):
                q_arr[i] = q_n.coeff(s, i)  # selects the terms multiplied by s^n which are synonymus for q0, .., qn
                u_arr[i] = u_n.coeff(s, i)  # selects the terms multiplied by s^n which are synonymus for u0, .., un
        else:
            print('input arrays have unequal length')

    if model == 'Gen. Maxwell':
        if len(p2) == len(p3):
            Q = p1 + sum((p2[:] * p3[:]) / (1.0 + p3[:] * s))
            Q_n = (Q).normal()  # writes retardance with common denominator

            u_n = sym.denom(Q_n)  # selects the denominator
            u_n = sym.expand(u_n)  # expands each term
            u_n = sym.collect(u_n, s)  # collect the terms with the same exponent of s

            q_n = sym.numer(Q_n)  # selects the numerator
            q_n = sym.expand(q_n)  # expands each term
            q_n = sym.collect(q_n, s)  # collect the terms with the same exponent of s

            q_arr = np.zeros(len(p2) + 1)  # initializes array with lenght J, tau + 1 to store q
            u_arr = np.zeros(len(p2) + 1)  # initializes array with lenght J, tau + 1 to store u

            for i in range(len(p2) + 1):
                q_arr[i] = q_n.coeff(s, i)  # selects the terms multiplied by s^n which are synonymus for q0, .., qn
                u_arr[i] = u_n.coeff(s, i)  # selects the terms multiplied by s^n which are synonymus for u0, .., un
        else:
            print('input arrays have unequal length')

    return u_arr, q_arr


def uq_prescribed(simultime, u ,q,z_indent,R=10.0e-9):
    """
    Description:
    This simulation gives the resulting force in time when a generalized viscoelastic material is indented with a known indentation history
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Parameter:
    :param simultime:   float 
                        total time of the simulation in seconds
    
    :param u:           numpy array 
                        coefficients of the generalized differential equation (operator equation, i.e., Eq 2 in López‐Guerra, et.al. JPolSciB, 55(10) )
    
    :param q:           numpy array
                        coefficients of the generalized differential equation (operator equation, i.e., Eq 2 in López‐Guerra, et.al. JPolSciB, 55(10) )
    
    :param z_indent:    numpy array
                        indentation history
    
    :param R:           float
                        optional, tip radius
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Return, function output:
    :param t:           numpy array
                        this is the time array
    
    :param z_indent:    numpy array
                        indentation history
     
    :param Fts:         numpy array
                        force history that corresponds to the input indentation history    
    """
    
    alpha = 16.0/3.0*np.sqrt(R)  #cell constant, from Lee and Radok equation, assumed material incompressibility (nu=0.5)
    ndt= len(z_indent) #number of timesteps
    t = np.linspace(0.0,simultime,ndt)   #defining time array
    dt = t[-1]/len(t) #timestep
    
    Y_ndot = np.zeros((len(t),len(q)))   #zero initialization of derivatives of indentation^1.5
    F_mdot = np.zeros((len(t),len(q)))  #zero initialization of derivatives of force
    Fts = np.zeros(len(t))      #initialization of Force array
        
    for k in np.arange(1,len(t),1): #advancing in time        
        Y_ndot[k,0] = z_indent[k]**1.5  #zero derivative of indentation^1.5 corresponds to sample deformation^1.5
        for i in np.arange(1,len(q),1):  #calculating higher derivatives for indentation^1.5
            Y_ndot[k,i] = (Y_ndot[k,i-1]-Y_ndot[k-1,i-1])/dt
        suma_q = 0.0
        for i in np.arange(0,len(q),1):#range(len(q)):
            suma_q = suma_q + alpha/u[-1]*q[i]*Y_ndot[k,i]
        #print('Suma Q: %2.12f'%suma_q)
        suma_u = 0.0
        for i in np.arange(0,len(q)-1,1):
            suma_u = suma_u + 1.0/u[-1]*u[i]*F_mdot[k-1,i]
        #print('Suma U: %2.12f'%suma_u)
        F_mdot[k,-1] = suma_q - suma_u  #Calculating highest order time derivative on Force
        for i in np.arange(len(q)-2,-1,-1): #calculating lower order time derivatives on Force from the highest one, using Euler scheme
            F_mdot[k,i] = F_mdot[k-1,i] + F_mdot[k,i+1]*dt        
        Fts[k] = F_mdot[k,0]  #zero derivative of force is Fts
    return t, z_indent, Fts


def contact_mode_gen(time, timestep, zb_init, vb, u, q, k_m1, fo1, Q1, vdw, R, nu=0.5, z_contact=0.0, F_trigger=800.0e-9, H=2.0e-19, a=0.2e-9):
    """
    Description:
    This function simulates the AFM technique - Contact Mode for an arbitrary number of characteristic
    times.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Parameters:
    :param time:           array of floats
                        time array including each iteration time step

    :param timestep:    float
                        time step of time array

    :param zb_init:     float
                        initial base position

    :param vb:          float
                        base velocity of approach

    :param u:           float array of floats
                        differential constants

    :param q:           array of floats
                        differential constants

    :param k_m1:        float
                        stiffness of cantilever

    :param fo1:         float
                        natural frequency of the free air cantilever

    :param Q1:          float
                        Quality factor

    :param vdw:         float
                        vdW Force trigger variable

    :param R:           float
                        radius of tip indentor or probe

    Default Parameters
    :param nu:          float
                        Poission's ration of sample material, assumption: incompressible -> nu=0.5

    :param z_contact:   float
                        constant indicating the sample surface location and therefore determines
                        when contact occurs

    :param F_trigger:   float
                        trigger force, max allowable force

    :param H:           float
                        Hammaker constant

    :param a:           float
                        intermolecular distance
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Return, function output:
    :return:            np.array(F_ts):   array of floats
                                          tip-sample force solution array
                        np.array(tip):    array of floats
                                          tip position solution array, (deformation of sample material)
                        np.array(base):   array of floats
                                          cantilever base position solution array
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    """
    alpha = 8.0 / 3 * np.sqrt(R) / (1 - nu)  # constant converting stress/strain to force/deformation
    m1 = k_m1 / pow((fo1 * 2 * np.pi), 2)  # equivalent mass of cantilever

    # initialize solution and calculation array
    F = np.zeros((len(time), len(u)))  # initialize force matrix, each column stores a derivative
    h = np.zeros((len(time), len(u)))  # initialize strain matrix, each column stores a derivative
    tip = np.zeros(len(time))  # initializes tip position solution array
    base = np.zeros(len(time))  # initializes base position of solution array
    F_ts = []

    # initialzes parameters for Verlet algorithm
    v1_z = vb  # cantilever velocity
    z1 = zb_init  # current cantilever position
    z1_old = zb_init  # previous cantilever positition
    zb_initial = zb_init  # initial base position of cantilever
    TipPos = zb_init  # tip position

    # Y_ndot = np.zeros((len(time),len(q)))   #zero initialization of derivatives of indentation^1.5
    # F_mdot = np.zeros((len(time),len(q)))  #zero initialization of derivatives of force
    Fts = 0  # initialization of Force array

    if zb_initial > z_contact:
        for k in np.arange(len(time)):  # advancing in time
            if Fts < F_trigger:
                # Euler Integration calculating the base position
                zb = zb_initial - vb * time[k]  # iterates base position

                # Cantilever dynamics - EOM
                a1_z = (-k_m1 * z1 - (m1 * (fo1 * 2 * np.pi) * v1_z / Q1) + k_m1 * zb + Fts) / m1  # calculates tip acc

                # Verlet algorithm to calculate z and central difference to calculate v
                z1_new = 2 * z1 - z1_old + a1_z * pow(timestep, 2)
                v1_z = (z1_new - z1_old) / (2 * timestep)

                z1_old = z1
                z1 = z1_new

                # Inizializes tip position
                TipPos = z1_new

                if TipPos < z_contact:

                    h[k, 0] = (-TipPos) ** 1.5  # zero derivative of indentation^1.5 corresponds to sample deformation^1.5
                    for i in np.arange(1, len(q), 1):  # calculating higher derivatives for indentation^1.5
                        h[k, i] = (h[k, i - 1] - h[k - 1, i - 1]) / timestep

                    suma_q = 0.0
                    for i in np.arange(0, len(q), 1):  # range(len(q)):
                        suma_q = suma_q + alpha / u[-1] * q[i] * h[k, i]
                        
                    suma_u = 0.0
                    for i in np.arange(0, len(q) - 1, 1):
                        suma_u = suma_u + 1.0 / u[-1] * u[i] * F[k - 1, i]

                    F[k, -1] = suma_q - suma_u  # Calculating highest order time derivative on Force
                    for i in np.arange(len(q) - 2, -1, -1):  # calculating lower order time derivatives on Force from the highest one, using Euler scheme
                        F[k, i] = F[k - 1, i] + F[k, i + 1] * timestep

                    Fts = F[k, 0]

                    if vdw < 0.5:  # no vdW interaction
                        F_ts.append(Fts)  # stored tip-sample force
                    else:
                        F_ts.append(Fts - H * R / (6 * pow(a, 2)))  # stored tip-sample force
                else:  # van der Vaal interaction
                    if vdw > 0.5:  # van der Waal interaction
                        F_ts.append(- H * R / (6 * (TipPos + a) ** 2))  # stored tip-sample force
                    else:  # no van der Waal interaction
                        F_ts.append(Fts)  # stored tip-sample force

                tip[k] = TipPos  # tip position of cantilever
                base[k] = zb  # base position of cantilever

                if k % 1000000 == 0:  # print ever 100,000th step to show the calc is running
                    print('Iteration:', k)
                    print('For:', Fts)
                    print('Tip:', TipPos)
                    print('Base:', zb)

            else:
                print('F_trigger:', F_trigger)
                print(k)
                break

    if zb_initial == z_contact:
        for k in np.arange(1, len(time), 1):  # advancing in time
            if Fts < F_trigger:
                # Euler Integration calculating the base position
                zb = zb_initial - vb * time[k]  # iterates base position

                # Cantilever dynamics - EOM
                a1_z = (-k_m1 * z1 - (m1 * (fo1 * 2 * np.pi) * v1_z / Q1) + k_m1 * zb + Fts) / m1  # calculates tip acc

                # Verlet algorithm to calculate z and central difference to calculate v
                z1_new = 2 * z1 - z1_old + a1_z * pow(timestep, 2)
                v1_z = (z1_new - z1_old) / (2 * timestep)

                z1_old = z1
                z1 = z1_new

                # Inizializes tip position
                TipPos = z1_new

                if TipPos < z_contact:

                    h[k, 0] = (-TipPos) ** 1.5  # zero derivative of indentation^1.5 corresponds to sample deformation^1.5
                    for i in np.arange(1, len(q), 1):  # calculating higher derivatives for indentation^1.5
                        h[k, i] = (h[k, i - 1] - h[k - 1, i - 1]) / timestep

                    suma_q = 0.0
                    for i in np.arange(0, len(q), 1):  # range(len(q)):
                        suma_q = suma_q + alpha / u[-1] * q[i] * h[k, i]

                    suma_u = 0.0
                    for i in np.arange(0, len(q) - 1, 1):
                        suma_u = suma_u + 1.0 / u[-1] * u[i] * F[k - 1, i]

                    F[k, -1] = suma_q - suma_u  # Calculating highest order time derivative on Force
                    for i in np.arange(len(q) - 2, -1,
                                       -1):  # calculating lower order time derivatives on Force from the highest one, using Euler scheme
                        F[k, i] = F[k - 1, i] + F[k, i + 1] * timestep

                    Fts = F[k, 0]

                    if vdw < 0.5:  # no vdW interaction
                        F_ts.append(Fts)  # stored tip-sample force
                    else:
                        F_ts.append(Fts - H * R / (6 * pow(a, 2)))  # stored tip-sample force
                else:  # van der Vaal interaction
                    if vdw > 0.5:  # van der Waal interaction
                        F_ts.append(- H * R / (6 * (TipPos + a) ** 2))  # stored tip-sample force
                    else:  # no van der Waal interaction
                        F_ts.append(Fts)  # stored tip-sample force

                tip[k] = TipPos  # tip position of cantilever
                base[k] = zb  # base position of cantilever

                if k % 1000000 == 0:  # print ever 100,000th step to show the calc is running
                    print('Iteration:', k)
                    print('For:', Fts)
                    print('Tip:', TipPos)
                    print('Base:', zb)

            else:
                print('F_trigger:', F_trigger)
                print(k)
                break

    return np.array(F_ts), tip, base


def contact_mode(t, timestep, zb_init, vb, u, q, k_m1, fo1, Q1, vdw, R, nu=0.5, z_contact=0.0, F_trigger=800.0e-9, H=2.0e-19, a=0.2e-9):
    """
    Description:
    This function simulates the AFM technique - Contact Mode for Standard 3 Parameter Model, 3 and 5-Arm
    Generalized Maxwell/Kelvin-Voigt Model.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Parameters:
    :param t:           array of floats
                        time array including each iteration time step

    :param timestep:    float
                        time step of time array

    :param zb_init:     float
                        initial base position

    :param vb:          float
                        base velocity of approach

    :param u:           float array of floats
                        differential constants

    :param q:           array of floats
                        differential constants

    :param k_m1:        float
                        stiffness of cantilever

    :param fo1:         float
                        natural frequency of the free air cantilever

    :param Q1:          float
                        Quality factor

    :param vdw:         float
                        vdW Force trigger variable

    :param R:           float
                        radius of tip indentor or probe
    
    Default Parameters:
    :param nu:          float
                        Poission's ration of sample material, assumption: incompressible -> nu=0.5

    :param z_contact:   float
                        constant indicating the sample surface location and therefore determines
                        when contact occurs

    :param F_trigger:   float
                        trigger force, max allowable force

    :param H:           float
                        Hammaker constant

    :param a:           float
                        intermolecular distance
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Return, function output:
    :return:            np.array(F_ts):   array of floats
                                          tip-sample force solution array
                        np.array(tip):    array of floats
                                          tip position solution array, (deformation of sample material)
                        np.array(base):   array of floats
                                          cantilever base position solution array
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    """
    alpha = 8.0 / 3 * np.sqrt(R) / (1 - nu)  # constant converting stress/strain to force/deformation
    m1 = k_m1 / pow((fo1 * 2 * np.pi), 2)  # equivalent mass of cantilever

    # initialize solution and calculation array
    F = np.zeros((len(t), len(u)))  # initialize force matrix, each column stores a derivative
    h = np.zeros((len(t), len(u)))  # initialize strain matrix, each column stores a derivative
    tip = []  # initializes tip position solution array
    base = []  # initializes base position of solution array
    F_ts = []  # initializes F solution array

    # initialzes parameters for Verlet algorithm
    #a1_z = 0.0  # cantilever acceleration
    v1_z = vb  # cantilever velocity
    z1 = zb_init  # current cantilever position
    z1_old = zb_init  # previous cantilever positition
    #z1_new = zb_init  # next cantilever position
    zb_initial = zb_init  # initial bae position of cantilever
    Fts = 0.0  # tip-sample force
    TipPos = zb_init  # tip position
    #zb = 0.0  # cantilever base position

    if len(u) == 2:  # Standard 3 Parameter Maxwell/Kelvin-Voigt
        print('Standard 3 Parameter Maxwell/Kelvin-Voigt')
        for i in range(len(t)):  # integration loop
            if Fts < F_trigger:  # set contidition for iteration and maximum tip sample force
                # Euler Integration calculating the base position
                zb = zb_initial - vb * t[i]  # iterates base position

                # Cantilever dynamics - EOM
                a1_z = (-k_m1 * z1 - (m1 * (fo1 * 2 * np.pi) * v1_z / Q1) + k_m1 * zb + Fts) / m1  # calculates tip acc

                # Verlet algorithm to calculate z and central difference to calculate v
                z1_new = 2 * z1 - z1_old + a1_z * pow(timestep, 2)
                v1_z = (z1_new - z1_old) / (2 * timestep)

                z1_old = z1
                z1 = z1_new

                # Inizializes tip position
                TipPos = z1_new

                if TipPos < z_contact:
                    h[i][0] = (-TipPos) ** 1.5  # lowest deformation derivative
                    # uses finite difference to calcuate higher order derivatives
                    h[i][1] = (h[i][0] - h[i - 1][0]) / timestep

                    # calcualtes highest force deriavtive first
                    F[i][1] = (alpha * (q[0] * h[i][0] + q[1] * h[i][1]) - u[0] * F[i - 1][0]) / u[1]
                    # uses euler integration to calculate lower order force derivatives
                    F[i][0] = F[i - 1][0] + F[i][1] * timestep

                    Fts = F[i][0]

                    if vdw < 0.5:  # no vdW interaction
                        F_ts.append(Fts)  # stored tip-sample force
                    else:
                        F_ts.append(Fts - H * R / (6 * pow(a, 2)))  # stored tip-sample force
                else:  # van der Vaal interaction
                    if vdw > 0.5:  # van der Waal interaction
                        F_ts.append(- H * R / (6 * (TipPos + a) ** 2))  # stored tip-sample force
                    else:  # no van der Waal interaction
                        F_ts.append(Fts)  # stored tip-sample force

                tip.append(TipPos)  # tip position of cantilever
                base.append(zb)  # base position of cantilever

                if i % 1000000 == 0:  # print ever 100,000th step to show the calc is running
                    print('Iteration:', i)
                    print('For:', Fts)
                    print('Tip:', TipPos)
                    print('Base:', zb)

            else:
                print('F_trigger:', F_trigger)
                print(i)
                break

    if len(u) == 4:  # Generalized 3 Arm Maxwell/Kelvin-Voigt
        print('Generalized 3 Arm Maxwell/Kelvin-Voigt')
        for i in range(len(t)):  # integration loop
            if Fts < F_trigger:  # set contidition for iteration and maximum tip sample force
                # Euler Integration calculating the base position
                zb = zb_initial - vb * t[i]  # iterates base position

                # Cantilever dynamics - EOM
                a1_z = (-k_m1 * z1 - (m1 * (fo1 * 2 * np.pi) * v1_z / Q1) + k_m1 * zb + Fts) / m1 # calculates tip acc

                # Verlet algorithm to calculate z and central difference to calculate v
                z1_new = 2 * z1 - z1_old + a1_z * pow(timestep, 2)
                v1_z = (z1_new - z1_old) / (2 * timestep)

                z1_old = z1
                z1 = z1_new

                # Inizializes tip position
                TipPos = z1_new

                if TipPos < z_contact:
                    h[i][0] = (-TipPos) ** 1.5  # lowest deformation derivative
                    # uses finite difference to calcuate higher order derivatives
                    h[i][1] = (h[i][0] - h[i - 1][0]) / timestep
                    h[i][2] = (h[i][1] - h[i - 1][1]) / timestep
                    h[i][3] = (h[i][2] - h[i - 1][2]) / timestep

                    # calcualtes highest force deriavtive first
                    F[i][3] = (alpha * (q[0] * h[i][0] + q[1] * h[i][1] + q[2] * h[i][2] + q[3] * h[i][3]) - u[0] *
                               F[i - 1][0] - u[1] * F[i - 1][1] - u[2] * F[i - 1][2]) / u[3]
                    # uses euler integration to calculate lower order force derivatives
                    F[i][2] = F[i - 1][2] + F[i][3] * timestep
                    F[i][1] = F[i - 1][1] + F[i][2] * timestep
                    F[i][0] = F[i - 1][0] + F[i][1] * timestep

                    Fts = F[i][0]

                    if vdw < 0.5:  # no vdW interaction
                        F_ts.append(Fts)  # stored tip-sample force
                    else:
                        F_ts.append(Fts - H * R / (6 * pow(a, 2)))  # stored tip-sample force
                else:  # van der Vaal interaction
                    if vdw > 0.5:  # van der Waal interaction
                        F_ts.append(- H * R / (6 * (TipPos + a) ** 2))  # stored tip-sample force
                    else:  # no van der Waal interaction
                        F_ts.append(Fts)  # stored tip-sample force

                tip.append(TipPos)  # tip position of cantilever
                base.append(zb)  # base position of cantilever

                if i % 1000000 == 0:  # print ever 100,000th step to show the calc is running
                    print('Iteration:', i)
                    print('For:', Fts)
                    print('Tip:', TipPos)
                    print('Base:', zb)

            else:
                print('F_trigger:', F_trigger)
                print(i)
                break

    if len(u) == 6:  # Generalized 5 Arm Maxwell/Kelvin-Voigt
        print('Generalized 5 Arm Maxwell/Kelvin-Voigt')
        for i in range(len(t)):  # integration loop
            if Fts < F_trigger:  # set contidition for iteration and maximum tip sample force
                # Euler Integration calculating the base position
                zb = zb_initial - vb * t[i]  # iterates base position

                # Cantilever dynamics - EOM
                a1_z = (-k_m1 * z1 - (m1 * (fo1 * 2 * np.pi) * v1_z / Q1) + k_m1 * zb + Fts) / m1  # calculates tip acc

                # Verlet algorithm to calculate z and central difference to calculate v
                z1_new = 2 * z1 - z1_old + a1_z * pow(timestep, 2)
                v1_z = (z1_new - z1_old) / (2 * timestep)

                z1_old = z1
                z1 = z1_new

                # Inizializes tip position
                TipPos = z1_new

                if TipPos < z_contact:
                    h[i][0] = (-TipPos) ** 1.5  # lowest deformation derivative
                    # uses finite difference to calcuate higher order derivatives
                    h[i][1] = (h[i][0] - h[i - 1][0]) / timestep
                    h[i][2] = (h[i][1] - h[i - 1][1]) / timestep
                    h[i][3] = (h[i][2] - h[i - 1][2]) / timestep
                    h[i][4] = (h[i][3] - h[i - 1][3]) / timestep
                    h[i][5] = (h[i][4] - h[i - 1][4]) / timestep

                    # calcualtes highest force deriavtive first
                    F[i][5] = (alpha * (q[0] * h[i][0] + q[1] * h[i][1] + q[2] * h[i][2] + q[3] * h[i][3] + q[4] *
                                        h[i][4] + q[5] * h[i][5]) - u[0] * F[i - 1][0] - u[1] * F[i - 1][1] - u[2] *
                               F[i - 1][2] - u[3] * F[i - 1][3] - u[4] * F[i - 1][4]) / u[5]
                    # uses euler integration to calculate lower order force derivatives
                    F[i][4] = F[i - 1][4] + F[i][5] * timestep
                    F[i][3] = F[i - 1][3] + F[i][4] * timestep
                    F[i][2] = F[i - 1][2] + F[i][3] * timestep
                    F[i][1] = F[i - 1][1] + F[i][2] * timestep
                    F[i][0] = F[i - 1][0] + F[i][1] * timestep

                    Fts = F[i][0]

                    if vdw < 0.5:  # no vdW interaction
                        F_ts.append(Fts)  # stored tip-sample force
                    else:
                        F_ts.append(Fts - H * R / (6 * pow(a, 2)))  # stored tip-sample force
                else:  # van der Vaal interaction
                    if vdw > 0.5:  # van der Waal interaction
                        F_ts.append(- H * R / (6 * (TipPos + a) ** 2))  # stored tip-sample force
                    else:  # no van der Waal interaction
                        F_ts.append(Fts)  # stored tip-sample force

                tip.append(TipPos)  # tip position of cantilever
                base.append(zb)  # base position of cantilever

                if i % 1000000 == 0:  # print ever 100,000th step to show the calc is running
                    print('Iteration:', i)
                    print('For:', Fts)
                    print('Tip:', TipPos)
                    print('Base:', zb)

            else:
                print('F_trigger:', F_trigger)
                print(i)
                break

    return np.array(F_ts), np.array(tip), np.array(base)


def verlet(zb, Fo1, Fo2, Fo3, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, time, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old, Fts, dt, fo1, fo2, fo3, f1, f2, f3):
    """
    Description: 
    This function perform the verlot algorihm to calculate the tip position of the cantilever.
    Parameters:
    
    Return:
    """
    a1 = ( -k_m1*z1 - (mass*(fo1*2*np.pi)*v1/Q1) + Fo1*np.cos((f1*2*np.pi)*time) + Fo2*np.cos((f2*2*np.pi)*time) + Fo3*np.cos((f3*2*np.pi)*time) + Fts ) /mass
    a2 = ( -k_m2*z2 - (mass*(fo2*2*np.pi)*v2/Q2) + Fo1*np.cos((f1*2*np.pi)*time) + Fo2*np.cos((f2*2*np.pi)*time) + Fo3*np.cos((f3*2*np.pi)*time) + Fts ) /mass
    a3 = ( -k_m3*z3 - (mass*(fo3*2*np.pi)*v3/Q3) + Fo1*np.cos((f1*2*np.pi)*time) + Fo2*np.cos((f2*2*np.pi)*time) + Fo3*np.cos((f3*2*np.pi)*time) + Fts ) /mass
    
    #Verlet algorithm (central difference) to calculate position of the tip
    z1_new = 2*z1 - z1_old + a1*pow(dt, 2)
    z2_new = 2*z2 - z2_old + a2*pow(dt, 2)
    z3_new = 2*z3 - z3_old + a3*pow(dt, 2)

    #central difference to calculate velocities
    v1 = (z1_new - z1_old)/(2*dt)
    v2 = (z2_new - z2_old)/(2*dt)
    v3 = (z3_new - z3_old)/(2*dt)
    
    #Updating z1_old and z1 for the next run
    z1_old = z1
    z1 = z1_new
    
    z2_old = z2
    z2 = z2_new
    
    z3_old = z3
    z3 = z3_new
    
    tip = zb + z1 + z2 + z3
    #tip_v = v1 + v2 + v3
    return tip, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old

numba_verlet = jit()(verlet)


def tapping(u, q, R, timestep, simultime, zb, A1, k_m1, fo1, printstep='default', Ndy = 1000, dmax = 10.0e-9, startprint ='default', Q1=100, Q2=200, Q3=300, H=2.0e-19, A2 = 0.0, A3 = 0.0, nu=0.5):
    """
    Description: 
    This function performs the Tapping Mode AFM measurement technique. 
    Parameters:
    
    Return:
    """
    fo2 = 6.27*fo1            # resonance frequency of the second eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    fo3 = 17.6*fo1           # resonance frequency of the third eigenmode (value taken from Garcia, R., & Herruzo, E. T. (2012). The emergence of multifrequency force microscopy. Nature nanotechnology, 7(4), 217-226.)
    k_m2 = k_m1*(fo2/fo1)**2
    k_m3 = k_m1*(fo3/fo1)**2
    mass = k_m1/(2.0*np.pi*fo1)**2  
    f1 = fo1  #excited at resonance
    f2 = fo2  #excited at resonance
    f3 = fo3  #excited at resonance
    #Calculating the force amplitude to achieve the given free amplitude from amplitude response of tip excited oscillator
    Fo1 = k_m1*A1/(fo1*2*np.pi)**2*(  (  (fo1*2*np.pi)**2 - (f1*2*np.pi)**2 )**2 + (fo1*2*np.pi*f1*2*np.pi/Q1)**2  )**0.5 #Amplitude of 1st mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo2 = k_m2*A2/(fo2*2*np.pi)**2*(  (  (fo2*2*np.pi)**2 - (f2*2*np.pi)**2 )**2 + (fo2*2*np.pi*f2*2*np.pi/Q2)**2  )**0.5  #Amplitude of 2nd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    Fo3 = k_m3*A3/(fo3*2*np.pi)**2*(  (  (fo3*2*np.pi)**2 - (f3*2*np.pi)**2 )**2 + (fo3*2*np.pi*f3*2*np.pi/Q3)**2  )**0.5    #Amplitude of 3rd mode's force to achieve target amplitude based on amplitude response of a tip excited harmonic oscillator
    
    alpha = 8.0 / 3 * np.sqrt(R) / (1 - nu)  # constant converting stress/strain to force/deformation
    m1 = k_m1 / pow((fo1 * 2 * np.pi), 2)  # equivalent mass of cantilever
    time = np.arange(0, simultime, timestep)
    Fts = 0
    z_contact = 0 
    zb = 0 
    
    # initialize solution and calculation array
    F = np.zeros((len(time), len(u)))  # initialize force matrix, each column stores a derivative
    h = np.zeros((len(time), len(u)))  # initialize strain matrix, each column stores a derivative
    tip = np.zeros(len(time))  # initializes tip position solution array
    base = np.zeros(len(time))  # initializes base position of solution array
    F_ts = []
    z_b = np.zeros(len(time))
    
    if zb_initial > z_contact:
        for k in range(len(time)):
            TipPos, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = numba_verlet(zb, Fo1, Fo2, Fo3, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, time[t], z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old, Fts, timestep, fo1, fo2, fo3, f1, f2, f3)
            
            tip[k] = TipPos  # keeps track of tip position history
            t_b[k] = zb
            F_ts.append(Fts)
            
            if TipPos > zb:  # no contact, Force = 0 
                suma_q = 0.0
                for i in np.arange(0, len(q) - 1, 1):
                    suma_u = suma_u + 1.0 / q[-1] * q[i] * h[k - 1, i]
                
                h[k, -1] = suma_q
                for i in np.arange(len(q) - 2, -1, -1):  # calculating lower order time derivatives of deformation from the highest one, using Euler scheme
                    h[k, i] = h[k - 1, i] + h[k, i + 1] * timestep
                
                if h[k, 0] > TipPos:  #contact, sample surface surpassed the tip position
                    zb = TipPos 
                    h[k, 0] = (-zb) ** 1.5  # zero derivative of indentation^1.5 corresponds to sample deformation^1.5
                    for i in np.arange(1, len(q), 1):  # calculating higher derivatives for indentation^1.5
                        h[k, i] = (h[k, i - 1] - h[k - 1, i - 1]) / timestep

                    suma_q = 0.0
                    for i in np.arange(0, len(q), 1):  # range(len(q)):
                        suma_q = suma_q + alpha / u[-1] * q[i] * h[k, i]

                    suma_u = 0.0
                    for i in np.arange(0, len(q) - 1, 1):
                        suma_u = suma_u + 1.0 / u[-1] * u[i] * F[k - 1, i]

                    F[k, -1] = suma_q - suma_u  # Calculating highest order time derivative on Force
                    for i in np.arange(len(q) - 2, -1, -1):  # calculating lower order time derivatives on Force from the highest one, using Euler scheme
                        F[k, i] = F[k - 1, i] + F[k, i + 1] * timestep

                    Fts = F[k, 0]
                    
                else:  # true non-contact
                    zb = h[k, 0] 
                    Fts = 0 
                    
            else:  # contact,  
                h[k, 0] = (-TipPos) ** 1.5  # zero derivative of indentation^1.5 corresponds to sample deformation^1.5
                for i in np.arange(1, len(q), 1):  # calculating higher derivatives for indentation^1.5
                    h[k, i] = (h[k, i - 1] - h[k - 1, i - 1]) / timestep

                suma_q = 0.0
                for i in np.arange(0, len(q), 1):  # range(len(q)):
                    suma_q = suma_q + alpha / u[-1] * q[i] * h[k, i]

                suma_u = 0.0
                for i in np.arange(0, len(q) - 1, 1):
                    suma_u = suma_u + 1.0 / u[-1] * u[i] * F[k - 1, i]

                F[k, -1] = suma_q - suma_u  # Calculating highest order time derivative on Force
                for i in np.arange(len(q) - 2, -1, -1):  # calculating lower order time derivatives of Force from the highest one, using Euler scheme
                    F[k, i] = F[k - 1, i] + F[k, i + 1] * timestep

                Fts = F[k, 0]
           # van der Waals Implementation
           #if TipPos > 0:  #overall non-contact
           #    F = -H*R/( 6.0*( (TipPos) + a )**2 )
            #else: #overall contact
            #    F = Fts - H*R/(6.0*a**2)  
            
    if zb_initial == z_contact:
        for k in range(1, len(time), 1):
            TipPos, z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old = numba_verlet(zb, Fo1, Fo2, Fo3, Q1, Q2, Q3, k_m1, k_m2, k_m3, mass, time[t], z1, z2, z3, v1, v2, v3, z1_old, z2_old, z3_old, Fts, timestep, fo1, fo2, fo3, f1, f2, f3)
            
            tip[k] = TipPos  # keeps track of tip position history
            z_b[k] = zb
            F_ts.append(Fts)
            
            if TipPos > zb:  # no contact, Force = 0 
                
                suma_q = 0.0
                for i in np.arange(0, len(q) - 1, 1):
                    suma_u = suma_u + 1.0 / q[-1] * q[i] * h[k - 1, i]
                
                h[k, -1] = suma_q
                for i in np.arange(len(q) - 2, -1, -1):  # calculating lower order time derivatives of deformation from the highest one, using Euler scheme
                    h[k, i] = h[k - 1, i] + h[k, i + 1] * timestep
                
                if h[k, 0] > TipPos:  #contact, sample surface surpassed the tip position
                    zb = TipPos 
                    h[k, 0] = (-zb) ** 1.5  # zero derivative of indentation^1.5 corresponds to sample deformation^1.5
                    for i in np.arange(1, len(q), 1):  # calculating higher derivatives for indentation^1.5
                        h[k, i] = (h[k, i - 1] - h[k - 1, i - 1]) / timestep

                    suma_q = 0.0
                    for i in np.arange(0, len(q), 1):  # range(len(q)):
                        suma_q = suma_q + alpha / u[-1] * q[i] * h[k, i]

                    suma_u = 0.0
                    for i in np.arange(0, len(q) - 1, 1):
                        suma_u = suma_u + 1.0 / u[-1] * u[i] * F[k - 1, i]

                    F[k, -1] = suma_q - suma_u  # Calculating highest order time derivative on Force
                    for i in np.arange(len(q) - 2, -1, -1):  # calculating lower order time derivatives on Force from the highest one, using Euler scheme
                        F[k, i] = F[k - 1, i] + F[k, i + 1] * timestep

                    Fts = F[k, 0]
                    
                else:  # true non-contact
                    zb = h[k, 0]
                    Fts = 0 
                    
            else:  # contact,  
                h[k, 0] = (-TipPos) ** 1.5  # zero derivative of indentation^1.5 corresponds to sample deformation^1.5
                for i in np.arange(1, len(q), 1):  # calculating higher derivatives for indentation^1.5
                    h[k, i] = (h[k, i - 1] - h[k - 1, i - 1]) / timestep

                suma_q = 0.0
                for i in np.arange(0, len(q), 1):  # range(len(q)):
                    suma_q = suma_q + alpha / u[-1] * q[i] * h[k, i]

                suma_u = 0.0
                for i in np.arange(0, len(q) - 1, 1):
                    suma_u = suma_u + 1.0 / u[-1] * u[i] * F[k - 1, i]

                F[k, -1] = suma_q - suma_u  # Calculating highest order time derivative on Force
                for i in np.arange(len(q) - 2, -1, -1):  # calculating lower order time derivatives of Force from the highest one, using Euler scheme
                    F[k, i] = F[k - 1, i] + F[k, i + 1] * timestep

                Fts = F[k, 0]
                
            # van der Waals Implementation
            #if TipPos > 0:  #overall non-contact
            #    F = -H*R/( 6.0*( (TipPos) + a )**2 )
            #else: #overall contact
            #    F = Fts - H*R/(6.0*a**2) 
            
    return np.array(F_ts), tip, z_b, 
