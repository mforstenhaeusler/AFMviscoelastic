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


def contact_mode(t, timestep, zb_init, vb, u, q, k_m1, fo1, Q1, vdw, R, nu, z_contact=0.0, F_trigger=800.0e-9, H=2.0e-19, a=0.2e-9):
    """
    Description:
    This function simulates tha AFM technique - Contact Mode.
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

    :param nu:          float
                        Poission's ration of sample material

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
