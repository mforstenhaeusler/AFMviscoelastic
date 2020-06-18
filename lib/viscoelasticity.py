"""
Created on Wed Jun 10th 2020

@author: Maximilian Forstenhaeusler
Description: This library contains core algorithms to analyse and fit viscoelstic models to different AFM data.
"""

import numpy as np
from lmfit import Parameters, minimize, report_fit


def G_storage(omega, Ge, G, tau):
    """
    Description:
    The function calculates the Storage Modulus G' using G as input.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Parameters:
    :param omega: array of float
                  array of Frequency values
    :param Ge:    float
                  equilibrium G
    :param G:     array of loats
                  Relaxance values
    :param tau:   array of floats
                  characteristic time
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Return, function output:
    :return:      G_prime:  array of loats
                            array of Storage Modulus G'
    """
    G_prime = np.zeros(len(omega))
    for i in range(len(omega)):
        G_prime[i] = Ge + sum((G[:] * pow(omega[:], 2) * pow(tau[:], 2)) / (1.0 + (pow(omega[i], 2) * pow(tau[:], 2))))

    return G_prime


def J_storage(omega, Jg, J, tau):
    """
    Description:
    The function calculates the Storage Modulus using J as input.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Parameters:
    :param omega: array of floats
                  array of Frequency values
    :param Jg:    float
                  classy compliance
    :param J:     array of floats
                  Compliance values
    :param tau:   array of floats
                  characteristic time
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Return, function ouput:
    :return:      J_prime:  array of floats
                            Storage Modulus J'
    """
    J_prime = np.zeros(len(omega))
    for i in range(len(omega)):
        if len(J) > 1:
            J_prime[i] = Jg + sum(J[:] / (1.0 + (pow(omega[i], 2) * pow(tau[:], 2))))
        else:
            J_prime[i] = Jg + (J / (1.0 + (pow(omega[i], 2) * pow(tau, 2))))

    return J_prime


def G_loss(omega, Ge, G, tau):
    """
    Description:
    The function calculates the Loss Modulus using G as input.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Parameters:
    :param omega: array of floats
                  array of Frequency values
    :param Ge:    float
                  equilibrium G
    :param G:     array of floats
                  Relaxance values
    :param tau:   array of floats
                  characteristic time
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Return, function ouput:
    :return:      G_dprime: array of floats
                            Loss Modulus G''
    """
    G_dprime = np.zeros(len(omega))
    for i in range(len(omega)):
        G_dprime[i] = Ge + sum((G[:]*omega[:]*tau[:])/(1.0 + (pow(omega[i], 2) * pow(tau, 2))))

    return G_dprime


def J_loss(omega, Jg, J, tau, phi=0):
    """
    Description:
    The function calculates the Loss Modulus using J as input.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Parameters:
    :param omega: array of floats
                  array of Frequency values
    :param Jg:    float
                  classy compliance
    :param J:     array of floats
                  Compliance values
    :param tau:   array of floats
                  characteristic time
    :param phi:   float
                  steady-state/ stead-flow fluidity
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Return, function ouput:
    :return:      J_dprime: array of floats
                            Loss Modulus J''
    """
    J_dprime = np.zeros(len(omega))
    for i in range(len(omega)):  # for Gen. Kelvin-Voigt with arbitrary number of arms
        if len(J) > 1:
            J_dprime[i] = sum(J[:] * omega[i] * tau[:] / (1.0 + (pow(omega[i], 2) * pow(tau[:], 2)))) + phi / omega[i]
        else:  # for SLS
            J_dprime[i] = (J * omega[i] * tau / (1.0 + (pow(omega[i], 2) * pow(tau, 2)))) + phi / omega[i]

    return J_dprime


def theta_loss(input_param, omega, p1, p2, p3, phi=0):
    """
    Description:
    The function calculates the Loss Angle, theta, using J oro G as input.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Parameters:
    :param omega: array of floats
                  array of Frequency values
    :param p1:    float
                  placeholder variable for classy compliance or equivalent G
    :param p2:    array of floats
                  placeholder variable for Compliance or Relaxance values
    :param p3:    array of floats
                  placeholder variable fot characteristic time
    :param phi:   float
                  steady-state/steady-flow fluidity
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Return, function ouput:
    :return:      theta:    array of floats
                            Loss Modulus theta
    """
    if input_param == 'G':
        Gloss = G_loss(omega, p1, p2, p3)
        Gstorage = G_storage(omega, p1, p2, p3)
        theta = np.arctan(Gloss / Gstorage) * 180 / np.pi

    if input_param == 'J':
        Jloss = J_loss(omega, p1, p2, p3, phi=0)
        Jstorage = J_storage(omega, p1, p2, p3)
        theta = np.arctan(Jloss / Jstorage) * 180 / np.pi

    return theta


def response(model, t, p1, p2, p3, phi = 0.0):
    """
    Description:
    The function calculates the Relaxance, Comppliance or Retardance in time.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Parameters:
    :param model: string
                  indicate either what to return, either 'Relaxance', 'Comppliance' or ’Retardance'
    :param t:     array of floats
                  time array
    :param p1:    float
                  placeholder variable for classy compliance or equivalent G
    :param p2:    array of floats
                  placeholder variable for Compliance or Relaxance values
    :param p3:    array of floats
                  placeholder variable fot characteristic time
    :param phi:   float
                  steady-state/steady-flow fluidity
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Return, function ouput:
    :return:      res:  array of floats
                        Relaxation, Comppliance or Retardance Modulus in time
    """
    if model == 'Relaxances':
        res = np.zeros(np.size(t))
        if np.size(p2) == 1:  # SLS
            for i in range(np.size(t)):
                res[i] = p1 + p2*np.exp(-t[i]/p3)
        else:  # Gen. Maxwell with arbitrary number of arms
            for i in range(np.size(t)):
                res[i] = p1 + sum(p2[:]*np.exp(-t[i]/p3[:]))

    if model == 'Compliance':
        res = np.zeros(np.size(t))
        if np.size(p2) == 1:  # SLS
            for i in range(np.size(t)):
                res[i] = p1 + p2*(1 - np.exp(-t[i]/p3)) + (phi*t[i])
        else:  # Gen. Kelvin-Voigt with arbitrary number of arms
            for i in range(np.size(t)):
                res[i] = p1 + sum(p2[:]*(1 - np.exp(-p2[i]/p3[:]))) + phi*t[i]

    if model == 'Retardance':
        res = np.zeros(np.size(t))
        if np.size(p2) == 1:  # SLS
            for i in range(np.size(t)):
                res[i] = p1 + p2/p3*np.exp(-t[i]/p3) + (phi)
        else:  # Gen. Kelvin Voigt with arbitrary number of arms
            for i in range(np.size(t)):
                res[i] = p1 + sum(p2[:]/p3[:]*(np.exp(-p2[i]/p3[:]))) + phi

    return res


def log_scale(x, t, tr=0.1, st=1.0, nn = 10):
    """
    Description:
    The function scales an array equally in each decade, weighting it in logarithmic scale.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Parameters:
    :param x:   array of floats
                fitst array to be scaled
    :param t:   array of floats
                time array to be scaled
    :param tr:

    :param st:

    :param nn:

    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Return, function ouput:
    :return:    np.array(x_log):    array of floats
                                    first array scaled in logaritmic scale
                np.array(t_log):    array of floats
                                    time array scaled in logaritmic scale
    """
    prints = 1
    nt = len(t)
    i =0
    x_log = []
    t_log = []
    while i < nt-1:
        if t[i] >= prints*tr and t[i]<=st :
            x_log.append(x[i])
            t_log.append(t[i])
            prints = prints + 1
        i = i + 1
        if prints == nn:
            tr = tr*10
            prints = 1

    return np.array(x_log), np.array(t_log)


def time_step(t):
    """
    Description:
    This function creates a time step from a time array.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Parameters:
    :param t:   array of floats
                time array
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Return, function ouput:
    :return:    dt:     float
                        time step dt
    """
    dt = t[-1]/len(t)
    return dt


def conv(model, t, F, p1, p2, p3):
    """
    Description:
    This function calculates the convolution of the retardance, U, or relaxance, Q, and the force, F, used to verifiy
    the RHS to LHS side of Equ. 13 from Calculation of Standard Viscoelastic.....
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Parameters:
    :param model:   string
                    state Viscoelastic model used, e.g. 'Gen. Kelvin-Voigt', 'Gen. Maxwell'
    :param t:       array of floats
                    time array
    :param F:       array of floats
                    Force array
    :param p1:      float
                    placeholder variable for Ge or Jg
    :param p2:      array of floats
                    placeholder variable for G or J
    :param p3:      array of floats
                    placeholder variable for charactersitic time, tau
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Return, function ouput:
    :return:        c:  array of float
                        Returns convolution U*F of retardance, U and force, F or the convolution Q*F
                        of relaxance, Q and force, F
    """
    dt = time_step(t)  # calculated delta from time array to pass into discrete convolution to give
                       # it information about the timestep
    if model == 'Gen. Kelvin-Voigt':
        U = np.zeros(len(t))
        for i in range(len(t)):  # Retardance function (Equ 12) from Calculation of Standard...
            U[i] = sum(p2[:] / p3[:] * np.exp(-t[i] / p3[:]))

        c = np.convolve(U, F, mode='full') * dt  # calculates convolution, dt adjust for timestep
        c = c[range(len(F))] + p1 * F  # only use range of given data adds Jg*F

    # if model == 'Gen. Maxwell':  # convolutioin needs to be solved
    #    U = np.zeros(len(t))
    #    for i in range(len(t)):
    #        U[i] = sum((*np.exp(-t[i]/tau_v[:]))

    #    c = np.convolve(U, F, mode='full')*dt  # calculates convolution, dt adjust for timestep
    #    c = c[range(len(F))] + Jg*F  # only use range of given data adds Jg*F

    return c


def conv_fitting_log(params, t, F, tip, arms, dt):
    """
    Description:
    The function is the residual function of the tip data and the convolution function used for the
    NLS fitting procedure using logarithmic scaled data in time.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Parameters:
    :param params:  dict file
                    Dictionary file storing the parameters
    :param t:       array of floats
                    time array scaled in logarithmic scale
    :param F:       array of floats
                    Force array sclaed in logarithmic scale
    :param tip:     array of floats
                    Tip position or deformation array scaled in logarithmic scale
    :param arms:    int
                    Number of amrs
    :param dt:      float
                    time step of logarithmic sclaed time array
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Return, function ouput:
    :return:        residual:   array of float
                                Returns the residual of the the experimental/simulated deformation data
                                and the convolution which is minimized by the lmfit algorithm minimize
                                in the NLS_fit algoritm
    """
    J = np.zeros(arms)  # initialize J array
    tau = np.zeros(arms)  # initialize tau array

    # loads oarameters from dictionary file into J and tau array
    p = params.valuesdict()
    Jg = p['Jg']
    J[0] = p['J1']
    tau[0] = p['tau1']
    if arms > 1:
        J[1] = p['J2']
        tau[1] = p['tau2']
        if arms > 2:
            J[2] = p['J3']
            tau[2] = p['tau3']
            if arms > 3:
                J[3] = p['J4']
                tau[3] = p['tau4']
                if arms > 4:
                    J[4] = p['J5']
                    tau[4] = p['tau5']

    U = np.zeros(len(t))
    for i in range(len(t)):  # Retardance function (Equ 12) from Calculation of Standard...
        U[i] = sum(J[:] / tau[:] * np.exp(-t[i] / tau[:]))

    model = np.convolve(U, F, mode='full') * dt  # convolves retardance with F and indicates
                                                 # time step to discrete convolution
    model = model[range(np.size(F))] + Jg * F  # selects range of data of size of force array
                                               # and adds the product of classy compliance
                                               # and force array
    residual = (tip - model) / tip  # calculates the residual to be minimized by teh NLS_fit
                                    # algorithm
    return residual


def conv_fitting_time(params, t, F, tip, arms, dt):
    """
    Description:
    The function is the residual function of the tip data and the convolution function used for the
    NLS fitting procedure using scaled data in time.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Parameters:
    :param params:  dict file
                    Dictionary file storing the parameters
    :param t:       array of floats
                    time array scaled
    :param F:       array of floats
                    Force array sclaed
    :param tip:     array of floats
                    Tip position or deformation array
    :param arms:    int
                    Number of amrs
    :param dt:      float
                    time step sclaed time array
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Return, function ouput:
    :return:        residual:   array of float
                                Returns the residual of the the experimental/simulated deformation data
                                and the convolution which is minimized by the lmfit algorithm minimize
                                in the NLS_fit algoritm
    """
    J = np.zeros(arms)  # initialize J array
    tau = np.zeros(arms)  # initialize tau array

    # loads oarameters from dictionary file into J and tau array
    p = params.valuesdict()
    Jg = p['Jg']
    J[0] = p['J1']
    tau[0] = p['tau1']
    if arms > 1:
        J[1] = p['J2']
        tau[1] = p['tau2']
        if arms > 2:
            J[2] = p['J3']
            tau[2] = p['tau3']
            if arms > 3:
                J[3] = p['J4']
                tau[3] = p['tau4']
                if arms > 4:
                    J[4] = p['J5']
                    tau[4] = p['tau5']

    U = np.zeros(len(t))
    for i in range(len(t)):  # Retardance function (Equ 12) from Calculation of Standard...
        U[i] = sum(J[:] / tau[:] * np.exp(-t[i] / tau[:]))

    model = np.convolve(U, F, mode='full') * dt  # convolves retardance with F and indicates
                                                 # time step to discrete convolution
    model = model[range(np.size(F))] + Jg * F  # selects range of data of size of force array
                                               # and adds the product of classy compliance
                                               # and force array
    residual = (tip - model) / tip  # calculates the residual to be minimized by teh NLS_fit
                                    # algorithm
    return residual


def NLS_fit(technique, Jg, J, tau, arms, t, tip, F, alfa, t_res, t_exp):
    """
    Descripition:
    Nonlinear Least Sqaure Fitting algorithm used to fit viscoelastic model parameter to experimental or simulated
    data/AFM data.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Parameters:
    :param technique: string
                      using Log sclaed or time scaled data
    :param Jg:        float
                      classy compliance
    :param J:         array of floats
                      Compliance
    :param tau:       array of floats
                      characteristic time
    :param arms:      int
                      number of model arms
    :param t:         array of floats
                      time array
    :param tip:       array of floats
                      tip position over time
    :param F:         array of floats
                      tip position over time
    :param alfa:      float
                      Constant converting stress/strain to force/defromation by adjusting for tip geometry
    :param t_res:     float
                      time resolution of experiment, used to log scale the data
    :param t_exp:     float
                      final time of experiment
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Return, function outout:
    :return:          Jg_fit:   float
                                classy compliance fitted
                      J_fit:    float
                                fitted complince terms
                      tau_fit:  float
                                fitted characteristic time
    """
    params = Parameters()  # initializes Parameters dictionary file
    # algorthimen initialzes parameters dictionary with viscoelastic model parameters
    params.add('Jg', value=Jg, min=0)
    if arms > 0:
        params.add('J1', value=J[0], min=0)
        params.add('tau1', value=tau[0], min=0)
        if arms > 1:
            params.add('J2', value=J[1], min=0)
            params.add('tau2', value=tau[1], min=0)
            if arms > 2:
                params.add('J3', value=J[2], min=0)
                params.add('tau3', value=tau[2], min=0)
                if arms > 3:
                    params.add('J4', value=J[3], min=0)
                    params.add('tau4', value=tau[3], min=0)
                    if arms > 4:
                        params.add('J5', value=J[4], min=0)
                        params.add('tau5', value=tau[4], min=0)
    if technique == 0:  # log scale

        tip_norm = alfa * tip ** 1.5  # RHS of equation ..., underlying fitting data
        tip_norm_log, t_log = log_scale(tip_norm, t, t_res, t_exp)  # normalized tip position scaled equally in per decade
        F_log, _ = log_scale(F, t, t_res, t_exp)   # tip-sample Force scaled scaled equally in per decad
        dt = time_step(t_log)  # calculated time step from input time array, necessary for discrete convolution

        result = minimize(conv_fitting_log, params, args=(t_log, F_log, tip_norm_log, arms, dt), method='leastsq')  # non-linear least sqaure fitting procedure minimizing the convolution function
        print(report_fit(result))  # prints the fit report

    if technique == 1:  # time scale
        tip_norm = alfa * tip ** 1.5
        dt = time_step(t)
        result = minimize(conv_fitting_time, params, args=(t, F, tip_norm, arms, dt), method='leastsq')
        print(report_fit(result))

    # stores fitted parameters
    Jg_fit = result.params['Jg'].value
    J_fit = np.zeros(arms)
    tau_fit = np.zeros(arms)
    J_fit[0] = result.params['J1'].value
    tau_fit[0] = result.params['tau1'].value
    if arms > 1:
        J_fit[1] = result.params['J2'].value
        tau_fit[1] = result.params['tau2'].value
        if arms > 2:
            J_fit[2] = result.params['J3'].value
            tau_fit[2] = result.params['tau3'].value
            if arms > 3:
                J_fit[3] = result.params['J4'].value
                tau_fit[3] = result.params['tau4'].value
                if arms > 4:
                    J_fit[4] = result.params['J5'].value
                    tau_fit[4] = result.params['tau5'].value

    return Jg_fit, J_fit, tau_fit
