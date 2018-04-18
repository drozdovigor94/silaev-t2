#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:00:52 2018

@author: igor
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from pyswarm import pso
import time

q = np.pi / 180
THETA_vec = np.arange(-90, 90, 0.01)

def E_nusla(THETA, THETA0, d):
    return np.abs(np.sum(np.exp(1j*2*np.pi*d*(np.sin(q*THETA) - np.sin(q*THETA0)))))

def NUSLA_SLR_opt(N=10, BW=20, FN_tolerance = 0.1):
    '''N = 10
    THETA0 = 0
    BW = 20'''

    b = 1/(N*np.sin(q*(BW/2)))
    
    d = np.arange(N)*b
    #BW = np.abs(np.arccos(-1/(N*b)) - np.arccos(1/(N*b)))/q
    FN = BW/2
    
    theta_SLL = np.append(np.arange(-90, -FN, 0.5), np.arange(FN, 90, 0.5))
    
    def errfun(dd):
        d_opt = d + np.append([0], dd)
        E_SLL = np.vectorize(lambda THETA:E_nusla(THETA, 0, d_opt))(theta_SLL)
        errSLL = np.max(E_SLL)
        # adjust number after less sign to change priority of BW over SLL
        if E_nusla(FN, 0, d_opt) < FN_tolerance:
            errBW = 0
        else:
            errBW = 20
        return errSLL + errBW
    consfun = lambda x: np.ediff1d(d + np.append([0],x))
    xopt, fopt = pso(errfun, [-1]*(N-1), [1]*(N-1), swarmsize=500, maxiter=1000, f_ieqcons=consfun)
    return (d, d + np.append([0], xopt))

def plot_results(THETA0, d, dopt):
    Ev = np.vectorize(lambda THETA:E_nusla(THETA, THETA0, d))(THETA_vec)
    Ev_opt = np.vectorize(lambda THETA:E_nusla(THETA, THETA0, d_opt))(THETA_vec)
    
    fig = plt.figure()
    ax1 = plt.subplot(221)
    ax1.plot(THETA_vec, Ev, THETA_vec, Ev_opt)
    ax1.set_xticks(np.arange(-90, 90+1, 10))
    ax1.set_xlabel(r'$\Theta$')
    ax1.set_ylabel(r'$E(\Theta)$')
    ax1.grid()
    
    ax2 = plt.subplot(222, projection='polar')
    ax2.set_theta_direction('clockwise')
    ax2.set_theta_zero_location('N')
    ax2.plot(THETA_vec*q, Ev, THETA_vec*q, Ev_opt)
    ax2.set_thetamin(-90)
    ax2.set_thetamax(90)
    ax2.set_xticks(np.arange(-90, 90+1, 10)*q)
    
    E0 = np.amax(Ev)
    ax3 = plt.subplot(223)
    ax3.plot(THETA_vec, 20*np.log10(Ev/E0), THETA_vec, 20*np.log10(Ev_opt/E0))
    ax3.set_xticks(np.arange(-90, 90+1, 10))
    ax3.set_xlabel(r'$\Theta$')
    ax3.set_ylabel(r'$E(\Theta)$')
    ax3.set_ylim(-50,0)
    ax3.grid()
    
    ax4 = plt.subplot(224, projection='polar')
    ax4.set_theta_direction('clockwise')
    ax4.set_theta_zero_location('N')
    ax4.plot(THETA_vec*q, 20*np.log10(Ev/E0), THETA_vec*q, 20*np.log10(Ev_opt/E0))
    ax4.set_thetamin(-90)
    ax4.set_thetamax(90)
    ax4.set_xticks(np.arange(-90, 90+1, 10)*q)
    ax4.set_ylim(-50,0)

start = time.time()
d, d_opt = NUSLA_SLR_opt(10, 40, 0.5)
stop = time.time()
print(stop-start)
THETA0 = 0
plot_results(THETA0, d, d_opt)
    