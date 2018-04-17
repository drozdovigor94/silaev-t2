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

q = np.pi / 180

def NUSLA_SLR_opt(N=10, THETA0=0, BW=20, FN_tolerance = 0.1):
    
    def E_nusla(THETA, d):
        return np.abs(np.sum(np.exp(1j*2*np.pi*d*(np.sin(q*THETA) - np.sin(q*THETA0)))))
    
    '''N = 10
    THETA0 = 0
    BW = 20'''
    THETA_vec = np.arange(-90, 90, 0.01)
    b = 1/(N*np.sin(q*(BW/2)))
    
    d = np.arange(N)*b
    #BW = np.abs(np.arccos(-1/(N*b)) - np.arccos(1/(N*b)))/q
    FNL = THETA0 - BW/2
    FNR = THETA0 + BW/2
    
    theta_SLL = np.append(np.arange(-90, FNL, 0.5), np.arange(FNR, 90, 0.5))
    
    def errfun(dd):
        d_opt = d + dd
        E_SLL = np.vectorize(lambda THETA:E_nusla(THETA, d_opt))(theta_SLL)
        errSLL = np.max(E_SLL)
        # adjust number after less sign to change priority of BW over SLL
        if E_nusla(FNR, d_opt) < FN_tolerance:
            errBW = 0
        else:
            errBW = 20
        return errSLL + errBW
    
    xopt, fopt = pso(errfun, [-0.2]*N, [0.2]*N, swarmsize=200, maxiter=300)
    
    E0 = E_nusla(THETA0, d)
    Ev = np.vectorize(lambda THETA:E_nusla(THETA, d))(THETA_vec)
    Ev_opt = np.vectorize(lambda THETA:E_nusla(THETA, d+xopt))(THETA_vec)
    return (THETA_vec, Ev, Ev_opt)

THETA_vec, Ev, Ev_opt = NUSLA_SLR_opt(20, 0, 50, 0.2)

def plot_results(THETA_vec, Ev, Ev_opt):
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
    
plot_results(THETA_vec, Ev, Ev_opt)
    