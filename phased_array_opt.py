# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 20:18:45 2018

@author: Igor
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

def phased_array_opt(N=8, THETA0=0, THETA_P=18):
    THETA_G = THETA0
    gamma = 0.5
    alpha = 2 * np.pi * gamma
    q = np.pi / 180
    
    beta = lambda THETA: alpha * (np.sin(q*THETA) - np.sin(q*THETA0))
    
    # unoptimized radiation pattern
    THETA_vec = np.arange(-90, 90, 0.01)
    beta_vec = (np.vectorize(beta))(THETA_vec)
    E_vec = np.abs(np.sin(0.5 * N * beta_vec) / np.sin(0.5 * beta_vec))
    
    # target function to be optimized
    def target_func(AB, THETA_G=0, THETA_P=18, sign=1.0):
        A1, A2, B2, B3 = AB
        ECG = A1 + A2 * np.cos(beta(THETA_G)) + \
              np.sum(np.cos(np.arange(2, N, 1)*beta(THETA_G)))
        ESG = B2*np.sin(beta(THETA_G)) + B3*np.sin(2*beta(THETA_G)) + \
              np.sum(np.sin(np.arange(3, N, 1)*beta(THETA_G)))
        EMG = np.sqrt(ECG**2 + ESG**2)
        ECP = A1 + A2 * np.cos(beta(THETA_P)) + \
              np.sum(np.cos(np.arange(2, N, 1)*beta(THETA_P)))
        ESP = B2*np.sin(beta(THETA_P)) + B3*np.sin(2*beta(THETA_P)) + \
              np.sum(np.sin(np.arange(3, N, 1)*beta(THETA_P)))
        EMP = np.sqrt(ECP**2 + ESP**2)
        GP = np.abs(EMG) / np.abs(EMP)
        return sign*GP
    
    # optimization
    res = optimize.minimize(target_func, (0.5, 0.5, 0.5, 0.5), \
                            args=(THETA_G,THETA_P,-1.0), \
                            bounds=((0,1),(0,1),(0,1),(0,1)))
    # optimization results
    GP = -res.fun
    GPD = 10*np.log10(GP)
    A1, A2, B2, B3 = res.x
    A = np.ones(N)
    B = np.ones(N)
    A[0] = A1
    A[1] = A2
    B[1] = B2
    B[2] = B3
    
    # optimized radiation pattern
    def EM(THETA):
        EC = np.sum(A*np.cos(np.arange(0, N, 1)*beta(THETA)))
        ES = np.sum(B*np.sin(np.arange(0, N, 1)*beta(THETA)))
        return np.sqrt(EC**2 + ES**2)
    EM_vec = np.vectorize(EM)(THETA_vec)
    
    R = EM(THETA_G) / EM(THETA_P)
    RD = 10*np.log10(R)
    
    # display results
    print('GP = {:.3f}\nGPD = {:.3f} дБ'.format(GP, GPD))
    print('R = {:.3f}\nRD = {:.3f} дБ'.format(R, RD))
    
    ymax = np.max(E_vec) * 1.1
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(THETA_vec, E_vec)
    ax1.set_xticks(np.arange(-90, 90+1, 10))
    ax1.set_ylim(0, ymax)
    ax1.set_xlabel(r'$\Theta$')
    ax1.set_ylabel(r'$E(\Theta)$')
    ax2.plot(THETA_vec, EM_vec)
    ax2.set_xticks(np.arange(-90, 90+1, 10))
    ax2.set_ylim(0, ymax)
    ax2.set_xlabel(r'$\Theta$')
    ax2.set_ylabel(r'$EM(\Theta)$')

phased_array_opt(8)