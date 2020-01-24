#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monday, 02 Dec 2019
@author: moreira
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

kwargs = {'origin':'lower', 'interpolation': 'sinc'}

dL = 0.5  # lattice parameter for discretization
# five point stencil - https://en.wikipedia.org/wiki/Five-point_stencil
stencil = (1.0 / (12.0 * dL * dL)) * np.array(
    [[0, 0, -1, 0, 0],
     [0, 0, 16, 0, 0],
     [-1, 16, -60, 16, -1],
     [0, 0, 16, 0, 0],
     [0, 0, -1, 0, 0]])

def smoothing(phi, tstep, stencil):
    nstep = 0
    V_target = phi.sum()
    dt = 0.001
    while (nstep <= tstep):  # this loop is integrating the Allen-Cahn function in time
        v = phi.sum()
        Lagrange = np.ones(phi.shape) * (V_target - v)
        phi = phi + dt * (convolve(phi, stencil, mode='constant') + phi * (1.0 - phi) * (
                phi - 0.5 + Lagrange))
        nstep += 1

    return phi

def container(field_1, field_2, alpha):
    """

    :param field_1: chemical field 1
    :param field_2: chemical field 2
    :return: repulsion between field_1 and field_2
    """
    mu = np.power(1. - field_2, 2) + alpha
    return field_1*mu* np.power(mu*(np.power(field_1, 2) + alpha), -0.5)


def maxwell_model(v, eta, dt, tstep, m, chi, gamma):
    dL = 0.5
    stencil_X = (1.0 / (6.0 * dL * dL)) * np.array(
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [-1, 16, -60, 16, -1],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]])

    stencil_Y = (1.0 / (6.0 * dL * dL)) * np.array(
        [[0, 0, -1, 0, 0],
         [0, 0, 16, 0, 0],
         [0, 0, -60, 0, 0],
         [0, 0, 16, 0, 0],
         [0, 0, -1, 0, 0]])

    stencil_mixed = (1.0 / (4.0 * dL * dL)) * np.array(
        [[-1, 0, 1],
         [0, 0, 0],
         [0, 0, 0],
         [1, 0, -1]])

    m = m/(1. + m)

    for nstep in range(0, tstep):

        v[0], v[1] = v[0] + dt * (-gamma*v[0] + (2. + eta) * convolve(v[0], stencil_X)
                                  + convolve(v[0], stencil_Y)
                                  + convolve(v[1], stencil_mixed)
                                  + chi * np.gradient(m, axis=0)), \
                     v[1] + dt * (-gamma*v[1] + (2. + eta) * convolve(v[1], stencil_Y)
                                  + convolve(v[1], stencil_X)
                                  + convolve(v[0], stencil_mixed)
                                  + chi * np.gradient(m, axis=1))

        return v


def init_fibronectin(gamma, value, L):
    for i in range(L[0]):
        for j in range(L[1]):
            if i > L[0]/2-5 and i < L[0]/2+5:
                gamma[i, j] = value
    return gamma

def integrate(tstep, tprint):


    bc = 'wrap'
    L = np.array([70, 70]) # default - 70, 70 - can go to 35
    myosin = np.zeros(L)  # myosins
    phi = np.zeros(L)
    chi_phi = 10.0
    alpha_m = 1.0
    chi_m = 50
    gamma = 0.0
    D_m = 0.5

    dt = 0.0005  # time interval
    nprint = 0  # counter to output data

    R = 20.  # cell radius - standard 20 - can go to 10
    V_target = np.pi * R ** 2  # volume target lagrange multiplier



    i_pos = np.array([12, 0]) # 12, 0


    # initializes the field phi with a circle
    for i in range(0, phi.size):
        s = np.asarray(np.unravel_index(i, phi.shape))
        if np.sqrt(np.sum((s - L/2) * (s - L/2))) <= R:
            phi.itemset(i, 1.0)
        if np.sqrt(np.sum((s - L / 2 - i_pos) * (s - L / 2 - i_pos))) <= R/8:
            myosin.itemset(i, 100.0)

    v = [np.random.rand(L[0], L[1]), np.random.rand(L[0], L[1])]
    phi = smoothing(phi, 1000, stencil)

    nstep = 0
    while nstep<=100:
        myosin = myosin + dt *  convolve(10. * myosin+ container(myosin, phi, 0.001), stencil, mode=bc)
        nstep+=1
    v = maxwell_model(v, -2 / 3, 0.01, 1000, myosin, chi_m, gamma)
    plt.imshow(myosin, alpha=0.8, **kwargs)
    plt.colorbar()
    plt.quiver(v[0], v[1], scale=10)
    plt.imshow(phi , **kwargs, alpha=0.5)
    plt.show()

    nstep = 0
    # main loop of integration
    while (nstep <= tstep):  # this loop is integrating the Allen-Cahn equation in time using the Euler's Method

        volume = phi.sum()  # calculating the cell volume
        Lagrange = np.ones(L)*(V_target-volume)  # lagrange multiplier for volume conservation
        grad_phi = np.gradient(phi)

        v = maxwell_model(v, -2/3, 0.01, 1, myosin, chi_m, gamma)
        # grad_myo = np.gradient(myosin)

        m_v = np.gradient(myosin * v[0], axis=0) + np.gradient(myosin * v[1], axis=1)

        v_phi = v[0]*grad_phi[0] + v[1]*grad_phi[1]

        phi, myosin = phi + dt * (- chi_phi * v_phi + 0.5 * convolve(phi, stencil, mode=bc)
                            + phi * (1.0 - phi) * (phi - 0.5 + 0.1*Lagrange)),\
                            myosin + dt * (-alpha_m * m_v
                            + convolve(D_m * myosin + container(myosin, phi, 0.001), stencil, mode=bc))



        nstep += 1
        nprint += 1
        if(nprint>=tprint):

            nprint=0

            print(nstep)

            plt.imshow(myosin, alpha=0.8, **kwargs)
            plt.colorbar()
            plt.quiver(v[0], v[1], scale=5)
            plt.imshow(phi, **kwargs, alpha=0.5)
            plt.show()
    return phi, myosin, v




if __name__ == '__main__':
    phi, myosin, v = integrate(200000, 10000)

