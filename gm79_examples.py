#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:26:40 2017

Example code for how to run the grant madsen solvers

@author: jadelson
"""

from scipy.optimize import minimize
import numpy as np
import crspy.crspy as cr
import gm79
import warnings

warnings.filterwarnings('ignore')

# Parameters
g = 9.81

#wave properties
omega = .4
H = 6
a0 = .4


def dispersion_relation(k):
    return np.abs(omega**2 - g*k*np.tanh(k*H))

if omega*np.sqrt(H/g) < 0.05:
    k = omega/np.sqrt(g*H)
    ub = omega*a0/(k*H)
elif omega*omega/g*H > 50:
    k = omega*omega/g
    ub = 0
else:
    kall = minimize(dispersion_relation,omega/(np.sqrt(g*H)),
                    options={'disp': False})
    k = kall.x[0]
    ub = omega*a0/np.sinh(k*H)
        
 


ucr = 1.3*.1
zr = 1
phiwc = 0
kb = 0.02

   
ustrc, ustrr, ustrwm, dwc, fwc, zoa = cr.m94( ub, omega, ucr, zr, phiwc, kb)
print('uc', ustrc, 'ucw', ustrr, 'uw', ustrwm, 'fwc', fwc, 'kbc', zoa*30)
#print(ucr, phiwc, ub, omega, kb, zr)

ubar = ustrc/.41*H*(np.log(H/zoa)- 1)
gm1 = gm79.gm79_depth_averaged(ubar, H, phiwc, ub, omega, kb)
gm2 = gm79.gm79(ucr, zr, phiwc, ub, omega, kb)
#uc_star_mag = ucr * kappa / np.log(zr/(kb/30))
print('uc', gm1['uc_star'], 'ucw', gm1['uw_star'], 'uw', gm1['ucw_star'], 'fcw', gm1['fcw'], 'kbc', gm1['kbc'])
print('uc', gm2['uc_star'], 'ucw', gm2['uw_star'], 'uw', gm2['ucw_star'], 'fcw', gm2['fcw'], 'kbc', gm2['kbc'])

#uc_star_mag = ucr * kappa / np.log(zr/(kb/30))

