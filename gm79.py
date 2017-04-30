#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 16:33:01 2017

@author: jadelson
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve, minimize
from scipy.special import ker, kei

eps = 1e-16
pi = np.pi
square = np.square
sqrt = np.sqrt
cos = np.cos
sin = np.sin
power = np.power
kappa = 0.41


def gm79(ucr, zr, phi_bar, ub_mag, omega, kb=0.12, rho=1000):
    """Solve grant-madsen 1979 for a flow with known wave conditions and 
    current at a known depth.
    
    :param ucr: current velocity at height zr above the bed
    :param zr: height above bed
    :param date: angle between the current and waves
    :param ub_mag: magnitude of bottom wave orbital velocity from potential theory
    :param omega: wave frequency
    :param kb: Nikuradse roughness
    :param rho: density
    
    :return : a dictionary of grant-madsen outputs
    """
    return  gm_worker(1, ucr, phi_bar, ub_mag, omega, kb, zr, rho)

def gm79_depth_averaged(ubar, H, phi_bar, ub_mag, omega, kb=0.12, rho=1000):
    """Solve grant-madsen 1979 for a flow with known wave conditions and 
    depth-averaged current velocity.
    
    :param ubar: depth-averaged velocity
    :param H: total depth
    :param date: angle between the current and waves
    :param ub_mag: magnitude of bottom wave orbital velocity from potential theory
    :param omega: wave frequency
    :param kb: Nikuradse roughness
    :param rho: density
    
    :return : a dictionary of grant-madsen outputs
    """
    return gm_worker(2, ubar, phi_bar, ub_mag, omega, kb, H, rho)

def gm79_internal(ua_mag, phi_c, ub_mag, omega, kb=0.12, rho=1000):
    """Solve grant-madsen 1979 for a flow given nondimensional ua/ub and phic
    
    :param ua_mag: ua_mag for ua/ub
    :param phi_c: phic is angle between ua and ub
    :param date: angle between the current and waves
    :param ub_mag: magnitude of bottom wave orbital velocity from potential theory
    :param omega: wave frequency
    :param kb: Nikuradse roughness
    :param rho: density
    
    :return : a dictionary of grant-madsen outputs
    """
    return gm_worker(0, ua_mag, phi_c, ub_mag, omega, kb, zr, rho)

def gm_worker(double_iterate, ucr, phi_init, ub_mag, omega, kb=0.12, zr=1, rho=1000):

    class GM:
        def process(self, ua_mag, phi_c):

            self.phi_c = phi_c
            
            # load functions for gx and gy as defined in equations 9 and 10
            gx = lambda _theta: sin(_theta) + ua_mag/ub_mag*cos(self.phi_c) #eqn 9 
            gy = lambda _theta: ua_mag/ub_mag*sin(self.phi_c) #eqn 10
            self.alpha = 1 + square(ua_mag/ub_mag) + 2 * ua_mag/ub_mag*cos(self.phi_c) #eqn 20
            self.Ab = ub_mag/omega
            if ua_mag/ub_mag >= np.abs(1/cos(self.phi_c)):
#                print('Warning: We gotta problem |ua|/|ub| > 1/cos φ. Loop: ', ua_mag/ub_mag)
                self.theta_star = pi/2
            else:
                self.theta_star = np.abs(np.arcsin(ua_mag/ub_mag*cos(self.phi_c))) #eqn 12
            
            # Integrals for equations 11, 14, and 17
            integrand_1 = lambda _theta: sqrt(power(gx(_theta),4) + square(gx(_theta)) * square(gy(_theta)))
            integrand_2 = lambda _theta: sqrt(power(gy(_theta),4) + square(gx(_theta)) * square(gy(_theta)))
            integral_a = quad(integrand_1, -1*self.theta_star, pi+self.theta_star)[0]
            integral_b = quad(integrand_1, pi+self.theta_star, 2*pi-self.theta_star)[0]
            integral_c = quad(integrand_2, 0, 2*pi)[0] 
            integral_d = quad(integrand_1, 0, 2*pi)[0]

            # Compute V2 and the simplfied V2 for large waves
            self.V1 = 1/ (2*pi) *sqrt(square(integral_d) + square(integral_c))
#            print(self.V1, self.alpha)
            self.V1 = self.alpha
            self.V2 = 1/ (2*pi) *sqrt(square(integral_a-integral_b) + square(integral_c)) #eqn 14
            
            self.V2_large_waves = 2/pi*(ua_mag/ub_mag)*sqrt(4-3*square(sin(self.phi_c))) #eqn. 16
        
            
            temp_17_denom = integral_a - integral_b
            if temp_17_denom == 0:
                self.phi_bar = 0
            else:    
                self.phi_bar = np.arctan((integral_c)/(temp_17_denom)) #eqn. 17    
            #This is the workhorse that solves for fcw by solving equation 54 
            def get_fcw(_fcw):                
                ucw_star_mag = sqrt(1/2*_fcw*self.V1)*ub_mag#eqn 50
                l = kappa*ucw_star_mag/omega #eqn 29
                zeta_0 = kb/30/l #eqn 31
        
                ztemp = 2*sqrt(zeta_0) #eqn 55
                K = 1/ztemp*1/sqrt(square(ker(ztemp)) + square(kei(ztemp)))#eqn 55
            
                temp_54_a = 0.0971*sqrt(kb/self.Ab)*K/power(_fcw,3.0/4.0) #eqn 54
#                temp_54_b = self.V2/2.0/power(self.alpha,1.0/4.0) #eqn 54
                temp_54_b = self.V2*power(self.V1,1.0/4.0)/2.0/sqrt(self.alpha) #eqn 54 THESIS!
                
                rhs_54 = square(temp_54_a) + 2.0*temp_54_a*temp_54_b*cos(self.phi_bar) #eqn 54
#                lhs_54 = power(self.alpha, 3.0/4.0)/4.0 - square(temp_54_b) #eqn 54
                lhs_54 = self.alpha*sqrt(self.V1)/4.0 - square(temp_54_b) #eqn 54
                self.uw_star_mag = sqrt(kappa*ucw_star_mag*ub_mag*sqrt(zeta_0)*K)
                
                return lhs_54 - rhs_54
            # Solve for fcw
            self.fcw = fsolve(get_fcw,1e-6)[0]
            
#            self.fcw = 1
            self.uc_star_mag = sqrt(1/2*self.fcw*self.V2)*ub_mag #eqn 15
            self.ucw_star_mag = sqrt(1/2*self.fcw*self.alpha)*ub_mag #eqn 21
            self.uw_star_mag = self.uw_star_mag[0]
#            self.uc_star_mag = sqrt(1/2*gm.fcw*gm.V2)*ub_mag #eqn 15

            self.beta = 1 - self.uc_star_mag/self.ucw_star_mag #eqn 49
            self.kbc = kb*power(24*self.ucw_star_mag/ub_mag*self.Ab/kb,self.beta) #eqn 49
#            self.kbc_guess = np.exp(np.log(30*zr/kb)-kappa*ua_mag/self.uc_star_mag_guess) #

            #return stuff
        
    if double_iterate == 0:
        ua_mag = ucr
        phi_c = phi_init
    elif double_iterate == 1 or double_iterate == 2:
        # when uc is specified at a specific depth
        if double_iterate == 1:
            def min_gm(x):
                gm = GM()
                gm.process(x[0], x[1])        
                return 10000*np.square(ucr * kappa / np.log(zr/(gm.kbc/30)) - gm.uc_star_mag) + square(square((gm.phi_bar - phi_init)/(pi/2)))
        # when uc is the depth averaged velocity
        else:
            def min_gm(x):
                gm = GM()
                gm.process(x[0], x[1])          
                return 10000*np.square(ucr * kappa /( zr*(np.log(zr/(gm.kbc/30)) - 1)) - gm.uc_star_mag) + square(square((gm.phi_bar - phi_init)/(pi/2)))
        cons = (
            {'type': 'ineq',
              'fun' : lambda x: np.array([x[0]]),
              'jac' : lambda x: np.array([1.0, 0.0])},
             {'type': 'ineq',
              'fun' : lambda x: np.array([x[1]]),
              'jac' : lambda x: np.array([0.0, 1.0])},
             {'type': 'ineq',
              'fun' : lambda x: np.array(pi/2 - x[1]),
              'jac' : lambda x: np.array([0.0, -1.0])}
             )
        uc_star_mag = ucr * kappa / np.log(zr/(kb/30))
        result = minimize(min_gm, [uc_star_mag, phi_init], constraints=cons)
        ua_mag = result.x[0]
        phi_c = result.x[1]
    else:
        print("Error, asking for an illegal operation mode of the gm_worker function.")
#    
    # Compute terms of interest
    gm = GM()
    
    gm.process(ua_mag, phi_c)
#    tau_c = 1/2 * rho *gm.fcw  *gm.V2*square(ub_mag) #eqn 11
#    uc_star_mag = sqrt(1/2*gm.fcw*gm.V2)*ub_mag #eqn 15
    tau_bmax = 1/2 * rho *gm.fcw * gm.alpha*square(ub_mag) #eqn 19
#    ucw_star_mag = sqrt(1/2*gm.fcw*gm.alpha)*ub_mag #eqn 21
#    l = kappa*ucw_star_mag/omega #eqn 29
#    zeta_0 = kb/30/l #eqn 31


    data = {
        'phi_c': gm.phi_c,
        'phi_bar': gm.phi_bar,
        'V2': gm.V2,
        'V1': gm.V1,
        'alpha': gm.alpha,
        'V2_large_waves': gm.V2_large_waves,
        'ua': ua_mag,
        'ub': ub_mag,
        'fcw': gm.fcw,
#        'tau_c': tau_c,
        'uc_star': gm.uc_star_mag,
        'ucw_star': gm.ucw_star_mag,
        'tau_b': tau_bmax,
        'uw_star': gm.uw_star_mag,
#        'l': l,
#        'zeta_0': zeta_0,
        'kbc': gm.kbc,
        'kb': kb,
        'Ab': gm.Ab,
        'theta_star': gm.theta_star
    }
    
    return data
    
   

#This main file recreates a bunch of figures from the paper

if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)


    KBAB = np.array([2e-4, 4e-4, 6e-4, 8e-4])
    for j in range(0,3):
        KBAB = np.append(KBAB, KBAB*10)
    KBAB = np.append(KBAB,1)

    
    #debug figure setup
    UAUB = np.arange(.01,1.01,.05)
    KBAB = np.array([2e-4, 4e-4, 8e-4, 2e-3, 4e-3, 8e-3, 2e-2, 4e-2, 6e-2, 2e-1, 4e-1, 6e-1, 1])
    THETA = [0]#np.arange(0,pi/2,.05)
    N = len(THETA)
    M = len(UAUB)
    L = len(KBAB)

#    #debug figure 2 setup
#    UAUB = np.arange(.1,1.01,.1)
#    KBAB = np.array([1])
#    THETA = np.arange(0,pi/2,.05)
#    N = len(THETA)
#    M = len(UAUB)
#    L = len(KBAB)
    
#    #figure 1 setup
#    UAUB = np.arange(.1,1.01,.1)
#    KBAB = [1]
#    THETA = np.arange(0,pi/2,.2)
#    N = len(THETA)
    
#    #figure 4 setup
#    UAUB = np.arange(.001,1.61,.1)
#    KBAB = np.array([2e-4, 4e-4, 8e-4, 2e-3, 4e-3, 8e-3, 2e-2, 4e-2, 6e-2, 2e-1, 4e-1, 6e-1, 1])
#    THETA = np.array([0+.0001])
#    M = len(UAUB)
#    N = len(THETA)
#    L = len(KBAB)
    
#    #table 1 setup
#    KBAB = [0.2, 0.0002]
#    UAUB = [0.025, 0.1, 0.6, 1.0, 1.2]
#    THETA = [pi/2]
#    N = 1


    UC = np.ones((N,M))#np.random.randn(N,M,L)#*sqrt(2)/2
    VC = np.ones((N,M))#np.random.randn(N,M,L)#*sqrt(2)/2
    fcw = np.zeros((N,M,L))
    phi_c = np.zeros((N,M,L))
    phi_bar = np.zeros((N,M,L))
    theta_star = np.zeros((N,M,L))
    V1 = np.zeros((N,M,L))
    V2 = np.zeros((N,M,L))
    V2_large = np.zeros((N,M,L))        
    alpha = np.zeros((N,M,L))
    kbc = np.zeros((N,M,L))
    newuaub = np.zeros((N,M,L))
    Ab = np.zeros((N,M,L))
    l = 0
    for kbAb in KBAB:
        
        j = 0                

        UW = UC*np.outer(cos(THETA),1/UAUB) - VC*np.outer(sin(THETA),1/UAUB)
        VW = UC*np.outer(sin(THETA),1/UAUB) + VC*np.outer(cos(THETA),1/UAUB)
        ua = sqrt(square(UC)+square(VC))
        ub = sqrt(square(UW)+square(VW))
        
        kb = 0.2
        omega = ub/(kb/kbAb)

        z0 = kb/30
        kappa = 0.4
        ustar_c = ua
#        ustar_c = ua*kappa/H/(np.log(H/z0) - 1)
#        print(ua - H*ustar_c*(np.log(H/z0) - 1)/kappa)
        zr = 1
        umine = ustar_c/kappa*np.log(zr/z0)
   
        for uaub in UAUB: 
            i = 0
            for theta in THETA:
                H = 3
                gm = gm79_internal(ua[i,j], theta, ub[i,j], omega[i,j], kb)
                V1[i,j,l] = gm['V1']
                V2[i,j,l] = gm['V2']
                V2_large[i,j,l] = gm['V2_large_waves']
                fcw[i,j,l] = gm['fcw']
                phi_c[i,j,l] = gm['phi_c']
                phi_bar[i,j,l] = gm['phi_bar']
                alpha[i,j,l] = gm['alpha']
                theta_star[i,j,l] = gm['theta_star']
                kbc[i,j,l] = gm['kbc']
                newuaub[i,j,l] =  gm['ua']/gm['ub']
                Ab[i,j,l] = gm['Ab']
#                #table 1 printout
#                print(uaub, gm['kb']/gm['Ab'] ,gm['ua']/gm['ub'], gm['kbc']/gm['kb'])

#ITERATOR                
                i = i + 1
#            #%% figure 1
#            line = ax.plot(UAB, V2, lw=.5)
#            ax.set_xlim([0,pi/2])
            
#            #%% figure 2
#            line = ax.plot(phi_c[:,j,l], phi_bar[:,j,l], lw=.5)
#            ax.set_xlim([0,pi/2])
#            ax.set_ylim([0,pi/2])
#            ax.sex_xlabel('phi_c')
#            ax.sex_ylabel('phi_bar')


#ITERATOR
            j = j + 1

        #figure 4
        line = ax.plot(newuaub[0,:,l], fcw[0,:,l], lw=.5)
        ax.set_yscale('log')
#        ax.set_xlim([0, 1])
#        ax.set_ylim([.001, 1]) 
        ax.set_xlabel('|ua|/|ub|')
        ax.set_ylabel('kbc/kb')
 
       
#ITERATOR
        l = l + 1
#    ax.set_yscale('log')
#    ax.set_xlim([0, 2]) 


    
