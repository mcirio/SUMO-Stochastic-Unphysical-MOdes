import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..','Functions' )
sys.path.append( mymodule_dir )

import numpy as np
from scipy.interpolate import interp1d
from integral import integral
from progressbar import progressbar

class matsubara_2():
    def __init__(self,beta,Omega,gamma,ll,N_mats,W_mats,integration_limit,t_corr_list):
        #self.mats = [self.compute_matsubara_2(beta,Omega,gamma,ll,t,N_mats,W_mats) for t in t_corr_list]
        self.beta = beta
        self.Omega = Omega
        self.gamma = gamma
        self.ll = ll
        self.N_mats = N_mats
        self.t_corr_list = t_corr_list

        Gamma = gamma / 2.
        self.a = (Omega + 1j * Gamma)**2
        self.aa = (Omega - 1j * Gamma)**2
        self.cof = - ll**2 * gamma / np.pi
        self.cof2 = - 2 * ll**2 * gamma

        self.mats = self.compute_matsubara_2_list(beta,Omega,gamma,ll,t_corr_list,N_mats,W_mats,integration_limit)
        self.mats_interp = interp1d(t_corr_list, self.mats, kind='cubic')
    def compute_matsubara_2(self,beta,Omega,gamma,ll,t,N_mats,W_mats,integration_limit):
        if beta == 'inf':
            return self.Matsubara_zeroT(Omega,gamma,ll,t,W_mats,integration_limit)
        Gamma = gamma / 2.
        res = 0
        k = 1
        while k < N_mats:
            wk = 2 * np.pi * k / beta
            res += wk * np.exp(-wk*np.abs(t)) / ((self.a+wk**2) * (self.aa+wk**2))
            k += 1
        return cof2 / beta * res
    def compute_matsubara_2_list(self,beta,Omega,gamma,ll,t_list,N_mats,W_mats,integration_limit):
        print("Computing Matsubara correlations ({length})".format(length = "t_corr_list"))
        res = []
        for t_index in progressbar(np.arange(len(t_list))):
            t = t_list[t_index]
            res.append(self.compute_matsubara_2(beta,Omega,gamma,ll,t,N_mats,W_mats,integration_limit))
        return res
    def Mats_to_integrate(self,w,Omega,gamma,ll,t):
        t = abs(t)
        return self.cof * w * np.exp(-w * t) / ((self.a + w**2) * (self.aa + w**2))
    def Matsubara_zeroT(self,Omega,gamma,ll,t,W_mats,integration_limit):
        return integral(self.Mats_to_integrate,Omega,gamma,ll,t, x_i=0, x_f=W_mats,limit=integration_limit)
    def save(self,dict):
        dict['beta'] = self.beta
        dict['Omega'] = self.Omega
        dict['ll'] = self.ll
        dict['gamma'] = self.gamma
        dict['N_mats'] = self.N_mats
        dict['t_corr_list'] = self.t_corr_list

        dict['mats_2'] = self.mats
        return dict