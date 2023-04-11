import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'Functions' )
sys.path.append( mymodule_dir )

import numpy as np
from scipy.interpolate import interp1d
from integral import integral
from progressbar import progressbar
from J_poly import J_poly

class matsubara():
    def __init__(self,beta,vec_p,vec_w,N_mats,W_mats,integration_limit,t_corr_list):
        self.beta = beta
        self.vec_p = vec_p
        self.vec_w = vec_w
        self.N_mats = N_mats
        self.t_corr_list = t_corr_list
        self.J = J_poly(vec_p,vec_w)

        #vec_w = self.regularize_poles(vec_w)
        #self.mats = [self.compute_matsubara(t,beta,vec_p,vec_w,N_mats,W_mats,integration_limit) for t in t_corr_list]
        self.mats = self.compute_matsubara_list(t_corr_list,beta,N_mats,W_mats,integration_limit)
        self.mats_interp = interp1d(t_corr_list, self.mats, kind='cubic')
    
    def compute_matsubara(self,t,beta,N_mats,W_mats,integration_limit):
        if beta == 'inf':
            return self.Matsubara_zeroT(vec_p,vec_w,W_mats,t,integration_limit)
        mats_freq = [2 * np.pi * k * 1j / beta for k in np.arange(1,N_mats)]
        res = 0
        for w_k in mats_freq:
            res += 2. * 1j / beta * self.J(w_k) * np.exp(-abs(w_k) * abs(t))
        return res
    def compute_matsubara_list(self,t_list,beta,N_mats,W_mats,integration_limit):
        print("Computing Matsubara correlations ({length})".format(length = "t_corr_list"))
        res = []
        for t_index in progressbar(np.arange(len(t_list))):
            t = t_list[t_index]
            res.append(self.compute_matsubara(t,beta,N_mats,W_mats,integration_limit))
        return res
    def Mats_to_integrate(self,w,t):
        t = abs(t)
        return 1j / np.pi * self.J(1j*w) * np.exp(-w * t)
    def Matsubara_zeroT(self,W_mats,t,integration_limit):
        return integral(self.Mats_to_integrate,t, x_i=0, x_f=W_mats,limit=integration_limit)
    def save(self,dict):
        dict['beta'] = self.beta
        dict['vec_p'] = self.vec_p
        dict['vec_w'] = self.vec_w
        dict['N_mats'] = self.N_mats
        dict['t_corr_list'] = self.t_corr_list

        dict['mats'] = self.mats
        return dict


class matsubara_2():
    def __init__(self,beta,Omega,gamma,ll,N_mats,W_mats,integration_limit,t_corr_list):
        #self.mats = [self.compute_matsubara_2(beta,Omega,gamma,ll,t,N_mats,W_mats) for t in t_corr_list]
        self.beta = beta
        self.Omega = Omega
        self.gamma = gamma
        self.ll = ll
        self.N_mats = N_mats
        self.t_corr_list = t_corr_list

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
            res += wk * np.exp(-wk*np.abs(t)) / (((Omega+1j*Gamma)**2+wk**2) * ((Omega-1j*Gamma)**2+wk**2))
            k += 1
        return - 2 * ll**2 * gamma / beta * res
    def compute_matsubara_2_list(self,beta,Omega,gamma,ll,t_list,N_mats,W_mats,integration_limit):
        print("Computing Matsubara correlations ({length})".format(length = "t_corr_list"))
        res = []
        for t_index in progressbar(np.arange(len(t_list))):
            t = t_list[t_index]
            res.append(self.compute_matsubara_2(beta,Omega,gamma,ll,t,N_mats,W_mats,integration_limit))
        return res
    def Mats_to_integrate(self,w,Omega,gamma,ll,t):
        t = abs(t)
        Gamma = gamma / 2.
        a = Omega + 1j * Gamma
        aa = Omega - 1j * Gamma
        return - ll**2 * gamma / np.pi * w * np.exp(-w * t) / ((a**2 + w**2) * (aa**2 + w**2))
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