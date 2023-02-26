import os
import sys

script_dir = os.path.dirname( __file__ )
#mymodule_dir = os.path.join( script_dir, '..', 'Classes' )
#sys.path.append( mymodule_dir )
mymodule_dir = os.path.join( script_dir, '..', 'Functions' )
sys.path.append( mymodule_dir )

from utility_functions import coth, sg, theta

import numpy as np
import cmath
from scipy import integrate
from functools import partial

class correlation_analytical():
    def __init__(self,vec_p,vec_w,t_list,beta,n_cut_mats,n_cut_noise_spectral,n_cut_noise,w_free):
        self.t_list = t_list
        self.beta = beta
        self.n_cut_mats = n_cut_mats
        self.n_cut_noise_spectral = n_cut_noise_spectral
        self.n_cut_noise = n_cut_noise
        self.w_free = w_free

        if np.mod(len(t_list),2) == 0: raise Exception('t_list should be symmetric around the initial time')

        t0_index = int((len(t_list)-1)/2.)
        t0 = t_list[t0_index]
        T = t_list[-1]

        self.residues = self.compute_residues(vec_p,vec_w)
        self.C_as = [self.analytical_antisymm_0(t,self.residues,vec_w) for t in t_list]
        self.C_as_2 = [self.analytical_antisymm_3(t,self.residues,vec_w,w_free) for t in t_list]
        self.C_s = [self.analytical_symm(t,beta,self.residues,vec_p,vec_w,n_cut_mats) for t in t_list]
        self.C_s_2 = [self.analytical_symm_4(t,beta,self.residues,vec_p,vec_w,n_cut_mats) for t in t_list]

        self.coeff_list = self.compute_coefficients_basis(beta,self.residues,vec_p,vec_w,n_cut_mats,T,n_cut_noise_spectral)
        self.C_s_reconstructed = [self.reconstructed_corr(t,self.coeff_list,n_cut_noise_spectral,T) for t in t_list]
        self.C_s_stochastic = self.average_xi(self.coeff_list,T,n_cut_noise_spectral,n_cut_noise,t0,t_list)
        self.error_symm = [np.sqrt((self.C_s[t0_index]**2 + abs(x)**2)/n_cut_noise) for x in self.C_s]

        self.C_Q = [self.analytical_antisymm_0(t,self.residues,vec_w) - self.analytical_fs(t,self.residues,vec_w,w_free) for t in t_list]
        self.C_s_plus_fs = [self.analytical_symm(t,beta,self.residues,vec_p,vec_w,n_cut_mats) + self.analytical_fs(t,self.residues,vec_w,w_free) for t in t_list]
        self.C_Q2 = [self.analytical_CQ(t,self.residues,vec_w,w_free) for t in t_list]
        self.C_s_plus_fs2 = [self.analytical_symm_plus_fs_3(t,beta,self.residues,vec_p,vec_w,n_cut_mats,w_free) for t in t_list]
        self.fs = [ self.analytical_fs(t,self.residues,vec_w,w_free) for t in t_list]


        n_list = np.arange(0,50)
        self.c_list = []
        for n in n_list:
            self.c_list.append(self.analytical_cn(beta,self.residues,vec_p,vec_w,n_cut_mats,w_free,n,T))
        self.C_rec = []
        for t in t_list:
            res = 0
            for n,c in enumerate(self.c_list):
                if n == 0:
                    res += c * np.cos(n*np.pi*t/T)
                else:
                    res += 2*c * np.cos(n*np.pi*t/T)
            self.C_rec.append(res)
        self.w_list = [n * np.pi/T for n in n_list]
        #self.parameters_antisymm = self.generate_PM_parameters_antisymm(self.residues,vec_w,w_free)
        #self.test_antisymm = [self.test_generate_PM_parameters_antisymm(t,self.residues,vec_w,w_free) for t in t_list]


    def J(self,w,vec_p,vec_w):
        res = 0
        for k,p_k in enumerate(vec_p):
            res += p_k * w**k
        for w_k in vec_w:
            res = res / (w - w_k)
        return res
    def compute_residues(self,vec_p,vec_w):
        res = []
        for w_k in vec_w:
            num = sum([p_j * w_k**j for j,p_j in enumerate(vec_p)])
            den = np.prod([(w_k - w_j) for w_j in vec_w if w_j is not w_k])
            res.append(num/den)
        return res          
    def analytical_antisymm_0(self,t,residues,vec_w):
        # Eq. (\ref{eq:C_as(t)_before_PM})
        eps = 10**-15
        res = 0
        for r_k,w_k in zip(residues,vec_w):
            if np.imag(w_k) > 0:
                wR_k = np.real(w_k)
                wI_k = np.imag(w_k)
                rR_k = np.real(r_k)
                rI_k = np.imag(r_k)
                if wR_k > 0:
                    temp= 0
                    temp += -1j * sg(t) * rR_k * np.cos(wR_k*t) * np.exp(-wI_k*abs(t))
                    temp += 1j * rI_k * np.sin(wR_k*t) * np.exp(-wI_k*abs(t))
                    res += 2 * temp
                if np.real(w_k) == 0:
                    res += (-1j * sg(t) * rR_k * np.cos(wR_k*t) * np.exp(-wI_k*abs(t)))
                    #res += 1j * rI_k * np.sin(wR_k*t) * np.exp(-wI_k*abs(t))                
        return res
    def analytical_antisymm(self,t,residues,vec_w):
        res = 0
        # Eq. (\ref{eq:C_antisymm_residues})
        # for r_k,w_k in zip(residues,vec_w):
        #    if np.imag(w_k) > 0: 
        #        res += - 1j / 2. * theta(t) * (r_k * np.exp(1j*w_k*t) + np.conj(r_k) * np.exp(-1j*np.conj(w_k)*t))
        #        res += 1j / 2. * theta(-t) * (r_k * np.exp(-1j*w_k*t) + np.conj(r_k) * np.exp(1j*np.conj(w_k)*t))
        
        # Eq. (\ref{eq:C_anti_PM})
        for r_k,w_k in zip(residues,vec_w):
            if np.imag(w_k) > 0:
                wR_k = np.real(w_k)
                wI_k = np.imag(w_k)
                rR_k = np.real(r_k)
                rI_k = np.imag(r_k)
                res += -1j * sg(t) * rR_k * np.cos(wR_k*t) * np.exp(-wI_k*abs(t))
                res += 1j * rI_k * np.sin(wR_k*t) * np.exp(-wI_k*abs(t))
        return res
    def analytical_antisymm_2(self,t,residues,vec_w,w_free):
        # Eq. (\ref{eq:C_as(t)_before_PM})
        W = w_free
        res = 0
        for r_k,w_k in zip(residues,vec_w):
            if np.imag(w_k) > 0:
                wR_k = np.real(w_k)
                wI_k = np.imag(w_k)
                rR_k = np.real(r_k)
                rI_k = np.imag(r_k)
                res += rR_k * np.sin(-1j*(W-wI_k)*t/2.+wR_k*t) * np.exp(-(W+wI_k)*abs(t)/2.)
                res += rR_k * np.sin(-1j*(W-wI_k)*t/2.-wR_k*t) * np.exp(-(W+wI_k)*abs(t)/2.)
                res += 1j * rI_k * np.sin(wR_k*t) * np.exp(-wI_k*abs(t))
                res += -1j * sg(t) * rR_k * np.cos(wR_k*t) * np.exp(-W*abs(t))
        return res
    def analytical_antisymm_3(self,t,residues,vec_w,w_free):
        # Eq. (\ref{eq:C_as(t)_before_PM})
        eps = 10**-15
        W = w_free
        res = 0
        for r_k,w_k in zip(residues,vec_w):
            if np.imag(w_k) > 0:
                wR_k = np.real(w_k)
                wI_k = np.imag(w_k)
                rR_k = np.real(r_k)
                rI_k = np.imag(r_k)
                res += (-1j * sg(t) * rR_k * np.cos(wR_k*t) * np.exp(-W*abs(t)))
                if wR_k > 0:
                    temp= 0
                    temp += rR_k * np.sin(-1j*(W-wI_k)*t/2.+wR_k*t) * np.exp(-(W+wI_k)*abs(t)/2.)
                    temp += rR_k * np.sin(-1j*(W-wI_k)*t/2.-wR_k*t) * np.exp(-(W+wI_k)*abs(t)/2.)
                    temp += 1j * rI_k * np.sin(wR_k*t) * np.exp(-wI_k*abs(t))
                    res += 2 * temp
                if np.real(w_k) == 0:
                    res += 2 * rR_k * np.sin(-1j*(W-wI_k)*t/2.) * np.exp(-(W+wI_k)*abs(t)/2.)
                    #res += 1j * rI_k * np.sin(wR_k*t) * np.exp(-wI_k*abs(t))                
        return res
    def analytical_fs(self,t,residues,vec_w,w_free):
        # Eq. (\ref{eq:C_as(t)_before_PM})
        eps = 10**-15
        W = w_free
        res = 0
        for r_k,w_k in zip(residues,vec_w):
            if np.imag(w_k) > 0:
                wR_k = np.real(w_k)
                wI_k = np.imag(w_k)
                rR_k = np.real(r_k)
                rI_k = np.imag(r_k)
                #res += (-1j * sg(t) * rR_k * np.cos(wR_k*t) * np.exp(-W*abs(t)))
                if wR_k > 0:
                    temp= 0
                    temp += rR_k * np.cos(-1j*(W-wI_k)*t/2.+wR_k*t) * np.exp(-(W+wI_k)*abs(t)/2.)
                    temp += rR_k * np.cos(-1j*(W-wI_k)*t/2.-wR_k*t) * np.exp(-(W+wI_k)*abs(t)/2.)
                    temp += 1j * rI_k * np.cos(wR_k*t) * np.exp(-wI_k*abs(t))
                    res += 2 * temp
                if np.real(w_k) == 0:
                    res += 2 * rR_k * np.cos(-1j*(W-wI_k)*t/2.) * np.exp(-(W+wI_k)*abs(t)/2.)
                    #res += 1j * rI_k * np.sin(wR_k*t) * np.exp(-wI_k*abs(t))                
        return -1j * res
    def analytical_CQ(self,t,residues,vec_w,w_free):
        # Eq. (\ref{eq:C_as(t)_before_PM})
        eps = 10**-15
        W = w_free
        res = 0
        for r_k,w_k in zip(residues,vec_w):
            if np.imag(w_k) > 0:
                wR_k = np.real(w_k)
                wI_k = np.imag(w_k)
                rR_k = np.real(r_k)
                rI_k = np.imag(r_k)
                #res += (-1j * sg(t) * rR_k * np.cos(wR_k*t) * np.exp(-W*abs(t)))
                if wR_k > 0:
                    temp= 0
                    temp += rR_k * np.exp(-1j*(-1j*(W-wI_k)*t/2.+wR_k*t)) * np.exp(-(W+wI_k)*abs(t)/2.)
                    temp += rR_k * np.exp(-1j*(-1j*(W-wI_k)*t/2.-wR_k*t)) * np.exp(-(W+wI_k)*abs(t)/2.)
                    temp += 1j * rI_k * np.exp(-1j*(wR_k*t)) * np.exp(-wI_k*abs(t))
                    res += 2 * temp
                if np.real(w_k) == 0:
                    res += 2 * rR_k * np.exp(-1j*(-1j*(W-wI_k)*t/2.)) * np.exp(-(W+wI_k)*abs(t)/2.)
                    #res += 1j * rI_k * np.sin(wR_k*t) * np.exp(-wI_k*abs(t))                
        return 1j * res
    def generate_PM_parameters_antisymm(self,residues,vec_w,w_free):
        W = w_free
        res = []
        for r_k,w_k in zip(residues,vec_w):
            if np.imag(w_k) > 0:
                wR_k = np.real(w_k)
                wI_k = np.imag(w_k)
                rR_k = np.real(r_k)
                rI_k = np.imag(r_k)
                n = - 1 / 2.
                ll = cmath.sqrt(rR_k / (2 * 1j * abs(n)))
                #ll = (rR_k / (2 * 1j * abs(n)))
                Omega = - (-1j*(W-wI_k)/2.+wR_k)
                Gamma = (W+wI_k)/2.
                res.append([ll,Omega,Gamma,n])
                n = - 1 / 2.
                ll = cmath.sqrt(rR_k / (2 * 1j * abs(n)))
                #ll = (rR_k / (2 * 1j * abs(n)))
                Omega = - (-1j*(W-wI_k)/2.-wR_k)
                Gamma = (W+wI_k)/2.
                res.append([ll,Omega,Gamma,n])
                n = - 1 / 2.
                ll = cmath.sqrt(1j * rI_k / (2 * 1j * abs(n)))
                #ll = (1j * rI_k / (2 * 1j * abs(n)))
                Omega = - (wR_k)
                Gamma = wI_k
                res.append([ll,Omega,Gamma,n])
        return res
    def test_generate_PM_parameters_antisymm(self,t,residues,vec_w,w_free):
        W = w_free
        res = 0
        for [ll,Omega,Gamma,n] in self.parameters_antisymm:
            res += ll**2 * ( (1+n) * np.exp(-1j*Omega*t) + n * np.exp(1j*Omega*t) ) * np.exp(-Gamma*abs(t))
        for r_k,w_k in zip(residues,vec_w):
            if np.imag(w_k) > 0:
                wR_k = np.real(w_k)
                wI_k = np.imag(w_k)
                rR_k = np.real(r_k)
                rI_k = np.imag(r_k)
                res += -1j * sg(t) * rR_k * np.cos(wR_k*t) * np.exp(-w_free*abs(t))
        return res
    def analytical_symm(self,t,beta,residues,vec_p,vec_w,n_cut_mats):
        # Eq. (\ref{eq:C_symm_res_mats})
        res = 0
        for r_k,w_k in zip(residues,vec_w):
            if np.imag(w_k) > 0: 
                res += 1j / 2. * theta(t) * (r_k * coth(beta,w_k/2.) * np.exp(1j*w_k*t) - np.conj(r_k) * coth(beta,np.conj(w_k)/2.) * np.exp(-1j*np.conj(w_k)*t))
                res += 1j / 2. * theta(-t) * (r_k * coth(beta,w_k/2.)* np.exp(-1j*w_k*t) - np.conj(r_k) * coth(beta,np.conj(w_k)/2.) * np.exp(1j*np.conj(w_k)*t))
        mats = self.compute_matsubara(t,beta,residues,vec_p,vec_w,n_cut_mats)
        return res + mats
    def analytical_symm_2(self,t,beta,residues,vec_p,vec_w,n_cut_mats):
        # Eq. (\ref{eq:C_symm_res_mats})
        res = 0
        for r_k,w_k in zip(residues,vec_w):
            rR_k = np.real(r_k * coth(beta,w_k/2.))
            rI_k = np.imag(r_k * coth(beta,w_k/2.))
            wR_k = np.real(w_k)
            wI_k = np.imag(w_k)
            if np.imag(w_k) > 0: 
                res += -  np.exp(-abs(wI_k)*abs(t)) * (rR_k * np.sin(wR_k * abs(t)) + rI_k * np.cos(wR_k * t))
        mats = self.compute_matsubara(t,beta,residues,vec_p,vec_w,n_cut_mats)
        return res + mats
    def analytical_symm_3(self,t,beta,residues,vec_p,vec_w,n_cut_mats):
        # Eq. (\ref{eq:C_symm_res_mats})
        res = 0
        for r_k,w_k in zip(residues,vec_w):
            if np.imag(w_k) > 0:
                rR_k = np.real(r_k * coth(beta,w_k/2.))
                rI_k = np.imag(r_k * coth(beta,w_k/2.))
                wR_k = np.real(w_k)
                wI_k = np.imag(w_k)
                if wR_k > 0:
                    temp= 0
                    temp += -  np.exp(-abs(wI_k)*abs(t)) * (rR_k * np.sin(wR_k * abs(t)) + rI_k * np.cos(wR_k * t))
                    res += 2 * temp
                if np.real(w_k) == 0:
                    res += -  np.exp(-abs(wI_k)*abs(t)) * rI_k 
                    #res += 1j * rI_k * np.sin(wR_k*t) * np.exp(-wI_k*abs(t))  
        mats = self.compute_matsubara(t,beta,residues,vec_p,vec_w,n_cut_mats)
        return res + mats
    def analytical_symm_4(self,t,beta,residues,vec_p,vec_w,n_cut_mats):
        # Eq. (\ref{eq:C_symm_res_mats})
        res = 0
        for r_k,w_k in zip(residues,vec_w):
            if np.imag(w_k) > 0:
                rR_k = np.real(r_k * coth(beta,w_k/2.))
                rI_k = np.imag(r_k * coth(beta,w_k/2.))
                wR_k = np.real(w_k)
                wI_k = np.imag(w_k)
                if wR_k > 0:
                    temp= 0
                    temp += 1j * r_k * coth(beta,w_k/2.) * np.exp(1j*w_k*abs(t)) + np.conj(1j * r_k * coth(beta,w_k/2.) * np.exp(1j*w_k*abs(t)) )
                    res += temp
                if np.real(w_k) == 0:
                    #print(r_k * coth(beta,w_k/2.))
                    #print(r_k)
                    #res += -  np.exp(-abs(wI_k)*abs(t)) * (-1j*r_k * coth(beta,w_k/2.))
                    res += (1j*r_k * coth(beta,w_k/2.)) * np.exp(1j * w_k*abs(t)) 
                    #res += 1j * rI_k * np.sin(wR_k*t) * np.exp(-wI_k*abs(t))  
        mats = self.compute_matsubara(t,beta,residues,vec_p,vec_w,n_cut_mats)
        return res + mats
    def analytical_symm_plus_fs(self,t,beta,residues,vec_p,vec_w,n_cut_mats,w_free):
        # Eq. (\ref{eq:C_symm_res_mats})
        W = w_free
        res = 0
        f = 0
        for r_k,w_k in zip(residues,vec_w):
            if np.imag(w_k) > 0:
                rR_k = np.real(r_k * coth(beta,w_k/2.))
                rI_k = np.imag(r_k * coth(beta,w_k/2.))
                wR_k = np.real(w_k)
                wI_k = np.imag(w_k)
                if wR_k > 0:
                    temp= 0
                    temp += 1j * r_k * coth(beta,w_k/2.) * np.exp(1j*w_k*abs(t)) + np.conj(1j * r_k * coth(beta,w_k/2.) * np.exp(1j*w_k*abs(t)) )
                    res += temp
                if np.real(w_k) == 0:
                    #print(r_k * coth(beta,w_k/2.))
                    #print(r_k)
                    #res += -  np.exp(-abs(wI_k)*abs(t)) * (-1j*r_k * coth(beta,w_k/2.))
                    res += (1j*r_k * coth(beta,w_k/2.)) * np.exp(1j * w_k*abs(t)) 
                    #res += 1j * rI_k * np.sin(wR_k*t) * np.exp(-wI_k*abs(t))  
            if np.imag(w_k) > 0:
                wR_k = np.real(w_k)
                wI_k = np.imag(w_k)
                rR_k = np.real(r_k)
                rI_k = np.imag(r_k)
                #res += (-1j * sg(t) * rR_k * np.cos(wR_k*t) * np.exp(-W*abs(t)))
                if wR_k > 0:
                    temp = 0
                    f += - 1j * rR_k * np.exp((W-wI_k)*abs(t)/2.+1j*wR_k*abs(t)) * np.exp(-(W+wI_k)*abs(t)/2.)
                    f += - 1j * rR_k * np.exp((W-wI_k)*abs(t)/2.-1j*wR_k*abs(t)) * np.exp(-(W+wI_k)*abs(t)/2.)
                    f += rI_k * np.exp(1j*wR_k*abs(t)) * np.exp(-wI_k*abs(t))
                    f += - 1j * rR_k * np.exp(-(W-wI_k)*abs(t)/2.-1j*wR_k*abs(t)) * np.exp(-(W+wI_k)*abs(t)/2.)
                    f += - 1j * rR_k * np.exp(-(W-wI_k)*abs(t)/2.+1j*wR_k*abs(t)) * np.exp(-(W+wI_k)*abs(t)/2.)
                    f += rI_k * np.exp(-1j*wR_k*abs(t)) * np.exp(-wI_k*abs(t))
                if np.real(w_k) == 0:
                    #f += - 1j * 2 * rR_k * np.cos(-1j*(W-wI_k)*t/2.) * np.exp(-(W+wI_k)*abs(t)/2.)
                    f += - 1j * rR_k * (np.exp((W-wI_k)*abs(t)/2.)) * np.exp(-(W+wI_k)*abs(t)/2.)
                    f += - 1j * rR_k * (np.exp(-(W-wI_k)*abs(t)/2.)) * np.exp(-(W+wI_k)*abs(t)/2.)

        mats = self.compute_matsubara(t,beta,residues,vec_p,vec_w,n_cut_mats)
        return res + mats + f
    def analytical_symm_plus_fs_2(self,t,beta,residues,vec_p,vec_w,n_cut_mats,w_free):
        # Eq. (\ref{eq:C_symm_res_mats})
        W = w_free
        res = 0
        f = 0
        for r_k,w_k in zip(residues,vec_w):
            if np.imag(w_k) > 0:
                rR_k = np.real(r_k * coth(beta,w_k/2.))
                rI_k = np.imag(r_k * coth(beta,w_k/2.))
                wR_k = np.real(w_k)
                wI_k = np.imag(w_k)
                if wR_k > 0:
                    temp= 0
                    temp += 1j * r_k * coth(beta,w_k/2.) * np.exp(1j*w_k*abs(t)) + np.conj(1j * r_k * coth(beta,w_k/2.) * np.exp(1j*w_k*abs(t)) )
                    res += temp
                if np.real(w_k) == 0:
                    #print(r_k * coth(beta,w_k/2.))
                    #print(r_k)
                    #res += -  np.exp(-abs(wI_k)*abs(t)) * (-1j*r_k * coth(beta,w_k/2.))
                    res += (1j*r_k * coth(beta,w_k/2.)) * np.exp(1j * w_k*abs(t)) 
                    #res += 1j * rI_k * np.sin(wR_k*t) * np.exp(-wI_k*abs(t))  
            if np.imag(w_k) > 0:
                wR_k = np.real(w_k)
                wI_k = np.imag(w_k)
                rR_k = np.real(r_k)
                rI_k = np.imag(r_k)
                #res += (-1j * sg(t) * rR_k * np.cos(wR_k*t) * np.exp(-W*abs(t)))
                if wR_k > 0:
                    temp = 0
                    f += - 1j * rR_k * np.exp(1j*(w_k) * abs(t))
                    f += - 1j * rR_k * np.exp((-1j*wR_k - W) * abs(t))
                    f += rI_k * np.exp(1j*(w_k) * abs(t)) + rI_k * np.exp(-1j*np.conj(w_k) * abs(t))
                    f += - 1j * rR_k * np.exp(-1j*np.conj(w_k) * abs(t))
                    f += - 1j * rR_k * np.exp((1j*wR_k - W) * abs(t))
                if np.real(w_k) == 0:
                    #f += - 1j * 2 * rR_k * np.cos(-1j*(W-wI_k)*t/2.) * np.exp(-(W+wI_k)*abs(t)/2.)
                    f += - 1j * rR_k * np.exp(-W * abs(t))
                    f += - 1j * rR_k * np.exp(-wI_k*abs(t))

        mats = self.compute_matsubara(t,beta,residues,vec_p,vec_w,n_cut_mats)
        return res + mats + f
    def analytical_symm_plus_fs_3(self,t,beta,residues,vec_p,vec_w,n_cut_mats,w_free):
        # Eq. (\ref{eq:C_symm_res_mats})
        W = w_free
        res = 0
        f = 0
        for r_k,w_k in zip(residues,vec_w):
            if np.imag(w_k) > 0:
                rR_k = np.real(r_k * coth(beta,w_k/2.))
                rI_k = np.imag(r_k * coth(beta,w_k/2.))
                wR_k = np.real(w_k)
                wI_k = np.imag(w_k)
                if wR_k > 0:
                    temp= 0
                    temp += 1j * (r_k * coth(beta,w_k/2.)-r_k) * np.exp(1j*w_k*abs(t)) -1j * (np.conj(r_k * coth(beta,w_k/2.))+r_k) * np.exp(-1j*np.conj(w_k)*abs(t))
                    res += temp
                if np.real(w_k) == 0:
                    #print(r_k * coth(beta,w_k/2.))
                    #print(r_k)
                    #res += -  np.exp(-abs(wI_k)*abs(t)) * (-1j*r_k * coth(beta,w_k/2.))
                    res += 1j*(r_k * coth(beta,w_k/2.)-r_k) * np.exp(1j * w_k*abs(t)) 
                    #res += 1j * rI_k * np.sin(wR_k*t) * np.exp(-wI_k*abs(t))  
            if np.imag(w_k) > 0:
                wR_k = np.real(w_k)
                wI_k = np.imag(w_k)
                rR_k = np.real(r_k)
                rI_k = np.imag(r_k)
                #res += (-1j * sg(t) * rR_k * np.cos(wR_k*t) * np.exp(-W*abs(t)))
                if wR_k > 0:
                    temp = 0
                    #f += - 1j * rR_k * np.exp(1j*(w_k) * abs(t))
                    f += - 0*1j * rR_k * np.exp((-1j*wR_k - W) * abs(t))
                    #f += rI_k * np.exp(1j*(w_k) * abs(t)) + rI_k * np.exp(-1j*np.conj(w_k) * abs(t))
                    #f += - 1j * rR_k * np.exp(-1j*np.conj(w_k) * abs(t))
                    f += - 0*1j * rR_k * np.exp((1j*wR_k - W) * abs(t))
                if np.real(w_k) == 0:
                    #f += - 1j * 2 * rR_k * np.cos(-1j*(W-wI_k)*t/2.) * np.exp(-(W+wI_k)*abs(t)/2.)
                    f += - 0*1j * rR_k * np.exp(-W * abs(t))
                    f += - 0*1j * rR_k * np.exp(-wI_k*abs(t))

        mats = self.compute_matsubara(t,beta,residues,vec_p,vec_w,n_cut_mats)
        return res + mats + f
    def weight(self,freq,n,T):
        return freq * T / (freq**2 * T**2 + n**2 * np.pi**2) * (np.exp(freq*T)*np.exp(1j*n*np.pi) - 1)
    def analytical_cn(self,beta,residues,vec_p,vec_w,n_cut_mats,w_free,n,T):
        # Eq. (\ref{eq:C_symm_res_mats})
        W = w_free
        res = 0
        f = 0
        for r_k,w_k in zip(residues,vec_w):
            if np.imag(w_k) > 0:
                rR_k = np.real(r_k * coth(beta,w_k/2.))
                rI_k = np.imag(r_k * coth(beta,w_k/2.))
                wR_k = np.real(w_k)
                wI_k = np.imag(w_k)
                if wR_k > 0:
                    temp= 0
                    freq = 1j*w_k
                    temp += 1j * (r_k * coth(beta,w_k/2.)-r_k) * self.weight(freq,n,T)
                    freq = -1j*np.conj(w_k)
                    temp += -1j * (np.conj(r_k * coth(beta,w_k/2.))+r_k) * self.weight(freq,n,T)
                    res += temp
                if np.real(w_k) == 0:
                    #print(r_k * coth(beta,w_k/2.))
                    #print(r_k)
                    #res += -  np.exp(-abs(wI_k)*abs(t)) * (-1j*r_k * coth(beta,w_k/2.))
                    freq = 1j * w_k
                    res += 1j*(r_k * coth(beta,w_k/2.)-r_k) * self.weight(freq,n,T) 
                    #res += 1j * rI_k * np.sin(wR_k*t) * np.exp(-wI_k*abs(t))  
   
        mats_freq = [2 * np.pi * k * 1j / beta for k in np.arange(1,n_cut_mats)]
        for w_k in mats_freq:
            freq = -abs(w_k)
            res += 2. * 1j / beta * self.J(w_k,vec_p,vec_w)  * self.weight(freq,n,T) 
        return res
    def compute_matsubara(self,t,beta,residues,vec_p,vec_w,n_cut_mats):
        mats_freq = [2 * np.pi * k * 1j / beta for k in np.arange(1,n_cut_mats)]
        res = 0
        for w_k in mats_freq:
            res += 2. * 1j / beta * self.J(w_k,vec_p,vec_w) * np.exp(-abs(w_k) * abs(t))
        return res

    def inner_product(self,f,g,T):
        def to_integrate(t):
            return np.real(1 / (2*T) * f(t) * g(t))
        return integrate.quad(to_integrate,-T,T,args=())[0] 
    def basis(self,t,T,n):
        return np.cos(n*np.pi*t/T)
    def compute_coefficients_basis(self,beta,residues,vec_p,vec_w,n_cut_mats,T,n_cut_noise_spectral):
        # list of coefficients has length $n_cut_noise_spectral + 1$ because it has also the zero term.
        res = []
        for n in np.arange(0,n_cut_noise_spectral+1):
            C = partial(self.analytical_symm,beta=beta,residues=residues,vec_p=vec_p,vec_w=vec_w,n_cut_mats=n_cut_mats)
            coeff = self.inner_product(partial(self.basis,T=T,n=n), C, T)
            res.append(coeff)
        return res
    def reconstructed_corr(self,t,coeff_list,n_cut_noise_spectral,T):
        # n_cut_noise_spectral has to be such that len(coeff_list) = n_cut_noise_spectral + 1
        res = 0
        for n,coeff in enumerate(coeff_list):
            if n == 0:
                res += coeff * self.basis(t,T,n)
            else:
                res += 2 * coeff * self.basis(t,T,n)
        return res

    def xi_field(self,coeff_list,n_cut_noise_spectral,T):
    # Eq. (\ref{eq:xi_spectral_representation_2})
    # n_cut_noise_spectral has to be such that len(coeff_list) = n_cut_noise_spectral + 1
        mu = 0
        sigma = 1
        xi_list = np.random.normal(mu, sigma, 2*n_cut_noise_spectral+1)
        def f(t):
            res = cmath.sqrt(coeff_list[0]) * xi_list[n_cut_noise_spectral]
            for n in np.arange(1,n_cut_noise_spectral+1):
                res += np.sqrt(2) * cmath.sqrt(coeff_list[n]) * (xi_list[n-1] * np.cos(n*np.pi*t/T) + xi_list[-n] * np.sin(n*np.pi*t/T))
            return res
        return f
    def average_xi(self,coeff_list,T,n_cut_noise_spectral,n_cut_noise,t0,t_list):
        res = 0 * np.array(t_list)
        for n in np.arange(0,n_cut_noise):
            xi = self.xi_field(coeff_list,n_cut_noise_spectral,T)
            res = res + np.array([xi(t0) * xi(t) for t in t_list])
        return res / n_cut_noise
    def save(self,dict):

        dict['C_as'] = self.C_as
        dict['C_as_2'] = self.C_as_2
        dict['C_s'] = self.C_s
        dict['C_s_2'] = self.C_s_2
        dict['coeff_list'] = self.coeff_list
        dict['C_s_reconstructed'] = self.C_s_reconstructed
        dict['C_s_stochastic'] = self.C_s_stochastic

        dict['t_list'] = self.t_list
        dict['beta'] = self.beta
        dict['n_cut_mats'] = self.n_cut_mats
        dict['n_cut_noise_spectral'] = self.n_cut_noise_spectral
        dict['n_cut_noise'] = self.n_cut_noise
        dict['w_free'] = self.w_free

        return dict
