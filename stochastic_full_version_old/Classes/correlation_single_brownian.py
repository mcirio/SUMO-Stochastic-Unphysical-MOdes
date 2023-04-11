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

class correlation_single_brownian():
    def __init__(self,w0,gamma,ll,t_list,beta,n_cut_mats,n_cut_noise_spectral,n_cut_noise):
        self.t_list = t_list
        self.beta = beta
        self.n_cut_mats = n_cut_mats
        self.n_cut_noise_spectral = n_cut_noise_spectral
        self.n_cut_noise = n_cut_noise

        if np.mod(len(t_list),2) == 0: raise Exception('t_list should be symmetric around the initial time')

        t0_index = int((len(t_list)-1)/2.)
        t0 = t_list[t0_index]
        T = t_list[-1]

        eps = 10**-15
        Gamma = gamma / 2.
        Omega_squared = w0**2 - Gamma**2
        regime = 'critical'
        Omega = 0
        if Omega_squared > eps:
            regime = 'underdamped'
            Omega = np.sqrt(Omega_squared)
        if Omega_squared < eps:
            regime = 'overdamped'
            Omega = 1j * np.sqrt(abs(Omega_squared))
        
        print('regime=', regime)
        self.C_as = [self.compute_antisymm(ll,Omega,Gamma,regime,t) for t in t_list]
        self.C_s = [self.compute_symm(t,beta,Omega,Gamma,w0,gamma,ll,n_cut_mats,regime) for t in t_list]

    def compute_antisymm(self,ll,Omega,Gamma,regime,t):
        if regime == 'underdamped':
            return ll**2 / (4 * Omega) * (np.exp(-1j * Omega*t) - np.exp(1j * Omega*t) ) * np.exp(-Gamma * abs(t))
        if regime == 'overdamped':
            return 1j * ll**2 / (4 * abs(Omega)) * (np.exp(-abs(Omega)*t) - np.exp(abs(Omega)*t) ) * np.exp(-Gamma * abs(t))
        if regime == 'critical':
            return - 1j * ll**2 / 2. * t * np.exp(-Gamma * abs(t))
    def compute_matsubara(self,t,beta,w0,gamma,ll,n_cut_mats):
        mats_freq = [2 * np.pi * k * 1j / beta for k in np.arange(1,n_cut_mats)]
        res = 0
        for w_k in mats_freq:
            res += 2. * 1j / beta * self.J(w_k,w0,gamma,ll) * np.exp(-abs(w_k) * abs(t))
        return res
    def J(self,w:float,w0,gamma,ll)->float:
        """spectral density"""
        return gamma * ll**2 * w / ((w**2-w0**2)**2+gamma**2 * w**2) 
    def compute_symm(self,t,beta,Omega,Gamma,w0,gamma,ll,n_cut_mats,regime):
        if regime == 'underdamped':
            w1 = Omega + 1j * Gamma
            w2 = -Omega + 1j * Gamma
            res = 0
            res += ll**2 / (4 * Omega) * (coth(beta,w1/2.) * np.exp(1j*Omega*abs(t))) * np.exp(-Gamma*abs(t))
            res -= ll**2 / (4 * Omega) * (coth(beta,w2/2.)* np.exp(-1j*Omega*abs(t)) ) * np.exp(-Gamma*abs(t))
        if regime == 'overdamped':
            w1 = Omega + 1j * Gamma
            w2 = -Omega + 1j * Gamma
            res = 0
            res += ll**2 / (4 * Omega) * (coth(beta,w1/2.) * np.exp(1j*Omega*abs(t))) * np.exp(-Gamma*abs(t))
            res -= ll**2 / (4 * Omega) * (coth(beta,w2/2.)* np.exp(-1j*Omega*abs(t)) ) * np.exp(-Gamma*abs(t))
            # Omega = abs(Omega)
            # w1 = Omega + Gamma
            # w2 = -Omega + Gamma
            # res = 0
            # res += -1j * ll**2 / (4 * Omega) * (1/np.tan(beta*w1/2.) * np.exp(-(Omega+Gamma)*abs(t))) 
            # res += 1j * ll**2 / (4 * Omega) * (1/np.tan(beta*w2/2.) * np.exp(-(Gamma-Omega)*abs(t))) 
        if regime == 'critical':
            w1 = Omega + 1j * Gamma
            w2 = -Omega + 1j * Gamma
            res = 0
            res += ll**2 / (4 * Omega) * (coth(beta,w1/2.) * np.exp(1j*Omega*abs(t))) * np.exp(-Gamma*abs(t))
            res -= ll**2 / (4 * Omega) * (coth(beta,w2/2.)* np.exp(-1j*Omega*abs(t)) ) * np.exp(-Gamma*abs(t))
 

        
        mats = self.compute_matsubara(t,beta,w0,gamma,ll,n_cut_mats)
        return res + mats

    


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
        #dict['C_as_2'] = self.C_as_2
        dict['C_s'] = self.C_s
        # dict['C_s_2'] = self.C_s_2
        # dict['coeff_list'] = self.coeff_list
        # dict['C_s_reconstructed'] = self.C_s_reconstructed
        # dict['C_s_stochastic'] = self.C_s_stochastic

        # dict['t_list'] = self.t_list
        # dict['beta'] = self.beta
        # dict['n_cut_mats'] = self.n_cut_mats
        # dict['n_cut_noise_spectral'] = self.n_cut_noise_spectral
        # dict['n_cut_noise'] = self.n_cut_noise
        # dict['w_free'] = self.w_free

        return dict
