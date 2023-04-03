import os
import sys

script_dir = os.path.dirname( __file__ )
#mymodule_dir = os.path.join( script_dir, '..', 'Classes' )
#sys.path.append( mymodule_dir )
mymodule_dir = os.path.join( script_dir, '..', 'Functions' )
sys.path.append( mymodule_dir )

from utility_functions import coth, sg, theta
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from functools import partial
import cmath
from scipy.interpolate import interp1d
import pickle

import numpy as np

class correlation_numerical():
    
    def __init__(self,J,t_list,beta,w_cut,n_cut_noise_spectral,n_cut_noise):
        self.t_list = t_list
        self.beta = beta
        self.w_cut = w_cut
        self.n_cut_noise_spectral = n_cut_noise_spectral
        self.n_cut_noise = n_cut_noise

        if np.mod(len(t_list),2) == 0: raise Exception('t_list should be symmetric around the initial time')

        self.J = J
        self.t_list = t_list
        self.beta = beta
        self.w_cut = w_cut
        T = t_list[-1]
        t0_index = int((len(t_list)-1)/2.)
        t0 = t_list[t0_index]

        self.C_as, self.C_s, self.C = self.compute_numerical_correlations(J,t_list,beta,w_cut)
        C_s_interp = interp1d(t_list, self.C_s, kind='cubic')
        self.coeff_list = self.compute_coefficients_basis(C_s_interp,T,n_cut_noise_spectral)
        self.C_s_reconstructed = [self.reconstructed_corr(t,self.coeff_list,n_cut_noise_spectral,T) for t in t_list]
        self.C_s_stochastic = self.average_xi(self.coeff_list,T,n_cut_noise_spectral,n_cut_noise,t0,t_list)
        self.error_symm = [np.sqrt((self.C_s[t0_index]**2 + abs(x)**2)/n_cut_noise) for x in self.C_s]


    def compute_numerical_correlations(self,J,t_list,beta,w_cut):
        eps = 10**-15
        num_real_antisymm = [integrate.quad(self.func_real_antisymm,eps,w_cut,args=(J,t))[0] for t in t_list]
        num_imag_antisymm = [integrate.quad(self.func_imag_antisymm,eps,w_cut,args=(J,t))[0] for t in t_list]
        num_real_symm = [integrate.quad(self.func_real_symm,eps,w_cut,args=(J,t,beta))[0] for t in t_list]
        num_imag_symm = [integrate.quad(self.func_imag_symm,eps,w_cut,args=(J,t,beta))[0] for t in t_list]

        C_as = [x[0] + 1j * x[1] for x in zip(num_real_antisymm,num_imag_antisymm)]
        C_s = [x[0] + 1j * x[1] for x in zip(num_real_symm,num_imag_symm)]
        C =  [x[0] + x[1] for x in zip(C_s,C_as)]

        return C_as, C_s, C

    def func_real_antisymm(self,w,J,t):
        res = -1j / (np.pi) * J(w) * np.sin(w*t)
        return np.real(res)
    def func_imag_antisymm(self,w,J,t):
        res = -1j / (np.pi) * J(w) * np.sin(w*t)
        return np.imag(res)

    def func_real_symm(self,w,J,t,beta):
        res = 1 / (np.pi) * J(w) * coth(beta,w/2.) * np.cos(w*t)
        return np.real(res)
    def func_imag_symm(self,w,J,t,beta):
        res = 1 / (np.pi) * J(w) * coth(beta,w/2.) * np.cos(w*t)
        return np.imag(res)

    def inner_product(self,f,g,T):
        def to_integrate(t):
            return np.real(1 / (2*T) * f(t) * g(t))
        return integrate.quad(to_integrate,-T,T,args=())[0] 
    def basis(self,t,T,n):
        return np.cos(n*np.pi*t/T)
    def compute_coefficients_basis(self,C,T,n_cut_noise_spectral):
        # list of coefficients has length $n_cut_noise_spectral + 1$ because it has also the zero term.
        res = []
        for n in np.arange(0,n_cut_noise_spectral+1):
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
        dict['C_s'] = self.C_s
        dict['C'] = self.C
        dict['coeff_list'] = self.coeff_list
        dict['C_s_reconstructed'] = self.C_s_reconstructed
        dict['C_s_stochastic'] = self.C_s_stochastic
        dict['error_symm'] = self.error_symm

        dict['t_list'] = self.t_list
        dict['beta'] = self.beta
        dict['w_cut'] = self.w_cut
        dict['n_cut_noise_spectral'] = self.n_cut_noise_spectral
        dict['n_cut_noise'] = self.n_cut_noise

        return dict