import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'Functions' )
sys.path.append( mymodule_dir )

import numpy as np
from functools import partial
import cmath
from scipy.interpolate import interp1d
from progressbar import progressbar
from integral import integral

class correlation_classical_last_2():
    def __init__(self,C_list,n_cut,n_noise,t_corr_list):
        # Parameters
        self.C_list = C_list
        self.n_cut = n_cut
        self.n_noise = n_noise
        self.t_corr_list = t_corr_list
        self.T = self.t_corr_list[-1]
        self.dt = t_corr_list[1] - t_corr_list[0]
        

        #self.C_interp = self.regularize_for_interpolation(interp1d(self.t_corr_list, self.C_list, kind='cubic'))
        # Computes the inner product between the function and trigonometric basis
        self.coeff_list = self.compute_coefficients_basis(C_list,self.n_cut,self.T)

        # Reconstructs the correlation. Here only errors are the finite elements in the trigonometric basis and the numerical evaluation of the inner product (which depends on time-spacing but could be done better)
        self.corr_reconstructed = self.reconstructed_corr_list(t_corr_list,self.coeff_list,self.T)

        # Extract random variables and uses the coeff_list above to generate fields.
        self.A = self.generate_A()
        self.xi_list = self.generate_xi_list()
        self.corr_stochastic = self.compute_correlations_from_xi_list(t_corr_list,self.xi_list)
        self.xi_interpolated_list = self.generate_interpolated_xi_list()

        if np.mod(len(t_corr_list),2) == 0: raise Exception("t_corr_list should be odd.")
        t0_index = int((len(t_corr_list)-1)/2.)
        self.expected_error = [np.sqrt((self.C_list[t0_index]**2 + abs(x)**2)/n_noise) for x in self.C_list]
        
    def inner_product(self,f,g,T):
        def to_integrate(x):
            return f(x) * g(x)
        return integral(to_integrate,x_i=-T,x_f=T,limit=100) / (2*T)
    def basis(self,t,T,n):
        return np.cos(n*np.pi*t/T)
    def compute_coefficients_basis(self,C_list,n_cut,T):
        # list of coefficients has length $n_cut + 1$ because it has also the zero term.
        print('Computing Coefficient Basis ({length})'.format(length="n_cut"))
        res = []
        for n in progressbar(np.arange(0,n_cut+1)):
            base = [self.basis(t,T,n) for t in self.t_corr_list]
            product = [x[0] * x[1] for x in zip(C_list,base)]
            #to_integrate = self.regularize_for_interpolation(interp1d(self.t_corr_list, product, kind='cubic'))
            #coeff = integral(to_integrate,x_i=-T,x_f=T,limit=1000) / (2*T)
            coeff = self.dt * sum(product[:-1]) / (2 * T)
            res.append(coeff)
        return res
    def generate_A(self):
        A = [] 
        for t in self.t_corr_list:
            row = []
            row.append(cmath.sqrt(self.coeff_list[0]))
            for n in np.arange(1,self.n_cut+1):
                row.append(cmath.sqrt(2.) * cmath.sqrt(self.coeff_list[n]) * np.cos(n*np.pi*t/self.T))
                row.append(cmath.sqrt(2.) * cmath.sqrt(self.coeff_list[n]) * np.sin(n*np.pi*t/self.T))
            A.append(row)
        return A
    def generate_xi(self):
        mu = 0
        sigma = 1
        xi_list = np.random.normal(mu, sigma, 2*self.n_cut+1)
        xi_field = np.dot(self.A,xi_list)
        return xi_field
    def compute_correlations_from_xi_list(self,t_corr_list,xi_list):
        n_noise = len(xi_list)
        res = 0 * np.array(t_corr_list)
        t0_index = int((len(t_corr_list)-1)/2.)
        for xi in xi_list:
            res = res + np.array([xi[t0_index] * value for value in xi])
        return res / n_noise
    def reconstructed_corr(self,t,coeff_list,T):
        # n_cut has to be such that len(coeff_list) = n_cut + 1
        res = 0
        for n,coeff in enumerate(coeff_list):
            if n == 0:
                res += coeff * self.basis(t,T,n)
            else:
                res += 2 * coeff * self.basis(t,T,n)
        return res
    def reconstructed_corr_list(self,t_list,coeff_list,T):
        print('Reconstructing Correlations ({length})'.format(length="t_corr_list"))
        res = []
        for t_index in progressbar(np.arange(len(t_list))):
            t = t_list[t_index]
            res.append(self.reconstructed_corr(t,coeff_list,T))
        return res
    def generate_xi_list(self):
        res = []
        print("Computing fields ({length})".format(length="n_noise"))
        for _ in progressbar(np.arange(self.n_noise)):
            res.append(self.generate_xi())
        return res
    def generate_interpolated_xi_list(self):
        xi_interpolated_list = []
        for xi in self.xi_list:
            f = self.give_arguments(interp1d(self.t_corr_list, xi, kind='cubic'))
            xi_interpolated_list.append(f)
        return xi_interpolated_list
    def give_arguments(self,f):
        def g(x,args):
            if x > self.T: x = self.T # To avoid rounding errors outside the interpolating domain
            if x < - self.T: x = - self.T
            return f(x)
        return g
    def regularize_for_interpolation(self,f):
        def g(x):
            if x > self.T: x = self.T # To avoid rounding errors outside the interpolating domain
            if x < - self.T: x = - self.T
            return f(x)
        return g
    def save(self,dict):
        #pickle_out = open("Test_correlation_classical_BrownianZeroT.dat",'wb')
        #dict = {}

        dict['C_list_class'] = self.C_list
        dict['n_cut_class'] = self.n_cut
        dict['n_noise_class'] = self.n_noise
        dict['t_corr_list_class'] = self.t_corr_list
        dict['T_class'] = self.T

        dict['t_corr_list_class'] = self.t_corr_list
        dict['coeff_list_class'] = self.coeff_list
        dict['corr_reconstructed_class'] = self.corr_reconstructed
        #dict['xi_list_class'] = self.xi_list
        dict['corr_stochastic_class'] = self.corr_stochastic
        dict['expected_error_class'] = self.expected_error

        #pickle.dump(dict,pickle_out)
        return dict
        #pickle_out.close()

