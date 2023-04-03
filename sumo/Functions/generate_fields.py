import os
import sys


import numpy as np
from scipy.optimize import least_squares

from sumo.Functions.integral import integral
from sumo.Functions.progressbar import progressbar
from sumo.Functions.utility_functions import coth
import cmath
from scipy.interpolate import interp1d
from  qutip.interpolate import Cubic_Spline
def generate_fields(t_corr_list,coeff_list,n_cut,n_noise):

    # def compute_coefficients_basis(t_corr_list,C_list,n_cut):
    #     T = t_corr_list[-1]
    #     dt = t_corr_list[1] - t_corr_list[0]
    #     print('Computing Coefficient Basis ({length})'.format(length="n_cut"))
    #     res = []
    #     for n in progressbar(np.arange(0,n_cut+1)):
    #         base = [np.cos(n*np.pi*t/T) for t in t_corr_list]
    #         product = [x[0] * x[1] for x in zip(C_list,base)]
    #         coeff = dt * sum(product[:-1]) / (2 * T)
    #         res.append(coeff)
    #     return res
    def generate_A(t_corr_list,coeff_list,n_cut):
        T = t_corr_list[-1]
        A = [] 
        print('Computing Matrix A ({length})'.format(length="t_corr_list"))
        for t_index in progressbar(np.arange(0,len(t_corr_list))):
            t = t_corr_list[t_index]
        # for t in t_corr_list:
            row = []
            row.append(cmath.sqrt(coeff_list[0]))
            for n in np.arange(1,n_cut+1):
                row.append(cmath.sqrt(2.) * cmath.sqrt(coeff_list[n]) * np.cos(n*np.pi*t/T))
                row.append(cmath.sqrt(2.) * cmath.sqrt(coeff_list[n]) * np.sin(n*np.pi*t/T))
            A.append(row)
        return A
    def generate_xi(A,n_cut):
        mu = 0
        sigma = 1
        xi_list = np.random.normal(mu, sigma, 2*n_cut+1)
        xi_field = np.dot(A,xi_list)
        return xi_field
    def generate_xi_list(A,n_cut,n_noise):
        res = []
        print("Computing fields ({length})".format(length="n_noise"))
        for _ in progressbar(np.arange(n_noise)):
            res.append(generate_xi(A,n_cut))
        return res
    def generate_interpolated_xi_list(xi_list,t_corr_list):
        T = t_corr_list[-1]
        xi_interpolated_list = []
        for xi in xi_list:
            #f = give_arguments(interp1d(t_corr_list, xi, kind='cubic'),T)
            #xi_interpolated_list.append(f)
            #NOTE: I had to switch to this to make parallization work.
            xi_interpolated_list.append(Cubic_Spline(t_corr_list[0],t_corr_list[-1],xi))
        return xi_interpolated_list
    def give_arguments(f,T):
        def g(x,args):
            if x > T: x = T # To avoid rounding errors outside the interpolating domain
            if x < - T: x = - T
            return f(x)
        return g
    def regularize_for_interpolation(f,T):
        def g(x):
            if x > T: x = T # To avoid rounding errors outside the interpolating domain
            if x < - T: x = - T
            return f(x)
        return g
    def compute_correlations_from_xi_list(t_corr_list,xi_list):
        n_noise = len(xi_list)
        res = 0 * np.array(t_corr_list)
        t0_index = int((len(t_corr_list)-1)/2.)
        for xi in xi_list:
            res = res + np.array([xi[t0_index] * value for value in xi])
        return res / n_noise

    # coeff_list = compute_coefficients_basis(t_corr_list,C_list,n_cut)
    A = generate_A(t_corr_list,coeff_list,n_cut)
    xi_list = generate_xi_list(A,n_cut,n_noise)
    xi_interpolated_list = generate_interpolated_xi_list(xi_list,t_corr_list)
    C_s_reconstructed_stochastic = compute_correlations_from_xi_list(t_corr_list,xi_list)
    return xi_interpolated_list, C_s_reconstructed_stochastic