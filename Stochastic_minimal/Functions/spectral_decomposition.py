import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'Functions' )
sys.path.append( mymodule_dir )

import numpy as np

def spectral_decomposition(t_corr_list,C_list,n_cut):

    def basis(t,T,n):
        return np.cos(n*t*np.pi/T)

    def compute_coefficients_basis(t_corr_list,C_list,n_cut):

        T = t_corr_list[-1]
        dt = t_corr_list[1] - t_corr_list[0]

        print('Computing Coefficient Basis ({length})'.format(length="n_cut"))

        n = np.arange(0, n_cut+1)
        t = np.array(t_corr_list).reshape(1, -1)
        C_list = np.array(C_list)

        base = basis(t, T, n.reshape(-1, 1))
        product = C_list*base
        coeff = dt * np.sum(product, axis=1) / (2 * T)
        
        return coeff

    def reconstructed_corr(t,coeff_list):
        T = t_corr_list[-1]
        # n_cut has to be such that len(coeff_list) = n_cut + 1
        res = 0
        for n,coeff in enumerate(coeff_list):
            if n == 0:
                res += coeff * basis(t,T,n)
            else:
                res += 2 * coeff * basis(t,T,n)
        return res
    def reconstructed_corr_list(t_list,coeff_list):

        print('Reconstructing Correlations ({length})'.format(length="t_corr_list"))
        t = np.array(t_list)
        return reconstructed_corr(t, coeff_list)

    coeff_list = compute_coefficients_basis(t_corr_list,C_list,n_cut)
    C_s_reconstructed = reconstructed_corr_list(t_corr_list,coeff_list)

    return coeff_list, C_s_reconstructed