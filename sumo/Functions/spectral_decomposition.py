import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'Functions' )
sys.path.append( mymodule_dir )

import numpy as np
from sumo.Functions.progressbar import progressbar

def spectral_decomposition(t_corr_list,C_list,n_cut):

    def basis(t,T,n):
        return np.cos(n*np.pi*t/T)
    def compute_coefficients_basis(t_corr_list,C_list,n_cut):
        T = t_corr_list[-1]
        dt = t_corr_list[1] - t_corr_list[0]
        print('Computing Coefficient Basis ({length})'.format(length="n_cut"))
        res = []
        for n in progressbar(np.arange(0,n_cut+1)):
            base = [basis(t,T,n) for t in t_corr_list]
            product = [x[0] * x[1] for x in zip(C_list,base)]
            coeff = dt * sum(product[:-1]) / (2 * T)
            res.append(coeff)
        return res
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
        T = t_corr_list[-1]
        print('Reconstructing Correlations ({length})'.format(length="t_corr_list"))
        res = []
        for t_index in progressbar(np.arange(len(t_list))):
            t = t_list[t_index]
            res.append(reconstructed_corr(t,coeff_list))
        return res

    coeff_list = compute_coefficients_basis(t_corr_list,C_list,n_cut)
    C_s_reconstructed = reconstructed_corr_list(t_corr_list,coeff_list)

    return coeff_list, C_s_reconstructed