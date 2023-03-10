# import os
# import sys

# script_dir = os.path.dirname( __file__ )
# mymodule_dir = os.path.join( script_dir, '..', 'Functions' )
# sys.path.append( mymodule_dir )

import numpy as np
from integral import integral
from progressbar import progressbar
from utility_functions import coth

def compute_correlations(J,beta,W_i,W_f,integration_limit,t_corr_list):
    def to_integrate_symmetric(w,t,J,beta):
        return 1 / (np.pi) * J(w) * coth(beta,w/2.) * np.cos(w * t)
    def to_integrate_antisymmetric(w,t,J):
        return - 1 / (np.pi) * 1j * J(w) * np.sin(w * t)
    def compute_symmetric(J,beta,W_i,W_f,integration_limit,t_corr_list):
        print('Computing symmetric correlations ({length})'.format(length="t_corr_list"))
        res = []
        for t_index in progressbar(np.arange(len(t_corr_list))):
            t = t_corr_list[t_index]
            res.append(integral(to_integrate_symmetric,t,J,beta,x_i=W_i,x_f=W_f,limit=integration_limit) )
        return res
    def compute_antisymmetric(J,W_i,W_f,integration_limit,t_corr_list):
        print('Computing antisymmetric correlations ({length})'.format(length="t_corr_list"))
        res = []
        for t_index in progressbar(np.arange(len(t_corr_list))):
            t = t_corr_list[t_index]
            res.append(integral(to_integrate_antisymmetric,t,J,x_i=W_i,x_f=W_f,limit=integration_limit) )
        return res
    C_s = compute_symmetric(J,beta,W_i,W_f,integration_limit,t_corr_list)
    C_as = compute_antisymmetric(J,W_i,W_f,integration_limit,t_corr_list)
    return C_s, C_as
