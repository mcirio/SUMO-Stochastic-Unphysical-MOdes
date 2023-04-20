# import os
# import sys

# script_dir = os.path.dirname( __file__ )
# mymodule_dir = os.path.join( script_dir, '..', 'Functions' )
# sys.path.append( mymodule_dir )

import numpy as np
from integral import integral
from progressbar import progressbar
from utility_functions import coth
import concurrent.futures

def compute_correlations(J,beta,W_i,W_f,integration_limit,t_corr_list):

    precision = 100000
    w = np.linspace(W_i+0.000000001, W_f, precision).reshape(precision,-1).T
    values = t_corr_list

    def to_integrate_symmetric(t):
        return np.sum(J(w) * coth(beta,w/2.)* np.cos(w*t)/np.pi , axis=1)* 20*np.pi/precision
    
    def to_integrate_antisymmetric(t):
        return np.sum(-1j * J(w)* np.sin(w*t)/ (np.pi) , axis=1)* 20*np.pi/precision
    
    def compute_symmetric(J,beta,W_i,W_f,integration_limit,t_corr_list):
        print('Computing symmetric correlations')

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(to_integrate_symmetric, values)
        return np.array(list(results))
    
    def compute_antisymmetric(J,W_i,W_f,integration_limit,t_corr_list):
        print('Computing antisymmetric correlations')
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(to_integrate_antisymmetric, values)
        return np.array(list(results))
    
    C_s = compute_symmetric(J,beta,W_i,W_f,integration_limit,t_corr_list)
    C_as = compute_antisymmetric(J,W_i,W_f,integration_limit,t_corr_list)
    print("completed")
    return C_s, C_as