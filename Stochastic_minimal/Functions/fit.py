import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'Functions' )
sys.path.append( mymodule_dir )

import numpy as np
from scipy.optimize import least_squares
from integral import integral
from progressbar import progressbar
from utility_functions import coth

# Fitting of a correlation which is antisymmetric in time.
def fit(t_corr_list,C_as,n_as_exp):

    def fit_function(params,t):
        res = 0
        for n in np.arange(0,n_as_exp):
            aR = params[6 * n + 0]
            aI = params[6 * n + 1]
            bR = params[6 * n + 2]
            bI = params[6 * n + 3]
            cR = params[6 * n + 4]
            cI = params[6 * n + 5]

            a = aR + 1j * aI
            b = bR + 1j * bI
            c = cR + 1j * cI

            res += a*np.sin(b*(t))*np.exp(-c*abs(t))
        return res
    def _penalty(params,t,y):
            res = fit_function(params,t)
            return y - res 
    def penalty(params,t,y):
        fx = _penalty(params,t,y)
        return np.array(list(np.absolute(fx.real))+list(np.absolute(fx.imag)))

    def extra_symmetric(params,t):
        res = 0
        for n in np.arange(0,n_as_exp):
            aR = params[6 * n + 0]
            aI = params[6 * n + 1]
            bR = params[6 * n + 2]
            bI = params[6 * n + 3]
            cR = params[6 * n + 4]
            cI = params[6 * n + 5]

            a = aR + 1j * aI
            b = bR + 1j * bI
            c = cR + 1j * cI

            res += - 1j * a*np.cos(b*(t))*np.exp(-c*abs(t))
        return res
    def generate_PM_parameters(t_corr_list,C_as):
        p0 = []
        b1 = []
        b2 = []
        for n in np.arange(0,n_as_exp):
            p0 = p0 + [0,1,1,0,1,0] # We start from imaginary overal coefficient because we know the correlation is [i * sin()]. All other parameters are set initially real.
            b1 = b1 + [-np.inf,-np.inf,-np.inf,-np.inf,0,-np.inf]#lambda,Omega,Gamma. Bound on Re[Omega] is because a sin(b) has a floating sign between a and b.
            b2 = b2 + [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]
        result = least_squares(penalty, p0,bounds=(b1,b2),args=(t_corr_list,C_as))
        return result.x
    def order_PM_parameters(PM_parameters):
        ordered_PM_parameters = []
        for n in np.arange(0,n_as_exp):
            single_PM_parameters = []
            single_PM_parameters.append(1j *(PM_parameters[6*n + 0]+1j*PM_parameters[6*n + 1])) # coupling
            single_PM_parameters.append( (PM_parameters[6*n + 2]+1j*PM_parameters[6*n + 3])) # frequency
            single_PM_parameters.append((PM_parameters[6*n + 4] + 1j * PM_parameters[6*n + 5])) # decay rate
            ordered_PM_parameters.append(single_PM_parameters)
        return ordered_PM_parameters
    
    PM_parameters = generate_PM_parameters(t_corr_list,C_as)
    ordered_PM_parameters = order_PM_parameters(PM_parameters)
    C_as_fit = [fit_function(PM_parameters,t) for t in t_corr_list]
    C_s_extra_fit = [extra_symmetric(PM_parameters,t) for t in t_corr_list]
    return ordered_PM_parameters, C_as_fit, C_s_extra_fit