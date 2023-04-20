import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..','Functions' )
sys.path.append( mymodule_dir )
import pickle

import numpy as np
from compute_correlations import compute_correlations
from fit import fit 
from spectral_decomposition import spectral_decomposition
from generate_fields import generate_fields
from generate_PM_model import generate_PM_model
from average_dynamics import average_dynamics

class Stochastic_model():
    def __init__(self,corr_params, system_params,PM_params,stoch_params):

        J = corr_params['J']
        beta = corr_params['beta'] 
        W_i = corr_params['W_i']
        W_f = corr_params['W_f'] 
        integration_limit = corr_params['integration_limit']
        t_corr_list = corr_params['t_corr_list']

        N_S = system_params['N_S']
        H_S = system_params['H_S']
        s = system_params['s']
        psi0_S = system_params['psi0_S']
        obs_list = system_params['obs_list']
        t_list = system_params['t_list']

        N_PM = PM_params['N_PM'] 
        n_as_exp = PM_params['n_as_exp'] 

        n_cut = stoch_params['n_cut'] 
        n_noise = stoch_params['n_noise'] 

        self.C_s, self.C_as = compute_correlations(J,beta,W_i,W_f,integration_limit,t_corr_list)
        self.C_s, self.C_as = self.C_s.reshape(2001,), self.C_as.reshape(2001,)
        self.ordered_PM_parameters, self.C_as_fit, self.C_s_extra = fit(t_corr_list,self.C_as,n_as_exp)

        self.C_s_plus_extra = [x[0] + x[1] for x in zip(self.C_s,self.C_s_extra)]
        self.C_as_minus_extra = [x[0] - x[1] for x in zip(self.C_as,self.C_s_extra)]

        self.coeff_list, self.C_s_reconstructed = spectral_decomposition(t_corr_list,self.C_s_plus_extra,n_cut)
        self.xi_interpolated_list, self.C_s_reconstructed_stochastic = generate_fields(t_corr_list,self.coeff_list,n_cut,n_noise)

        self.H_S, self.s, self.H_xi, self.L, self.psi0, self.new_obs_list, self.c_list = generate_PM_model(H_S,psi0_S,s,self.ordered_PM_parameters,obs_list,N_S,N_PM)
        self.dynamics_average, self.sigma, self.dynamics_list = average_dynamics(self.L,self.H_xi,self.xi_interpolated_list,self.c_list,t_list,self.psi0,n_noise,self.new_obs_list)