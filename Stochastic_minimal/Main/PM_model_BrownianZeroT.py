import time

start_time = time.time()

import os
import sys
import pickle

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'Classes' )
sys.path.append( mymodule_dir )
mymodule_dir = os.path.join( script_dir, '..', 'Functions' )
sys.path.append( mymodule_dir )

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.optimize import curve_fit
from J_poly import J_poly
from regularize_poles import regularize_poles
from compute_correlations import compute_correlations
from fit import fit
from generate_fields import generate_fields
from spectral_decomposition import spectral_decomposition
from generate_PM_model import generate_PM_model
from average_dynamics import average_dynamics
from PM_model_Brownian_ZeroT import PM_model_Brownian_ZeroT
from PM_model_Brownian_ZeroT_noMats import PM_model_Brownian_ZeroT_noMats
from Stochastic_model import Stochastic_model
# from PM_model import PM_model
# from matsubara import matsubara_2
# from superoperator_functions import Hamiltonian_single
# from superoperator_functions import Lindblad_single
# from superoperator_functions import create_PM_single

# Frequency unit
w0 = 2 * np.pi 
# Cut offs
W_mats = 10 * w0
N_mats = 1000
N_corr = 1000#1000
N_M = 2
N_R = 2
n_cut = 500
n_noise = 100
n_as_exp = 1
n_s_cut = 100 #500
W_i = 0
W_f = 10 * w0
integration_limit = 500
# Parameters
omega0 = w0
ws = w0
gamma = .05 * w0
Gamma = gamma / 2.
Omega = np.sqrt(omega0**2 - Gamma**2)
ll = .2 / np.sqrt(2*np.pi) * w0**(3/2.) #0.45 * w0 #* np.sqrt(2*Omega)
T = 60. * 2 * np.pi / w0
T_corr = T 
vec_p = [0,2 * Gamma * ll**2]
vec_w = [Omega + 1j*Gamma]
vec_w = regularize_poles(vec_w)
J = J_poly(vec_p,vec_w)
beta = 'inf'
# Other units
units_freq = ws
units_time = 1 / ws
units_corr = ll**2 / (2 * Omega)

# Lists
t_list = np.linspace(0,T,500)
t_list_dynamics = t_list
t_list_scaled = [x * w0 / (2*np.pi) for x in t_list] #Time in units of 2\pi/w0
t_list_corr = np.linspace(-T_corr,T_corr,2*N_corr+1)
t_corr_list = np.linspace(-T_corr,T_corr,2*N_corr+1)
t_list_interp = np.linspace(-1.1*T,1.1*T,5000)
w_list = np.linspace(0,3*w0,100)
W_list = np.linspace(W_i,W_f,200)

# Full PM model
print('-Computing PM model')
PM_full = PM_model_Brownian_ZeroT(ws,omega0,gamma,ll,N_R,N_mats,W_mats,integration_limit,t_corr_list,t_list,N_M)
dynamics_PM_full = PM_full.dynamics_full

# PM model (no Mats)
print('-Computing PM model (no Mats)')
PM_noMats = PM_model_Brownian_ZeroT_noMats(ws,omega0,gamma,ll,t_list,N_R)
dynamics_PM_noMats = PM_noMats.dynamics_full

# Stochastic
print('-Computing Stochastic PM model')
N_S = 2
N_PM = 3
sigmaz = basis(2,1) * basis(2,1).dag() - basis(2,0) * basis(2,0).dag()
sigmax = basis(2,1) * basis(2,0).dag() + basis(2,0) * basis(2,1).dag()
H_S = ws / 2. * sigmaz
s = sigmax
psi0_S = basis(2,1) * basis(2,1).dag()
obs_list = [sigmaz]

corr_params = {}
corr_params['J'] = J
corr_params['beta'] = beta
corr_params['W_i'] = W_i
corr_params['W_f'] = W_f
corr_params['integration_limit'] = integration_limit
corr_params['t_corr_list'] = t_corr_list

system_params = {}
system_params['N_S'] = N_S
system_params['H_S'] = H_S
system_params['s'] = s
system_params['psi0_S'] = psi0_S
system_params['obs_list'] = obs_list
system_params['t_list'] = t_list

PM_params = {}
PM_params['N_PM'] = N_PM
PM_params['n_as_exp'] = n_as_exp
stoch_params = {}
stoch_params['n_cut'] = n_cut
stoch_params['n_noise'] = n_noise

stoch = Stochastic_model(corr_params, system_params,PM_params,stoch_params)

C_s, C_as = stoch.C_s, stoch.C_as # correlations
ordered_PM_parameters, C_as_fit, C_s_extra, C_s_plus_extra, C_as_minus_extra = stoch.ordered_PM_parameters, stoch.C_as_fit, stoch.C_s_extra, stoch.C_s_plus_extra,stoch.C_as_minus_extra # fitting
coeff_list, C_s_reconstructed, C_s_reconstructed_stochastic = stoch.coeff_list, stoch.C_s_reconstructed, stoch.C_s_reconstructed_stochastic # classical
dynamics_average, sigma, dynamics_list = stoch.dynamics_average, stoch.sigma, stoch.dynamics_list # dynamics

# C_s, C_as = compute_correlations(J,beta,W_i,W_f,integration_limit,t_corr_list)
# ordered_PM_parameters, C_as_fit, C_s_extra = fit(t_corr_list,C_as,n_as_exp)

# C_s_plus_extra = [x[0] + x[1] for x in zip(C_s,C_s_extra)]
# C_as_minus_extra = [x[0] - x[1] for x in zip(C_as,C_s_extra)]

# coeff_list, C_s_reconstructed = spectral_decomposition(t_corr_list,C_s_plus_extra,n_cut)
# xi_interpolated_list, C_s_reconstructed_stochastic = generate_fields(t_corr_list,coeff_list,n_cut,n_noise)

# H_S, s, H_xi, L, psi0, new_obs_list, c_list = generate_PM_model(H_S,psi0_S,s,ordered_PM_parameters,obs_list,N_S,N_PM)
# dynamics_average, sigma, dynamics_list = average_dynamics(L,H_xi,xi_interpolated_list,c_list,t_list,psi0,n_noise,new_obs_list)

# Saving
pickle_out = open("Git\SUMO-Stochastic-Unphysical-MOdes\Stochastic_minimal\Main\Data\data.dat",'wb')
saved_dict = {}

saved_dict['units_freq'] = units_freq
saved_dict['units_time'] = units_time
saved_dict['units_corr'] = units_corr
saved_dict['t_corr_list'] = t_corr_list
saved_dict['t_list'] = t_list
saved_dict['C_s'] = C_s
saved_dict['C_as'] = C_as
saved_dict['ordered_PM_parameters'] = ordered_PM_parameters
saved_dict['C_as_fit'] = C_as_fit
saved_dict['C_s_extra'] = C_s_extra
saved_dict['C_s_plus_extra'] = C_s_plus_extra
saved_dict['C_as_minus_extra'] = C_as_minus_extra
saved_dict['coeff_list'] = coeff_list
saved_dict['C_s_reconstructed'] = C_s_reconstructed
saved_dict['C_s_reconstructed_stochastic'] = C_s_reconstructed_stochastic
saved_dict['dynamics_average'] = dynamics_average
saved_dict['sigma'] = sigma
saved_dict['dynamics_list'] = dynamics_list
saved_dict['dynamics_PM_full'] = dynamics_PM_full
saved_dict['dynamics_PM_noMats'] = dynamics_PM_noMats
pickle.dump(saved_dict,pickle_out)
pickle_out.close()


end_time = time.time()
total_time = end_time - start_time
print("Time taken:", total_time, "seconds")