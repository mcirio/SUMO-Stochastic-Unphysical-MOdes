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
from functools import partial
import cmath
from qutip.ui.progressbar import BaseProgressBar
from J_poly import J_poly
from PM_model import PM_model
from regularize_poles import regularize_poles
from matsubara import matsubara_2
from utility_functions import coth
from superoperator_functions import Hamiltonian_single,Lindblad_single,create_PM_state,create_PM_single
from J_ohmic import J_ohmic
from correlation_numerical_dynamics_physical_3 import correlation_numerical_dynamics_physical_3

# Frequency unit
w0 = 2 * np.pi 
# Cut offs
W_mats = 10 * w0
N_mats = 1000
N_corr = 500 #500#500#1000
N_cut = 1000
N_M = 3
N_M2 = 3
N_R = 5
W_free = 100 * w0
W_cut = 10 * w0
n_cut = 500
mats_cut = 500
n_noise = 100
n_as_exp = 3
n_s_cut = 500
n_s_noise = 10  #50#100#500
W_i = 0
W_f = 10 * w0
integration_limit = 500
# Parameters
w_tilde = W_mats
gamma = .1 * w0
Gamma = gamma / 2.
Omega = np.sqrt(w0**2 - Gamma**2)
ll = np.pi * .01 * w0 #* np.sqrt(2*Omega)
wc = 3 * w0
T = 5. * 2 * np.pi / w0
time_stamp = 0
norm = 1.
vec_p = [0,2 * Gamma * ll**2]
vec_w = [Omega + 1j*Gamma]
vec_w = regularize_poles(vec_w)
J = J_ohmic(ll,wc)
beta = 'inf'
Omega_S = w0
#th = -1
#th = (np.exp(-beta*Omega_S)-1) / (np.exp(-beta*Omega_S)+1)
# Lists
t_list = np.linspace(0,T,500)
t_list_dynamics = t_list
t_list_scaled = [x * w0 / (2*np.pi) for x in t_list] #Time in units of 2\pi/w0
t_list_corr = np.linspace(-T,T,2*N_corr+1)
t_corr_list = np.linspace(-T,T,2*N_corr+1)
t_list_interp = np.linspace(-1.1*T,1.1*T,5000)
w_list = np.linspace(0,3*w0,100)
W_list = np.linspace(W_i,W_f,200)
## Operators 
N_S = 2
N_PM = 4
sigmaz = basis(2,1) * basis(2,1).dag() - basis(2,0) * basis(2,0).dag()
sigmax = basis(2,1) * basis(2,0).dag() + basis(2,0) * basis(2,1).dag()
H_S = Omega_S / 2. * sigmaz
s = sigmax
psi0_S = basis(2,1) * basis(2,1).dag()
obs_list = [sigmaz]

###################################################
#PM = PM_model_analytical(Omega,ll,gamma,t_corr_list,beta,n_s_cut,n_s_noise,H_S,s,N_S,N_PM,psi0_S,t_list,obs_list,integration_limit,N_mats,W_mats)
interpolation = 'no'
PM = PM_model(J,W_i,W_f,t_corr_list,beta,n_as_exp,n_s_cut,n_s_noise,H_S,s,N_S,N_PM,psi0_S,t_list,obs_list,integration_limit,interpolation)
#C = correlation_numerical_dynamics_physical_3(J,W_i,W_f,t_corr_list,beta,n_as_exp,n_s_cut,n_s_noise,integration_limit)
###################################################
fig0, axs0 = plt.subplots(3, 1,figsize=(17,8))
axs0[0].set_title('Antisymmetric - extra')
axs0[0].plot(t_corr_list,np.real(PM.C.C_as_minus_extra),'b',label='antisymmetric (T=0)-corrected real')
axs0[0].plot(t_corr_list,np.real(PM.C.C_as_minus_extra_fit),'r--',label='fit antisymmetric (T=0)-corrected real')
axs0[0].legend()
axs0[1].set_title('Antisymmetric - extra')
axs0[1].plot(t_corr_list,np.imag(PM.C.C_as_minus_extra),'k',label='antisymmetric (T=0)-corrected imag')
axs0[1].plot(t_corr_list,np.imag(PM.C.C_as_minus_extra_fit),'g--',label='fit antisymmetric (T=0)-corrected imag')
axs0[1].legend()

axs0[2].set_title('Symmetric + extra')
axs0[2].plot(t_corr_list,PM.C.C_s_plus_extra,label='symmetric (T=0)-corrected')
axs0[2].plot(t_corr_list,PM.C.C_s_plus_extra_reconstructed,'r--',label='reconstructed')
axs0[2].plot(t_corr_list,PM.C.C_s_plus_extra_stochastic,'g--',label='reconstructed')
axs0[2].legend()

plt.show()

eta = np.sqrt(ll**2/(2*Omega))/Omega_S
th = 2*eta**2/4.-1
##################################################################
pickle_out = open("./PM_3/Tests/Data/Test_PM_model_Ohmic.dat",'wb')
dict = {}
dict_2 = {}
#dict = PM.save(dict)
#dict['dynamics_full'] = dynamics_full
dict = PM.save(dict_2)
#dict['dynamics_full'] = dynamics_full
#dict['dynamics_no_Matsubara'] = dynamics_no_Matsubara
dict['Omega_S'] = Omega_S
dict['ll'] = ll
dict['wc'] = wc
#dict['Gamma'] = Gamma
to_save = [dict,th]
pickle.dump(to_save,pickle_out)
pickle_out.close()
###############################################################
fig, axs = plt.subplots(4, 1,figsize=(17,12))
axs[0].set_title('Symmetric + extra')
#axs[0].plot(t_corr_list,np.real(PM.C_s_list),'b',label='analytical (beta={})'.format(beta))
#axs[0].plot(t_corr_list,np.real(PM.C_s_reconstructed),'r--',label='reconstructed')
#axs[0].plot(t_corr_list,np.real(PM.C_s_stochastic),'g--',label='stochastic')
axs[0].legend()

axs[1].set_title('Symmetric + extra (two methods)')
axs[1].plot(t_corr_list,PM.C.C_s_plus_extra,'b')
axs[1].plot(t_corr_list,PM.C.C_s_plus_extra_stochastic,'g--')
axs[1].plot(t_corr_list,PM.C.C_s_plus_extra_reconstructed,'r--')

axs[2].set_title('Antisymmetric - extra (two methods)')
axs[2].plot(t_corr_list,PM.C.C_as_minus_extra,'b')
axs[2].plot(t_corr_list,PM.C.C_as_fit,'b')
#axs[2].plot(t_corr_list,[ll**2/(2*Omega)*np.exp(-1j*Omega*t-Gamma*abs(t)) for t in t_corr_list],'r--')

axs[3].set_title('Dynamics')
#axs[3].plot(t_list,dynamics_full,'b',label='full PM')
axs[3].plot(t_list,PM.dynamics,'g--',label='stochastic')
#axs[3].plot(t_list,PM.dynamics_average,'r--',label='stochastic')
#axs[3].plot(t_list,[th for _ in t_list],color='darkgrey',linestyle='--')
axs[3].legend()
#axs[1].set_ylim(-1.01,-.93)

plt.show()