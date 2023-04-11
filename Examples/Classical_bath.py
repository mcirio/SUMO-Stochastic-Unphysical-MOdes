# Dynamics of a single two-level system subject to magnetic field having both a constant and a colored-noise component

import os
import sys
script_dir = os.path.dirname( __file__ )
import numpy as np
import matplotlib.pyplot as plt
from qutip import *


from sumo.Functions.spectral_decomposition import spectral_decomposition
from sumo.Functions.generate_fields import generate_fields
from sumo.Functions.quantum_functions import Hamiltonian_single
from sumo.Functions.average_dynamics import average_dynamics, average_dynamics_parallel

from sumo.Functions.progressbar import progressbar
from sumo.Functions.timedeprates import TimeDepRates

# Frequency unit
w0 = 2 * np.pi 
# Cut offs
N_corr = 1000 # time discretization
n_cut = 100 # number of spectral components in the field
n_noise = 500 # stochastic average
# Parameters
T = 25 / w0 # Total time
w_corr = .5 * w0 # correlation frequency
Gamma = .2 * w0 # correlatino decay
ll = .02 * w0 # correlation strength
ws = w0 # two-level system frequency


Delta = 0.1 * w0 # two-level system coherent field strength
# Other units
units_freq = w0 # arbitrary
units_time = 1 / w0 # arbitrary
units_corr = w0**2 # arbitrary
# Operators
sigmaz = basis(2,1) * basis(2,1).dag() - basis(2,0) * basis(2,0).dag()
sigmax = basis(2,1) * basis(2,0).dag() + basis(2,0) * basis(2,1).dag()
# Lists
t_corr_list = np.linspace(-T,T,2*N_corr+1) # correlation times. From -T to T
t_dynamics_list = np.linspace(0,T,N_corr)  # Dynamics time. From 0 to T
c_corr_list = [ll * np.cos(w_corr * t) * np.exp(-Gamma*abs(t)) for t in t_corr_list] # correlations (arbitrary choice)




# Classical noise
coeff_list, c_corr_reconstructed = spectral_decomposition(t_corr_list,c_corr_list,n_cut) # spectral decomposition
xi_interpolated_list, c_corr_reconstructed_stochastic = generate_fields(t_corr_list,coeff_list,n_cut,n_noise) # field
# Dynamics
s = sigmax # coupling operator
H_S = ws / 2. * sigmaz + Delta * s # system Hamiltonian
psi0 = basis(2,1) * basis(2,1).dag() # initial state
obs_list = [sigmaz] # observable to be plotted
L = Hamiltonian_single(H_S) # Hamilonian as a superoperator
H_xi = Hamiltonian_single(s) # coupling operator as a superoperator
c_list = [] # system decay
dynamics_no_noise = mesolve(L, psi0, t_dynamics_list, c_list, obs_list).expect[0] # dynamics without noise
dynamics_average, sigma_dynamics, dynamics_list = average_dynamics_parallel(L,H_xi,xi_interpolated_list,c_list,t_dynamics_list,psi0,n_noise,obs_list) # dynamics with noise

###############
#Attempt to construct purely rate-based version
#follow PRA 95, 052126 equation 42 onwards
# V = c_corr_list * s 

def corr(t):
    return ll * np.cos(w_corr * t) * np.exp(-Gamma*abs(t))

L = TimeDepRates(H_S,s,corr,t_dynamics_list)
print("Solving master equation with TD rates")
dynamics_gam = mesolve(L, psi0, t_dynamics_list, [], obs_list).expect[0] # dynamics with rates


# Rescaling 
t_corr_list_rescaled = [t / units_time for t in t_corr_list]
t_dynamics_list_rescaled = [t / units_time for t in t_dynamics_list]
c_corr_list_rescaled = [c / units_corr for c in c_corr_list]
c_corr_reconstructed_rescaled = [c / units_corr for c in c_corr_reconstructed]
c_corr_reconstructed_stochastic_rescaled = [c / units_corr for c in c_corr_reconstructed_stochastic]
# Plots
## Text size
dim_labels = 20
dim_legends = 10
dim_titles = 20
dim_ticks_major = 15
dim_ticks_minor = 10
dim_linewidth = 2
## Figure
fig, axs = plt.subplots(1, 2,figsize=(16,6))
plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.65)
## Correlations
axs[0].set_title('Correlations',fontsize=dim_titles)
axs[0].plot(t_corr_list_rescaled,c_corr_list_rescaled,'b',linewidth=dim_linewidth,label='numerical')
axs[0].plot(t_corr_list_rescaled,c_corr_reconstructed_rescaled,'r--',linewidth=dim_linewidth,label='spectral reconstruction') 
axs[0].plot(t_corr_list_rescaled,c_corr_reconstructed_stochastic_rescaled,'g--',linewidth=dim_linewidth,label='stochastic')
axs[0].legend(fontsize=dim_legends,loc='upper right')
axs[0].set_xlabel('time [$1/\omega_0$]', fontsize=dim_labels)
axs[0].set_ylabel(r'$C(t)[1/\omega_0^2]$', fontsize=dim_labels)
## Dynamics
axs[1].set_title('Dynamics',fontsize=dim_titles)
axs[1].plot(t_dynamics_list_rescaled,dynamics_no_noise,'b',linewidth=dim_linewidth,label='no noise')
axs[1].plot(t_dynamics_list_rescaled,dynamics_list[0],'r--',linewidth=dim_linewidth,label='single trajectory')
axs[1].plot(t_dynamics_list_rescaled,dynamics_average,color='darkgreen', linestyle='--',linewidth=dim_linewidth,label='average')
axs[1].plot(t_dynamics_list_rescaled,dynamics_gam,color='y', linestyle='-',linewidth=dim_linewidth,label='time dep rates')

#axs[1].plot(t_dynamics_list_rescaled,expect(sigmaz,rho_PD),color='g', linestyle='--',linewidth=dim_linewidth,label='PD sol')

axs[1].fill_between(t_dynamics_list_rescaled, np.array(dynamics_average)-np.array(sigma_dynamics), np.array(dynamics_average)+np.array(sigma_dynamics),alpha=0.4, edgecolor='green', facecolor='lightgreen')
axs[1].set_xlabel('time [$1/\omega_0$]', fontsize=dim_labels)
axs[1].set_ylabel(r'$\langle\sigma_z\rangle$', fontsize=dim_labels)
axs[1].legend(fontsize=dim_legends,loc='lower right')
plt.show()