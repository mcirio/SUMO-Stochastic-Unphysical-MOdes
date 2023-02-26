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
from J_array import J_array
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

integration_limit = 500
# Parameters
w_tilde = W_mats
# gamma = .1 * w0
# Gamma = gamma / 2.
Omega = w0
ll = .05 * w0 #* np.sqrt(2*Omega)

W_i = Omega - 2 * ll
W_f = Omega + 2 * ll


wc = 3 * w0
T = 20. * 2 * np.pi / w0
time_stamp = 0
norm = 1.
# vec_p = [0,2 * Gamma * ll**2]
# vec_w = [Omega + 1j*Gamma]
# vec_w = regularize_poles(vec_w)
J = J_array(ll,Omega,ll)
#J = J_ohmic(ll,wc)
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
J_list = [J(w) for w in W_list]
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
interpolation = 'no'
C = correlation_numerical_dynamics_physical_3(J,W_i,W_f,t_corr_list,beta,n_as_exp,n_s_cut,n_s_noise,integration_limit,interpolation)
###################################################
dim_labels = 18
dim_titles = 20
dim_ticks = 18
dim_legend = 13
dim_letters = 15
line_width = 2
x_lim = 20

fig0, axs0 = plt.subplots(1, 1,figsize=(16,6))
axs0.plot(W_list,J_list)
axs0.set_ylim(0,1)

fig, axs = plt.subplots(2, 2,figsize=(16,6))
plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.65)
#plt.tight_layout(h_pad=0, w_pad=5)
#axs[0,0].set_title('Correlation (real)')
#axs[0,0].set_title(r'$\mathrm{Re}[C_\mathrm{s}(t)]$',fontsize=dim_titles)
axs[0,0].set_title('Symmetric Correlation',fontsize=dim_titles)
#axs[0,0].plot(t_corr_list,np.real(C_as),'b',label='antisymmetric')
axs[0,0].plot(t_corr_list,np.real(C.C_s),'b',label='symmetric',linewidth=line_width)
axs[0,0].set_xlabel(r'time [$1/\omega_S$]', fontsize=dim_labels)
#axs[0,0].set_ylabel('$C(t)$ [$\omega_c^2$]', fontsize=dim_labels)
axs[0,0].set_ylabel(r'$\mathrm{Re}\{C_\mathrm{s}(t)[\omega_c^2]\}$', fontsize=dim_labels)
axs[0,0].tick_params(axis='both', which='major', labelsize=dim_ticks)
axs[0,0].set_xlim(-x_lim,x_lim)
#axs[0,0].set_xlim(-100,100)
#axs[0,0].set_ylim(-1,1)
#axs[0,0].legend(loc='lower right')

#axs[0,1].set_title('Correlation (imag)')
#axs[0,1].set_title(r'$\mathrm{Im}[C_\mathrm{as}(t)]$',fontsize=dim_titles)
axs[0,1].set_title('Imaginary Correlation',fontsize=dim_titles)
axs[0,1].plot(t_corr_list,np.imag(C.C_as),'b',label='numerical',linewidth=line_width)
#axs[0,1].plot(t_corr_list,np.imag(C_s),'g',label='symmetric')
axs[0,1].plot(t_corr_list,np.imag(-1j*np.array(C.check_fit)),'r--',label='fit, exp = 2',linewidth=line_width)
# axs[0,1].plot(t_corr_list,np.imag(-1j*np.array(C.check_fit_2)),'g--',label='fit, exp = 1',linewidth=line_width)
axs[0,1].set_xlabel(r'time [$1/\omega_S$]', fontsize=dim_labels)
#axs[0,1].set_ylabel('$C(t)$ [$\omega^2_c$]', fontsize=dim_labels)
axs[0,1].set_ylabel(r'$\mathrm{Im}\{C_\mathrm{as}(t)[\omega^2_c]\}$', fontsize=dim_labels)
#axs[0,1].set_xlim(-100,100)
axs[0,1].legend(loc='upper right', prop={'size': dim_legend})
axs[0,1].tick_params(axis='both', which='major', labelsize=dim_ticks)
axs[0,1].set_xlim(-x_lim,x_lim)
######################

#axs[1,1].set_title('Antisymmetric - extra (real)')
#axs[1,1].set_title(r'$C_\mathrm{Q}(t)$',fontsize=dim_titles)
axs[1,1].set_title('Quantum correlation',fontsize=dim_titles)
axs[1,1].plot(t_corr_list,np.real(C.C_as_minus_extra),'b',label='numerical (real)',linewidth=line_width)
axs[1,1].plot(t_corr_list,np.real(C.C_as_minus_extra_fit),'r--',label='fit (real',linewidth=line_width)

axs[1,1].plot(t_corr_list,np.imag(C.C_as_minus_extra),'g',label='numerical (imag)',linewidth=line_width)
axs[1,1].plot(t_corr_list,np.imag(C.C_as_minus_extra_fit),color='orange',linestyle='dashed',label='fit (imag',linewidth=line_width)
axs[1,1].set_xlabel(r'time [$1/\omega_S$]', fontsize=dim_labels)
#axs[1,1].set_ylabel('$C_{as}(t)$ [$\omega^2_c$]', fontsize=dim_labels)
axs[1,1].set_ylabel(r'$C_\mathrm{Q}(t)[\omega^2_c]$', fontsize=dim_labels)
#axs[1,1].set_xlim(-100,100)
axs[1,1].legend(loc='upper right', prop={'size': dim_legend})
axs[1,1].tick_params(axis='both', which='major', labelsize=dim_ticks)
axs[1,1].set_xlim(-x_lim,x_lim)

# axs[2,1].set_title('Antisymmetric - extra (imag)')

# axs[2,1].plot(t_corr_list,np.imag(C_as_minus_extra),'k',label='numerical')
# axs[2,1].plot(t_corr_list,np.imag(C_as_minus_extra_fit),'g--',label='fit')
# axs[2,1].set_xlabel('time [$1/\omega_S$]')
# axs[2,1].set_ylabel('$C_{as}(t)$ [$\omega^2_c$]')
# #axs[2,1].set_xlim(-100,100)
# axs[2,1].legend(loc='lower right')

#axs[1,0].set_title('Symmetric + extra (real)')
#axs[1,0].set_title(r'$\mathrm{Re}[C_\mathrm{class}(t)]$',fontsize=dim_titles)
axs[1,0].set_title('Classical Correlation',fontsize=dim_titles)

axs[1,0].plot(t_corr_list,np.real(C.C_s_plus_extra),label='numerical',linewidth=line_width)
axs[1,0].plot(t_corr_list,np.real(C.C_s_plus_extra_reconstructed),'r--',label='expected',linewidth=line_width)
axs[1,0].plot(t_corr_list,np.real(C.C_s_plus_extra_stochastic),'g--',label='empirical',linewidth=line_width)
axs[1,0].fill_between(t_corr_list, np.array(C.C_s_plus_extra_reconstructed)-np.array(C.C_s_plus_extra_class.expected_error), np.array(C.C_s_plus_extra_reconstructed)+np.array(C.C_s_plus_extra_class.expected_error),alpha=0.4, edgecolor='darkgreen', facecolor='lightgreen')
#axs[1,0].set_xlim(-100,100)
axs[1,0].set_xlabel(r'time [$1/\omega_S$]', fontsize=dim_labels)
#axs[1,0].set_ylabel('$C_{s}(t)$ [$\omega^2_c$]', fontsize=dim_labels)
axs[1,0].set_ylabel(r'$\mathrm{Re}\{C_\mathrm{class}(t)\omega^2_c]\}$', fontsize=dim_labels)
axs[1,0].legend(loc='lower right', prop={'size': dim_legend})
axs[1,0].tick_params(axis='both', which='major', labelsize=dim_ticks)
axs[1,0].set_xlim(-x_lim,x_lim)
fig.align_labels() 
fig.text(.1, .87, r'$(a)$', fontsize=20)
fig.text(.59, .87, r'$(b)$', fontsize=20)
fig.text(.1, .35, r'$(c)$', fontsize=20)
fig.text(.59, .35, r'$(d)$', fontsize=20)

plt.show()