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
from PM_model import PM_model
from matsubara import matsubara_2
from superoperator_functions import Hamiltonian_single
from superoperator_functions import Lindblad_single
from superoperator_functions import create_PM_single

# Frequency unit
w0 = 2 * np.pi 
# Cut offs
W_mats = 10 * w0
N_mats = 1000
N_corr = 1000#1000
N_cut = 1000
N_M = 2
N_R = 2
W_free = 100 * w0
W_cut = 10 * w0
n_cut = 500
mats_cut = 500
n_noise = 100
n_as_exp = 1
n_s_cut = 1000 #500
#n_s_cut = 1000
#N_corr = round(n_s_cut * np.pi) #dt=T/N_corr should be comparable (smaller) with 1/max freq=T/(pi n_s_cut). If not, the numerical integral to compute the spectral decomposition might get wrong.
n_s_noise = 100#1000
W_i = 0
W_f = 10 * w0
integration_limit = 500
# Parameters
w_tilde = W_mats
gamma = .05 * w0
Gamma = gamma / 2.
Omega = np.sqrt(w0**2 - Gamma**2)
ll = .2 * w0 #0.45 * w0 #* np.sqrt(2*Omega)
T = 60. * 2 * np.pi / w0
T_corr = T 
time_stamp = 0
norm = 1.
vec_p = [0,2 * Gamma * ll**2]
vec_w = [Omega + 1j*Gamma]
vec_w = regularize_poles(vec_w)
J = J_poly(vec_p,vec_w)
beta = 'inf'
# Lists
t_list = np.linspace(0,T,500)
t_list_dynamics = t_list
t_list_scaled = [x * w0 / (2*np.pi) for x in t_list] #Time in units of 2\pi/w0
t_list_corr = np.linspace(-T_corr,T_corr,2*N_corr+1)
t_corr_list = np.linspace(-T_corr,T_corr,2*N_corr+1)
t_list_interp = np.linspace(-1.1*T,1.1*T,5000)
w_list = np.linspace(0,3*w0,100)
W_list = np.linspace(W_i,W_f,200)
## Operators 
N_S = 2
N_PM = 6
sigmaz = basis(2,1) * basis(2,1).dag() - basis(2,0) * basis(2,0).dag()
sigmax = basis(2,1) * basis(2,0).dag() + basis(2,0) * basis(2,1).dag()
H_S = w0 / 2. * sigmaz
s = sigmax
psi0_S = basis(2,1) * basis(2,1).dag()
obs_list = [sigmaz]

###################################################
interpolation = 'yes'
PM = PM_model(J,W_i,W_f,t_corr_list,beta,n_as_exp,n_s_cut,n_s_noise,H_S,s,N_S,N_PM,psi0_S,t_list,obs_list,integration_limit,interpolation)
###################################################
## PM model
#Fit
N_par = 4
def bi_exponential(x, a1, b1, a2,b2):
     return a1*np.exp(-b1*abs(x)) + a2*np.exp(-b2*abs(x))
mats = matsubara_2(beta,Omega,gamma,ll,N_mats,W_mats,integration_limit,t_corr_list)
popt, pcov = curve_fit(bi_exponential, t_corr_list, np.abs(np.real(mats.mats)),maxfev=1000,bounds=([0 for _ in np.arange(0,N_par)], [100 for _ in np.arange(0,N_par)]))
freq_array = np.array([popt[1],popt[3]])
coeff_array = np.array([popt[0],popt[2]])

#print(freq_array)
#print(coeff_array)

Omega_res = Omega
ll_res = np.sqrt(ll**2 / (2 * Omega))
Gamma_res = Gamma
n_res = 0
Omega_m1 = 0
Omega_m2 = 0
ll_m1 = 1j * np.sqrt(abs(coeff_array[0]))
ll_m2 = 1j * np.sqrt(abs(coeff_array[1]))
Gamma_m1 = freq_array[0]
Gamma_m2 = freq_array[1]
n_m1 = 0
n_m2 =0

# Operators
sigmaz = basis(2,1) * basis(2,1).dag() - basis(2,0) * basis(2,0).dag()
sigmax = basis(2,1) * basis(2,0).dag() + basis(2,0) * basis(2,1).dag()

a = tensor(qeye(2),destroy(N_R),qeye(N_M),qeye(N_M))
b_m1 = tensor(qeye(2),qeye(N_R),destroy(N_M),qeye(N_M))
b_m2 = tensor(qeye(2),qeye(N_R),qeye(N_M),destroy(N_M))
sz = tensor(sigmaz,qeye(N_R),qeye(N_M),qeye(N_M))
sx = tensor(sigmax,qeye(N_R),qeye(N_M),qeye(N_M))

# Lindblad, observables and initial state
c_list = []
obs_list = [sz]
psi0 = tensor(basis(2,1),basis(N_R,0),basis(N_M,0),basis(N_M,0))

H_S = w0 / 2. * sz 
L = Hamiltonian_single(H_S)
L = L + create_PM_single(sx,a,Omega_res,ll_res,Gamma_res,n_res)
L = L + create_PM_single(sx,b_m1,Omega_m1,ll_m1,Gamma_m1,n_m1)
L = L + create_PM_single(sx,b_m2,Omega_m2,ll_m2,Gamma_m2,n_m2)

#Matusbara Correlations
#mats = [Matsubara_zeroT(Omega,gamma,ll,t,W_mats) for t in t_list]

#Dynamics
args={}
dynamics_full = mesolve(L, psi0, t_list, c_list, obs_list,args=args).expect[0]
print('Finished PM model')
##################################################################
## No Matsubara

# Operators

a = tensor(qeye(2),destroy(N_R))
sz = tensor(sigmaz,qeye(N_R))
sx = tensor(sigmax,qeye(N_R))

# Lindblad, observables and initial state
c_list = []
obs_list = [sz]
psi0 = tensor(basis(2,1),basis(N_R,0))

# Hamiltonian
Omega_S = w0
H_S = Omega_S / 2. * sz 
L = Hamiltonian_single(H_S)
L = L + create_PM_single(sx,a,Omega_res,ll_res,Gamma_res,n_res)

# Dynamics
args={}
dynamics_no_Matsubara = mesolve(L, psi0, t_list, c_list, obs_list,args=args).expect[0]
print('Finished PM (no Matsubara) model')
##################################################################
pickle_out = open("./PM_3/Tests/Data/Test_PM_model_BrownianZeroT.dat",'wb')
dict = {}
dict = PM.save(dict)
dict['dynamics_full'] = dynamics_full
dict['dynamics_no_Matsubara'] = dynamics_no_Matsubara
dict['Omega_S'] = Omega_S
dict['ll'] = ll
dict['Omega'] = Omega
dict['Gamma'] = Gamma
pickle.dump(dict,pickle_out)
pickle_out.close()
##################################################################
fig, axs = plt.subplots(3, 1,figsize=(17,8))
axs[0].set_title('Antisymmetric - extra')
axs[0].plot(t_corr_list,np.real(PM.C.C_as_minus_extra),'b',label='antisymmetric (T=0)-corrected real')
axs[0].plot(t_corr_list,np.imag(PM.C.C_as_minus_extra),'k',label='antisymmetric (T=0)-corrected imag')

axs[0].plot(t_corr_list,np.real(PM.C.C_as_minus_extra_fit),'r--',label='fit antisymmetric (T=0)-corrected real')
axs[0].plot(t_corr_list,np.imag(PM.C.C_as_minus_extra_fit),'g--',label='fit antisymmetric (T=0)-corrected imag')
axs[0].legend()

axs[0].set_title('Symmetric + extra')
axs[1].plot(t_corr_list,PM.C.C_s_plus_extra,label='symmetric (T=0)-corrected')
axs[1].plot(t_corr_list,PM.C.C_s_plus_extra_reconstructed,'r--',label='reconstructed')
axs[1].plot(t_corr_list,PM.C.C_s_plus_extra_stochastic,'g--',label='reconstructed')
axs[1].legend()

axs[2].set_title('Dynamics')
axs[2].plot(t_list,dynamics_full,'b',label='full PM')
axs[2].plot(t_list,dynamics_no_Matsubara,'k',label="no Matsubara")
#axs.plot(t_list,dynamics_average,'r--',label='Stochastic')
axs[2].plot(t_list,PM.dynamics,'r--',label='stochastic')
axs[2].legend()
axs[2].set_ylim(-1.01,-.93)

plt.show()