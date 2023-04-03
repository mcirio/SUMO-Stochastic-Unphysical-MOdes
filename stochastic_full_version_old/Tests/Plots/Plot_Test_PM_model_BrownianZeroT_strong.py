import os
import sys
import pickle

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..\..', 'Classes' )
sys.path.append( mymodule_dir )
mymodule_dir = os.path.join( script_dir, '..\..', 'Functions' )
sys.path.append( mymodule_dir )


import numpy as np
import matplotlib.pyplot as plt
from rescale import rescale
from regularize_poles import regularize_poles
from J_poly import J_poly
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

pickle_in = open("./PM_3/Tests/Data/Test_PM_model_BrownianZeroT_strong.dat",'rb')
dict = pickle.load(pickle_in)
pickle_in.close()

print('---------------------------------------')
for k in dict.keys(): print(k)
print('---------------------------------------')
t_corr_list = dict['t_corr_list']
t_list = dict['t_list']
C_as = dict['C_as']
C_s = dict['C_s']
C_as_minus_extra = dict['C_as_minus_extra'] 
C_as_minus_extra_fit = dict['C_as_minus_extra_fit'] 
C_s_plus_extra = dict['C_s_plus_extra'] 
C_s_plus_extra_reconstructed = dict['C_s_plus_extra_reconstructed'] 
C_s_plus_extra_stochastic = dict['C_s_plus_extra_stochastic'] 
dynamics = dict['dynamics'] 
dynamics_full = dict['dynamics_full'] 
dynamics_no_Matsubara = dict['dynamics_no_Matsubara'] 
Omega_S = dict['Omega_S'] 
ll = dict['ll']
Omega = dict['Omega'] 
Gamma = dict['Gamma'] 
check = dict['check']

gamma = 2 * Gamma
vec_p = [0,2 * Gamma * ll**2]
vec_w = [Omega + 1j*Gamma]
vec_w = regularize_poles(vec_w)
J = J_poly(vec_p,vec_w)
w_list = np.linspace(0,3*Omega_S,1000)
J_list = [J(w) for w in w_list]

T_unit = 1 / Omega
C_unit = ll**2 / Omega
W_unit = Omega_S
t_corr_list = rescale(t_corr_list,T_unit)
t_list =rescale(t_list,T_unit)
C_as = rescale(C_as,C_unit)
C_s = rescale(C_s,C_unit)
C_as_minus_extra = rescale(C_as_minus_extra,C_unit)
C_as_minus_extra_fit = rescale(C_as_minus_extra_fit,C_unit)
C_s_plus_extra = rescale(C_s_plus_extra,C_unit)
C_s_plus_extra_reconstructed = rescale(C_s_plus_extra_reconstructed,C_unit)
C_s_plus_extra_stochastic = rescale(C_s_plus_extra_stochastic,C_unit)
check = rescale(check,C_unit)
w_list = rescale(w_list,W_unit)
J_list = rescale(J_list,W_unit)
expected_error = dict['expected_error'] 
sigma_dynamics = dict['sigma_dynamics'] 
dynamics_list = dict['dynamics_list'] 
xi_list = dict['xi_list']



fig, axs = plt.subplots(3, 2,figsize=(16,7))
axs[0,0].set_title('Correlation (real)')
axs[0,0].plot(t_corr_list,np.real(C_as),'b',label='antisymmetric')
axs[0,0].plot(t_corr_list,np.real(C_s),'g',label='symmetric')
axs[0,0].set_xlabel('time [$1/\omega_S$]')
axs[0,0].set_ylabel('$C(t)$ [$\lambda^2/\Omega$]')
#axs[0,0].set_xlim(-100,100)
axs[0,0].set_ylim(-1,1)
axs[0,0].legend(loc='lower right')

axs[0,1].set_title('Correlation (imag)')
axs[0,1].plot(t_corr_list,np.imag(C_as),'b',label='antisymmetric')
axs[0,1].plot(t_corr_list,np.imag(C_s),'g',label='symmetric')
axs[0,1].set_xlabel('time [$1/\omega_S$]')
axs[0,1].set_ylabel('$C(t)$ [$\lambda^2/\Omega$]')
#axs[0,1].set_xlim(-100,100)
axs[0,1].legend(loc='lower right')
######################

axs[1,1].set_title('Antisymmetric - extra (real)')
axs[1,1].plot(t_corr_list,np.real(C_as_minus_extra),'b',label='numerical')
axs[1,1].plot(t_corr_list,np.real(C_as_minus_extra_fit),'r--',label='fit')
axs[1,1].plot(t_corr_list,np.real(check),color='g',linestyle='dotted',label='check real')
axs[1,1].set_xlabel('time [$1/\omega_S$]')
axs[1,1].set_ylabel('$C_{as}(t)$ [$\lambda^2/\Omega$]')
#axs[1,1].set_xlim(-100,100)
axs[1,1].legend(loc='lower right')

axs[2,1].set_title('Antisymmetric - extra (imag)')
axs[2,1].plot(t_corr_list,np.imag(C_as_minus_extra),'k',label='numerical')

axs[2,1].plot(t_corr_list,np.imag(C_as_minus_extra_fit),'g--',label='fit')
axs[2,1].plot(t_corr_list,np.imag(check),color='r',linestyle='dotted',label='check imag')
axs[2,1].set_xlabel('time [$1/\omega_S$]')
axs[2,1].set_ylabel('$C_{as}(t)$ [$\lambda^2/\Omega$]')
#axs[2,1].set_xlim(-100,100)
axs[2,1].legend(loc='lower right')

axs[1,0].set_title('Symmetric + extra (real)')
axs[1,0].plot(t_corr_list,np.real(C_s_plus_extra),label='numerical')
axs[1,0].plot(t_corr_list,np.real(C_s_plus_extra_reconstructed),'r--',label='reconstructed')
axs[1,0].plot(t_corr_list,np.real(C_s_plus_extra_stochastic),'g--',label='stochastic')
#axs[1,0].set_xlim(-100,100)
axs[1,0].set_xlabel('time [$1/\omega_S$]')
axs[1,0].set_ylabel('$C_{s}(t)$ [$\lambda^2/\Omega$]')
axs[1,0].legend(loc='lower right')

axs[2,0].set_title('Symmetric + extra (real)')
axs[2,0].plot(t_corr_list,np.imag(C_s_plus_extra),label='numerical')
axs[2,0].plot(t_corr_list,np.imag(C_s_plus_extra_reconstructed),'r--',label='reconstructed')
axs[2,0].plot(t_corr_list,np.imag(C_s_plus_extra_stochastic),'g--',label='stochastic')
#axs[2,0].set_xlim(-100,100)
axs[2,0].set_ylim(-1.5,1)
axs[2,0].set_xlabel('time [$1/\omega_S$]')
axs[2,0].set_ylabel('$C_{s}(t)$ [$\lambda^2/\Omega$]')
axs[2,0].legend(loc='lower right')

plt.tight_layout()

fig2, axs2 = plt.subplots(1, 1,figsize=(16,7))
#axs2.set_title('Dynamics')
eta = np.sqrt(ll**2/(2*Omega))/Omega_S
axs2.plot(t_list,[2*eta**2/4.-1 for t in t_list],color='darkgrey',linestyle='dashed',linewidth=3)
axs2.plot(t_list,dynamics_full,'b',label='full PM',linewidth=4)
axs2.plot(t_list,dynamics_no_Matsubara,'k--',label="no Matsubara",linewidth=4)
axs2.plot(t_list,dynamics,linestyle='dashed',color='r',linewidth=4,label='stochastic')
axs2.fill_between(t_list, np.array(dynamics)-np.array(sigma_dynamics), np.array(dynamics)+np.array(sigma_dynamics),alpha=0.4, edgecolor='crimson', facecolor='coral')
axs2.set_xlabel('time [$1/\omega_S$]', fontsize=30)
axs2.set_ylabel(r'$\langle\sigma_z\rangle$', fontsize=30)
#axs2.legend(loc='upper right', prop={'size': 20})
axs2.tick_params(axis='both', which='major', labelsize=25)
axs2.tick_params(axis='both', which='minor', labelsize=20)
fig2.text(.118, .9, r'$(b)$', fontsize=30)
axs2.set_ylim(-1.01,-.3)

left, bottom, width, height = [0.586, 0.545, 0.4, 0.4]
axs3 = fig2.add_axes([left, bottom, width, height])

#print(ll**2/(2*Omega))

#fig3, axs3 = plt.subplots(1, 1,figsize=(16,7))
#axs3.set_title('Spectral density', fontsize=30)
axs3.plot(w_list,J_list,'b',linewidth='8')
axs3.axvline(Omega_S/W_unit,color='k',linestyle='--')
# axs3.set_title('Matsubara Correlation', fontsize=30)
# axs3.plot(t_corr_list,np.real(C_s_plus_extra),linewidth=8)
# axs3.set_xlabel('time [$1/\omega_S$]', fontsize=30)
# axs3.set_ylabel('$C_{as}(t)$ [$\lambda^2/\Omega$]', fontsize=30)

axs3.set_xlabel('frequency [$\omega_S$]', fontsize=28)
axs3.set_ylabel('$J(\omega) [(\gamma\lambda^2)^{1/4}]$', fontsize=28)
axs3.tick_params(axis='both', which='major', labelsize=20)
axs3.tick_params(axis='both', which='minor', labelsize=15)
plt.tight_layout(h_pad=0, w_pad=5)

plt.show()

fig_dir = os.path.join( script_dir, 'Figures' )
fig2.savefig(os.path.join(fig_dir, 'brownian_zeroT_strong.pdf'))  

#fig_dir = os.path.join( script_dir, 'Figures' )
#fig2.savefig(os.path.join(fig_dir, 'PM_model_Brownian_zeroT_strong_dynamics.svg'))  
#fig3.savefig(os.path.join(fig_dir, 'PM_model_Brownian_zeroT_strong_spectralDensity.svg'))  
#fig3.savefig(os.path.join(fig_dir, 'Mats.svg'))  