import os
import sys
import pickle

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..','Classes' )
print(mymodule_dir)
sys.path.append( mymodule_dir )
mymodule_dir = os.path.join( script_dir, '..','Functions' )
sys.path.append( mymodule_dir )

import numpy as np
import matplotlib.pyplot as plt

pickle_in = open("Git\SUMO-Stochastic-Unphysical-MOdes\Stochastic_minimal\Main\Data\data.dat",'rb')
saved_dict = pickle.load(pickle_in)
pickle_in.close()


units_freq = saved_dict['units_freq'] 
units_time = saved_dict['units_time'] 
units_corr = saved_dict['units_corr'] 
t_corr_list = saved_dict['t_corr_list'] 
t_list = saved_dict['t_list'] 
C_s = saved_dict['C_s'] 
C_as = saved_dict['C_as'] 
ordered_PM_parameters = saved_dict['ordered_PM_parameters'] 
C_as_fit = saved_dict['C_as_fit']
C_s_extra = saved_dict['C_s_extra'] 
C_s_plus_extra = saved_dict['C_s_plus_extra'] 
C_as_minus_extra = saved_dict['C_as_minus_extra'] 
coeff_list = saved_dict['coeff_list'] 
C_s_reconstructed = saved_dict['C_s_reconstructed'] 
C_s_reconstructed_stochastic = saved_dict['C_s_reconstructed_stochastic'] 
dynamics_average = saved_dict['dynamics_average'] 
sigma = saved_dict['sigma'] 
dynamics_list = saved_dict['dynamics_list'] 
dynamics_PM_full = saved_dict['dynamics_PM_full']
dynamics_PM_noMats = saved_dict['dynamics_PM_noMats'] 

t_corr_list_rescaled = [t / units_time for t in t_corr_list]
t_list_rescaled = [t / units_time for t in t_list]
C_as_rescaled = [x / units_corr for x in C_as]
C_as_fit_rescaled = [x / units_corr for x in C_as_fit]
C_s_plus_extra_rescaled = [x / units_corr for x in C_s_plus_extra]
C_s_reconstructed_rescaled = [x / units_corr for x in C_s_reconstructed]
C_s_reconstructed_stochastic_rescaled = [x / units_corr for x in C_s_reconstructed_stochastic]

dim_labels = 20
dim_legends = 10
dim_titles = 20
dim_ticks_major = 15
dim_ticks_minor = 10
dim_linewidth = 2
# dim_labels_inset = 28
# dim_ticks_major_inset = 20
# dim_ticks_minor_inset = 15


fig, axs = plt.subplots(1, 2,figsize=(16,6))
plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.65)

axs[0].set_title('Antisymmetric Correlation (imaginary part)',fontsize=dim_titles)
axs[0].plot(t_corr_list_rescaled,np.imag(C_as_rescaled),'b',linewidth=dim_linewidth,label='numerical')
# plt.plot([J(w) for w in w_list])
axs[0].plot(t_corr_list_rescaled,np.imag(C_as_fit_rescaled),'r--',linewidth=dim_linewidth,label='fit')
axs[0].legend(fontsize=dim_legends,loc='lower right')
axs[0].set_xlabel('time [$1/\omega_S$]', fontsize=dim_labels)
axs[0].set_ylabel(r'$\langle\sigma_z\rangle$', fontsize=dim_labels)

axs[1].set_title('Quantum Correlation (real part)',fontsize=dim_titles)
axs[1].plot(t_corr_list_rescaled,C_s_plus_extra_rescaled,'b',linewidth=dim_linewidth,label='numerical')
axs[1].plot(t_corr_list_rescaled,C_s_reconstructed_rescaled,'r--',linewidth=dim_linewidth,label='reconstructed')
axs[1].plot(t_corr_list_rescaled,C_s_reconstructed_stochastic_rescaled,'g--',linewidth=dim_linewidth,label='stochastic')
axs[1].legend(fontsize=dim_legends,loc='lower right')
axs[1].set_xlabel('time [$1/\omega_S$]', fontsize=dim_labels)
axs[1].set_ylabel(r'$C(t)[\lambda^2/2\Omega]$', fontsize=dim_labels)
plt.show()

fig2, axs2 = plt.subplots(1, 1,figsize=(16,9))
fig2.suptitle('Dynamics',fontsize=dim_titles)
#axs2.set_title('Dynamics')
# eta = np.sqrt(ll**2/(2*Omega))/Omega_S
# axs2.plot(t_list,[2*eta**2/4.-1 for t in t_list],color='darkgrey',linestyle='dashed',linewidth=3)
# axs2.plot(t_list,dynamics_full,'b',label='full PM',linewidth=4)
# axs2.plot(t_list,dynamics_no_Matsubara,'k--',label="no Matsubara",linewidth=4)
axs2.plot(t_list_rescaled,dynamics_PM_full,linestyle='solid',color='b',linewidth=2*dim_linewidth,label='full PM')
axs2.plot(t_list_rescaled,dynamics_PM_noMats,linestyle='dashed',color='k',linewidth=dim_linewidth,label='PM (no Mats)')
axs2.plot(t_list_rescaled,dynamics_average,linestyle='dashed',color='r',linewidth=dim_linewidth,label='stochastic')
axs2.fill_between(t_list_rescaled, np.array(dynamics_average)-np.array(sigma), np.array(dynamics_average)+np.array(sigma),alpha=0.4, edgecolor='crimson', facecolor='coral')
axs2.set_xlabel('time [$1/\omega_S$]', fontsize=dim_labels)
axs2.set_ylabel(r'$\langle\sigma_z\rangle$', fontsize=dim_labels)
#axs2.legend(loc='upper right', prop={'size': 20})
axs2.tick_params(axis='both', which='major', labelsize=dim_ticks_major)
axs2.tick_params(axis='both', which='minor', labelsize=dim_ticks_minor)
axs2.legend(fontsize=dim_legends,loc='upper right')
axs2.set_ylim(-1,-.98)
plt.show()