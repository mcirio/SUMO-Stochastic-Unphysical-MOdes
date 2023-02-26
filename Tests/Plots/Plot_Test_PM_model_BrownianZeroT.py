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
#rcParams['text.usetex'] = True
#rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] 

pickle_in = open("./PM_3/Tests/Data/Test_PM_model_BrownianZeroT.dat",'rb')
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
expected_error = dict['expected_error'] 
sigma_dynamics = dict['sigma_dynamics'] 
dynamics_list = dict['dynamics_list'] 
xi_list = dict['xi_list']
n_s_noise = dict['n_s_noise'] 

# res = np.array([0*x for x in dynamics])
# dynamics_list = dynamics_list[0:10]
# for d in dynamics_list:
#     res = res + (np.array(d)-np.array(dynamics))**2
# res = np.sqrt(res / len(dynamics_list)) 
# sigma_dynamics = res

gamma = 2 * Gamma
vec_p = [0,2 * Gamma * ll**2]
vec_w = [Omega + 1j*Gamma]
vec_w = regularize_poles(vec_w)
J = J_poly(vec_p,vec_w)
w_list = np.linspace(0,3*Omega_S,1000)
J_list = [J(w) for w in w_list]

T_unit = 1 / Omega_S
T_unit_corr = 1 / Omega
C_unit = ll**2 / Omega
W_unit = Omega_S
J_unit = ((gamma * ll**2)**(-1/4.))
t_corr_list = rescale(t_corr_list,T_unit_corr)
t_list =rescale(t_list,T_unit)
C_as = rescale(C_as,C_unit)
C_s = rescale(C_s,C_unit)
C_as_minus_extra = rescale(C_as_minus_extra,C_unit)
C_as_minus_extra_fit = rescale(C_as_minus_extra_fit,C_unit)
C_s_plus_extra = rescale(C_s_plus_extra,C_unit)
C_s_plus_extra_reconstructed = rescale(C_s_plus_extra_reconstructed,C_unit)
C_s_plus_extra_stochastic = rescale(C_s_plus_extra_stochastic,C_unit)
w_list = rescale(w_list,W_unit)
J_list = rescale(J_list,W_unit)
expected_error = rescale(expected_error,C_unit)

fig, axs = plt.subplots(3, 2,figsize=(16,7))
plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.65)
#plt.tight_layout(h_pad=0, w_pad=5)
axs[0,0].set_title('Correlation (real)')
axs[0,0].plot(t_corr_list,np.real(C_as),'b',label='antisymmetric')
axs[0,0].plot(t_corr_list,np.real(C_s),'g',label='symmetric')
axs[0,0].set_xlabel('time [$1/\omega_S$]')
axs[0,0].set_ylabel('$C(t)$ [$\lambda^2/\Omega$]')
axs[0,0].set_xlim(-100,100)
axs[0,0].set_ylim(-1,1)
axs[0,0].legend(loc='lower right')

axs[0,1].set_title('Correlation (imag)')
axs[0,1].plot(t_corr_list,np.imag(C_as),'b',label='antisymmetric')
axs[0,1].plot(t_corr_list,np.imag(C_s),'g',label='symmetric')
axs[0,1].set_xlabel('time [$1/\omega_S$]')
axs[0,1].set_ylabel('$C(t)$ [$\lambda^2/\Omega$]')
axs[0,1].set_xlim(-100,100)
axs[0,1].legend(loc='lower right')
######################

axs[1,1].set_title('Antisymmetric - extra (real)')
axs[1,1].plot(t_corr_list,np.real(C_as_minus_extra),'b',label='numerical')
axs[1,1].plot(t_corr_list,np.real(C_as_minus_extra_fit),'r--',label='fit')
axs[1,1].set_xlabel('time [$1/\omega_S$]')
axs[1,1].set_ylabel('$C_{as}(t)$ [$\lambda^2/\Omega$]')
axs[1,1].set_xlim(-100,100)
axs[1,1].legend(loc='lower right')

axs[2,1].set_title('Antisymmetric - extra (imag)')
axs[2,1].plot(t_corr_list,np.imag(C_as_minus_extra),'k',label='numerical')
axs[2,1].plot(t_corr_list,np.imag(C_as_minus_extra_fit),'g--',label='fit')
axs[2,1].set_xlabel('time [$1/\omega_S$]')
axs[2,1].set_ylabel('$C_{as}(t)$ [$\lambda^2/\Omega$]')
axs[2,1].set_xlim(-100,100)
axs[2,1].legend(loc='lower right')

axs[1,0].set_title('Symmetric + extra (real)')
axs[1,0].plot(t_corr_list,np.real(C_s_plus_extra),label='numerical')
axs[1,0].plot(t_corr_list,np.real(C_s_plus_extra_reconstructed),'r--',label='reconstructed')
axs[1,0].plot(t_corr_list,np.real(C_s_plus_extra_stochastic),'g--',label='stochastic')
axs[1,0].fill_between(t_corr_list, np.array(C_s_plus_extra_reconstructed)-np.array(expected_error), np.array(C_s_plus_extra_reconstructed)+np.array(expected_error),alpha=0.4, edgecolor='darkgreen', facecolor='lightgreen')
axs[1,0].set_xlim(-100,100)
axs[1,0].set_xlabel('time [$1/\omega_S$]')
axs[1,0].set_ylabel('$C_{s}(t)$ [$\lambda^2/\Omega$]')
axs[1,0].legend(loc='lower right')

axs[2,0].set_title('Symmetric + extra (real)')
axs[2,0].plot(t_corr_list,np.imag(C_s_plus_extra),label='numerical')
axs[2,0].plot(t_corr_list,np.imag(C_s_plus_extra_reconstructed),'r--',label='reconstructed')
axs[2,0].plot(t_corr_list,np.imag(C_s_plus_extra_stochastic),'g--',label='stochastic')
axs[2,0].set_xlim(-100,100)
axs[2,0].set_ylim(-1.5,1)
axs[2,0].set_xlabel('time [$1/\omega_S$]')
axs[2,0].set_ylabel('$C_{s}(t)$ [$\lambda^2/\Omega$]')
axs[2,0].legend(loc='lower right')

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



fig2.text(.135, .915, r'$(a)$', fontsize=30)

axs2.set_yticks([-1,-.98,-.96,-.94])
axs2.set_ylim(-1.0,-.93)



left, bottom, width, height = [0.588, 0.575, 0.4, 0.4]
axs3 = fig2.add_axes([left, bottom, width, height])

#fig3, axs3 = plt.subplots(1, 1,figsize=(16,7))
#axs3.set_title('Spectral density', fontsize=30)
axs3.plot(w_list,J_list,'b',linewidth='6')
axs3.axvline(Omega_S/W_unit,color='k',linestyle='--')
axs3.set_xlabel('frequency [$\omega_S$]', fontsize=28)
axs3.set_ylabel('$J(\omega) [(\gamma\lambda^2)^{1/4}]$', fontsize=28)
axs3.tick_params(axis='both', which='major', labelsize=20)
axs3.tick_params(axis='both', which='minor', labelsize=15)
plt.tight_layout(h_pad=0, w_pad=5)
#axs3.legend(loc='upper right')





# nn=1
# ress = []
# #for n in np.arange(0,len(dynamics_list)):
# for n in np.arange(0,1000):
#     res = np.array([0*x for x in dynamics])
#     #res = np.average(res.reshape(-1, nn), axis=1)
#     dynamics_list_temp = dynamics_list[0:n]
#     for d in dynamics_list_temp:
#         temp = np.array(d)**2-np.array(dynamics)
#         #temp = np.average(temp.reshape(-1, nn), axis=1)
#         res = res + abs(temp)
#     res = (res / len(dynamics_list_temp))
#     ress.append(res[100])
# fig4, axs4 = plt.subplots(1, 1,figsize=(16,7))
# axs4.plot(ress)
# res = []
# for n in np.arange(len(dynamics_list)):
#     dynamics_list_temp = dynamics_list[0:n]
#     sigma = 0 * np.array(dynamics_list[0])
#     dynamics_av = 0 * np.array(dynamics_list[0])
#     for d in dynamics_list_temp:
#         sigma += d**2 
#         dynamics_av += d
#     sigma = sigma / len(dynamics_list_temp)
#     dynamics_av = dynamics_av / len(dynamics_list_temp)
#     s = abs(sigma - dynamics_av**2)
#     #s = abs(sigma - dynamics_full**2)
#     res.append(s[10])
# res2 = 0 * np.array(dynamics_list[0])
# for d in dynamics_list:
#     res2 += np.array(d)**2
# res2 = res2 / len(dynamics_list)
# fig4, axs4 = plt.subplots(1, 1,figsize=(16,7))
# # axs4.plot(sigma-dynamics_full**2)
# #axs4.plot(sigma-dynamics**2)
# axs4.plot(np.sqrt(res))
# axs4.plot([np.sqrt(res[1]/(n))  for n,x in enumerate(res)])
# axs4.plot([sigma_dynamics[10] for _ in res])

# axs4.plot(abs(res2-dynamics_full**2))


# n_noise = len(xi_list)
# res = 0 * np.array(t_corr_list)
# t0_index = int((len(t_corr_list)-1)/2.)
# for xi in xi_list[:10]:
#     res = res + np.array([xi[t0_index] for value in xi])
# res / len(xi_list[:10])
# print(xi_list[0][t0_index])
# fig4, axs4 = plt.subplots(2, 1,figsize=(16,7))
# axs4[0].plot(res)
# axs4[1].plot(xi_list[0])
# axs4[1].plot(xi_list[1])

#for x in dynamics_list:
#    axs4.plot(x)
# axs4.plot(dynamics,'k')
#axs4.plot([x[1] for x in dynamics_list])
#axs4.set_ylim(.98,1)
#axs4.set_xlim(0,2)

dim_labels = 18
dim_titles = 20
dim_ticks = 18
dim_legend = 15
dim_letters = 15
line_width = 2

fig3, axs3 = plt.subplots(2, 2,figsize=(16,7))
plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.65)
#plt.tight_layout(h_pad=0, w_pad=5)
axs3[0,0].set_title('Symmetric Correlation',fontsize=dim_titles)
#axs3[0,0].plot(t_corr_list,np.real(C_as),'b',label='antisymmetric')
axs3[0,0].plot(t_corr_list,np.real(C_s),'b',label='symmetric',linewidth=line_width)
axs3[0,0].axvline(0,color='k',linestyle='dashed')
axs3[0,0].set_xlabel(r'time$[1/\Omega]$', fontsize=dim_labels)
axs3[0,0].set_ylabel(r'$\mathrm{Re}\{C_\mathrm{s}(t)[\lambda^2/\Omega]\}$', fontsize=dim_labels)
axs3[0,0].set_xlim(-75,75)
axs3[0,0].tick_params(axis='both', which='major', labelsize=dim_ticks)
#axs3[0,0].set_ylim(-1,1)
#axs3[0,0].legend(loc='lower right')

axs3[0,1].set_title('Antisymmetric Correlation',fontsize=dim_titles)
axs3[0,1].plot(t_corr_list,np.imag(C_as),color='orange',label='antisymmetric',linewidth=line_width)
axs3[0,1].axvline(0,color='k',linestyle='dashed')
#axs3[0,1].plot(t_corr_list,np.imag(C_s),'g',label='symmetric')
axs3[0,1].set_ylabel(r'$\mathrm{Im}\{C_\mathrm{as}(t)[\lambda^2/\Omega]\}$', fontsize=dim_labels)
axs3[0,1].set_xlabel(r'time$[1/\Omega]$', fontsize=dim_labels)
axs3[0,1].set_xlim(-75,75)
axs3[0,1].tick_params(axis='both', which='major', labelsize=dim_ticks)
#axs3[0,1].legend(loc='lower right')
######################

axs3[1,1].set_title('Quantum Correlation',fontsize=dim_titles)
axs3[1,1].plot(t_corr_list,np.real(C_as_minus_extra),'b',label=r'$\mathrm{Re}[C_\mathrm{Q}(t)]$',linewidth=line_width)
axs3[1,1].plot(t_corr_list,np.imag(C_as_minus_extra),color='darkorange',label=r'$\mathrm{Im}[C_\mathrm{Q}(t)]$',linewidth=line_width)
axs3[1,1].axvline(0,color='k',linestyle='dashed')
#axs3[1,1].plot(t_corr_list,np.real(C_as_minus_extra_fit),'r--',label='fit')
axs3[1,1].set_xlabel(r'time [$1/\Omega$]', fontsize=dim_labels)
axs3[1,1].set_ylabel(r'$C_\mathrm{Q}(t)[\lambda^2/\Omega]$', fontsize=dim_labels)
axs3[1,1].set_xlim(-75,75)
axs3[1,1].legend(loc='lower right', prop={'size': dim_legend})
axs3[1,1].tick_params(axis='both', which='major', labelsize=dim_ticks)


axs3[1,0].set_title('Classical Correlation',fontsize=dim_titles)
axs3[1,0].plot(t_corr_list,np.real(C_s_plus_extra),label='numerical',linewidth=line_width)
axs3[1,0].plot(t_corr_list,np.real(C_s_plus_extra_reconstructed),'r--',label='expected',linewidth=line_width)
axs3[1,0].plot(t_corr_list,np.real(C_s_plus_extra_stochastic),'g--',label='empirical',linewidth=line_width)
axs3[1,0].fill_between(t_corr_list, np.array(C_s_plus_extra_reconstructed)-np.array(expected_error), np.array(C_s_plus_extra_reconstructed)+np.array(expected_error),alpha=0.4, edgecolor='darkgreen', facecolor='lightgreen')
axs3[1,0].axvline(0,color='k',linestyle='dashed')
axs3[1,0].set_xlim(-75,75)
axs3[1,0].set_xlabel(r'time [$1/\Omega$]', fontsize=dim_labels)
axs3[1,0].set_ylabel(r'$\mathrm{Re}\{C_\mathrm{class}(t)[\lambda^2/\Omega]\}$', fontsize=dim_labels)
axs3[1,0].legend(loc='lower right', prop={'size': dim_legend})
axs3[1,0].tick_params(axis='both', which='major', labelsize=dim_ticks)

fig3.text(.103, .905, r'$(a)$', fontsize=dim_letters)
fig3.text(.585, .905, r'$(b)$', fontsize=dim_letters)
fig3.text(.103, .415, r'$(c)$', fontsize=dim_letters)
fig3.text(.585, .415, r'$(d)$', fontsize=dim_letters)
fig3.align_labels()
# fig4, axs4 = plt.subplots(1, 1,figsize=(16,7))
# axs4.plot(t_list,xi_list[20])

plt.show()

fig_dir = os.path.join( script_dir, 'Figures' )
fig2.savefig(os.path.join(fig_dir, 'brownian_zeroT_weak.pdf'))  
# # fig2.savefig(os.path.join(fig_dir, 'PM_model_Brownian_zeroT_dynamics.svg'))  
fig3.savefig(os.path.join(fig_dir, 'PM_model_Brownian_zeroT_correlations.pdf'))  
# print(np.sqrt(ll**2/(2*Omega))/Omega_S)


