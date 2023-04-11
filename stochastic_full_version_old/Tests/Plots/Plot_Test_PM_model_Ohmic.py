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
from J_ohmic import J_ohmic
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

pickle_in = open("./PM_3/Tests/Data/Test_PM_model_Ohmic.dat",'rb')
dict,th = pickle.load(pickle_in)
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
#dynamics_full = dict['dynamics_full'] 
#dynamics_no_Matsubara = dict['dynamics_no_Matsubara'] 
Omega_S = dict['Omega_S'] 
ll = dict['ll']
wc = dict['wc'] 
expected_error = dict['expected_error'] 
sigma_dynamics = dict['sigma_dynamics'] 
dynamics_list = dict['dynamics_list'] 
xi_list = dict['xi_list']
n_s_noise = dict['n_s_noise'] 
ordered_PM_parameters = dict['ordered_PM_parameters'] 
check = dict['check'] 
check_fit = dict['check_fit']


######################################################################
pickle_in = open("./PM_3/Tests/Data/Test_PM_model_Ohmic_nexp1.dat",'rb')
dict,th = pickle.load(pickle_in)
pickle_in.close()
dynamics_2 = dict['dynamics'] 
C_as_2 = [x for x in rescale(dict['C_as'],wc**2)]
check_fit_2 = [x for x in rescale(dict['check_fit'],wc**2)]
sigma_dynamics_2 = dict['sigma_dynamics']
####################################################################

# res = np.array([0*x for x in dynamics])
# dynamics_list = dynamics_list[0:10]
# for d in dynamics_list:
#     res = res + (np.array(d)-np.array(dynamics))**2
# res = np.sqrt(res / len(dynamics_list)) 
# sigma_dynamics = res

J = J_ohmic(ll,wc)
w_list = np.linspace(0,3*Omega_S,1000)
J_list = [J(w) for w in w_list]

T_unit = 1 / Omega_S
C_unit = wc**2
W_unit = Omega_S
J_unit = wc
t_corr_list = rescale(t_corr_list,T_unit)
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
check_fit = rescale(check_fit,C_unit)

dim_labels = 18
dim_titles = 20
dim_ticks = 18
dim_legend = 13
dim_letters = 15
line_width = 2
x_lim = 5

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
axs[0,0].plot(t_corr_list,np.real(C_s),'b',label='symmetric',linewidth=line_width)
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
axs[0,1].plot(t_corr_list,np.imag(C_as),'b',label='numerical',linewidth=line_width)
#axs[0,1].plot(t_corr_list,np.imag(C_s),'g',label='symmetric')
axs[0,1].plot(t_corr_list,np.imag(-1j*np.array(check_fit)),'r--',label='fit, exp = 2',linewidth=line_width)
axs[0,1].plot(t_corr_list,np.imag(-1j*np.array(check_fit_2)),'g--',label='fit, exp = 1',linewidth=line_width)
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
axs[1,1].plot(t_corr_list,np.real(C_as_minus_extra),'b',label='numerical (real)',linewidth=line_width)
axs[1,1].plot(t_corr_list,np.real(C_as_minus_extra_fit),'r--',label='fit (real',linewidth=line_width)

axs[1,1].plot(t_corr_list,np.imag(C_as_minus_extra),'g',label='numerical (imag)',linewidth=line_width)
axs[1,1].plot(t_corr_list,np.imag(C_as_minus_extra_fit),color='orange',linestyle='dashed',label='fit (imag',linewidth=line_width)
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

axs[1,0].plot(t_corr_list,np.real(C_s_plus_extra),label='numerical',linewidth=line_width)
axs[1,0].plot(t_corr_list,np.real(C_s_plus_extra_reconstructed),'r--',label='expected',linewidth=line_width)
axs[1,0].plot(t_corr_list,np.real(C_s_plus_extra_stochastic),'g--',label='empirical',linewidth=line_width)
axs[1,0].fill_between(t_corr_list, np.array(C_s_plus_extra_reconstructed)-np.array(expected_error), np.array(C_s_plus_extra_reconstructed)+np.array(expected_error),alpha=0.4, edgecolor='darkgreen', facecolor='lightgreen')
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


# axs[2,0].set_title('Symmetric + extra (real)')
# axs[2,0].plot(t_corr_list,np.imag(C_s_plus_extra),label='numerical')
# axs[2,0].plot(t_corr_list,np.imag(C_s_plus_extra_reconstructed),'r--',label='reconstructed')
# axs[2,0].plot(t_corr_list,np.imag(C_s_plus_extra_stochastic),'g--',label='stochastic')
# #axs[2,0].set_xlim(-100,100)
# axs[2,0].set_ylim(-1.5,1)
# axs[2,0].set_xlabel('time [$1/\omega_S$]')
# axs[2,0].set_ylabel('$C_{s}(t)$ [$\omega^2_c$]')
# axs[2,0].legend(loc='lower right')

fig2, axs2 = plt.subplots(1, 1,figsize=(16,7))
#axs2.set_title('Dynamics')
#eta = np.sqrt(ll**2/(2*Omega))/Omega_S
#axs2.plot(t_list,[th for t in t_list],color='darkgrey',linestyle='dashed',linewidth=3)
#axs2.plot(t_list,dynamics_full,'b',label='full PM',linewidth=4)
#axs2.plot(t_list,dynamics_no_Matsubara,'k--',label="no Matsubara",linewidth=4)
axs2.plot(t_list,dynamics,linestyle='dashed',color='r',linewidth=4,label='exp = 2')
axs2.plot(t_list,dynamics_2,linestyle='dashed',color='g',linewidth=4,label='exp = 1')
axs2.fill_between(t_list, np.array(dynamics)-np.array(sigma_dynamics), np.array(dynamics)+np.array(sigma_dynamics),alpha=0.4, edgecolor='crimson', facecolor='coral')
axs2.fill_between(t_list, np.array(dynamics_2)-np.array(sigma_dynamics_2), np.array(dynamics_2)+np.array(sigma_dynamics_2),alpha=0.4, edgecolor='green', facecolor='lightgreen')
axs2.legend(loc='lower left',prop={'size': 1.5*dim_legend})
####################################################
from qutip import *
sigmaz = basis(2,1) * basis(2,1).dag() - basis(2,0) * basis(2,0).dag()
sigmax = basis(2,1) * basis(2,0).dag() + basis(2,0) * basis(2,1).dag()
N_M = 4
sz = tensor(sigmaz,qeye(N_M))
sx = tensor(sigmax,qeye(N_M))
a = tensor(qeye(2),destroy(N_M))

ll = ordered_PM_parameters[0][0]
Omega = ordered_PM_parameters[0][1]
Gamma = ordered_PM_parameters[0][2]


H = Omega_S * sz + Omega * a.dag() * a + np.sqrt(ll) * sx * (a+a.dag())
gs = H.eigenstates()[1][0]
exp = (gs.dag() * sz * gs)[0,0]

######################################################
axs2.plot(t_list,[exp for t in t_list],color='darkgrey',linestyle='dashed',linewidth=3)


#axs2.fill_between(t_list, np.array(dynamics)-np.array(sigma_dynamics), np.array(dynamics)+np.array(sigma_dynamics),alpha=0.4, edgecolor='darkgreen', facecolor='lightgreen')
# axs2.set_xlabel('time [$1/\omega_S$]', fontsize=30)
# axs2.set_ylabel(r'$\langle\sigma_z\rangle$', fontsize=30)
# #axs2.legend(loc='upper right', prop={'size': 20})
# axs2.tick_params(axis='both', which='major', labelsize=25)
# axs2.tick_params(axis='both', which='minor', labelsize=20)

dim_labels = 35
dim_ticks_major = 25
dim_ticks_minor = 20
dim_labels_inset = 28
dim_ticks_major_inset = 20
dim_ticks_minor_inset = 15
axs2.set_xlabel('time [$1/\omega_S$]', fontsize=dim_labels)
axs2.set_ylabel(r'$\langle\sigma_z\rangle$', fontsize=dim_labels)
#axs2.legend(loc='upper right', prop={'size': 20})
# axs2.tick_params(axis='both', which='major', labelsize=25)
# axs2.tick_params(axis='both', which='minor', labelsize=20)
axs2.tick_params(axis='both', which='major', labelsize=dim_ticks_major)
axs2.tick_params(axis='both', which='minor', labelsize=dim_ticks_minor)
axs2.set_ylim(-1.1,1.1)



#fig2.text(.1, .92, r'$(a)$', fontsize=30)

#axs2.set_yticks([-1,-.98,-.96,-.94])
#axs2.set_ylim(-1.0,-.93)



left, bottom, width, height = [0.588, 0.575, 0.4, 0.4]
axs3 = fig2.add_axes([left, bottom, width, height])

#fig3, axs3 = plt.subplots(1, 1,figsize=(16,7))
#axs3.set_title('Spectral density', fontsize=30)
axs3.plot(w_list,J_list,'b',linewidth=6)
axs3.axvline(Omega_S/W_unit,color='k',linestyle='--')
axs3.set_xlabel('frequency [$\omega_S$]', fontsize=dim_labels_inset)
axs3.set_ylabel('$J(\omega) [\omega_c]$', fontsize=dim_labels_inset)
axs3.tick_params(axis='both', which='major', labelsize=dim_ticks_major_inset)
axs3.tick_params(axis='both', which='minor', labelsize=dim_ticks_minor_inset)
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

plt.show()

fig_dir = os.path.join( script_dir, 'Figures' )
fig2.savefig(os.path.join(fig_dir, 'ohmic.pdf'))  
fig.savefig(os.path.join(fig_dir, 'PM_model_Ohmic_corr.pdf'))  
fig2.savefig(os.path.join(fig_dir, 'PM_model_Ohmic_dynamics.pdf'))  
# print(np.sqrt(ll**2/(2*Omega))/Omega_S)

# w0=2*np.pi
# beta = 1 / w0
# from utility_functions import coth
# R = np.real(coth(beta,(Omega+1j*Gamma)/2.))
# I = np.imag(coth(beta,(Omega+1j*Gamma)/2.))
# Omega_res = Omega
# ll_res = np.sqrt(ll**2 / (2 * Omega))
# Gamma_res = Gamma
# n_res = (R - 1) / 2.
# beta_res = np.log(1+1/n_res) / Omega_res

# print(n_res)
# print(np.exp(-3*beta*Omega_res))


print(ordered_PM_parameters)



# C_tot = np.array([x[0] + x[1] for x in zip(C_as_minus_extra_fit , C_s_plus_extra_reconstructed)])
# C_tot = np.array([x[0] + x[1] for x in zip(C_s , C_s)])

# dt = t_corr_list[1] - t_corr_list[0]
# N0 = int((len(t_corr_list)-1)/2.)
# C_tot_m = C_tot[:N0+1]
# C_tot_p = C_tot[:N0+1][::-1]
# # plt.plot(C_tot_p)
# # plt.plot(C_tot_m)

# C_tot_symm = []
# for n,c in enumerate(C_tot_p):
#     C_tot_symm.append(C_tot_p[n]+C_tot_m[-n])
# print(N0)
# # plt.plot(C_tot_symm)
# C_2 = []
# for N in np.arange(1,len(C_tot_symm)):
#     C_2.append(sum(C_tot_symm[1:N+1]) * dt)
# Dec = []
# for N in np.arange(1,len(C_2)):
#     Dec.append(2*sum(C_2[1:N+1]) * dt)

# def integrate_num(dt,y_list):
#     int_list = []
#     res = []
#     for y in y_list[1:]:
#         res.append(y)
#         int_list.append(sum(res)*dt)
#     return  int_list

# C_2 = integrate_num(t_corr_list[1]-t_corr_list[0],C_tot[N0:]+C_tot[:N0+1][::-1])
# C_3 = integrate_num(t_corr_list[1]-t_corr_list[0],C_2)
# plt.plot(C_2)

# new_t_list = []
# res_list = []
# for N,tt in enumerate(t_corr_list[N0+1:]):
#     new_t_list.append(tt-dt/2.)
#     res = 0
#     for n,t in enumerate(t_corr_list[N0+1:N0+1+N]):
#         res += C_tot_symm[n] * dt
#     res_list.append(res)
# res = 0
# for c in res_list[1:]:
#     res += c * dt
# print(res)

res = np.array([ll / np.pi**2 * np.log(1+wc**2 * t**2) for t in t_list])
res2 = np.array([2 * ll * np.log(1+wc**2 * t**2) for t in t_list])
plt.plot(np.exp(-res))
plt.plot(np.exp(-res2))

# plt.plot(C_tot_symm)
# plt.plot(res)
plt.show()


