import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'Functions' )
sys.path.append( mymodule_dir )

import numpy as np
from scipy.optimize import least_squares
from correlation_classical_last import correlation_classical_last
from correlation_classical_interpolation import correlation_classical_interpolation
from integral import integral
from progressbar import progressbar
from utility_functions import coth

class correlation_numerical_dynamics_physical_3():
    def __init__(self,J,W_i,W_f,t_corr_list,beta,n_as_exp,n_s_cut,n_s_noise,integration_limit,interpolation):
        # Parameters
        self.J = J
        self.W_i = W_i
        self.W_f = W_f
        self.t_corr_list = t_corr_list
        self.beta = beta
        self.n_as_exp = n_as_exp
        self.n_s_cut = n_s_cut
        self.n_s_noise = n_s_noise
        self.integration_limit = integration_limit

        # Compute symmetric and antisymmetric part
        self.C_s = self.compute_symmetric()
        self.C_as = self.compute_antisymmetric()

        # Compute fit parameters for the antisymmetric part
        self.PM_parameters = self.generate_PM_parameters()
        self.ordered_PM_parameters = self.order_PM_parameters()

        #  Use the fit parameters to define the extra symmetric correlation
        self.C_as_fit = [self.fit_function(self.PM_parameters,t) for t in self.t_corr_list]
        self.C_s_extra = [self.extra_symmetric(self.PM_parameters,t) for t in self.t_corr_list]

        # Add and subtract that extra symmetric correlation
        self.C_s_plus_extra = [x[0] + x[1] for x in zip(self.C_s,self.C_s_extra)]
        self.C_as_minus_extra = [x[0] - x[1] for x in zip(self.C_as,self.C_s_extra)]
        self.C_as_minus_extra_fit =  [x[0] - x[1] for x in zip(self.C_as_fit,self.C_s_extra)]

        # Take the newly defined symmetric part and compute the classical fields
        if interpolation == 'yes':
            self.C_s_plus_extra_class = correlation_classical_interpolation(self.C_s_plus_extra,self.n_s_cut,self.n_s_noise,self.t_corr_list)
        if interpolation == 'no':
            self.C_s_plus_extra_class = correlation_classical_last(self.C_s_plus_extra,self.n_s_cut,self.n_s_noise,self.t_corr_list)
        self.C_s_plus_extra_reconstructed = self.C_s_plus_extra_class.corr_reconstructed
        self.C_s_plus_extra_stochastic = self.C_s_plus_extra_class.corr_stochastic
        self.xi_list_s_plus_extra = self.C_s_plus_extra_class.xi_interpolated_list

        # Some checks
        self.check = [self.check_fun(t) for t in t_corr_list]
        self.check_fit = [self.check_fit(t) for t in t_corr_list]

    def check_fun(self,t):
        res = 0
        for param in self.ordered_PM_parameters:
            res += param[0]* np.exp(-1j*param[1]*t) * np.exp(-param[2]*abs(t))
        return res
    def check_fit(self,t):
        res = 0
        for param in self.ordered_PM_parameters:
            res += param[0]* np.sin(param[1]*t) * np.exp(-param[2]*abs(t))
        return res
    def to_integrate_symmetric(self,w,t):
        return 1 / (np.pi) * self.J(w) * coth(self.beta,w/2.) * np.cos(w * t)
    def to_integrate_antisymmetric(self,w,t):
        return - 1 / (np.pi) * 1j * self.J(w) * np.sin(w * t)
    def compute_symmetric(self):
        #return [integral(self.to_integrate_symmetric,t,x_i=self.W_i,x_f=self.W_f,limit=self.integration_limit) for t in self.t_corr_list]
        print('Computing symmetric correlations ({length})'.format(length="t_corr_list"))
        res = []
        for t_index in progressbar(np.arange(len(self.t_corr_list))):
            t = self.t_corr_list[t_index]
            res.append(integral(self.to_integrate_symmetric,t,x_i=self.W_i,x_f=self.W_f,limit=self.integration_limit) )
        return res
    def compute_antisymmetric(self):
        #return [integral(self.to_integrate_antisymmetric,t,x_i=self.W_i,x_f=self.W_f,limit=self.integration_limit) for t in self.t_corr_list]
        print('Computing antisymmetric correlations ({length})'.format(length="t_corr_list"))
        res = []
        for t_index in progressbar(np.arange(len(self.t_corr_list))):
            t = self.t_corr_list[t_index]
            res.append(integral(self.to_integrate_antisymmetric,t,x_i=self.W_i,x_f=self.W_f,limit=self.integration_limit) )
        return res
    def fit_penalty(self,params,t,y):
        #if np.mod(len(params),6) != 0: raise Exception('parameters should be multiple of 6')
        #n_exp = int(len(params) / 6.)
        res = 0
        for n in np.arange(0,self.n_as_exp):
            aR = params[6 * n + 0]
            aI = params[6 * n + 1]
            bR = params[6 * n + 2]
            bI = params[6 * n + 3]
            cR = params[6 * n + 4]
            cI = params[6 * n + 5]

            a = aR + 1j * aI
            b = bR + 1j * bI
            c = cR + 1j * cI

            res += a*np.sin(b*(t))*np.exp(-c*abs(t))# +100 * (1-self.sg(np.real(c)))#+ 100*np.real(a)**2#+ 10*np.imag(b)**2 + 10*np.imag(c)**2 #?#
        return y - res 
    def fit_penalty_2(self,params,t,y):
        fx = self.fit_penalty(params,t,y)
        return np.array(list(np.absolute(fx.real))+list(np.absolute(fx.imag)))
    def fit_function(self,params,t):
        res = 0
        for n in np.arange(0,self.n_as_exp):
            aR = params[6 * n + 0]
            aI = params[6 * n + 1]
            bR = params[6 * n + 2]
            bI = params[6 * n + 3]
            cR = params[6 * n + 4]
            cI = params[6 * n + 5]

            a = aR + 1j * aI
            b = bR + 1j * bI
            c = cR + 1j * cI

            res += a*np.sin(b*(t))*np.exp(-c*abs(t))
        return res
    def extra_symmetric(self,params,t):
        res = 0
        for n in np.arange(0,self.n_as_exp):
            aR = params[6 * n + 0]
            aI = params[6 * n + 1]
            bR = params[6 * n + 2]
            bI = params[6 * n + 3]
            cR = params[6 * n + 4]
            cI = params[6 * n + 5]

            a = aR + 1j * aI
            b = bR + 1j * bI
            c = cR + 1j * cI

            res += - 1j * a*np.cos(b*(t))*np.exp(-c*abs(t))
        return res
    def generate_PM_parameters(self):
        p0 = []
        b1 = []
        b2 = []
        for n in np.arange(0,self.n_as_exp):
            p0 = p0 + [0,1,1,0,1,0] # We start from imaginary overal coefficient because we know the correlation is [i * sin()]. All other parameters are set initially real.
            b1 = b1 + [-np.inf,-np.inf,-np.inf,-np.inf,0,-np.inf]#lambda,Omega,Gamma. Bound on Re[Omega] is because a sin(b) has a floating sign between a and b.
            b2 = b2 + [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]
        # b1 = [-np.inf,-np.inf,-np.inf,-np.inf,0,-np.inf]
        # b1.extend([-np.inf,-np.inf,-np.inf,-np.inf,0,-np.inf])
        # b1.extend([-np.inf,-np.inf,-np.inf,-np.inf,0,-np.inf])
        # b1.extend([-np.inf,-np.inf,-np.inf,-np.inf,0,-np.inf])
        # b2 = [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]
        # b2.extend([np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
        # b2.extend([np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
        # b2.extend([np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
        #result = least_squares(self.fit_penalty_2, p0,bounds=(b1,b2),method='dogbox',args=(self.t_corr_list_normalized,self.C_as_normalized))
        result = least_squares(self.fit_penalty_2, p0,bounds=(b1,b2),args=(self.t_corr_list,self.C_as))

        return result.x
    def order_PM_parameters(self):
        ordered_PM_parameters = []
        for n in np.arange(0,self.n_as_exp):
            single_PM_parameters = []
            single_PM_parameters.append(1j *(self.PM_parameters[6*n + 0]+1j*self.PM_parameters[6*n + 1])) # coupling
            single_PM_parameters.append( (self.PM_parameters[6*n + 2]+1j*self.PM_parameters[6*n + 3])) # frequency
            single_PM_parameters.append((self.PM_parameters[6*n + 4] + 1j * self.PM_parameters[6*n + 5])) # decay rate
            ordered_PM_parameters.append(single_PM_parameters)
        return ordered_PM_parameters
 