#Python libraries

import os
import logging
import requests 
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.interpolate import interp1d
import scipy.optimize as optimize

#GWFish libraries

import GWFish as gw 
import GWFish.modules.constants as cst
import GWFish.modules.auxiliary as aux
import GWFish.modules.fft as fft
import GWFish.modules.waveforms as wf
from GWFish.modules.waveforms import Waveform


################################################################################
################################ TAYLORF2_PPE ##################################
########################## with spin corrections ###############################

class TaylorF2_PPE(Waveform):

    """ GWFish implementation of TaylorF2_PPE """
    def __init__(self, name, gw_params, data_params):
        super().__init__(name, gw_params, data_params)
        self._maxn = None
        self.psi = None
        if self.name != 'TaylorF2_PPE':
            logging.warning('Different waveform name passed to TaylorF2_PPE: '+ self.name)

    @property
    def maxn(self):
        if self._maxn is None:
            if 'maxn' in self.data_params:
                self._maxn = self.data_params['maxn']
            else:
                self._maxn = 8
            if type(self._maxn) is not int:
                return ValueError('maxn must be integer')
        return self._maxn


    def calculate_phase(self): 

        M, mu, Mc, delta_mass, eta, eta2, eta3, chi_eff, chi_PN, chi_s, chi_a, C, ff = wf.Waveform.get_param_comb(self)

        f_isco = aux.fisco(self.gw_params)  #inner stable circular orbit 

        ones = np.ones((len(ff), 1)) 
        
        #PPE phase parameters

        PN = self.gw_params['PN']
        beta = self.gw_params['beta']
        
        #gIMR phase parameters
        delta_phi_0 = self.gw_params['delta_phi_0']
        delta_phi_1 = self.gw_params['delta_phi_1']
        delta_phi_2 = self.gw_params['delta_phi_2']
        delta_phi_3 = self.gw_params['delta_phi_3']
        delta_phi_4 = self.gw_params['delta_phi_4']
        delta_phi_5 = self.gw_params['delta_phi_5']
        delta_phi_6 = self.gw_params['delta_phi_6']
        delta_phi_7 = self.gw_params['delta_phi_7']
        delta_phi_8 = self.gw_params['delta_phi_8']
        delta_phi_9 = self.gw_params['delta_phi_9']

        #f_cut = cut_order * f_isco
        cut = self.gw_params['cut']


        ################################################################################ 
        ############################## PHASE CORRECTIONS ###############################
        ############################# PN expansion of phase ############################

        # We have to add delta_phi_ppe as in gIMRPhenomD (arXiv:1603.08955)
        # phi ---> phi*(1+delta_phi_i)
        # phi is a combination of phi_i, i=0,....,7 and i=2PN
        # We want to modify phi for each b one by one and b = i-5 

        psi_TF2, psi_TF2_prime, psi_TF2_f1, psi_TF2_prime_f1 = wf.TaylorF2.calculate_phase(self)

        phi_0 = 1.
        phi_1 = 0.
        phi_2 = 3715./756. + 55./9.*eta
        phi_3 = -16.*np.pi + 113./3.*delta_mass*chi_a + (113./3. - 76./3.*eta)*chi_s
        phi_4 = 15293365./508032. + 27145./504.*eta + 3085./72.*eta2 + (-(405./8.) +\
                200*eta)*chi_a**2 - 405./4.*delta_mass*chi_a*chi_s + (-(405./8.) + 5./2.*eta)*chi_s**2
        phi_5 = 38645./756.*np.pi - 65./9.*np.pi*eta + delta_mass*(-(732985./2268.) -\
                140./9.*eta)*chi_a + (-(732985./2268.) + 24260./81.*eta + 340./9.*eta2)*chi_s
        phi_5_l = 3.*phi_5
        phi_6 = 11583231236531./4694215680. - 6848./21.*C - (640.*np.pi**2)/3. +\
                (-15737765635./3048192. + 2255.*np.pi**2/12.)*eta + 76055.*eta2/1728. - 127825.*eta3/1296. +\
                2270./3.*np.pi*delta_mass*chi_a + (2270.*np.pi/3. - 520.*np.pi*eta)*chi_s -6848./63.*np.log(64.)
        phi_6_l = - 6848./63.*3.
        phi_7 = (77096675./254016. + 378515./1512.*eta - 74045./756.*eta2)*np.pi +\
                delta_mass*(-(25150083775./3048192.) + 26804935./6048.*eta - 1985./48.*eta2)*chi_a +\
                (-(25150083775./3048192.) + 10566655595./762048.*eta - 1042165./3024.*eta2 + 5345./36.*eta3)*chi_s

        psi_gIMR = 3./(128.*eta)*(delta_phi_0*(np.pi*ff)**(-5./3.) +\
                delta_phi_1*(np.pi*ff)**(-4./3.)+\
                phi_2*delta_phi_2*(np.pi*ff)**(-1.) +\
                phi_3*delta_phi_3*(np.pi*ff)**(-2./3.) +\
                phi_4*delta_phi_4*(np.pi*ff)**(-1./3.) +\
                phi_5*delta_phi_5 +\
                phi_6*delta_phi_6*(np.pi*ff)**(1./3.) +\
                phi_7*delta_phi_7*(np.pi*ff)**(2./3.)) 
        
        psi_ppe = beta*((np.pi*(ff * cst.c**3/(cst.G*M))*Mc)**((2*PN-5.)/3.))  #ppe correction at every b order

        psi_EI = psi_TF2 + psi_ppe + psi_gIMR

        #Depending on the choice on gw_params you recover psi_tot = psi_TF2 + psi_ppe
                                                         #psi_tot = psi_TF2 + psi_gIMR

        ################################################################################ 
        # Evaluate PHASE and DERIVATIVE at the INTERFACE between ins and int >>>>>>>>>>>
        ################################################################################ 

        f1 = 0.018

        psi_gIMR_f1 = 3./(128.*eta)*(delta_phi_0*(np.pi*f1)**(-5./3.) +\
                    delta_phi_1*(np.pi*f1)**(-4./3.)+\
                    phi_2*delta_phi_2*(np.pi*f1)**(-1.) +\
                    phi_3*delta_phi_3*(np.pi*f1)**(-2./3.) +\
                    phi_4*delta_phi_4*(np.pi*f1)**(-1./3.) +\
                    phi_5*delta_phi_5 + phi_5_l*delta_phi_8*np.log(np.pi*f1) +\
                    (phi_6*delta_phi_6*+ phi_6_l*delta_phi_9*np.log(np.pi*f1))*((np.pi*f1)**(1./3.)) +\
                    phi_7*delta_phi_7*(np.pi*f1)**(2./3.))
                
        psi_ppe_f1 = beta*((np.pi*(f1/(cst.G*M/cst.c**3)*Mc))**((2*PN-5.)/3.))

        psi_EI_f1 = psi_TF2_f1 + psi_ppe_f1 + psi_gIMR_f1
        

        psi_gIMR_prime = 3./(128.*eta)*((np.pi)**(-5./3.)*(-5./3.*ff**(-8./3.)) +\
                        delta_phi_1*(np.pi)**(-4./3.)*(-4./3.*ff**(-7./3.)) +\
                        phi_2*delta_phi_2*(np.pi)**(-1.)*(-1.*ff**(-2.)) +\
                        phi_3*delta_phi_3*(np.pi)**(-2./3.)*(-2./3.*ff**(-5./3.)) +\
                        phi_4*delta_phi_4*(np.pi)**(-1./3.)*(-1./3.*ff**(-4./3.)) +\
                        phi_6*delta_phi_6*(np.pi)**(1./3.)*(1./3.*ff**(-2./3.)) +\
                        phi_6_l*delta_phi_9*((np.pi*ff)**(1./3.)*(np.pi*ff**(-1.)) +\
                                             np.log(np.pi*ff)*(np.pi)**(1./3.)*(1./3.*ff**(-2./3.)) +\
                        phi_7*delta_phi_7*(np.pi)**(2./3.)*(2./3.*ff**(-1./3.))))

        psi_gIMR_prime_f1 = 3./(128.*eta)*((np.pi)**(-5./3.)*(-5./3.*f1**(-8./3.)) +\
                        delta_phi_1*(np.pi)**(-4./3.)*(-4./3.*f1**(-7./3.)) +\
                        phi_2*delta_phi_2*(np.pi)**(-1.)*(-1.*f1**(-2.)) +\
                        phi_3*delta_phi_3*(np.pi)**(-2./3.)*(-2./3.*f1**(-5./3.)) +\
                        phi_4*delta_phi_4*(np.pi)**(-1./3.)*(-1./3.*f1**(-4./3.)) +\
                        phi_5_l*delta_phi_8*(np.pi)*f1**(-1.) +\
                        phi_6*delta_phi_6*(np.pi)**(1./3.)*(1./3.*f1**(-2./3.)) +\
                        phi_6_l*delta_phi_9*((np.pi*f1)**(1./3.)*(np.pi*ff**(-1.)) +\
                                             np.log(np.pi*f1)*(np.pi)**(1./3.)*(1./3.*f1**(-2./3.))) +\
                        phi_7*delta_phi_7*(np.pi)**(2./3.)*(2./3.*f1**(-1./3.)))

        psi_ppe_prime = beta*(2*PN-5.)/3.*((np.pi*(ff/(cst.G*M/cst.c**3))*Mc)**((2*PN-8.)/3.))

        psi_ppe_prime_f1 = beta*(2*PN-5.)/3.*((np.pi*(f1/(cst.G*M/cst.c**3))*Mc)**((2*PN-8.)/3.))
        
        psi_EI_prime = psi_TF2_prime + psi_gIMR_prime + psi_ppe_prime
        psi_EI_prime_f1 = psi_TF2_prime_f1 + psi_gIMR_prime_f1 + psi_ppe_prime_f1

        return psi_EI, psi_EI_prime, psi_EI_f1, psi_EI_prime_f1

    def calculate_frequency_domain_strain(self):

        M, mu, Mc, delta_mass, eta, eta2, eta3, chi_eff, chi_PN, chi_s, chi_a, C, ff = wf.Waveform.get_param_comb(self)

        cut = self.gw_params['cut']
        f_isco = aux.fisco(self.gw_params)

        psi, psi_prime, psi_f1, psi_prime_f1 = TaylorF2_PPE.calculate_phase(self)

        hp, hc = wf.TaylorF2.calculate_amplitude(self)
        ############################### PHASE OUTPUT ###############################

        phase = np.exp(1.j * psi)

        ############################## STRAIN OUTPUT ###############################
        
        polarizations = np.hstack((hp * phase, hc * 1.j * phase))

        # Very crude high-f cut-off which can be an input parameter 'cut', default = 4*f_isco
        f_cut = cut*f_isco*cst.G*M/cst.c**3
 
        polarizations[np.where(ff[:,0] > f_cut), :] = 0.j

        self._frequency_domain_strain = polarizations
        
    ################################################################################
    ############################# Amplitude & phase plot ###########################
    ################################################################################
        
    def plot (self, output_folder='./'):

        psi, psi_prime, psi_f1, psi_prime_f1 = TaylorF2_PPE.calculate_phase(self)
        
        plt.figure()
        plt.loglog(self.frequencyvector, \
                   np.abs(self.frequency_domain_strain[:, 0]), label=r'$h_+$')
        plt.loglog(self.frequencyvector, \
                   np.abs(self.frequency_domain_strain[:, 1]), label=r'$h_\times$')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel(r'Fourier amplitude [$Hz^{-1}$]')
        plt.grid(which='both', color='lightgray', alpha=0.5, linestyle='dashed', linewidth=0.5)
        #plt.axis(axis)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_folder + 'amp_tot_TF2_PPE.png')
        plt.close()

        plt.figure()
        plt.semilogx(self.frequencyvector, psi)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Phase [rad]')
        plt.grid(which='both', color='lightgray', alpha=0.5, linestyle='dashed', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(output_folder + 'phase_tot_TF2_PPE.png')
        plt.close()


################################################################################

################################################################################
############################## IMRPhenomD_PPE ##################################
################################################################################

class IMRPhenomD_PPE(Waveform):
    
    """ GWFish implementation of IMRPhenomD_PPE """
    def __init__(self, name, gw_params, data_params):
        super().__init__(name, gw_params, data_params)
        self._maxn = None
        self.psi = None
        if self.name != 'IMRPhenomD_PPE':
            logging.warning('Different waveform name passed to IMRPhenomD_PPE: '+\
                             self.name)


    def calculate_phase(self): 

        M, mu, Mc, delta_mass, eta, eta2, eta3, chi_eff, chi_PN, chi_s, chi_a, C, ff = wf.Waveform.get_param_comb(self)

        f_isco = aux.fisco(self.gw_params)  #inner stable circular orbit 

        ones = np.ones((len(ff), 1)) 

        #PPE phase parameters

        PN = self.gw_params['PN']
        beta = self.gw_params['beta']
        
        #gIMR phase parameters
        delta_phi_0 = self.gw_params['delta_phi_0']
        delta_phi_1 = self.gw_params['delta_phi_1']
        delta_phi_2 = self.gw_params['delta_phi_2']
        delta_phi_3 = self.gw_params['delta_phi_3']
        delta_phi_4 = self.gw_params['delta_phi_4']
        delta_phi_5 = self.gw_params['delta_phi_5']
        delta_phi_6 = self.gw_params['delta_phi_6']
        delta_phi_7 = self.gw_params['delta_phi_7']
        delta_phi_8 = self.gw_params['delta_phi_8']
        delta_phi_9 = self.gw_params['delta_phi_9']

        psi, psi_prime, psi_f1, psi_prime_f1 = TaylorF2_PPE.calculate_phase(self)
        
        #LATE INSPIRAL Phase Coefficients >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #(sigma0=sigma1=0 due to phase translation)
        
        sigma2 = -10114.056472621156 - 44631.01109458185*eta\
                + (chi_PN - 1)*(-6541.308761668722 - 266959.23419307504*eta + 686328.3229317984*eta2)\
                + (chi_PN - 1)**2*(3405.6372187679685 - 437507.7208209015*eta + 1.6318171307344697e6*eta2)\
                + (chi_PN - 1)**3*(-7462.648563007646 - 114585.25177153319*eta + 674402.4689098676*eta2)
        sigma3 = 22933.658273436497 + 230960.00814979506*eta\
                + (chi_PN - 1)*(14961.083974183695 + 1.1940181342318142e6*eta - 3.1042239693052764e6*eta2)\
                + (chi_PN - 1)**2*(-3038.166617199259 + 1.8720322849093592e6*eta - 7.309145012085539e6*eta2)\
                + (chi_PN - 1)**3*(42738.22871475411 + 467502.018616601*eta - 3.064853498512499e6*eta2)
        sigma4 = -14621.71522218357 - 377812.8579387104*eta\
                + (chi_PN - 1)*(-9608.682631509726 - 1.7108925257214056e6*eta + 4.332924601416521e6*eta2)\
                + (chi_PN - 1)**2*(-22366.683262266528 - 2.5019716386377467e6*eta + 1.0274495902259542e7*eta2)\
                + (chi_PN - 1)**3*(-85360.30079034246 - 570025.3441737515*eta + 4.396844346849777e6*eta2)

        psi_late_ins = + 1./eta*(3./4.*sigma2*ff**(4./3.) +\
                            3./5.*sigma3*ff**(5./3.) +\
                            1./2.*sigma4*ff**2)
        
        ################################################################################ 
        # Evaluate PHASE and DERIVATIVE at the INTERFACE between ins and int >>>>>>>>>>>
        ################################################################################ 

        f1 = 0.018
        
        psi_late_ins_f1 = 1./eta*(3./4.*sigma2*f1**(4./3.) + 3./5.*sigma3*f1**(5./3.) + 1./2.*sigma4*f1**2)

        psi_late_ins_prime = 1./eta*(sigma2*ff**(1./3.) + sigma3*ff**(2./3.) + sigma4*ff)
        psi_late_ins_prime_f1 = 1./eta*(sigma2*f1**(1./3.) + sigma3*f1**(2./3.) + sigma4*f1)

        #Total INSPIRAL PART OF THE PHASE (and its DERIVATIVE), with also late inspiral terms
        ################################################################################ 
        
        psi_ins = psi + psi_late_ins
        psi_ins_f1 = psi_f1 + psi_late_ins_f1

        psi_ins_prime = psi_prime + psi_late_ins_prime
        psi_ins_prime_f1 = psi_prime_f1 + psi_late_ins_prime_f1

        ################################################################################ 
        # PN coefficients for the INTERMEDIATE PHASE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        ################################################################################ 
        #beta0 and beta1 are fixed by the continuity conditions 
        
        beta2 = -3.282701958759534 - 9.051384468245866*eta\
                + (chi_PN - 1)*(-12.415449742258042 + 55.4716447709787*eta - 106.05109938966335*eta2)\
                + (chi_PN - 1)**2*(-11.953044553690658 + 76.80704618365418*eta - 155.33172948098394*eta2)\
                + (chi_PN - 1)**3*(-3.4129261592393263 + 25.572377569952536*eta - 54.408036707740465*eta2)
        beta3 = -0.000025156429818799565 + 0.000019750256942201327*eta\
                + (chi_PN - 1)**2*(-0.000018370671469295915 + 0.000021886317041311973*eta + 0.00008250240316860033*eta2)\
                + (chi_PN - 1)**2*(7.157371250566708e-6 - 0.000055780000112270685*eta + 0.00019142082884072178*eta2)\
                + (chi_PN - 1)**3*(5.447166261464217e-6 - 0.00003220610095021982*eta + 0.00007974016714984341*eta2)


        ####################### INS-INT PHASE CONTINUITY CONDITIONS ###################
        # Impose C1 conditions at the interface (same conditions as in IMRPhenomD but with different psi_ins & psi_ins_prime)

        beta1 = eta*psi_ins_prime_f1 - beta2*f1**(-1.) - beta3*f1**(-4.)  # psi_ins_prime_f1 = psi_int_prime_f1
        beta0 = eta*psi_ins_f1 - beta1*f1 - beta2*np.log(f1) + beta3/3.*f1**(-3.) #psi_ins_f1 = psi_int_f1
      
        # Evaluate full psi intermediate and its analytical derivative
        psi_int = 1./eta*(beta0 + beta1*ff + beta2*np.log(ff) - 1./3.*beta3*ff**(-3.))
        psi_int_prime = 1./eta*(beta1 + beta2*ff**(-1.) + beta3*ff**(-4.))

        # Frequency at the interface between intermediate and merger-ringdown phases
        ff_RD, ff_damp = wf.IMRPhenomD.RD_damping(self)
        f2 = 0.5*ff_RD

        psi_int_f2 = 1./eta*(beta0 + beta1*f2 + beta2*np.log(f2) - 1./3.*beta3*f2**(-3.))
        
        psi_int_prime_f2 = 1./eta*(beta1 + beta2*f2**(-1.) + beta3*f2**(-4.))

        ################################################################################ 
        # PN coefficients for the MERGER-RINGDOWN PHASE>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        ################################################################################ 
        #alpha0 and alpha1 are fixed by the continuity conditions
        
        alpha2 = -0.07020209449091723 - 0.16269798450687084*eta\
                + (chi_PN - 1)*(-0.1872514685185499 + 1.138313650449945*eta - 2.8334196304430046*eta2)\
                + (chi_PN - 1)**2*(-0.17137955686840617 + 1.7197549338119527*eta - 4.539717148261272*eta2)\
                + (chi_PN - 1)**3*(-0.049983437357548705 + 0.6062072055948309*eta - 1.682769616644546*eta2)
        alpha3 = 9.5988072383479 - 397.05438595557433*eta\
                + (chi_PN - 1)*(16.202126189517813 - 1574.8286986717037*eta + 3600.3410843831093*eta2)\
                + (chi_PN - 1)**2*(27.092429659075467 - 1786.482357315139*eta + 5152.919378666511*eta2)\
                + (chi_PN - 1)**3*(11.175710130033895 - 577.7999423177481*eta + 1808.730762932043*eta2)
        alpha4 =  -0.02989487384493607 + 1.4022106448583738*eta\
                + (chi_PN - 1)*(-0.07356049468633846 + 0.8337006542278661*eta + 0.2240008282397391*eta2)\
                + (chi_PN - 1)**2*(-0.055202870001177226 + 0.5667186343606578*eta + 0.7186931973380503*eta2)\
                + (chi_PN - 1)**3*(-0.015507437354325743 + 0.15750322779277187*eta + 0.21076815715176228*eta2)
        alpha5 = 0.9974408278363099 - 0.007884449714907203*eta\
                + (chi_PN - 1)*(-0.059046901195591035 + 1.3958712396764088*eta - 4.516631601676276*eta2)\
                + (chi_PN - 1)**2*(-0.05585343136869692 + 1.7516580039343603*eta - 5.990208965347804*eta2)\
                + (chi_PN - 1)**3*(-0.017945336522161195 + 0.5965097794825992*eta - 2.0608879367971804*eta2)

        
        ####################### IN-MERG PHASE CONTINUITY CONDITIONS ###################
        
        alpha1 = psi_int_prime_f2 - alpha2*f2**(-2.) - alpha3*f2**(-1./4.) -\
                (alpha4*ff_damp)/(ff_damp**2. + (f2 - alpha5*ff_RD)**2.) # psi_int_prime_f2 = psi_MR_prime_f2
        alpha0 = psi_int_f2 - alpha1*f2 + alpha2*f2**(-1.) -\
                4./3.*alpha3*f2**(3./4.) - alpha4*np.arctan((f2 - alpha5*ff_RD)/ff_damp) #psi_int_f2 = psi_MR_f2

        # Evaluate full merger-ringdown phase and its analytical derivative
        psi_MR = 1./eta*(alpha0 + alpha1*ff - alpha2*ff**(-1.) + 4./3.*alpha3*ff**(3./4.) +\
                                alpha4*np.arctan((ff - alpha5*ff_RD)/ff_damp))
        
        psi_MR_prime = 1./eta*(alpha1 + alpha2*ff**(-2.) + alpha3*ff**(-1./4.) + alpha4*ff_damp/(ff_damp**2. +\
                        (ff - alpha5*ff_RD)**2.))

        # Conjunction functions
        ff1 = 0.018*ones
        ff2 = 0.5*ff_RD*ones

        
    
        theta_minus1 = 0.5*(1*ones - wf.step_function(ff,ff1))
        theta_minus2 = 0.5*(1*ones - wf.step_function(ff,ff2))
    
        theta_plus1 = 0.5*(1*ones + wf.step_function(ff,ff1))
        theta_plus2 = 0.5*(1*ones + wf.step_function(ff,ff2))

        psi_ins, psi_ins_prime, psi_ins_f1, psi_ins_prime_f1 = wf.IMRPhenomD.calculate_ins_phase(self)
        psi_int, psi_int_prime, psi_int_f2, psi_int_prime_f2 = wf.IMRPhenomD.calculate_int_phase(self)
        psi_MR, psi_MR_prime = wf.IMRPhenomD.calculate_MR_phase(self)


        ########################### PHASE COMPONENTS ############################
        ###################### written continuosly in frequency #################

        psi_ins = psi_ins*theta_minus1
        psi_int = theta_plus1*psi_int*theta_minus2
        psi_MR = psi_MR*theta_plus2

        psi_tot = psi_ins + psi_int + psi_MR
        
        psi_prime_tot = psi_ins_prime*theta_minus1+theta_minus2*psi_int_prime*theta_plus1+theta_plus2*psi_MR_prime

        return psi_tot, psi_prime_tot

        
        

    def calculate_frequency_domain_strain(self): 

        psi, psi_prime = IMRPhenomD_PPE.calculate_phase(self)

        hp, hc = wf.IMRPhenomD.calculate_amplitude(self)
        ########################### PHASE OUTPUT ###############################
         
        phase = np.exp(1.j * psi)
 
        ########################################################################

        polarizations = np.hstack((hp * phase, hc * 1.j * phase))

        ############################### OUTPUT #################################

        self._frequency_domain_strain = polarizations

        ########################################################################
       
#GWFISH

