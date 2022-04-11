import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.interpolate import interp1d
import scipy.optimize as optimize

try:
    import lalsimulation as lalsim
    import lal
    from lal import CreateREAL8Vector
except ModuleNotFoundError as err:
    uselal = err
    print('LAL package is not installed. Only GWFish waveforms available.')

import GWFish.modules.constants as cst
import GWFish.modules.auxiliary as aux

def hphc_amplitudes(waveform, parameters, frequencyvector, plot=None):
    parameters = parameters.copy()

    if waveform == 'gwfish_TaylorF2':
        hphc = TaylorF2(parameters, frequencyvector, plot=plot)
    elif waveform == 'gwfish_IMRPhenomD':
        hphc = IMRPhenomD(parameters, frequencyvector, plot=plot)
    elif waveform[0:7] == 'lalsim_':
        hphc = lal_caller(waveform[7:], frequencyvector, **parameters)
    else:
        print(str(waveform) + ' is not a valid waveform.')
        print('Valid options are gwfish_TaylorF2, gwfish_IMRPhenomD, lalsim_XXX.')

    t_of_f = t_of_f_PN(parameters, frequencyvector)

    return hphc, t_of_f

def convert_args_list_to_float(*args_list):
    """ Converts inputs to floats, returns a list in the same order as the input"""
    # copied from https://git.ligo.org/lscsoft/bilby/, March 21, 2022
    try:
        args_list = [float(arg) for arg in args_list]
    except ValueError:
        raise ValueError("Unable to convert inputs to floats")
    return args_list

def lalsim_SimInspiralTransformPrecessingNewInitialConditions(
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
        reference_frequency, phase):
    # copied from https://git.ligo.org/lscsoft/bilby/, March 21, 2022
    from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions

    args_list = convert_args_list_to_float(
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
        reference_frequency, phase)

    return SimInspiralTransformPrecessingNewInitialConditions(*args_list)

@np.vectorize
def transform_precessing_spins(theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1,
                               a_2, mass_1, mass_2, reference_frequency, phase):
    # copied from https://git.ligo.org/lscsoft/bilby/, March 21, 2022
    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = (
        lalsim_SimInspiralTransformPrecessingNewInitialConditions(
            theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
            reference_frequency, phase))

    return iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z

def bilby_to_lalsimulation_spins(
        # copied from https://git.ligo.org/lscsoft/bilby/, March 21, 2022
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
        reference_frequency, phase):
    if (a_1 == 0 or tilt_1 in [0, np.pi]) and (a_2 == 0 or tilt_2 in [0, np.pi]):
        spin_1x = 0
        spin_1y = 0
        spin_1z = a_1 * np.cos(tilt_1)
        spin_2x = 0
        spin_2y = 0
        spin_2z = a_2 * np.cos(tilt_2)
        iota = theta_jn
    else:
        iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = \
            transform_precessing_spins(
                theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1,
                mass_2, reference_frequency, phase)
    return iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z

def t_of_f_PN(parameters, frequencyvector):
    # t(f) is required to calculate slowly varying antenna pattern as function of instantaneous frequency.
    # This FD approach follows Marsat/Baker arXiv:1806.10734v1; equation (22) neglecting the phase term, which does not
    # matter for SNR calculations.

    z = parameters['redshift']
    M1 = parameters['mass_1'] * (1 + z) * cst.Msol
    M2 = parameters['mass_2'] * (1 + z) * cst.Msol

    M = M1 + M2
    mu = M1 * M2 / M

    Mc = cst.G * mu ** 0.6 * M ** 0.4 / cst.c ** 3

    t_of_f = -5./(256.*np.pi**(8/3))/Mc**(5/3)/frequencyvector**(8/3)

    return t_of_f+parameters['geocent_time']

def lal_caller(waveform, frequencyvector, mass_1, mass_2, luminosity_distance, redshift, theta_jn, phase, geocent_time,
           a_1=0, tilt_1=0, phi_12=0, a_2=0, tilt_2=0, phi_jl=0, lambda_1=0, lambda_2=0, **kwargs):
    params_lal = lal.CreateDict()
    approx_lal = lalsim.GetApproximantFromString(waveform)

    if lambda_1 != 0:
        from lalsimulation import SimInspiralWaveformParamsInsertTidalLambda1
        SimInspiralWaveformParamsInsertTidalLambda1(params_lal, float(lambda_1))
    if lambda_2 != 0:
        from lalsimulation import SimInspiralWaveformParamsInsertTidalLambda2
        SimInspiralWaveformParamsInsertTidalLambda2(params_lal, float(lambda_2))

    frequencyvector = frequencyvector.copy().flatten()

    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby_to_lalsimulation_spins(
        theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=mass_1, mass_2=mass_2,
        reference_frequency=50., phase=phase)

    frequency_array = CreateREAL8Vector(len(frequencyvector))
    frequency_array.data = frequencyvector

    h_plus, h_cross = lalsim.SimInspiralChooseFDWaveformSequence(
        phase,
        mass_1 * lal.MSUN_SI * (1 + redshift),  # in [kg]
        mass_2 * lal.MSUN_SI * (1 + redshift),  # in [kg]
        spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
        50.,  # reference frequency
        luminosity_distance * lal.PC_SI * 1e6,  # in [m]
        iota,
        params_lal,
        approx_lal,
        frequency_array
    )

    # Add initial 2pi*f*tc - phic - pi/4 to phase
    phi_in = np.exp(1.j*(2*frequencyvector*np.pi*geocent_time))
    hp = phi_in * np.conjugate(h_plus.data.data)  # it's already multiplied by the phase
    hc = phi_in * np.conjugate(h_cross.data.data)

    hp = hp[:, np.newaxis]
    hc = hc[:, np.newaxis]
    polarizations = np.hstack((hp, hc))

    return polarizations

def TaylorF2(parameters, frequencyvector, maxn=8, plot=None):
    ff = frequencyvector
    ones = np.ones((len(ff), 1))

    phic = parameters['phase']
    tc = parameters['geocent_time']
    z = parameters['redshift']
    r = parameters['luminosity_distance'] * cst.Mpc
    iota = parameters['theta_jn']
    M1 = parameters['mass_1'] * (1 + z) * cst.Msol
    M2 = parameters['mass_2'] * (1 + z) * cst.Msol

    M = M1 + M2
    mu = M1 * M2 / M

    Mc = cst.G * mu ** 0.6 * M ** 0.4 / cst.c ** 3

    # compute GW amplitudes (https://arxiv.org/pdf/2012.01350.pdf)
    hp = cst.c / (2. * r) * np.sqrt(5. * np.pi / 24.) * Mc ** (5. / 6.) / (np.pi * ff) ** (7. / 6.) * (
            1. + np.cos(iota) ** 2.)
    hc = cst.c / (2. * r) * np.sqrt(5. * np.pi / 24.) * Mc ** (5. / 6.) / (np.pi * ff) ** (7. / 6.) * 2. * np.cos(iota)

    C = 0.57721566  # Euler constant
    eta = mu / M

    f_isco = aux.fisco(parameters)

    v = (np.pi * cst.G * M / cst.c ** 3 * ff) ** (1. / 3.)
    v_isco = (np.pi * cst.G * M / cst.c ** 3 * f_isco) ** (1. / 3.)

    # coefficients of the PN expansion (https://arxiv.org/pdf/0907.0700.pdf)
    pp = np.hstack((1. * ones, 0. * ones, 20. / 9. * (743. / 336. + eta * 11. / 4.) * ones, -16 * np.pi * ones,
                    10. * (3058673. / 1016064. + 5429. / 1008. * eta + 617. / 144. * eta ** 2) * ones,
                    np.pi * (38645. / 756. - 65. / 9. * eta) * (1 + 3. * np.log(v / v_isco)),
                    11583231236531. / 4694215680. - 640. / 3. * np.pi ** 2 - 6848. / 21. * (C + np.log(4 * v))
                    + (
                            -15737765635. / 3048192. + 2255. / 12. * np.pi ** 2) * eta + 76055. / 1728. * eta ** 2 - 127825. / 1296. * eta ** 3,
                    np.pi * (77096675. / 254016. + 378515. / 1512. * eta - 74045. / 756. * eta ** 2) * ones))

    psi = 0.

    for k in np.arange(maxn):
        PNc = pp[:, k]
        psi += PNc[:, np.newaxis] * v ** k

    psi *= 3. / (128. * eta * v ** 5)
    psi += 2. * np.pi * ff * tc - phic - np.pi / 4.

    phase = np.exp(1.j * psi)
    polarizations = np.hstack((hp * phase, hc * 1.j * phase))
    polarizations[np.where(ff > 4 * f_isco), :] = 0.j  # very crude high-f cut-off

    if plot is not None:
        plt.figure()
        plt.loglog(frequencyvector, np.abs(polarizations[:, 0]), label=r'$h_+$')
        plt.loglog(frequencyvector, np.abs(polarizations[:, 1]), label=r'$h_\times$')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel(r'Fourier amplitude [$Hz^{-1}$]')
        plt.grid(which='both', color='lightgray', alpha=0.5, linestyle='dashed', linewidth=0.5)
        plt.axis(plot)
        plt.legend()
        plt.tight_layout()
        plt.savefig('amp_tot_TF2.png')
        plt.close()

        plt.figure()
        plt.semilogx(frequencyvector, psi)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Phase [rad]')
        plt.grid(which='both', color='lightgray', alpha=0.5, linestyle='dashed', linewidth=0.5)
        plt.tight_layout()
        plt.savefig('phase_tot_TF2.png')
        plt.close()

    return polarizations

def step_function(f1, f2):
    vec = []
    for i in range(len(f1)):
        if (f1[i] < f2[i]):
            vec.append(-1.)
        else:
            vec.append(+1.)
    vec = np.array(vec)
    vec = vec[:, np.newaxis]
    return vec

def kerr_isco(chi):
    Z1 = 1 + (1 - chi**2)**(1/3)*((1 + chi)**(1/3) + (1 - chi)**(1/3))
    Z2 = (3*chi**2 + Z1**2)**(0.5)
    return (3 + Z2 - np.sign(chi)*((3 - Z1)*(3 + Z1 + 2*Z2))**(0.5))

def epsilon_chi(x):
    return (1. - 2./kerr_isco(x) + x/(kerr_isco(x))**(3/2))/(1 - 3./kerr_isco(x) + 2*x/(kerr_isco(x))**(3/2))**(0.5)

def j_chi(x):
    return 2./(3*kerr_isco(x))**(0.5)*(3*(kerr_isco(x))**(0.5) -2*x)

def chi_final_func(x, M, eta, s, delta_m, Delta, L0, L1, L2a, L2b, L2c, L2d, L3a, L3b, L3c, L3d, L4a, L4b, L4c, L4d, L4e, L4f, L4g, L4h, L4i):
    return (4*eta)**2*(L0 +L1*s + L2a*Delta*delta_m + L2b*s**2 + L2c*Delta**2 + L2d*delta_m**2 + \
    L3a*Delta*s*delta_m + L3b*s*Delta**2 + L3c*s**3 + L3d*s*delta_m**2 + L4a*s**2*Delta*delta_m + L4b*Delta**3*delta_m + \
    L4c*Delta**4 + L4d*s**4 + L4e*Delta**2*s**2 + L4f*delta_m**4 + L4g*Delta*delta_m**3 + L4h*Delta**2*delta_m**2 + \
    L4i*s**2*delta_m**2 + s*(1+8*eta)*delta_m**4 + eta*j_chi(x)*delta_m**6) - x

def final_bh(M1, M2, chi1, chi2):
    M, eta = [M1 + M2, (M1*M2)/(M1 + M2)**2]
    s, delta_m, Delta = [(chi1*M1**2 + chi2*M2**2)/M**2, (M1 - M2)/M, (chi2*M2 - chi1*M1)/M]

    M0, K1, K2a, K2b, K2c, K2d, K3a, K3b, K3c, K3d, K4a, K4b, K4c, K4d, K4e, K4f, K4g, K4h, K4i = [0.951507, -0.051379,\
    -0.004804, -0.054522, -0.000022, 1.995246, 0.007064, -0.017599, -0.119175, 0.025000, -0.068981, -0.011383, -0.002284,\
    -0.165658, 0.019403, 2.980990, 0.020250, -0.004091, 0.078441]
    L0, L1, L2a, L2b, L2c, L2d, L3a, L3b, L3c, L3d, L4a, L4b, L4c, L4d, L4e, L4f, L4g, L4h, L4i = [0.686710, 0.613247,\
    -0.145427, -0.115689, -0.005254, 0.801838, -0.073839, 0.004759, -0.078377, 1.585809, -0.003050, -0.002968, 0.004364,\
    -0.047204, -0.053099, 0.953458, -0.067998, 0.001629, -0.066693]

    chi_f = optimize.fsolve(chi_final_func, 0.5, args = (M, eta, s, delta_m, Delta, L0, L1, L2a, L2b, L2c, L2d, L3a, L3b,\
    L3c, L3d, L4a, L4b, L4c, L4d, L4e, L4f, L4g, L4h, L4i))
    m_f = M*(4*eta)**2*(M0 + K1*s + K2a*Delta*delta_m + K2b*s**2 + K2c*Delta**2 + K2d*delta_m**2 + K3a*Delta*s*delta_m +  \
    K3b*s*Delta**2 + K3c*s**3 + K3d*s*delta_m**2 + K4a*s**2*Delta*delta_m + K4b*Delta**3*delta_m + K4c*Delta**4 + K4d*s**4 + \
    K4e*Delta**2*s**2 + K4f*delta_m**4 + K4g*Delta*delta_m**3 + K4h*Delta**2*delta_m**2 + K4i*s**2*delta_m**2 + \
    (1+eta*(epsilon_chi(chi_f) + 11))*delta_m**6)

    return chi_f, m_f

def phenomD_amp_MR(f, parameters, f_damp, f_RD, gamma1, gamma2, gamma3):
    ff = sp.symbols('ff', real=True)
    amp_MR = gamma1*(gamma3*f_damp)/((ff - f_RD)**2. + (gamma3*f_damp)**2)*\
            sp.exp(-gamma2*(ff - f_RD)/(gamma3*f_damp))
    
    amp_MR_f = amp_MR.evalf(subs={ff: f})
    amp_MR_prime = sp.diff(amp_MR, ff)
    #print('amp_MR_prime = ', sp.simplify(amp_MR_prime))
    amp_MR_prime_f = amp_MR_prime.evalf(subs={ff: f})

    
    return amp_MR_f, amp_MR_prime_f

def IMRPhenomD(parameters, frequencyvector, plot=None):
    phic = parameters['phase']
    tc = parameters['geocent_time']
    z = parameters['redshift']
    r = parameters['luminosity_distance'] * cst.Mpc
    iota = parameters['theta_jn']
    M1 = parameters['mass_1'] * (1 + z) * cst.Msol
    M2 = parameters['mass_2'] * (1 + z) * cst.Msol
    if (M1 < M2):
        aux_mass = M1
        M1 = M2
        M2 = aux_mass

    if 'a_1' in parameters:
        chi_1 = parameters['a_1']
    else:
        chi_1 = 0.0
    if 'a_2' in parameters:
        chi_2 = parameters['a_2']
    else:
        chi_2 = 0.0
    M = M1 + M2
    mu = M1 * M2 / M
    Mc = cst.G * mu ** 0.6 * M ** 0.4 / cst.c ** 3
    delta_mass = (M1 - M2)/M
    
    ff = frequencyvector*cst.G*M/cst.c**3
    ones = np.ones((len(ff), 1))

    C = 0.57721566  # Euler constant
    eta = mu / M
    eta2 = eta*eta
    eta3 = eta2*eta

    chi_eff = (M1*chi_1 + M2*chi_2)/M
    chi_PN = chi_eff - 38/113*eta*(chi_1 + chi_2)
    chi_s = 0.5*(chi_1 + chi_2)
    chi_a = 0.5*(chi_1 - chi_2)


    #########################################################################################################
    #########################################################################################################
    #########################################################################################################

    # PHASE

    # PN expansion of phase
    # PN coefficients:
    phi_0 = 1.
    phi_1 = 0.
    phi_2 = 3715./756. + 55./9.*eta
    phi_3 = -16.*np.pi + 113./3.*delta_mass*chi_a + (113./3. - 76./3.*eta)*chi_s
    phi_4 = 15293365./508032. + 27145./504.*eta + 3085./72.*eta2 + (-(405./8.) + 200*eta)*chi_a**2 - \
            405./4.*delta_mass*chi_a*chi_s + (-(405./8.) + 5./2.*eta)*chi_s**2
    phi_5 = (1 + np.log(np.pi*ff))*(38645./756.*np.pi - 65./9.*np.pi*eta + \
            delta_mass*(-(732985./2268.) - 140./9.*eta)*chi_a + (-(732985./2268.) + 24260./81.*eta + 340./9.*eta2)*chi_s)
    phi_6 = 11583231236531./4694215680. - 6848./21.*C - (640.*np.pi**2)/3. + (-15737765635./3048192. + 2255.*np.pi**2/12.)*eta +\
            76055.*eta2/1728. - 127825.*eta3/1296. - 6848./63.*np.log(64*np.pi*ff) + 2270./3.*np.pi*delta_mass*chi_a +\
            (2270.*np.pi/3. - 520.*np.pi*eta)*chi_s
    phi_7 = (77096675./254016. + 378515./1512.*eta - 74045./756.*eta2)*np.pi +\
            delta_mass*(-(25150083775./3048192.) + 26804935./6048.*eta - 1985./48.*eta2)*chi_a +\
            (-(25150083775./3048192.) + 10566655595./762048.*eta - 1042165./3024.*eta2 + 5345./36.*eta3)*chi_s
   
    psi_TF2 = 2.*np.pi*ff*cst.c**3/(cst.G*M)*tc - phic*ones - np.pi/4.*ones + 3./(128.*eta)*((np.pi*ff)**(-5./3.) +\
            phi_2*(np.pi*ff)**(-1.) + phi_3*(np.pi*ff)**(-2./3.) + phi_4*(np.pi*ff)**(-1./3.) +\
            phi_5 + phi_6*(np.pi*ff)**(1./3.) + phi_7*(np.pi*ff)**(2./3.))
    
    # Coefficients for the late ispiral phase
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

    psi_ins = psi_TF2 + 1./eta*(3./4.*sigma2*ff**(4./3.) + 3./5.*sigma3*ff**(5./3.) +\
                    1./2.*sigma4*ff**2)
    
    #psi_ins_prime = psi_TF2_prime + 1./eta*(sigma2*ff**(1./3.) + sigma3*ff**(2./3.) + sigma4*ff)

    # Evaluate phase and its derivate at the interface between inspiral and intermediate phase
    f1 = 0.018
    psi_ins_gradient = interp1d(ff[:,0], np.gradient(psi_ins[:,0]))

    phi_5_f1 = (1 + np.log(np.pi*f1))*(38645./756.*np.pi - 65./9.*np.pi*eta + \
            delta_mass*(-(732985./2268.) - 140./9.*eta)*chi_a + (-(732985./2268.) + 24260./81.*eta + 340./9.*eta2)*chi_s)
    phi_6_f1 = 11583231236531./4694215680. - 6848./21.*C - (640.*np.pi**2)/3. + (-15737765635./3048192. + 2255.*np.pi**2/12.)*eta +\
            76055.*eta2/1728. - 127825.*eta3/1296. - 6848./63.*np.log(64*np.pi*f1) + 2270./3.*np.pi*delta_mass*chi_a +\
            (2270.*np.pi/3. - 520.*np.pi*eta)*chi_s
    psi_ins_f1 = 2.*np.pi*f1/(cst.G*M)*cst.c**3*tc - phic - np.pi/4. + 3./(128.*eta)*(np.pi*f1)**(-5/3)*(phi_0 +\
            phi_2*(np.pi*f1)**(2./3.) + phi_3*(np.pi*f1) +\
            phi_4*(np.pi*f1)**(4./3.) + phi_5_f1*(np.pi*f1)**(5./3.) +\
            phi_6_f1*(np.pi*f1)**2. + phi_7*(np.pi*f1)**(7./3.)) +\
            1./eta*(3./4.*sigma2*f1**(4./3.) + 3./5.*sigma3*f1**(5./3.) +\
            1./2.*sigma4*f1**2)

    psi_ins_prime_f1 = psi_ins_gradient(f1)


    # Coefficients for the intermediate phase
    beta2 = -3.282701958759534 - 9.051384468245866*eta\
            + (chi_PN - 1)*(-12.415449742258042 + 55.4716447709787*eta - 106.05109938966335*eta2)\
            + (chi_PN - 1)**2*(-11.953044553690658 + 76.80704618365418*eta - 155.33172948098394*eta2)\
            + (chi_PN - 1)**3*(-3.4129261592393263 + 25.572377569952536*eta - 54.408036707740465*eta2)
    beta3 = -0.000025156429818799565 + 0.000019750256942201327*eta\
            + (chi_PN - 1)**2*(-0.000018370671469295915 + 0.000021886317041311973*eta + 0.00008250240316860033*eta2)\
            + (chi_PN - 1)**2*(7.157371250566708e-6 - 0.000055780000112270685*eta + 0.00019142082884072178*eta2)\
            + (chi_PN - 1)**3*(5.447166261464217e-6 - 0.00003220610095021982*eta + 0.00007974016714984341*eta2)
  
    # Impose C1 conditions at the interface
    beta1 = eta*psi_ins_prime_f1 - beta2*f1**(-1.) - beta3*f1**(-4.)  # psi_ins_prime_f1 = psi_int_prime_f1
    beta0 = eta*psi_ins_f1 - beta1*f1 - beta2*np.log(f1) + beta3/3.*f1**(-3.) #psi_ins_f1 = psi_int_f1
   
    # Evaluate full psi intermediate and its analytical derivative
    psi_int = 1./eta*(beta0 + beta1*ff + beta2*np.log(ff) - 1./3.*beta3*ff**(-3.))
    psi_int_prime = 1./eta*(beta1 + beta2*ff**(-1.) + beta3*ff**(-4.))
    
    # Coefficients for the merger-ringdown phase
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
    
    # Interpolate from dataset to evaluate damping and ringdown frequencies
    chi_f, m_f = final_bh(M1, M2, chi_1, chi_2)

    data_ff = np.loadtxt('n1l2m2.dat', unpack = True)
    M_omega = interp1d(data_ff[0, :], data_ff[1, :])
    tau_omega = interp1d(data_ff[0, :], data_ff[2, :])

    ff_RD = (M_omega(chi_f)/(2*np.pi)*M/m_f)[0]
    ff_damp = (-tau_omega(chi_f)/(2*np.pi)*M/m_f)[0]
    
    # Frequency at the interface between intermediate and merger-ringdown phases
    f2 = 0.5*ff_RD
    # Impose C1 conditions at the interface
    alpha1 = (beta1 + beta2*f2**(-1.) + beta3*f2**(-4.)) - alpha2*f2**(-2.) - alpha3*f2**(-1./4.) -\
            (alpha4*ff_damp)/(ff_damp**2. + (f2 - alpha5*ff_RD)**2.) # psi_int_prime_f2 = psi_MR_prime_f2
    alpha0 = (beta0 + beta1*f2 + beta2*np.log(f2) - beta3/3.*f2**(-3.)) - alpha1*f2 + alpha2*f2**(-1.) -\
            4./3.*alpha3*f2**(3./4.) - alpha4*np.arctan((f2 - alpha5*ff_RD)/ff_damp) #psi_int_f2 = psi_MR_f2

    # Evaluate full merger-ringdown phase and its analytical derivative
    psi_MR = 1./eta*(alpha0 + alpha1*ff - alpha2*ff**(-1.) + 4./3.*alpha3*ff**(3./4.) +\
                            alpha4*np.arctan((ff - alpha5*ff_RD)/ff_damp))
    psi_MR_prime = 1./eta*(alpha1 + alpha2*ff**(-2.) + alpha3*ff**(-1./4.) + alpha4*ff_damp/(ff_damp**2. +\
                    (ff - alpha5*ff_RD)**2.))

    # Conjunction functions
    ff1 = 0.018*ones
    ff2 = 0.5*ff_RD*ones

    theta_minus1 = 0.5*(1*ones - step_function(ff,ff1))
    theta_minus2 = 0.5*(1*ones - step_function(ff,ff2))

    theta_plus1 = 0.5*(1*ones + step_function(ff,ff1))
    theta_plus2 = 0.5*(1*ones + step_function(ff,ff2))
  
    psi_ins = psi_ins*theta_minus1
    psi_int = theta_plus1*psi_int*theta_minus2
    psi_MR = psi_MR*theta_plus2

   
    psi_tot = psi_ins + psi_int + psi_MR
    psi_prime_tot = psi_ins_gradient(ff)*theta_minus1+theta_minus2*psi_int_prime*theta_plus1+theta_plus2*psi_MR_prime


    # Construct the phase
    phase = np.exp(1.j * psi_tot)
    #########################################################################################################
    #########################################################################################################
    #########################################################################################################

    # AMPLITUDE

    # PN coefficients:
    a_0 = 1.
    a_1 = 0.
    a_2 = -323./224. + 451./168.*eta
    a_3 = 27./8.*delta_mass*chi_a + (27./8. - 11./6.*eta)*chi_s
    a_4 = 105271./24192.*eta2 - 1975055./338688.*eta - 27312085./8128512. + (-(81./32.) + 8.*eta)*chi_a**2. -\
        81./16.*delta_mass*chi_a*chi_s + (-(81./32.) + 17./8.*eta)*chi_s**2.
    a_5 = (85.*np.pi)/64.*(4*eta - 1.) + delta_mass*(285197./16128. - 1579./4032.*eta)*chi_a +\
        (285197./16128. - 15317./672.*eta - 2227./1008.*eta2)*chi_s
    a_6 = 34473079./6386688.*eta3 - 3248849057./178827264.*eta2 + 545384828789./5007163392.*eta - 205./48.*eta*np.pi**2. -\
        177520268561./8583708672. + (1614569./64512. - 1873643./16128.*eta + 2167./42.*eta2)*chi_a**2. +\
        (31./12.*np.pi - 7./3.*np.pi*eta)*chi_s + (1614569./64512. - 61391./1344.*eta + 57451./4032.*eta2)*chi_s**2. +\
        delta_mass*chi_a*(31./12.*np.pi + (1614569./32256. - 165961./2688.*eta)*chi_s)
    
    amp_PN = (a_0 + a_2*(np.pi*ff)**(2./3.) + a_3*(np.pi*ff) + a_4*(np.pi*ff)**(4./3.) +\
            a_5*(np.pi*ff)**(5./3.) + a_6*(np.pi*ff)**2.)

    # Late inspiral coefficients
    rho1 = 3931.8979897196696 - 17395.758706812805*eta\
            + (chi_PN - 1)*(3132.375545898835 + 343965.86092361377*eta - 1.2162565819981997e6*eta2)\
            + (chi_PN - 1)**2*(-70698.00600428853 + 1.383907177859705e6*eta - 3.9662761890979446e6*eta2)\
            + (chi_PN - 1)**3*(-60017.52423652596 + 803515.1181825735*eta - 2.091710365941658e6*eta2)
    rho2 = -40105.47653771657 + 112253.0169706701*eta\
            + (chi_PN - 1)*(23561.696065836168 - 3.476180699403351e6*eta + 1.137593670849482e7*eta2)\
            + (chi_PN - 1)**2*(754313.1127166454 - 1.308476044625268e7*eta + 3.6444584853928134e7*eta2)\
            + (chi_PN - 1)**3*(596226.612472288 - 7.4277901143564405e6*eta + 1.8928977514040343e7*eta2)
    rho3 = 83208.35471266537 - 191237.7264145924*eta\
            + (chi_PN - 1)*(-210916.2454782992 + 8.71797508352568e6*eta - 2.6914942420669552e7*eta2)\
            + (chi_PN - 1)**2*(-1.9889806527362722e6 + 3.0888029960154563e7*eta - 8.390870279256162e7*eta2)\
            + (chi_PN - 1)**3*(-1.4535031953446497e6 + 1.7063528990822166e7*eta - 4.2748659731120914e7*eta2)

    amp_ins = amp_PN + (rho1*(ff)**(7./3.) + rho2*(ff)**(8./3.) + rho3*(ff)**3.)

    # Merger-ringdown coefficients
    gamma1 = 0.006927402739328343 + 0.03020474290328911*eta\
            + (chi_PN - 1)*(0.006308024337706171 - 0.12074130661131138*eta + 0.26271598905781324*eta2)\
            + (chi_PN - 1)**2*(0.0034151773647198794 - 0.10779338611188374*eta + 0.27098966966891747*eta2)\
            + (chi_PN - 1)**3*(0.0007374185938559283 - 0.02749621038376281*eta + 0.0733150789135702*eta2)
    gamma2 = 1.010344404799477 + 0.0008993122007234548*eta\
            + (chi_PN - 1)*(0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2)\
            + (chi_PN - 1)**2*(0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)\
            + (chi_PN - 1)**3*(0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)
    gamma3 = 1.3081615607036106 - 0.005537729694807678*eta\
            + (chi_PN - 1)*(-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2)\
            + (chi_PN - 1)**2*(-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)\
            + (chi_PN - 1)**3*(-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)

    # Intermediate phase 
    v2 = 0.8149838730507785 + 2.5747553517454658*eta\
            + (chi_PN - 1)*(1.1610198035496786 - 2.3627771785551537*eta + 6.771038707057573*eta2)\
            + (chi_PN - 1)**2*(0.7570782938606834 - 2.7256896890432474*eta + 7.1140380397149965*eta2)\
            + (chi_PN - 1)**3*(0.1766934149293479 - 0.7978690983168183*eta + 2.1162391502005153*eta2)

    # Conjunction frequencies
    f1_amp = 0.014
    f3_amp = (np.abs(ff_RD + (ff_damp*gamma3*(np.sqrt(1-gamma2**2.) - 1)/gamma2)))
    f2_amp = (f1_amp + f3_amp)/2.


    amp_MR = gamma1*(gamma3*ff_damp*ones)/((ff - ff_RD*ones)**2. +\
            (gamma3*ff_damp*ones)**2)*np.exp(-gamma2*(ff - ff_RD*ones)/(gamma3*ff_damp*ones))

    amp_ins_f1 = a_0 + a_2*(np.pi*f1_amp)**(2./3.) + a_3*(np.pi*f1_amp) + a_4*(np.pi*f1_amp)**(4./3.) +\
            a_5*(np.pi*f1_amp)**(5./3.) + a_6*(np.pi*f1_amp)**2. + rho1*f1_amp**(7./3.) +\
            rho2*f1_amp**(8./3.) + rho3*f1_amp**3.

    amp_ins_prime_f1 = 2./3.*a_2*np.pi**(2./3.)*f1_amp**(-1./3.) + a_3*np.pi + 4./3.*a_4*np.pi**(4./3.)*f1_amp**(1./3.) +\
                        5./3.*a_5*np.pi**(5./3.)*f1_amp**(2./3.) + 2*a_6*np.pi**2.*f1_amp + 7./3.*rho1*f1_amp**(4./3.) +\
                        8./3.*rho2*f1_amp**(5./3.) + 3.*rho3*f1_amp**2.

    amp_MR_f3, amp_MR_prime_f3 = phenomD_amp_MR(f3_amp, parameters, ff_damp, ff_RD, gamma1, gamma2, gamma3)
    amp_MR_f3 = float(amp_MR_f3)
    amp_MR_prime_f3 = float(amp_MR_prime_f3)
   
    # Solve for delta coefficients (intermediate phase)
    A = np.array([[1., f1_amp, f1_amp**2., f1_amp**3., f1_amp**4.],\
                    [1., f2_amp, f2_amp**2., f2_amp**3., f2_amp**4.],\
                    [1., f3_amp, f3_amp**2., f3_amp**3., f3_amp**4.],\
                    [0., 1., 2.*f1_amp, 3.*f1_amp**2., 4.*f1_amp**3.],\
                    [0., 1., 2.*f3_amp, 3.*f3_amp**2., 4.*f3_amp**3.]])
    b = np. array([amp_ins_f1, v2, amp_MR_f3, amp_ins_prime_f1, amp_MR_prime_f3])
    delta = np.linalg.solve(A, b)

    # Full intermediate amplitude
    amp_int = (delta[0] + delta[1]*(ff) + delta[2]*(ff)**2. + delta[3]*(ff)**3. +\
            delta[4]*(ff)**4.)
  

    ff1_amp = f1_amp*ones
    ff3_amp = f3_amp*ones

    theta_minus1_amp = 0.5*(1*ones - step_function(ff,ff1_amp))
    theta_minus2_amp = 0.5*(1*ones - step_function(ff,ff3_amp))

    theta_plus1_amp = 0.5*(1*ones + step_function(ff,ff1_amp))
    theta_plus2_amp = 0.5*(1*ones + step_function(ff,ff3_amp))

    # Overall (2,2) mode factor and its derivative
    A0 = 1./(np.pi**(2./3.))*(5./24.)**(0.5)*cst.c/r*Mc**(5./6.)*frequencyvector**(-7./6.)
    
    amp_ins = amp_ins*theta_minus1_amp*A0
    amp_int = theta_plus1_amp*amp_int*theta_minus2_amp*A0
    amp_MR = theta_plus2_amp*amp_MR*A0

    amp_tot = amp_ins + amp_int + amp_MR
    

    hp = amp_tot*0.5*(1 + np.cos(iota)**2.)
    hc = amp_tot*np.cos(iota)
    polarizations = np.hstack((hp * phase, hc * 1.j * phase))

    if plot is not None:
        plt.figure()
        y_height = plot[3]/10
        plt.loglog(frequencyvector, np.abs(polarizations[:, 0]), linewidth=2, color='blue', label=r'$h_+$')
        plt.loglog(frequencyvector, np.abs(polarizations[:, 1]), linewidth=2, color='blue', label=r'$h_\times$')
        plt.axvline(x=f1_amp*cst.c**3/(M*cst.G), color = 'orange', linestyle = '--', linewidth = 2)
        plt.axvline(x=f2_amp*cst.c**3/(M*cst.G), color = 'orange', linestyle = '--', linewidth = 2)
        plt.axvline(x=f3_amp*cst.c**3/(M*cst.G), color = 'orange', linestyle = '--', linewidth = 2)
        plt.text(1.05*f1_amp*cst.c**3/(M*cst.G), y_height, 'f1_match', rotation=90, fontsize=10, color = 'orange')
        plt.text(1.05*f3_amp*cst.c**3/(M*cst.G), y_height, 'f3_match', rotation=90, fontsize=10, color = 'orange')
        plt.text(1.05*f2_amp*cst.c**3/(M*cst.G), y_height, 'f2_match', rotation=90, fontsize=10, color = 'orange')
        plt.legend(fontsize=8)
        plt.axis(plot)
        plt.grid(which='both', color='lightgray', alpha=0.5, linestyle='dashed', linewidth=0.5)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel(r'Fourier amplitude [$Hz^{-1}$]')
        plt.savefig('amp_phenomD.png')
        plt.close()

        plt.figure()
        plt.semilogx(frequencyvector, psi_prime_tot, linewidth = 2, color = 'blue', label='PhenomD')
        y_loc = (1 + 1e-9)*psi_prime_tot[0,0]
        plt.axvline(x=0.018*cst.c**3/(cst.G*M), color = 'orange', linestyle = '--', linewidth = 2)
        plt.axvline(x=ff_RD*cst.c**3/(cst.G*M), color = 'orange', linestyle = '--', linewidth = 2)
        plt.text(1.05*0.018*cst.c**3/(cst.G*M), y_loc, '$Mf = 0.018$', rotation=90, fontsize=12, color='orange')
        plt.text(1.05*ff_RD*cst.c**3/(cst.G*M), y_loc, '$f_{RD}$', rotation=90, fontsize=12, color='orange')
        plt.legend(fontsize = 8)
        plt.grid(which = 'both', color = 'lightgray', alpha = 0.5, linestyle = 'dashed', linewidth = 0.5)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('$\phi$_prime')
        plt.savefig('psi_prime_phenomD.png')
        plt.close()

        plt.figure()
        freq_lim_vec = frequencyvector[frequencyvector > 0.018*cst.c**3/(cst.G*M)]
        psi_lim_vec = psi_tot[len(psi_tot)-len(freq_lim_vec):, 0]
        fig, ax = plt.subplots(figsize=[8, 5])
        ax.loglog(frequencyvector, psi_tot, linewidth = 2, color = 'blue', label = 'PhenomD')
        axins = ax.inset_axes([0.5, +0.1, 0.47, 0.47])
        axins.plot(freq_lim_vec, psi_lim_vec, color='blue', linewidth=2)
        axins.set_xscale('log')
        axins.set_yscale('log')
        axins.set_xticklabels('')
        axins.set_yticklabels('')
        axins.grid(which = 'both', color = 'lightgray', alpha = 0.5, linestyle = 'dashed', linewidth = 0.5)
        ax.indicate_inset_zoom(axins, edgecolor="black")
        plt.legend(fontsize = 8)
        plt.grid(which = 'both', color = 'lightgray', alpha = 0.5, linestyle = 'dashed', linewidth = 0.5)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('$\phi$')
        plt.savefig('psi_phenomD_zoomed.png')
        plt.close()


    return polarizations
