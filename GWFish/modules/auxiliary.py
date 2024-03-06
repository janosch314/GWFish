import scipy.optimize as optimize
import numpy as np

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=69.6, Om0=0.286)

import GWFish.modules.detection as det
import GWFish.modules.constants as cst

def from_mChirp_q_to_m1_m2(mChirp, q):
    """
    Compute the transformation from mChirp, q to m1, m2
    """
    m1 = mChirp * (1 + q)**(1/5) * q**(-3/5)
    m2 = mChirp * (1 + q)**(1/5) * q**(2/5)
    
    return m1, m2

def check_and_convert_to_mass_1_mass_2(parameters):
    if ('chirp_mass' in parameters.keys()) and ('mass_ratio' in parameters.keys()):
            parameters['mass_1'], parameters['mass_2'] = from_mChirp_q_to_m1_m2(parameters['chirp_mass'], parameters['mass_ratio'])
    if ('chirp_mass_source' in parameters.keys()) and ('mass_ratio' in parameters.keys()):
            parameters['mass_1_source'], parameters['mass_2_source'] = from_mChirp_q_to_m1_m2(parameters['chirp_mass'], parameters['mass_ratio'])
    if ('mass_1_source' in parameters.keys()) or ('mass_2_source' in parameters.keys()):
        if 'redshift' not in parameters.keys():
            raise ValueError('If using source-frame masses, one must specify the redshift parameter')
        else:
            parameters['mass_1'] = parameters['mass_1_source'] * (1 + parameters['redshift'])
            parameters['mass_2'] = parameters['mass_2_source'] * (1 + parameters['redshift'])


def fisco(parameters):
    local_params = parameters.copy()
    check_and_convert_to_mass_1_mass_2(local_params)

    M = (local_params['mass_1'] + local_params['mass_2']) * cst.Msol

    return 1 / (np.pi) * cst.c ** 3 / (cst.G * M) / 6 ** 1.5  # frequency of innermost stable circular orbit

def horizon(network, parameters, frequencyvector, detSNR, T, fmax):
    ff = frequencyvector

    def dSNR(z, detector, SNR0):
        z = np.max([0.05, z[0]])

        r = cosmo.luminosity_distance(z).value * cst.Mpc

        local_params = parameters.copy()
        check_and_convert_to_mass_1_mass_2(local_params)

        # define necessary variables, multiplied with solar mass, parsec, etc.
        M = (local_params['mass_1'] + local_params['mass_2']) * cst.Msol
        mu = (local_params['mass_1'] * local_params['mass_2'] / (
                local_params['mass_1'] + local_params['mass_2'])) * cst.Msol

        local_params['redshift'] = z
        f_isco_z = fisco(local_params)

        Mc = cst.G * mu ** 0.6 * M ** 0.4 / cst.c ** 3

        # compute GW amplitudes (https://arxiv.org/pdf/2012.01350.pdf) with optimal orientation
        hp = cst.c / r * np.sqrt(5. * np.pi / 24.) * Mc ** (5. / 6.) / (np.pi * ff) ** (7. / 6.)
        hp[
            ff > 5 * f_isco_z] = 0  # very crude, but reasonable high-f cut-off; matches roughly IMR spectra (in qadrupole order)
        print(5 * f_isco_z)

        hc = 1.j * hp

        hpij = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
        hcij = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])

        components = detector.components

        # project signal onto the detector
        proj = np.zeros((len(hp), len(components)), dtype=complex)

        for k in np.arange(len(components)):
            if detector.name == 'ET':
                n = components[k].ifo_id
                az = n * np.pi * 2. / 3.
                opening_angle = np.pi / 3.
                e1 = np.array([np.cos(az), np.sin(az), 0.])
                e2 = np.array([np.cos(az + opening_angle), np.sin(az + opening_angle), 0.])
            else:
                e1 = np.array([1., 0., 0.])
                e2 = np.array([0., 1., 0.])

            proj[:, k] = 0.5 * hp[:, 0] * (e1 @ hpij @ e1 - e2 @ hpij @ e2) \
                         + 0.5 * hc[:, 0] * (e1 @ hcij @ e1 - e2 @ hcij @ e2)

        SNRs = det.SNR(detector, proj)
        SNRtot = np.sqrt(np.sum(SNRs ** 2))

        # print('z = ' + str(z) + ', r = ' + str(cosmo.luminosity_distance(z).value) + 'Mpc, SNR = '+str(SNRtot))

        return SNRtot - SNR0

    for d in np.arange(len(network.detectors)):
        zmax = optimize.root(lambda x: dSNR(x, network.detectors[d], detSNR[1]), 5).x[0]

        print(network.detectors[d].name + ' horizon (time-invariant antenna pattern; M={:.3f}; SNR>{:.2f}): z={:.3f}'
              .format(local_params['mass_1'] + local_params['mass_2'], detSNR[1], zmax))


def scalar_product(deriv1, deriv2, detector):
    components = detector.components
    ff = detector.frequencyvector

    if deriv1.ndim == 1:
        deriv1 = deriv1[:, np.newaxis]
        deriv2 = deriv2[:, np.newaxis]

    if ff.ndim == 1:
        ff = ff[:, np.newaxis]

    df = ff[1, 0] - ff[0, 0]

    scalar_prods = np.zeros(len(components))
    for k in np.arange(len(components)):
        scalar_prods[k] = 4 * np.trapz(
                np.real(deriv1[:, k] * np.conjugate(deriv2[:, k])) / components[k].Sn(ff[:, 0]), ff[:, 0], axis=0)

    return scalar_prods
