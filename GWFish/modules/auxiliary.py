import scipy.optimize as optimize
import numpy as np

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=69.6, Om0=0.286)

import detection as det
import constants as cst

def fisco(parameters):
    M = (parameters['mass_1'] + parameters['mass_2']) * cst.Msol * (1 + parameters['redshift'])

    return 1 / (np.pi) * cst.c ** 3 / (cst.G * M) / 6 ** 1.5  # frequency of innermost stable circular orbit


def horizon(network, parameters, frequencyvector, detSNR, T, fmax):
    ff = frequencyvector

    def dSNR(z, detector, SNR0):
        z = np.max([0.05, z[0]])

        r = cosmo.luminosity_distance(z).value * cst.Mpc

        # define necessary variables, multiplied with solar mass, parsec, etc.
        M = (parameters['mass_1'] + parameters['mass_2']) * cst.Msol * (1 + z)
        mu = (parameters['mass_1'] * parameters['mass_2'] / (
                parameters['mass_1'] + parameters['mass_2'])) * cst.Msol * (1 + z)

        parameters['redshift'] = z
        f_isco_z = fisco(parameters)

        Mc = cst.G * mu ** 0.6 * M ** 0.4 / cst.c ** 3

        # compute GW amplitudes (https://arxiv.org/pdf/2012.01350.pdf) with optimal orientation
        hp = cst.c / r * np.sqrt(5. * np.pi / 24.) * Mc ** (5. / 6.) / (np.pi * ff) ** (7. / 6.)
        hp[
            ff > 5 * f_isco_z] = 0  # very crude, but reasonable high-f cut-off; matches roughly IMR spectra (in qadrupole order)
        print(5 * f_isco_z)

        hc = 1.j * hp

        hpij = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
        hcij = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])

        interferometers = detector.interferometers

        # project signal onto the detector
        proj = np.zeros((len(hp), len(interferometers)), dtype=complex)

        for k in np.arange(len(interferometers)):
            if detector.name == 'ET':
                n = interferometers[k].ifo_id
                az = n * np.pi * 2. / 3.
                opening_angle = np.pi / 3.
                e1 = np.array([np.cos(az), np.sin(az), 0.])
                e2 = np.array([np.cos(az + opening_angle), np.sin(az + opening_angle), 0.])
            else:
                e1 = np.array([1., 0., 0.])
                e2 = np.array([0., 1., 0.])

            proj[:, k] = 0.5 * hp[:, 0] * (e1 @ hpij @ e1 - e2 @ hpij @ e2) \
                         + 0.5 * hc[:, 0] * (e1 @ hcij @ e1 - e2 @ hcij @ e2)

        SNRs = det.SNR(interferometers, proj, ff)
        SNRtot = np.sqrt(np.sum(SNRs ** 2))

        # print('z = ' + str(z) + ', r = ' + str(cosmo.luminosity_distance(z).value) + 'Mpc, SNR = '+str(SNRtot))

        return SNRtot - SNR0

    for d in np.arange(len(network.detectors)):
        zmax = optimize.root(lambda x: dSNR(x, network.detectors[d], detSNR[1]), 5).x[0]

        print(network.detectors[d].name + ' horizon (time-invariant antenna pattern; M={:.3f}; SNR>{:.2f}): z={:.3f}'
              .format(parameters['mass_1'] + parameters['mass_2'], detSNR[1], zmax))


def scalar_product(deriv1, deriv2, interferometers, ff):
    if deriv1.ndim == 1:
        deriv1 = deriv1[:, np.newaxis]
        deriv2 = deriv2[:, np.newaxis]

    if ff.ndim == 1:
        ff = ff[:, np.newaxis]

    df = ff[1, 0] - ff[0, 0]

    scalar_prods = np.zeros(len(interferometers))
    for k in np.arange(len(interferometers)):
        scalar_prods[k] = 4 * df * np.sum(
            np.real(deriv1[:, k] * np.conjugate(deriv2[:, k])) / interferometers[k].Sn(ff[:, 0]), axis=0)

    return scalar_prods