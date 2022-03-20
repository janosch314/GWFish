import matplotlib.pyplot as plt
import numpy as np

import GWFish.modules.constants as cst
import GWFish.modules.auxiliary as aux
import GWFish.modules.lalsim_interface as lalsim

def hphc_amplitudes(waveform, parameters, frequencyvector):
    parameters = parameters.copy()
    if waveform=='gwfish_TaylorF2':
        hphc, t_of_f = TaylorF2(parameters, frequencyvector)
    elif waveform[0:7]=='lalbbh_':
        tc = parameters['geocent_time']
        parameters['geocent_time'] = 0
        hphc = lalsim.BBH(waveform[7:], frequencyvector, **parameters)
        phi = np.angle(hphc[:, 0])
        t_of_f = np.diff(phi, axis=0) / (2.*np.pi*(frequencyvector[1]-frequencyvector[0]))
        t_of_f = tc + np.vstack((t_of_f, [t_of_f[-1]]))
    elif waveform[0:7]=='lalbns_':
        tc = parameters['geocent_time']
        parameters['geocent_time'] = 0
        print(parameters)
        hphc = lalsim.BNS(waveform[7:], frequencyvector, **parameters)
        phi = np.angle(hphc[:, 0])
        t_of_f = np.diff(phi, axis=0) / (2. * np.pi * (frequencyvector[1] - frequencyvector[0]))
        t_of_f = tc + np.vstack((t_of_f, [t_of_f[-1]]))
    else:
        print(str(waveform) + ' is not a valid waveform Choose v')
    return hphc, t_of_f

def TaylorF2(parameters, frequencyvector, maxn=8, plot=None):
    ff = frequencyvector
    ones = np.ones((len(ff), 1))

    phic = parameters['phase']
    tc = parameters['geocent_time']
    z = parameters['redshift']
    r = parameters['luminosity_distance'] * cst.Mpc
    iota = parameters['iota']
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

    # t(f) is required to calculate slowly varying antenna pattern as function of instantaneous frequency.
    # This FD approach follows Marsat/Baker arXiv:1806.10734v1; equation (22) neglecting the phase term, which does not
    # matter for SNR calculations.
    t_of_f = np.diff(psi, axis=0) / (2. * np.pi * (ff[1] - ff[0]))
    t_of_f = tc + np.vstack((t_of_f, [t_of_f[-1]]))

    psi += 2. * np.pi * ff * tc - phic - np.pi / 4.
    phase = np.exp(1.j * psi)
    polarizations = np.hstack((hp * phase, hc * 1.j * phase))
    polarizations[np.where(ff > 4 * f_isco), :] = 0.j  # very crude high-f cut-off

    if plot != None:
        plt.figure()
        plt.semilogx(ff, t_of_f - tc)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('t(f)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('t_of_f.png')
        plt.close()

    return polarizations, t_of_f
