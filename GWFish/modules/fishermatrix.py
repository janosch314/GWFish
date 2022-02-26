import numpy as np
import GWFish.modules.waveforms as wf
import GWFish.modules.detection as det
import GWFish.modules.auxiliary as aux
import GWFish.modules.constants as cst

def invertSVD(matrix):
    dm = np.sqrt(np.diag(matrix))
    normalizer = np.outer(dm, dm)
    matrix_norm = matrix / normalizer

    [U, S, Vh] = np.linalg.svd(matrix_norm)
    thresh = 1e-10
    kVal = sum(S > thresh)
    matrix_inverse_norm = U[:, 0:kVal] @ np.diag(1. / S[0:kVal]) @ Vh[0:kVal, :]
    # print(matrix @ (matrix_inverse_norm / normalizer))

    return matrix_inverse_norm / normalizer


def derivative(parameters, p, detector, max_time_until_merger):
    """
    Calculates derivatives with respect to geocent_time, merger phase, and distance analytically.
    Derivatives of other parameters are calculated numerically.
    """

    fisher_parameters = ['ra', 'dec', 'psi', 'iota', 'luminosity_distance', 'mass_1', 'mass_2', 'geocent_time', 'phase']
    # fisher_parameters = ['ra', 'dec', 'psi', 'iota', 'redshift', 'mass_1', 'mass_2', 'geocent_time', 'phase']

    local_params = parameters.copy()

    tc = local_params['geocent_time']

    if p == 4:  # derivative wrt luminosity distance
        wave, t_of_f = wf.TaylorF2(local_params, detector.frequencyvector, maxn=8)
        r = local_params['luminosity_distance'] * cst.Mpc
        derivative = -1. / r * det.projection(local_params, detector, wave, t_of_f, max_time_until_merger)
    elif p == 7:    # derivative wrt merger time
        wave, t_of_f = wf.TaylorF2(local_params, detector.frequencyvector, maxn=8)

        derivative = 2j * np.pi * detector.frequencyvector * det.projection(local_params, detector, wave, t_of_f, max_time_until_merger)
    elif p == 8:    # derivative wrt phase parameter
        wave, t_of_f = wf.TaylorF2(local_params, detector.frequencyvector, maxn=8)

        derivative = -1j * det.projection(local_params, detector, wave, t_of_f, max_time_until_merger)
    else:
        pv = local_params[fisher_parameters[p]]
        eps = 1e-5  # this follows the simple "cube root of numerical precision" recommendation, which is 1e-16 for double
        dp = np.maximum(eps, eps * pv)

        pv_set1 = parameters.copy()
        pv_set2 = parameters.copy()

        pv_set1[fisher_parameters[p]] = pv - dp / 2.
        pv_set2[fisher_parameters[p]] = pv + dp / 2.

        if p < 3:  # these parameters do not influence the waveform
            wave, t_of_f = wf.TaylorF2(local_params, detector.frequencyvector, maxn=8)

            signal1 = det.projection(pv_set1, detector, wave, t_of_f, max_time_until_merger)
            signal2 = det.projection(pv_set2, detector, wave, t_of_f, max_time_until_merger)

            derivative = (signal2 - signal1) / dp
        else:
            pv_set1['geocent_time'] = 0.  # to improve precision of numerical differentiation
            pv_set2['geocent_time'] = 0.
            wave1, t_of_f1 = wf.TaylorF2(pv_set1, detector.frequencyvector, maxn=8)
            wave2, t_of_f2 = wf.TaylorF2(pv_set2, detector.frequencyvector, maxn=8)

            pv_set1['geocent_time'] = tc
            pv_set2['geocent_time'] = tc
            signal1 = det.projection(pv_set1, detector, wave1, t_of_f1+tc, max_time_until_merger)
            signal2 = det.projection(pv_set2, detector, wave2, t_of_f2+tc, max_time_until_merger)

            derivative = np.exp(2j * np.pi * detector.frequencyvector * tc) * (signal2 - signal1) / dp

    # print(fisher_parameters[p] + ': ' + str(derivative))
    return derivative


def FisherMatrix(parameters, detector, max_time_until_merger):
    num_p = 9
    fm = np.zeros((num_p, num_p))

    for p1 in np.arange(num_p):
        deriv1 = derivative(parameters, p1, detector, max_time_until_merger)
        # sum Fisher matrices from different components of same detector (e.g., in the case of ET)
        fm[p1, p1] = np.sum(aux.scalar_product(deriv1, deriv1, detector), axis=0)
        for p2 in np.arange(p1 + 1, num_p):
            deriv2 = derivative(parameters, p2, detector, max_time_until_merger)
            fm[p1, p2] = np.sum(aux.scalar_product(deriv1, deriv2, detector), axis=0)
            fm[p2, p1] = fm[p1, p2]

    return fm


def analyzeFisherErrors(network, parameters, population, networks_ids):
    """
    Analyze parameter errors with respect to the following list of waveform parameters:
        ['ra', 'dec', 'psi', 'iota', 'redshift', 'mass_1', 'mass_2', 'geocent_time', 'phase']
    """
    # param_names = ['ra', 'dec', 'psi', 'iota', 'redshift', 'mass_1', 'mass_2', 'geocent_time', 'phase']
    # units = ['rad', 'rad', 'rad', 'rad', '', 'M_sol', 'M_sol', 's', 'rad']
    param_names = ['ra', 'dec', 'psi', 'iota', 'luminosity_distance', 'mass_1', 'mass_2', 'geocent_time', 'phase']
    units = ['rad', 'rad', 'rad', 'rad', 'Mpc', 'M_sol', 'M_sol', 's', 'rad']

    npar = len(param_names)
    ns = len(network.detectors[0].fisher_matrix[:, 0, 0])  # number of signals
    N = len(networks_ids)

    detect_SNR = network.detection_SNR

    network_names = []
    for n in np.arange(N):
        network_names.append('_'.join([network.detectors[k].name for k in networks_ids[n]]))

    for n in np.arange(N):
        parameter_errors = np.zeros((ns, npar))
        sky_localization = np.zeros((ns,))
        networkSNR = np.zeros((ns,))
        for d in networks_ids[n]:
            networkSNR += network.detectors[d].SNR ** 2
        networkSNR = np.sqrt(networkSNR)

        for k in np.arange(ns):
            network_fisher_matrix = np.zeros((npar, npar))

            if networkSNR[k] > detect_SNR[1]:
                for d in networks_ids[n]:
                    if network.detectors[d].SNR[k] > detect_SNR[0]:
                        network_fisher_matrix += np.squeeze(network.detectors[d].fisher_matrix[k, :, :])

            if network_fisher_matrix[0, 0] > 0:
                network_fisher_matrix[4, :] *= cst.Mpc  # changing to D_lum error unit to Mpc
                network_fisher_matrix[:, 4] *= cst.Mpc
                network_fisher_inverse = invertSVD(network_fisher_matrix)
                parameter_errors[k, :] = np.sqrt(np.diagonal(network_fisher_inverse))
                sky_localization[k] = 2. * np.pi * np.abs(np.cos(parameters['dec'].iloc[k])) \
                                      * np.sqrt(
                    network_fisher_inverse[0, 0] * network_fisher_inverse[1, 1] - network_fisher_inverse[0, 1] ** 2)

        # ii = np.array(np.where(networkSNR > detect_SNR[1]))
        ii = np.where(networkSNR > detect_SNR[1])[0]
        save_data = np.c_[networkSNR[ii], parameters['redshift'].iloc[ii], parameters[param_names].iloc[ii],
                          sky_localization[ii], parameter_errors[ii, :]]
        np.savetxt('Errors_' + network_names[n] + '_' + population + '_SNR' + str(detect_SNR[1]) + '.txt', save_data,
                   delimiter=' ')
