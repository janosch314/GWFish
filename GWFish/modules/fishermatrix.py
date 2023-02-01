import numpy as np
import GWFish.modules.waveforms as wf
import GWFish.modules.detection as det
import GWFish.modules.auxiliary as aux

import lalsimulation as lalsim
import lal

# TEMPORARY, REMOVE
import copy

def invertSVD(matrix):
    thresh = 1e-10

    dm = np.sqrt(np.diag(matrix))
    normalizer = np.outer(dm, dm)
    matrix_norm = matrix / normalizer

    [U, S, Vh] = np.linalg.svd(matrix_norm)

    kVal = sum(S > thresh)
    matrix_inverse_norm = U[:, 0:kVal] @ np.diag(1. / S[0:kVal]) @ Vh[0:kVal, :]

    # print(matrix @ (matrix_inverse_norm / normalizer))

    return matrix_inverse_norm / normalizer, S


def derivative(waveform, parameter_values, pp, detector, time_domain=False, eps=1e-5):

    """
    eps: 1e-5, this follows the simple "cube root of numerical precision" recommendation, which is 1e-16 for double

    Calculates derivatives with respect to geocent_time, merger phase, and distance analytically.
    Derivatives of other parameters are calculated numerically.
    """

    local_params = parameter_values.copy()

    tc = local_params['geocent_time']

    # Just for time series waveforms
    wave, t_of_f = wf.hphc_amplitudes(waveform, local_params, detector.frequencyvector, preserve_lal_timeseries=time_domain)
    if time_domain:
        strain_lal_template = wave[1]
        wave = wave[0]

    if pp == 'luminosity_distance':
        wave, t_of_f = wf.hphc_amplitudes(waveform, local_params, detector.frequencyvector, time_domain=time_domain)
        derivative = -1. / local_params[pp] * det.projection(local_params, detector, wave, t_of_f)
    elif pp == 'geocent_time' and not time_domain:
        wave, t_of_f = wf.hphc_amplitudes(waveform, local_params, detector.frequencyvector, time_domain=time_domain)
        derivative = 2j * np.pi * detector.frequencyvector * det.projection(local_params, detector, wave, t_of_f)
    elif pp == 'phase' and not time_domain:
        wave, t_of_f = wf.hphc_amplitudes(waveform, local_params, detector.frequencyvector, time_domain=time_domain)
        derivative = -1j * det.projection(local_params, detector, wave, t_of_f)
    else:
        pv = local_params[pp]

        dp = np.maximum(eps, eps * pv)

        pv_set1 = parameter_values.copy()
        pv_set2 = parameter_values.copy()

        if pp == 'mass_ratio':
          print('mass ratio')
          pv_set1[pp] = pv
          pv_set2[pp] = pv + dp
        else:
          pv_set1[pp] = pv - dp / 2.
          pv_set2[pp] = pv + dp / 2.

        if pp in ['ra', 'dec', 'psi']:  # these parameters do not influence the waveform
            wave, t_of_f = wf.hphc_amplitudes(waveform, local_params, detector.frequencyvector, time_domain=time_domain)

            signal1 = det.projection(pv_set1, detector, wave, t_of_f)
            signal2 = det.projection(pv_set2, detector, wave, t_of_f)

            derivative = (signal2 - signal1) / dp
        else:
            pv_set1['geocent_time'] = 0.  # to improve precision of numerical differentiation
            pv_set2['geocent_time'] = 0.
            wave1, t_of_f1 = wf.hphc_amplitudes(waveform, pv_set1, detector.frequencyvector, time_domain=time_domain)
            wave2, t_of_f2 = wf.hphc_amplitudes(waveform, pv_set2, detector.frequencyvector, time_domain=time_domain)

            pv_set1['geocent_time'] = tc
            pv_set2['geocent_time'] = tc
            signal1 = det.projection(pv_set1, detector, wave1, t_of_f1+tc)
            signal2 = det.projection(pv_set2, detector, wave2, t_of_f2+tc)

            if not time_domain:
                derivative = np.exp(2j * np.pi * detector.frequencyvector * tc) * (signal2 - signal1) / dp
            else:
                derivative = (signal2 - signal1) / dp # THIS MUST BE FIXED

    # print(fisher_parameters[p] + ': ' + str(derivative))

    if not time_domain:
        return derivative
    else:
        derivative = np.real(derivative) # Removing zero imaginary part added in det.projection()
        n_times, n_detectors = derivative.shape
        dstrain_dtheta_at_detector = []
        for ii_det in range(n_detectors):
            derivative_REAL8Sequence = lal.CreateREAL8Sequence(n_times)
            derivative_REAL8Sequence.data = derivative[:,ii_det]
            strain_lal_template.data = derivative_REAL8Sequence
            strain_lal_template.name = 'STRAIN_DERIVATIVE'
            dstrain_dtheta_at_detector.append(copy.copy(strain_lal_template))
        return dstrain_dtheta_at_detector

# IMPORTANT NOTE FOR HIGH-ORDER DERIVATIVE:
# Perhaps one-sided derivatives are better than two-sided?
# This can also cause error for parameters symmetric around maximum-likelihood points, e.g. q=1, theta_jn=0.

#def derivative(waveform, parameter_values, p, detector):
#
#    """
#    Calculates derivatives with respect to geocent_time, merger phase, and distance analytically.
#    Derivatives of other parameters are calculated numerically.
#    """
#
#    local_params = parameter_values.copy()
#    local_params_deriv = parameter_values.copy()
#
#    tc = local_params['geocent_time']
#
#    # Can be pre-determined:
#    err_order = 8
#    grr = int(err_order/2) # grid range
#    stencil_coeffs = {-4: 1/280, -3: -4/105, -2: 1/5, -1: -4/5, 0: 0, 1: 4/5, 2: -1/5, 3: 4/105, 4: -1/280}
#
#    if p == 'luminosity_distance':
#        wave, t_of_f = wf.hphc_amplitudes(waveform, local_params, detector.frequencyvector)
#        derivative = -1. / local_params[p] * det.projection(local_params, detector, wave, t_of_f)
#    elif p == 'geocent_time':
#        wave, t_of_f = wf.hphc_amplitudes(waveform, local_params, detector.frequencyvector)
#        derivative = 2j * np.pi * detector.frequencyvector * det.projection(local_params, detector, wave, t_of_f)
#    elif p == 'phase':
#        wave, t_of_f = wf.hphc_amplitudes(waveform, local_params, detector.frequencyvector)
#        derivative = -1j * det.projection(local_params, detector, wave, t_of_f)
#    else:
#        pv = local_params[p]
#        eps = 1e-5  # 1e-5, this follows the simple "cube root of numerical precision" recommendation, which is 1e-16 for double
#        dp = np.maximum(eps, eps * pv)
#
#        pv_stencils = {key: local_params[p]+dp*stencil_coeffs[key] for key in range(-grr,grr+1)}
#
#        if p in ['ra', 'dec', 'psi']:  # these parameters do not influence the waveform
#            wave, t_of_f = wf.hphc_amplitudes(waveform, local_params, detector.frequencyvector)
#            signal_stencils = []
#            for pvs in pv_stencils.values():
#                local_params_deriv.update({p:pvs})
#                signal_stencils.append(det.projection(local_params_deriv, detector, wave, t_of_f))
#            derivative = np.sum(signal_stencils,axis=0) / dp
#        else:
#            signal_stencils = []
#            for pvs in pv_stencils.values():
#                local_params_deriv.update({p:pvs})
#                local_params_deriv['geocent_time'] = 0.
#                wave, t_of_f = wf.hphc_amplitudes(waveform, local_params_deriv, detector.frequencyvector)
#                local_params_deriv['geocent_time'] = tc
#                signal_stencils.append(det.projection(local_params_deriv, detector, wave, t_of_f+tc))
#
#            derivative = np.exp(2j * np.pi * detector.frequencyvector * tc) * np.sum(signal_stencils,axis=0) / dp
#
#    # print(fisher_parameters[p] + ': ' + str(derivative))
#    return derivative

# ========= Functions for LAL FFD of derivatives of time-domain waveforms ========= #

def fft_lal_timeseries(lal_timeseries, delta_f, f_start=0.):
    """

    f_start: not recommended to change, f_start=0 is in lalsim.SimInspiralFD when calling time-domain waveforms

    Applying a Fourier transform, as in LALSimulation
    https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiral.c#L3044
    Note, time-domain data is previously conditioned (tapered) in lalsim.SimInspiralTD, 
    and resized in waveform.td_lal_caller(), as in lalsim.SimInspiralFD.
    """

    chirplen = lal_timeseries.data.length
    lal_frequency_series = lal.CreateCOMPLEX16FrequencySeries('FD_H', lal_timeseries.epoch, f_start, delta_f, 
                           lal.DimensionlessUnit,int(chirplen / 2 + 1))
    plan = lal.CreateForwardREAL8FFTPlan(chirplen,0)
    lal.REAL8TimeFreqFFT(lal_frequency_series,lal_timeseries,plan)
    return lal_frequency_series

def fft_derivs_at_detectors(deriv_list, frequency_vector):
    """
    A wrapper for fft_lal_timeseries
    """
    delta_f = frequency_vector[1,0] - frequency_vector[0,0]
    ffd_deriv_list = []
    for deriv in deriv_list:
        ffd_deriv_list.append(fft_lal_timeseries(deriv, delta_f, f_start=0.).data.data)

    # Because f_start = 0 Hz, we need to mask some frequencies
    idx_f_low = int(frequency_vector[0,0]/delta_f)
    idx_f_high = int(frequency_vector[-1,0]/delta_f)

    return np.vstack(ffd_deriv_list).T[idx_f_low:idx_f_high+1,:]

# ================================================================================ # 

def FisherMatrix(waveform, parameter_values, fisher_parameters, detector, time_domain=False):

    nd = len(fisher_parameters)
    fm = np.zeros((nd, nd))

    for p1 in np.arange(nd):
        deriv1_p = fisher_parameters[p1]
        deriv1 = derivative(waveform, parameter_values, deriv1_p, detector, time_domain=time_domain)
        if time_domain:
            deriv1 = fft_derivs_at_detectors(deriv1, detector.frequencyvector)

        # sum Fisher matrices from different components of same detector (e.g., in the case of ET)

        fm[p1, p1] = np.sum(aux.scalar_product(deriv1, deriv1, detector), axis=0)
        for p2 in np.arange(p1+1, nd):
            deriv2_p = fisher_parameters[p2]
            deriv2 = derivative(waveform, parameter_values, deriv2_p, detector, time_domain=time_domain)
            if time_domain:
                deriv2 = fft_derivs_at_detectors(deriv2, detector.frequencyvector)

            fm[p1, p2] = np.sum(aux.scalar_product(deriv1, deriv2, detector), axis=0)

            fm[p2, p1] = fm[p1, p2]

    return fm


def analyzeFisherErrors(network, parameter_values, fisher_parameters, population, networks_ids):
    """
    Analyze parameter errors.
    """

    # Check if sky-location parameters are part of Fisher analysis. If yes, sky-location error will be calculated.
    signals_havesky = False
    if ('ra' in fisher_parameters) and ('dec' in fisher_parameters):
        signals_havesky = True
        i_ra = fisher_parameters.index('ra')
        i_dec = fisher_parameters.index('dec')
    signals_haveids = False
    if 'id' in parameter_values.columns:
        signals_haveids = True
        signal_ids = parameter_values['id']
        parameter_values.drop('id', inplace=True, axis=1)


    npar = len(fisher_parameters)
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
        fishers = np.zeros((ns, npar, npar))
        inv_fishers = np.zeros((ns, npar, npar))
        sing_values = np.zeros((ns, npar))
        
        for d in networks_ids[n]:
            networkSNR += network.detectors[d].SNR ** 2
        networkSNR = np.sqrt(networkSNR)

        for k in np.arange(ns):
            network_fisher_matrix = np.zeros((npar, npar))

            if networkSNR[k] > detect_SNR[1]:
                for d in networks_ids[n]:
                    if network.detectors[d].SNR[k] > detect_SNR[0]:
                        network_fisher_matrix += np.squeeze(network.detectors[d].fisher_matrix[k, :, :])

                if npar > 0:
                    network_fisher_inverse, S = invertSVD(network_fisher_matrix)
                    fishers[k, :, :] = network_fisher_matrix
                    inv_fishers[k, :, :] = network_fisher_inverse
                    sing_values[k, :] = S
                    parameter_errors[k, :] = np.sqrt(np.diagonal(network_fisher_inverse))

                    if signals_havesky:
                        sky_localization[k] = np.pi * np.abs(np.cos(parameter_values['dec'].iloc[k])) \
                                              * np.sqrt(network_fisher_inverse[i_ra, i_ra]*network_fisher_inverse[i_dec, i_dec]
                                                        -network_fisher_inverse[i_ra, i_dec]**2)
        delim = " "
        header = 'network_SNR '+delim.join(parameter_values.keys())+" "+delim.join(["err_" + x for x in fisher_parameters])

        ii = np.where(networkSNR > detect_SNR[1])[0]
        save_data = np.c_[networkSNR[ii], parameter_values.iloc[ii], parameter_errors[ii, :]]
        fishers = fishers[ii, :, :]
        inv_fishers = inv_fishers[ii, :, :]
        sing_values = sing_values[ii, :]
        
        np.save('Fishers_'+ network_names[n] + '_' + population + '_SNR' + str(detect_SNR[1]) + '.npy', fishers)
        np.save('Inv_Fishers_'+ network_names[n] + '_' + population + '_SNR' + str(detect_SNR[1]) + '.npy', inv_fishers)
        np.save('Sing_Values_'+ network_names[n] + '_' + population + '_SNR' + str(detect_SNR[1]) + '.npy', sing_values)
        
        if signals_havesky:
            header += " err_sky_location"
            save_data = np.c_[save_data, sky_localization[ii]]
        if signals_haveids:
            header = "signal "+header
            save_data = np.c_[signal_ids.iloc[ii], save_data]

        file_name = 'Errors_' + network_names[n] + '_' + population + '_SNR' + str(detect_SNR[1]) + '.txt'

        if signals_haveids and (len(save_data) > 0):
            np.savetxt('Errors_' + network_names[n] + '_' + population + '_SNR' + str(detect_SNR[1]) + '.txt',
                       save_data, delimiter=' ', fmt='%s ' + "%.3E " * (len(save_data[0, :]) - 1), header=header, comments='')
        else:
            np.savetxt('Errors_' + network_names[n] + '_' + population + '_SNR' + str(detect_SNR[1]) + '.txt',
                       save_data, delimiter=' ', fmt='%s ' + "%.3E " * (len(save_data[0, :]) - 1), header=header, comments='')

