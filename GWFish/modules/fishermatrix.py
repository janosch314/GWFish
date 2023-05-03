import numpy as np
import GWFish.modules.waveforms as wf
import GWFish.modules.detection as det
import GWFish.modules.auxiliary as aux
import GWFish.modules.fft as fft

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

def fft_derivs_at_detectors(deriv_list, frequency_vector):
    """
    A wrapper for fft_lal_timeseries
    """
    delta_f = frequency_vector[1,0] - frequency_vector[0,0]
    ffd_deriv_list = []
    for deriv in deriv_list:
        ffd_deriv_list.append(fft.fft_lal_timeseries(deriv, delta_f, f_start=0.).data.data)

    # Because f_start = 0 Hz, we need to mask some frequencies
    idx_f_low = int(frequency_vector[0,0]/delta_f)
    idx_f_high = int(frequency_vector[-1,0]/delta_f)

    return np.vstack(ffd_deriv_list).T[idx_f_low:idx_f_high+1,:]

class Derivative:
    """
    Standard GWFish waveform derivative class, based on finite differencing in frequency domain.
    Calculates derivatives with respect to geocent_time, merger phase, and distance analytically.
    Derivatives of other parameters are calculated numerically.

    eps: 1e-5, this follows the simple "cube root of numerical precision" recommendation, which is 1e-16 for double
    """
    def __init__(self, waveform, parameters, detector, eps=1e-5, waveform_class=wf.Waveform):
        self.waveform = waveform
        self.detector = detector
        self.eps = eps
        self.waveform_class = waveform_class
        self.data_params = {'frequencyvector': detector.frequencyvector}
        self.waveform_object = waveform_class(waveform, parameters, self.data_params)
        self.waveform_at_parameters = None
        self.projection_at_parameters = None

        # For central parameters and their epsilon-neighbourhood
        self.local_params = parameters.copy()
        self.pv_set1 = parameters.copy()
        self.pv_set2 = parameters.copy()

        self.tc = self.local_params['geocent_time']

    @property
    def waveform_at_parameters(self):
        """
        Return a waveform at the point in parameter space determined by the parameters argument.

        Returns tuple, (wave, t_of_f).
        """
        if self._waveform_at_parameters is None:
            wave = self.waveform_object()
            t_of_f = self.waveform_object.t_of_f
            self._waveform_at_parameters = (wave, t_of_f)
        return self._waveform_at_parameters

    @waveform_at_parameters.setter
    def waveform_at_parameters(self, new_waveform_data):
        self._waveform_at_parameters = new_waveform_data

    @property
    def projection_at_parameters(self):
        if self._projection_at_parameters is None:
            self._projection_at_parameters = det.projection(self.local_params, self.detector,
                                                            self.waveform_at_parameters[0], # wave
                                                            self.waveform_at_parameters[1]) # t(f)
        return self._projection_at_parameters

    @projection_at_parameters.setter
    def projection_at_parameters(self, new_projection_data):
        self._projection_at_parameters = new_projection_data

    def with_respect_to(self, target_parameter):
        """
        Return a derivative with respect to target_parameter at the point in 
        parameter space determined by the argument parameters.
        """
        if target_parameter == 'luminosity_distance':
            derivative = -1. / self.local_params[target_parameter] * self.projection_at_parameters
        elif target_parameter == 'geocent_time':
            derivative = 2j * np.pi * self.detector.frequencyvector * self.projection_at_parameters
        elif target_parameter == 'phase':
            derivative = -1j * self.projection_at_parameters
        else:
            pv = self.local_params[target_parameter]

            dp = np.maximum(self.eps, self.eps * pv)

            self.pv_set1 = self.local_params.copy()
            self.pv_set2 = self.local_params.copy()
            self.pv_set1[target_parameter] = pv - dp / 2.
            self.pv_set2[target_parameter] = pv + dp / 2.

            if target_parameter in ['ra', 'dec', 'psi']:  # these parameters do not influence the waveform
    
                signal1 = det.projection(self.pv_set1, self.detector, 
                                         self.waveform_at_parameters[0], 
                                         self.waveform_at_parameters[1])
                signal2 = det.projection(self.pv_set2, self.detector, 
                                         self.waveform_at_parameters[0], 
                                         self.waveform_at_parameters[1])
    
                derivative = (signal2 - signal1) / dp
            else:
                # to improve precision of numerical differentiation
                self.pv_set1['geocent_time'] = 0.
                self.pv_set2['geocent_time'] = 0.

                self.waveform_object.update_gw_params(self.pv_set1)
                wave1 = self.waveform_object()
                t_of_f1 = self.waveform_object.t_of_f

                self.waveform_object.update_gw_params(self.pv_set2)
                wave2 = self.waveform_object()
                t_of_f2 = self.waveform_object.t_of_f                

                self.pv_set1['geocent_time'] = self.tc
                self.pv_set2['geocent_time'] = self.tc
                signal1 = det.projection(self.pv_set1, self.detector, wave1, t_of_f1 + self.tc)
                signal2 = det.projection(self.pv_set2, self.detector, wave2, t_of_f2 + self.tc)
    

                derivative = np.exp(2j * np.pi * self.detector.frequencyvector \
                                    * self.tc) * (signal2 - signal1) / dp

        return derivative

    def __call__(self, target_parameter):
        return self.with_respect_to(target_parameter)

class FisherMatrix:
    def __init__(self, waveform, parameters, fisher_parameters, detector, eps=1e-5, waveform_class=wf.Waveform):
        self.fisher_parameters = fisher_parameters
        self.detector = detector
        self.derivative = Derivative(waveform, parameters, detector, eps=eps, waveform_class=waveform_class)
        self.nd = len(fisher_parameters)
        self.fm = None

    def update_fm(self):
        self._fm = np.zeros((self.nd, self.nd))
        for p1 in np.arange(self.nd):
            deriv1_p = self.fisher_parameters[p1]
            deriv1 = self.derivative(deriv1_p)
            self._fm[p1, p1] = np.sum(aux.scalar_product(deriv1, deriv1, self.detector), axis=0)
            for p2 in np.arange(p1+1, self.nd):
                deriv2_p = self.fisher_parameters[p2]
                deriv2 = self.derivative(deriv2_p)
                self._fm[p1, p2] = np.sum(aux.scalar_product(deriv1, deriv2, self.detector), axis=0)
                self._fm[p2, p1] = self._fm[p1, p2]

    @property
    def fm(self):
        if self._fm is None:
            self.update_fm()
        return self._fm

    @fm.setter
    def fm(self, hardcode_fm):
        self._fm = hardcode_fm

    def __call__(self):
        return self.fm

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
                       save_data, delimiter=' ', fmt='%s' + " %.3E" * (len(save_data[0, :]) - 1), header=header, comments='')
        else:
            np.savetxt('Errors_' + network_names[n] + '_' + population + '_SNR' + str(detect_SNR[1]) + '.txt',
                       save_data, delimiter=' ', fmt='%s' + " %.3E" * (len(save_data[0, :]) - 1), header=header, comments='')

