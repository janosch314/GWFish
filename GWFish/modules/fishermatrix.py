import numpy as np
import GWFish.modules.waveforms as wf
import GWFish.modules.detection as det
import GWFish.modules.auxiliary as aux
import GWFish.modules.fft as fft

import copy
import pandas as pd
from typing import Optional, Union

from tqdm import tqdm

import logging
from pathlib import Path

def invertSVD(matrix):
    thresh = 1e-10

    dm = np.sqrt(np.diag(matrix))
    normalizer = np.outer(dm, dm)
    matrix_norm = matrix / normalizer

    [U, S, Vh] = np.linalg.svd(matrix_norm)

    kVal = sum(S > thresh)
    
    logging.debug(f'Inverting a matrix keeping {kVal}/{len(S)} singular values')
    
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
    def __init__(self, waveform, parameters, detector, eps=1e-5, eps_mass=1e-8, waveform_class=wf.Waveform, f_ref=wf.DEFAULT_F_REF):
        self.waveform = waveform
        self.detector = detector
        self.eps = eps
        self.eps_mass = eps_mass
        self.waveform_class = waveform_class
        self.data_params = {'frequencyvector': detector.frequencyvector, 'f_ref': f_ref}
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

            if target_parameter in ['chirp_mass', 'chirp_mass_source', 'mass_1', 'mass_2', 'mass_1_source', 'mass_2_source']:
                dp = self.eps_mass * pv
            else:
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

                waveform_obj1 = self.waveform_class(self.waveform, self.pv_set1, self.data_params)
                wave1 = waveform_obj1()
                t_of_f1 = waveform_obj1.t_of_f

                waveform_obj2 = self.waveform_class(self.waveform, self.pv_set2, self.data_params)
                wave2 = waveform_obj2()
                t_of_f2 = waveform_obj2.t_of_f                

                self.pv_set1['geocent_time'] = self.tc
                self.pv_set2['geocent_time'] = self.tc
                signal1 = det.projection(self.pv_set1, self.detector, wave1, t_of_f1 + self.tc)
                signal2 = det.projection(self.pv_set2, self.detector, wave2, t_of_f2 + self.tc)
    

                derivative = np.exp(2j * np.pi * self.detector.frequencyvector \
                                    * self.tc) * (signal2 - signal1) / dp
                                    
        self.waveform_object.update_gw_params(self.local_params)

        return derivative

    def __call__(self, target_parameter):
        return self.with_respect_to(target_parameter)

class FisherMatrix:
    def __init__(self, waveform, parameters, fisher_parameters, detector, eps=1e-5, eps_mass=1e-8, waveform_class=wf.Waveform, f_ref=wf.DEFAULT_F_REF):
        self.fisher_parameters = fisher_parameters
        self.detector = detector
        self.derivative = Derivative(waveform, parameters, detector, eps=eps, eps_mass=eps_mass, waveform_class=waveform_class, f_ref=f_ref)
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

def sky_localization_area(
    network_fisher_inverse: np.ndarray,
    declination_angle: np.ndarray,
    right_ascension_index: int,
    declination_index: int,
) -> float:
    """
    Compute the 1-sigma sky localization ellipse area starting
    from the full network Fisher matrix inverse and the inclination.
    """
    return (
        np.pi
        * np.abs(np.cos(declination_angle))
        * np.sqrt(
            network_fisher_inverse[right_ascension_index, right_ascension_index]
            * network_fisher_inverse[declination_index, declination_index]
            - network_fisher_inverse[right_ascension_index, declination_index] ** 2
        )
    )

def sky_localization_percentile_factor(
    percentile: float=90.) -> float:
    """Conversion factor $C_{X\%}$ to go from the sky localization area provided 
    by GWFish (one sigma, in steradians) to the X% contour, in degrees squared.
    
    $$ \Delta \Omega_{X\%} = C_{X\%} \Delta \Omega_{\\text{GWFish output}} $$
    
    :param percentile: Percentile of the sky localization area.
    
    :return: Conversion factor $C_{X\%}$
    """
    
    return - 2 * np.log(1 - percentile / 100.) * (180 / np.pi)**2

def compute_detector_fisher(
    detector: det.Detector,
    signal_parameter_values: Union[pd.DataFrame, dict[str, float]],
    fisher_parameters: Optional[list[str]] = None,
    f_ref = wf.DEFAULT_F_REF,
    waveform_model: str = wf.DEFAULT_WAVEFORM_MODEL,
    waveform_class: type(wf.Waveform) = wf.LALFD_Waveform,
    use_duty_cycle: bool = False,
    redefine_tf_vectors: bool = False,
    long_wavelength: bool = True,
    eps: float = 1e-5,
    eps_mass: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """Compute the Fisher matrix and SNR for a single detector.
    
    Example usage:
    
    ```
    >>> from GWFish.modules.detection import Detector
    >>> detector = Detector('ET')
    >>> params = {
    ...    'mass_1': 10.,
    ...    'mass_2': 10.,
    ...    'luminosity_distance': 1000.,
    ...    'theta_jn': 0.,
    ...    'ra': 0.,
    ...    'dec': 0.,
    ...    'phase': 0.,
    ...    'psi': 0.,
    ...    'geocent_time': 1e9,
    ...    }
    >>> fisher, detector_SNR_square = compute_detector_fisher(detector, params)
    >>> print(fisher.shape)
    (9, 9)
    >>> print(f'{np.sqrt(detector_SNR_square):.0f}')
    260
    
    ```
    
    :param detector: The detector to compute the Fisher matrix for
    :param signal_parameter_values: The parameter values for the signal. They can be a dictionary of parameter names and values, or a single-row pandas DataFrame with the parameter names as columns.
    :param fisher_parameters: The parameters to compute the Fisher matrix for. If None, all parameters are used.
    :param waveform_model: The waveform model to use (see [choosing an approximant](../how-to/choosing_an_approximant.md));
    :param waveform_class: The waveform class to use (see [choosing an approximant](../how-to/choosing_an_approximant.md));
    :param use_duty_cycle: Whether to use the detector duty cycle (i.e. stochastically set the SNR to zero some of the time); defaults to `False`
    :param redefine_tf_vectors: Whether to redefine the time-frequency vectors in order to correctly model signals with small frequency evolution. Defaults to `False`.
    
    :return: The Fisher matrix, and the square of the detector SNR.
    """
    data_params = {
        'frequencyvector': detector.frequencyvector,
        'f_ref': f_ref
    }
    waveform_obj = waveform_class(waveform_model, signal_parameter_values, data_params)
    wave = waveform_obj()
    t_of_f = waveform_obj.t_of_f

    if redefine_tf_vectors:
        signal, timevector, frequencyvector = det.projection(signal_parameter_values, detector, wave, t_of_f, redefine_tf_vectors=True, long_wavelength_approx = long_wavelength)
    else:
        signal = det.projection(signal_parameter_values, detector, wave, t_of_f, long_wavelength_approx = long_wavelength)
        frequencyvector = detector.frequencyvector[:, 0]

    component_SNRs = det.SNR(detector, signal, use_duty_cycle, frequencyvector=frequencyvector)
    detector_SNR_square = np.sum(component_SNRs ** 2)

    if fisher_parameters is None:
        if isinstance(signal_parameter_values, dict):
            fisher_parameters = list(signal_parameter_values.keys())
        else:
            fisher_parameters = signal_parameter_values.columns

    return FisherMatrix(waveform_model, signal_parameter_values, fisher_parameters, detector, waveform_class=waveform_class, f_ref=f_ref, eps=eps, eps_mass=eps_mass).fm, detector_SNR_square

def compute_network_errors(
    network: det.Network,
    parameter_values: pd.DataFrame,
    fisher_parameters: Optional[list[str]] = None,
    f_ref = wf.DEFAULT_F_REF,
    waveform_model: str = wf.DEFAULT_WAVEFORM_MODEL,
    waveform_class = wf.LALFD_Waveform,
    use_duty_cycle: bool = False,
    redefine_tf_vectors: bool = False,
    save_matrices: bool = False,
    save_matrices_path: Union[Path, str] = Path('.'),
    matrix_naming_postfix: str = '',
    long_wavelength: bool = True,
    eps: float = 1e-5,
    eps_mass: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Compute Fisher matrix errors for a network whose
    SNR and Fisher matrices have already been calculated.

    Will only return output for the `n_above_thr` signals 
    for which the network SNR is above `network.detection_SNR[1]`.
    
    :param network: detector network to use
    :param parameter_values: dataframe with parameters for one or more signals
    :param fisher_parameters: list of parameters to use for the Fisher matrix analysis - if `None` (default), all waveform parameters are used
    :param waveform_model: waveform model to use - refer to [choosing an approximant](../how-to/choosing_an_approximant.md)
    :param waveform_model: waveform class to use - refer to [choosing an approximant](../how-to/choosing_an_approximant.md)
    :param redefine_tf_vectors: Whether to redefine the time-frequency vectors in order to correctly model signals with small frequency evolution. Defaults to `False`.
    :param use_duty_cycle: Whether to use the detector duty cycle (i.e. stochastically set the SNR to zero some of the time); defaults to `False`
    :param save_matrices: Whether to save the Fisher matrices and their inverses to disk; defaults to `False`
    :param save_matrices_path: Path (expressed with Pathlib or through a string) where  to save the Fisher matrices and their inverses to disk; defaults to `Path('.')` (the current folder)
    :param matrix_naming_postfix: string to be appended to the names of the Fisher matrices and their inverses: they will look like `fisher_matrices_postfix.npy` and `inv_fisher_matrices_postfix.npy`
    
    :return:
    - `detected`: array with shape `(n_above_thr,)` - array of indices for the detected signals.
    - `network_snr`: array with shape `(n_signals,)` - Network SNR for all signals.
    - `parameter_errors`: array with shape `(n_signals, n_parameters)` - One-sigma     Fisher errors for the parameters.
    - `sky_localization`: array with shape `(n_signals,)` or `None` - One-sigma sky localization area in steradians, returned if the signals have both right ascension and declination, or `None` otherwise.
    """

    if fisher_parameters is None:
        fisher_parameters = list(parameter_values.keys())
        
    if 'max_frequency_cutoff' in fisher_parameters:
        fisher_parameters.remove('max_frequency_cutoff')
    
    if 'redshift' in fisher_parameters:
        fisher_parameters.remove('redshift')

    n_params = len(fisher_parameters)
    n_signals = len(parameter_values)

    assert n_params > 0
    assert n_signals > 0
    
    if isinstance(save_matrices_path, str):
        save_matrices_path = Path(save_matrices_path)
    
    if save_matrices:
        save_matrices_path.mkdir(parents=True, exist_ok=True)
        fisher_matrices = np.zeros((n_signals, n_params, n_params))
        inv_fisher_matrices = np.zeros((n_signals, n_params, n_params))

    signals_havesky = False
    if ("ra" in fisher_parameters) and ("dec" in fisher_parameters):
        signals_havesky = True
        i_ra = fisher_parameters.index("ra")
        i_dec = fisher_parameters.index("dec")

    detector_snr_thr, network_snr_thr = network.detection_SNR

    parameter_errors = np.zeros((n_signals, n_params))
    if signals_havesky:
        sky_localization = np.zeros((n_signals,))
    network_snr = np.zeros((n_signals,))

    for k in tqdm(range(n_signals)):
        network_fisher_matrix = np.zeros((n_params, n_params))

        network_snr_square = 0.
        
        signal_parameter_values = parameter_values.iloc[k]

        for detector in network.detectors:
            
            detector_fisher, detector_snr_square = compute_detector_fisher(detector, signal_parameter_values, fisher_parameters, f_ref, waveform_model, 
                                                                           waveform_class, use_duty_cycle, long_wavelength = long_wavelength,
                                                                           eps=eps, eps_mass=eps_mass)
            
            network_snr_square += detector_snr_square
        
            if np.sqrt(detector_snr_square) > detector_snr_thr:
                network_fisher_matrix += detector_fisher

        network_fisher_inverse, _ = invertSVD(network_fisher_matrix)
        
        if save_matrices:
            fisher_matrices[k, :, :] = network_fisher_matrix
            inv_fisher_matrices[k, :, :] = network_fisher_inverse
        
        parameter_errors[k, :] = np.sqrt(np.diagonal(network_fisher_inverse))

        network_snr[k] = np.sqrt(network_snr_square)

        if signals_havesky:
            sky_localization[k] = sky_localization_area(
                network_fisher_inverse, parameter_values["dec"].iloc[k], i_ra, i_dec
            )

    detected, = np.where(network_snr > network_snr_thr)

    if save_matrices:
        
        if matrix_naming_postfix != '':
            if not matrix_naming_postfix.startswith('_'):
                matrix_naming_postfix = f'_{matrix_naming_postfix}'
        
        fisher_matrices = fisher_matrices[detected, :, :]
        inv_fisher_matrices = inv_fisher_matrices[detected, :, :]
        
        np.save(save_matrices_path /  f"fisher_matrices{matrix_naming_postfix}.npy", fisher_matrices)
        np.save(save_matrices_path /  f"inv_fisher_matrices{matrix_naming_postfix}.npy", inv_fisher_matrices)

    if signals_havesky:
        return (
            detected,
            network_snr,
            parameter_errors,
            sky_localization,
        )

    return detected, network_snr, parameter_errors, None


def errors_file_name(
    network: det.Network, sub_network_ids: list[int], population_name: str
) -> str:

    sub_network = "_".join([network.detectors[k].name for k in sub_network_ids])

    return (
        f"Errors_{sub_network}_{population_name}_SNR{network.detection_SNR[1]:.0f}")


def output_to_txt_file(
    parameter_values: pd.DataFrame,
    network_snr: np.ndarray,
    parameter_errors: np.ndarray,
    sky_localization: Optional[np.ndarray],
    fisher_parameters: list[str],
    filename: Union[str, Path],
    decimal_output_format: str = '%.3E'
) -> None:

    if isinstance(filename, str):
        filename = Path(filename)
    
    delim = " "
    header = (
        "network_SNR "
        + delim.join(parameter_values.keys())
        + " "
        + delim.join(["err_" + x for x in fisher_parameters])
    )
    save_data = np.c_[network_snr, parameter_values, parameter_errors]
    if sky_localization is not None:
        header += " err_sky_location"
        save_data = np.c_[save_data, sky_localization]

    row_format = "%s " + " ".join([decimal_output_format for _ in range(save_data.shape[1] - 1)])

    np.savetxt(
        filename.with_suffix(".txt"),
        save_data,
        delimiter=" ",
        header=header,
        comments="",
        fmt=row_format,
    )

def analyze_and_save_to_txt(
    network: det.Network,
    parameter_values: pd.DataFrame,
    fisher_parameters: list[str],
    sub_network_ids_list: list[list[int]],
    population_name: str,
    save_path: Optional[Union[Path, str]] = None,
    save_matrices: bool = False,
    decimal_output_format: str = '%.3E',
    **kwargs
) -> None:
    
    if save_path is None:
        save_path = Path().resolve()
    if isinstance(save_path, str):
        save_path = Path(save_path)

    for sub_network_ids in sub_network_ids_list:

        partial_network = network.partial(sub_network_ids)

        filename = errors_file_name(
            network=network,
            sub_network_ids=sub_network_ids,
            population_name=population_name,
        )
        
        detected, network_snr, errors, sky_localization = compute_network_errors(
            network=partial_network,
            parameter_values=parameter_values,
            fisher_parameters=fisher_parameters,
            save_matrices=save_matrices,
            save_matrices_path=save_path,
            matrix_naming_postfix='_'.join(filename.split('_')[1:]),
            **kwargs,
        )

        output_to_txt_file(
            parameter_values=parameter_values.iloc[detected],
            network_snr=network_snr[detected],
            parameter_errors=errors[detected, :],
            sky_localization=(
                sky_localization[detected] if sky_localization is not None else None
            ),
            fisher_parameters=fisher_parameters,
            filename=save_path/filename,
            decimal_output_format=decimal_output_format,
        )

