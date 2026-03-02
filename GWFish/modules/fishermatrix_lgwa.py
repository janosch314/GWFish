### We are building a function that returns the Fisher matrix for the only lgwa case 
### note: in the lgwa case, the waveform has to be fixed!

### The parameters needed are: binary parameters, parameters on which we want to run the fisher


### This module is a modification of the fishermatrix.py module in GWFish, just for the case in which lgwa is needed
### The functions do not have any modification wrt the base case, are commented and include via import from the base code


import numpy as np
#import GWFish.modules.waveforms as wf
#import GWFish.modules.detection as det
#import GWFish.modules.auxiliary as aux
#import GWFish.modules.fft as fft
import lgwa_response.lunar_coordinates as lunar_coord

from lgwa_response.likelihood import LunarLikelihood
from lgwa_response.likelihood import noise_weighted_inner_product

import copy
import pandas as pd
from typing import Optional, Union

from tqdm import tqdm

import logging
from pathlib import Path

def from_q_to_eta(q):
    """
    Symetric mass ratio from mass ratio q. 
    """
    return q / (1.+q)**2

def from_lambda_eff_2_l1l2(q, lambda_eff, d_lambda):
    """
    Function for the conversion of parameters' name from gwfish convetions to lgwa conventions
    """    
    eta = from_q_to_eta(q)
    c_1 = (1. + 7. * eta - 31. * eta**2)
    c_2 = (1. - 4. * eta)**0.5 * (1 + 9 * eta - 11. * eta**2)
    c_3 = (1. - 4. * eta)**0.5 *\
                    (1. - 13272. / 1319. * eta + 8944. / 1319. * eta**2)
    c_4 = (1. - 15910. / 1319. * eta + 32850. / 1319. * eta**2 +
                     3380. / 1319. * eta**3)
    lambda_1 =\
        (13. * lambda_eff / 8. * (c_3 - c_4) -
         2. * d_lambda * (c_1 - c_2))\
        / ((c_1 + c_2) * (c_3 - c_4) -
           (c_1 - c_2) * (c_3 + c_4))
    lambda_2 =\
        (13. * lambda_eff / 8. * (c_3 + c_4) -
         2. * d_lambda * (c_1 + c_2)) \
        / ((c_1 - c_2) * (c_3 + c_4) -
           (c_1 + c_2) * (c_3 - c_4))

    return lambda_1, lambda_2


def from_l1l2_2_lambda_eff(q, lambda_1, lambda_2):
    """
    Function for the conversion of parameters' name from gwfish convetions to lgwa conventions
    """
    eta = from_q_to_eta(q)
    c_1 = (1. + 7. * eta - 31. * eta**2)
    c_2 = (1. - 4. * eta)**0.5 * (1 + 9 * eta - 11. * eta**2)
    c_3 = (1. - 4. * eta)**0.5 *\
                    (1. - 13272. / 1319. * eta + 8944. / 1319. * eta**2)
    c_4 = (1. - 15910. / 1319. * eta + 32850. / 1319. * eta**2 +
                     3380. / 1319. * eta**3)
    
    lambda_eff = 8. / 13. * ( (c_1*(lambda_1+lambda_2)) + (c_2*(lambda_1-lambda_2)) )
    d_lambda = 0.5 * ( (c_3*(lambda_1+lambda_2)) + (c_4*(lambda_1-lambda_2)) )

    return lambda_eff, d_lambda


def from_lgwa2gwfish_paramnames(lgwa_params):
    """
    Function for the conversion of parameters' name from lgwa convetions to gwfish conventions
    """

    lambda_1, lambda_2 =  from_lambda_eff_2_l1l2(lgwa_params['mass_ratio'], lgwa_params['lambda_eff'], lgwa_params['d_lambda'])

    if 'geocent_time' in lgwa_params.keys():
        parameters = {
            "chirp_mass": lgwa_params['chirp_mass'],
            "mass_ratio": lgwa_params['mass_ratio'],
            "time_at_center": np.array([788572813.0]),
            #"geocent_time":lgwa_params['geocent_time'],
            "inclination": lgwa_params['inclination'],
            "theta_jn": lgwa_params['inclination'],
            "phase": lgwa_params['phase'],
            "ra": lgwa_params['ra'],
            "dec": lgwa_params['dec'],
            "psi": lgwa_params['psi'],
            "luminosity_distance": lgwa_params['luminosity_distance'],
            #"spin_1z": lgwa_params['spin_1z'],
            #"spin_2z": lgwa_params['spin_2z'],
            #"a_1": lgwa_params['spin_1z'],
            #"a_2": lgwa_params['spin_2z'],
            "lambda_eff": lgwa_params['lambda_eff'],
            "d_lambda": lgwa_params['d_lambda'],
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
        }   

    else:
        parameters = {
            "chirp_mass": lgwa_params['chirp_mass'],
            "mass_ratio": lgwa_params['mass_ratio'],
            "time_at_center": lgwa_params['time_at_center'],
            "geocent_time":lgwa_params['geocent_time'],
            "inclination": lgwa_params['inclination'],
            "theta_jn": lgwa_params['inclination'],
            "phase": lgwa_params['phase'],
            "ra": lgwa_params['ra'],
            "dec": lgwa_params['dec'],
            "psi": lgwa_params['psi'],
            "luminosity_distance": lgwa_params['luminosity_distance'],
            #"spin_1z": lgwa_params['spin_1z'],
            #"spin_2z": lgwa_params['spin_2z'],
            #"a_1": lgwa_params['spin_1z'],
            #"a_2": lgwa_params['spin_2z'],
            "lambda_eff": lgwa_params['lambda_eff'],
            "d_lambda": lgwa_params['d_lambda'],
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
        }
        
    return parameters



def from_gwfish2lgwa_paramnames(gwfish_params):
    """
    Function for the conversion of parameters' name from gwfish convetions to lgwa conventions
    """    
    
    if ('lambda_1' or 'lambda_2') in gwfish_params.keys():

        lambda_eff, d_lambda = from_lambda_eff_2_l1l2(gwfish_params['mass_ratio'], gwfish_params['lambda_1'], gwfish_params['lambda_2'])

        parameters = {
            "chirp_mass": gwfish_params['chirp_mass'],
            "mass_ratio": gwfish_params['mass_ratio'],
            "time_at_center": gwfish_params['geocent_time'],
            "geocent_time": gwfish_params['geocent_time'],
            "inclination": gwfish_params['theta_jn'],
            "theta_jn": gwfish_params['theta_jn'],
            "phase": gwfish_params['phase'],
            "ra": gwfish_params['ra'],
            "dec": gwfish_params['dec'],
            "psi": gwfish_params['psi'],
            "luminosity_distance": gwfish_params['luminosity_distance'],
            "spin_1z": 0.,
            "spin_2z": 0.,
            #"a_1": gwfish_params['a_1'],
            #"a_2": gwfish_params['a_2'],
            "lambda_eff": lambda_eff,
            "d_lambda": d_lambda,
            "lambda_1": gwfish_params['lambda_1'], 
            "lambda_2": gwfish_params['lambda_2'],
        }

    else:
        parameters = {
            "chirp_mass": gwfish_params['chirp_mass'],
            "mass_ratio": gwfish_params['mass_ratio'],
            "time_at_center": gwfish_params['geocent_time'],
            "geocent_time": gwfish_params['geocent_time'],
            "inclination": gwfish_params['theta_jn'],
            "theta_jn": gwfish_params['theta_jn'],
            "phase": gwfish_params['phase'],
            "ra": gwfish_params['ra'],
            "dec": gwfish_params['dec'],
            "psi": gwfish_params['psi'],
            "luminosity_distance": gwfish_params['luminosity_distance'],
            "spin_1z": 0.,
            "spin_2z": 0.,
            #"a_1": gwfish_params['a_1'],
            #"a_2": gwfish_params['a_2'],
            "lambda_eff": 0.,
            "d_lambda": 0.,
            "lambda_1": 0., 
            "lambda_2": 0.,
        }
    
    return parameters


class Derivative:
    """
    Standard GWFish waveform derivative class, based on finite differencing in frequency domain.
    Calculates derivatives with respect to geocent_time, merger phase, and distance analytically.
    Derivatives of other parameters are calculated numerically.

    eps: 1e-5, this follows the simple "cube root of numerical precision" recommendation, which is 1e-16 for double
    """
    def __init__(self, f, parameters, lunarLikelihood, eps=1e-5, eps_mass=1e-8):
        #self.waveform = waveform
        self.detector = 'lgwa'
        self.eps = eps
        self.eps_mass = eps_mass
        self.frequencyvector = f
        self.lunarLikelihood = lunarLikelihood
        #self.waveform_class = waveform_class
        #self.data_params = {'frequencyvector': detector.frequencyvector, 'f_ref': f_ref}
        #self.waveform_object = waveform_class(waveform, parameters, self.data_params)
        self.waveform_at_parameters = None
        self.projection_at_parameters = None

        # For central parameters and their epsilon-neighbourhood
        self.local_params = parameters.copy()
        self.pv_set1 = parameters.copy()
        self.pv_set2 = parameters.copy()

    @property
    def projection_at_parameters(self):
        if self._projection_at_parameters is None:
            self._projection_at_parameters = self.lunarLikelihood.projected_waveform(self.frequencyvector, self.local_params)
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
        #elif target_parameter == 'geocent_time':
        #    derivative = 2j * np.pi * self.frequencyvector * self.projection_at_parameters
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

            self.pv_set1 = from_gwfish2lgwa_paramnames(self.pv_set1)
            self.pv_set2 = from_gwfish2lgwa_paramnames(self.pv_set2)

            self.pv_set1[target_parameter] = pv - dp / 2.
            self.pv_set2[target_parameter] = pv + dp / 2.

            self.pv_set1 = from_gwfish2lgwa_paramnames(self.pv_set1)
            self.pv_set2 = from_gwfish2lgwa_paramnames(self.pv_set2)
    
            signal1 = self.lunarLikelihood.projected_waveform(self.frequencyvector, self.pv_set1)
            signal2 = self.lunarLikelihood.projected_waveform(self.frequencyvector, self.pv_set2)

            derivative = (signal2 - signal1) / dp
            
        return derivative
    
    def __call__(self, target_parameter):
        return self.with_respect_to(target_parameter)

class FisherMatrix:
    def __init__(self, f, parameters, lunarLikelihood, fisher_parameters, eps=1e-5, eps_mass=1e-8):
        self.fisher_parameters = fisher_parameters
        self.frequencyvector = f
        self.lunarLikelihood = lunarLikelihood
        #self.detector = detector
        self.derivative = Derivative(f, parameters, self.lunarLikelihood, eps=eps, eps_mass=eps_mass)
        self.nd = len(fisher_parameters)
        self.fm = None

    def update_fm(self):
        self._fm = np.zeros((self.nd, self.nd))
        for p1 in np.arange(self.nd):
            deriv1_p = self.fisher_parameters[p1]
            deriv1 = self.derivative(deriv1_p)
            
            #self._fm[p1, p1] = np.sum(aux.scalar_product(deriv1, deriv1, self.detector), axis=0)
            self._fm[p1, p1] = np.sum(noise_weighted_inner_product(deriv1, deriv1, self.lunarLikelihood.psd(self.frequencyvector), self.frequencyvector), axis=0)
            for p2 in np.arange(p1+1, self.nd):
                deriv2_p = self.fisher_parameters[p2]
                deriv2 = self.derivative(deriv2_p)
                #self._fm[p1, p2] = np.sum(aux.scalar_product(deriv1, deriv2, self.detector), axis=0)
                self._fm[p1, p2] = np.sum(noise_weighted_inner_product(deriv1, deriv2, self.lunarLikelihood.psd(self.frequencyvector), self.frequencyvector), axis=0)
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
    f,
    lunarLikelihood,
    signal_parameter_values: Union[pd.DataFrame, dict[str, float]],
    fisher_parameters: Optional[list[str]] = None,
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
    detector_SNR_square = lunarLikelihood.optimal_snr(f, signal_parameter_values)**2

    if fisher_parameters is None:
        if isinstance(signal_parameter_values, dict):
            fisher_parameters = list(signal_parameter_values.keys())
        else:
            fisher_parameters = signal_parameter_values.columns
    
    return FisherMatrix(f, signal_parameter_values, lunarLikelihood, fisher_parameters, eps=eps, eps_mass=eps_mass).fm, detector_SNR_square