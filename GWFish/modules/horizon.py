"""
This module is aimed at computing detection horizons.

The thing we want to compute is the luminosity distance at which a given
signal will be detected (with a given SNR).
"""

from warnings import warn

import numpy as np

from astropy.cosmology import Planck18
import astropy.cosmology as cosmology
import astropy.units as u

from .detection import SNR, Detector, projection
from .waveforms import hphc_amplitudes

WAVEFORM_MODEL = 'gwfish_TaylorF2'

def horizon(
    params: dict,
    detector: Detector,
    target_SNR: int = 9, 
    waveform_model: str = WAVEFORM_MODEL,
    cosmology_model: cosmology.Cosmology = Planck18
    ):
    """
    Given the parameters for a GW signal and a detector, this function 
    computes the luminosity distance and corresponding redshift 
    (as connected by a given cosmology) which the signal would 
    need to be at in order to have a given redshift - by default, 9.
    
    Returns:
    
    distance in Mpc, redshift
    """
    
    relative_error = np.inf
    redshift = 1e-2
    distance = 40
    
    if 'redshift' in params or 'luminosity_distance' in params:
        warn('The redshift and distance parameters will not be used in this function.')
    
    while True:
        
        params = params | {'redshift': redshift, 'luminosity_distance': distance}
        
        polarizations, timevector = hphc_amplitudes(
            waveform_model, 
            params, 
            detector.frequencyvector, 
            plot=None
        )
        
        signal = projection(
            params,
            detector,
            polarizations,
            timevector
        )
        
        component_SNRs = SNR(detector, signal)
        this_SNR = np.sqrt(np.sum(component_SNRs**2))

        if abs(np.log(this_SNR / target_SNR)) < 1e-3:
            break
        
        distance *= this_SNR / target_SNR
        redshift = cosmology.z_at_value(cosmology_model.luminosity_distance, distance * u.Mpc).value
        
    return distance, redshift