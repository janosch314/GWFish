from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.cosmology import Planck18

from GWFish.modules.detection import Detector, Network
from GWFish.modules.fishermatrix import compute_network_errors

def from_m1_m2_to_mChirp_q(m1, m2):
    """
    Compute the transformation from m1, m2 to mChirp, q
    """
    mChirp = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
    q = m2 / m1
    return mChirp, q

def test_snr_different_mass_inputs():

    detectors = ['ET']
    network = Network(detector_ids = detectors, detection_SNR = (0., 8.))
    waveform_model = 'IMRPhenomD_NRTidalv2'

    # Fixed parameters
    z = np.array([0.00980])
    extra_params = {'redshift': z,
        'luminosity_distance': Planck18.luminosity_distance(z).value,
        'theta_jn': np.array([2.545065595974997]),
        'ra': np.array([3.4461599999999994]),
        'dec': np.array([-0.4080839999999999]),
        'psi': np.array([0.]),
        'phase': np.array([0.]),
        'geocent_time': np.array([1187008882.4]),
        'a_1':np.array([0.005136138323169717]), 
        'a_2':np.array([0.003235146993487445]), 
        'lambda_1':np.array([368.17802383555687]), 
        'lambda_2':np.array([586.5487031450857])}


    # Change mass parameters parametrization
    masses_source = {
        'mass_1_source': np.array([1.4957673]), 
        'mass_2_source': np.array([1.24276395])}

    masses_det = {
        'mass_1': np.array([1.4957673]) * (1 + z), 
        'mass_2': np.array([1.24276395]) * (1 + z)}


    chirp_mass, mass_ratio = from_m1_m2_to_mChirp_q(1.4957673, 1.24276395)
    masses_chirpq = {
        'chirp_mass': np.array([chirp_mass]) * (1 + z), 
        'mass_ratio': np.array([mass_ratio])}

    parameters_source = pd.DataFrame(masses_source | extra_params)
    parameters_det = pd.DataFrame(masses_det | extra_params)
    parameters_chirp = pd.DataFrame(masses_chirpq | extra_params)

    _, snr_source, _, _ = compute_network_errors(
        network = network,
        parameter_values = parameters_source,
        waveform_model = waveform_model
        )
    
    _, snr_det, _, _ = compute_network_errors(
        network = network,
        parameter_values = parameters_det,
        waveform_model = waveform_model
        )
    
    _, snr_chirp, _, _ = compute_network_errors(
        network = network,
        parameter_values = parameters_chirp,
        waveform_model = waveform_model
        )
    
    assert np.isclose(snr_source, snr_det, rtol=1e-5)
    assert np.isclose(snr_source, snr_chirp, rtol=1e-5)
