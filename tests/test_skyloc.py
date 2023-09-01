import GWFish.modules as gw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

from .test_projection import BNS_PARAMS
NINETY_PERCENT_INTERVAL = (180 / np.pi) ** 2 * 4.605170185988092

FISHER_PARAMETERS = [
    "mass_1",
    "mass_2",
    "luminosity_distance",
    "theta_jn",
    "dec",
    "ra",
    "psi",
    "phase",
    "geocent_time",
    "lambda_1",
    "lambda_2",
]

def test_compute_sky_localization():
    population = "BNS"

    detectors = ['LBI-SUS']

    networks = '[[0]]'

    detectors_ids = detectors
    networks_ids = json.loads(networks)
    duty_cycle = False
    
    waveform_model = "IMRPhenomD_NRTidalv2"

    params = BNS_PARAMS

    parameters = pd.DataFrame({k: v * np.array([1]) for k, v in params.items()})

    threshold_SNR = np.array([0.0, 8.0])
    fisher_parameters = FISHER_PARAMETERS
    network = gw.detection.Network(
        detectors_ids,
        detection_SNR=threshold_SNR,
        parameters=parameters,
        fisher_parameters=fisher_parameters,
    )
    waveform_class = gw.waveforms.LALFD_Waveform

    networkSNR_sq = 0
    for k in np.arange(len(parameters)):
        parameter_values = parameters.iloc[k]

        networkSNR_sq = 0
        for d in np.arange(len(network.detectors)):
            data_params = {
                "frequencyvector": network.detectors[d].frequencyvector,
                "f_ref": 50.0,
            }
            waveform_obj = waveform_class(waveform_model, parameter_values, data_params)
            wave = waveform_obj()
            t_of_f = waveform_obj.t_of_f

            signal = gw.detection.projection(
                parameter_values, network.detectors[d], wave, t_of_f
            )
            if any(np.isnan(signal[:, 0])):
                breakpoint()

            SNRs = gw.detection.SNR(network.detectors[d], signal, duty_cycle=duty_cycle)
            print(f"{network.detectors[d].name}: SNR={np.sqrt(np.sum(SNRs ** 2)):.1f}")
            networkSNR_sq += np.sum(SNRs**2)
            network.detectors[d].SNR[k] = np.sqrt(np.sum(SNRs**2))

            network.detectors[d].fisher_matrix[k, :, :] = gw.fishermatrix.FisherMatrix(
                waveform_model,
                parameter_values,
                fisher_parameters,
                network.detectors[d],
                waveform_class=waveform_class,
            ).fm

        network.SNR[k] = np.sqrt(networkSNR_sq)

    gw.detection.analyzeDetections(network, parameters, population, networks_ids)

    skylocs = gw.fishermatrix.analyzeFisherErrors(
        network, parameters, fisher_parameters, population, networks_ids
    )
    skyloc = skylocs[0] * NINETY_PERCENT_INTERVAL
    assert np.isclose(skyloc, 0.00071518, rtol=0.02)