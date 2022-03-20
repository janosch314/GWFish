#!/usr/bin/env python

import numpy as np
import pandas as pd

from numpy.random import default_rng

import time
import json

from tqdm import tqdm
from astropy.cosmology import FlatLambdaCDM

import argparse

import GWFish.modules as gw

cosmo = FlatLambdaCDM(H0=69.6, Om0=0.286)

rng = default_rng()

def main():
    # example to run with command-line arguments:
    # python CBC_Simulation.py --pop_file=CBC_pop.hdf5 --detectors ET CE2 --networks [[0,1],[0],[1]]

    folder = './injections/'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pop_file', type=str, default='CBC_pop.hdf5', nargs=1,
        help='Population to run the analysis on. Runs on CBC_pop.hdf5 if no argument given.')
    parser.add_argument(
        '--pop_id', type=str, default='BBH', nargs=1,
        help='Short population identifier for file names. Uses BBH if no argument given.')
    parser.add_argument(
        '--detectors', type=str, default=['ET'], nargs='+',
        help='Detectors to analyze. Uses ET as default if no argument given.')
    parser.add_argument(
        '--networks', default='[[0]]', nargs=1,
        help='Network IDs. Uses [[0]] as default if no argument given.')
    parser.add_argument(
        '--config', type=str, default='GWFish/detectors.yaml',
        help='Configuration file where the detector specifications are stored. Uses GWFish/detectors.yaml as default if no argument given.')
   

    args = parser.parse_args()
    ConfigDet = args.config

    threshold_SNR = np.array([0., 9.])  # [min. individual SNR to be included in PE, min. network SNR for detection]
    calculate_errors = True   # whether to calculate Fisher-matrix based PE errors
    duty_cycle = False  # whether to consider the duty cycle of detectors

    waveform = 'lalbns_IMRPhenomPv2_NRTidal'

    #fisher_parameters = ['ra', 'dec', 'psi', 'iota', 'luminosity_distance', 'mass_1', 'mass_2', 'geocent_time', 'phase']
    fisher_parameters = ['luminosity_distance','ra','dec']

    pop_file = args.pop_file
    population = args.pop_id

    detectors_ids = args.detectors
    networks_ids = json.loads(args.networks)

    parameters = pd.read_hdf(folder+pop_file)

    network = gw.detection.Network(detectors_ids, detection_SNR=threshold_SNR, parameters=parameters,
                                   fisher_parameters=fisher_parameters, config=ConfigDet)

    # lisaGWresponse(network.detectors[0], frequencyvector)
    # exit()

    # horizon(network, parameters.iloc[0], frequencyvector, threshold_SNR, 1./df, fmax)
    # exit()

    print('Processing CBC population')
    for k in tqdm(np.arange(len(parameters))):
        parameter_values = parameters.iloc[k]
        parameter_values = dict(
            mass_1=1.5, mass_2=1.3, chi_1=0.02, chi_2=0.02, luminosity_distance=50.,
            theta_jn=0.4, psi=2.659, phase=1.3, geocent_time=1126259642.413,
            ra=1.375, dec=-1.2108, lambda_1=400, lambda_2=450)

        networkSNR_sq = 0
        for d in np.arange(len(network.detectors)):
            wave, t_of_f = gw.waveforms.hphc_amplitudes(waveform, parameter_values, network.detectors[d].frequencyvector)
            signal = gw.detection.projection(parameter_values, network.detectors[d], wave, t_of_f)

            SNRs = gw.detection.SNR(network.detectors[d], signal, duty_cycle=duty_cycle)
            networkSNR_sq += np.sum(SNRs ** 2)
            network.detectors[d].SNR[k] = np.sqrt(np.sum(SNRs ** 2))

            if calculate_errors:
                network.detectors[d].fisher_matrix[k, :, :] = \
                    gw.fishermatrix.FisherMatrix(waveform, parameter_values, fisher_parameters, network.detectors[d])

        network.SNR[k] = np.sqrt(networkSNR_sq)

    gw.detection.analyzeDetections(network, parameters, population, networks_ids)

    if calculate_errors:
        gw.fishermatrix.analyzeFisherErrors(network, parameters, fisher_parameters, population, networks_ids)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
