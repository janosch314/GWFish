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
        '--pop_file', type=str, default=['CBC_pop.hdf5'], nargs=1,
        help='Population to run the analysis on.'
             'Runs on BBH_injections_1e6.hdf5 if no argument given.')
    parser.add_argument(
        '--pop_id', type=str, default=['BBH'], nargs=1,
        help='Short population identifier for file names.'
             'Uses BBH if no argument given.')
    parser.add_argument(
        '--detectors', type=str, default=['ET'], nargs='+',
        help='Detectors to analyze. Uses ET as default if no argument given.')
    parser.add_argument(
        '--networks', default='[[0]]', help='Network IDs. Uses [[0]] as default if no argument given.')
    
    parser.add_argument(
        '--config', type=str, default=['detConfig_1.yaml'], help='Configuration file where the detector specificationa are stored. Uses detConfig.yaml as default if no argument given.')
    args = parser.parse_args()
    threshold_SNR = np.array([0., 9.])  # [min. individual SNR to be included in PE, min. network SNR for detection]
    #print('threshold_SNR = ',threshold_SNR)
    max_time_until_merger = 10 * 3.16e7  # used for LISA, where observation times of a signal can be limited by mission lifetime
    calculate_errors = True   # whether to calculate Fisher-matrix based PE errors
    duty_cycle = False  # whether to consider the duty cycle of detectors

    pop_file = args.pop_file[0]
    # pop_file = 'CBC_pop.hdf5'
    population = args.pop_id[0]
    # population = 'BBH'

    detectors_ids = args.detectors
    networks_ids = json.loads(args.networks[0])

    parameters = pd.read_hdf(folder+pop_file)
    ConfigDet=args.config[0]
    
    ns = len(parameters)

    network = gw.detection.Network(detectors_ids, number_of_signals=ns, detection_SNR=threshold_SNR, parameters=parameters, Config=ConfigDet)

    # lisaGWresponse(network.detectors[0], frequencyvector)
    # exit()

    # horizon(network, parameters.iloc[0], frequencyvector, threshold_SNR, 1./df, fmax)
    # exit()

    #print(parameters.iloc[0])
    print('Processing CBC population')

    for k in tqdm(np.arange(len(parameters))):
        one_parameters = parameters.iloc[k]

        networkSNR_sq = 0
        for d in np.arange(len(network.detectors)):
            wave, t_of_f = gw.waveforms.TaylorF2(one_parameters, network.detectors[d].frequencyvector, maxn=8)
            signal = gw.detection.projection(one_parameters, network.detectors[d], wave, t_of_f, max_time_until_merger)

            SNRs = gw.detection.SNR(network.detectors[d], signal, duty_cycle=duty_cycle)
            networkSNR_sq += np.sum(SNRs ** 2)
            network.detectors[d].SNR[k] = np.sqrt(np.sum(SNRs ** 2))
            if calculate_errors:
                network.detectors[d].fisher_matrix[k, :, :] = \
                    gw.fishermatrix.FisherMatrix(one_parameters, network.detectors[d], max_time_until_merger)

        network.SNR[k] = np.sqrt(networkSNR_sq)

    gw.detection.analyzeDetections(network, parameters, population, networks_ids)

    if calculate_errors:
        gw.fishermatrix.analyzeFisherErrors(network, parameters, population, networks_ids)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
