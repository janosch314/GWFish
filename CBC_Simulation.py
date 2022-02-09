#!/usr/bin/env python

import numpy as np
import pandas as pd

from numpy.random import default_rng

import time
import json
import progressbar

from tqdm import tqdm
from astropy.cosmology import FlatLambdaCDM

import argparse

import GWFish as gw

cosmo = FlatLambdaCDM(H0=69.6, Om0=0.286)

rng = default_rng()

def main():
    # example to run with command-line arguments:
    # python CBC_Simulation.py --pop_file=CBC_pop.hdf5 --detectors ET CE2 --networks [[0,1],[0],[1]]

    folder = './injections/'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pop_file', type=str, default=['CBC_pop_trial1.hdf5'], nargs=1,
        help='Population to run the analysis on.'
             'Runs on BBH_injections_1e6.hdf5 if no argument given.')
    parser.add_argument(
        '--pop_id', type=str, default=['BBH'], nargs=1,
        help='Short population identifier for file names.'
             'Uses BBH if no argument given.')
    parser.add_argument(
        '--detectors', type=str, default=['ET0'], nargs='+',
        help='Detectors to analyze. Uses ET as default if no argument given.')
    parser.add_argument(
        '--networks', default=['[[0]]'], help='Network IDs. Uses [[0]] as default if no argument given.')
    args = parser.parse_args()
    

    d = args.detectors
    
    if ('ET' in d):
        fmin = 10
        fmax = 2048
        df = 1./4.
        #df = 1. / 16. # for binary NSs
    elif ('LGWA' in d):
        fmin = 1e-3
        fmax = 4
        df = 1. / 4096.
    elif ('LISA' in d):
        fmin = 1e-3
        fmax = 0.3
        df = 1e-4
    else:
        fmin = 8
        fmax = 1024
        df = 1. / 4.

  

    frequencyvector = np.linspace(fmin, fmax, int((fmax - fmin) / df) + 1)
    #print('f_vec before = ',frequencyvector)
    frequencyvector = frequencyvector[:, np.newaxis]
    #print('f_vec after = ',frequencyvector)

    threshold_SNR = np.array([0., 9.])  # [min. individual SNR to be included in PE, min. network SNR for detection]
    #print('threshold_SNR = ',threshold_SNR)
    max_time_until_merger = 10 * 3.16e7  # used for LISA, where observation times of a signal can be limited by mission lifetime
    calculate_errors = True    # whether to calculate Fisher-matrix based PE errors
    duty_cycle = False  # whether to consider the duty cycle of detectors

    pop_file = args.pop_file[0]
    # pop_file = 'CBC_pop.hdf5'
    population = args.pop_id[0]
    # population = 'BBH'

    detectors_ids = args.detectors
    # detectors_ids = ['ET', '1kkk',...]
    print(detectors_ids)
    networks_ids = json.loads(args.networks[0])
    print(networks_ids)

    parameters = pd.read_hdf(folder+pop_file)


    print(parameters)
    parameters['iota'] = np.pi/2

    ns = len(parameters)

    network = gw.Network(detectors_ids, number_of_signals=ns, detection_SNR=threshold_SNR, parameters=parameters)

    # lisaGWresponse(network.detectors[0], frequencyvector)
    # exit()

    # horizon(network, parameters.iloc[0], frequencyvector, threshold_SNR, 1./df, fmax)
    # exit()

    #print(parameters.iloc[0])
    print('Processing CBC population')
    bar = progressbar.ProgressBar(max_value=len(parameters))

    for k in tqdm(np.arange(len(parameters))):
        one_parameters = parameters.iloc[k]
        wave, t_of_f = gw.TaylorF2(one_parameters, frequencyvector, maxn=8)

        networkSNR_sq = 0
        for d in np.arange(len(network.detectors)):
            signal = gw.projection(one_parameters, network.detectors[d], wave, t_of_f, frequencyvector,
                                max_time_until_merger)

            SNRs = gw.SNR(network.detectors[d].interferometers, signal, frequencyvector, duty_cycle=duty_cycle)
            networkSNR_sq += np.sum(SNRs ** 2)
            network.detectors[d].SNR[k] = np.sqrt(np.sum(SNRs ** 2))
            if calculate_errors:
                network.detectors[d].fisher_matrix[k, :, :] = \
                    gw.FisherMatrix(one_parameters, network.detectors[d], frequencyvector, max_time_until_merger)

        network.SNR[k] = np.sqrt(networkSNR_sq)

        bar.update(k)

    bar.finish()

    gw.analyzeDetections(network, parameters, population, networks_ids)

    if calculate_errors:
        gw.analyzeFisherErrors(network, parameters, population, networks_ids)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
