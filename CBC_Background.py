#!/usr/bin/env python

import numpy as np
import pandas as pd

from numpy.random import default_rng

import time
import json
from tqdm import tqdm

import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

import pickle
import os
import argparse

import GWFish.modules as gw

cosmo = FlatLambdaCDM(H0=69.6, Om0=0.286)

rng = default_rng()

def analyzeForeground(network, h_of_f, dT):

    for d in np.arange(len(network.detectors)):
        ff = network.detectors[d].frequencyvector[:,0]

        H0 = 2.4e-18  # 72km/s/Mpc
        Omega = 2e-10 * np.power(ff / 10, 2. / 3.)  # Regimbau et al: https://arxiv.org/pdf/2002.05365.pdf
        h_astro = np.sqrt(3 * H0 ** 2 / (10 * np.pi ** 2) * Omega / ff ** 3)

        components = network.detectors[d].components
        psd_astro_all = np.abs(np.squeeze(h_of_f[:, d, :])) ** 2 / dT
        N = len(psd_astro_all[0, :])

        plotrange = components[0].plotrange

        bb = np.logspace(-28, -22, 100)
        hist = np.empty((len(bb) - 1, len(ff)))
        for i in range(len(ff)):
            hist[:, i] = np.histogram(np.sqrt(psd_astro_all[i, :]), bins=bb)[0]
        bb = np.delete(bb, -1)

        # calculate percentiles
        hist_norm = hist / N
        hist_norm[np.isnan(hist_norm)] = 0
        histsum = np.cumsum(hist_norm, axis=0) / (np.sum(hist_norm, axis=0)[np.newaxis, :])
        ii10 = np.argmin(np.abs(histsum - 0.1), axis=0)
        ii50 = np.argmin(np.abs(histsum - 0.5), axis=0)
        ii90 = np.argmin(np.abs(histsum - 0.9), axis=0)

        hist[hist == 0] = np.nan

        fig = plt.figure(figsize=(9, 6))
        plt.figure(figsize=(9, 6))
        cmap = plt.get_cmap('RdYlBu_r')
        cm = plt.pcolor(np.transpose(ff), bb, hist, cmap=cmap)
        # plt.loglog(ff, h_astro)
        plt.loglog(ff, np.sqrt(components[0].Sn(ff)), color='green')
        plt.loglog(ff, bb[ii10], 'w-')
        plt.loglog(ff, bb[ii50], 'w-')
        plt.loglog(ff, bb[ii90], 'w-')
        plt.loglog(ff, bb[ii10], 'k--')
        plt.loglog(ff, bb[ii50], 'k--')
        plt.loglog(ff, bb[ii90], 'k--')
        plt.xlabel('Frequency [Hz]', fontsize=20)
        plt.ylabel(r"Strain spectra [$1/\sqrt{\rm Hz}$]", fontsize=20)
        plt.xlim((plotrange[0], plotrange[1]))
        plt.ylim((plotrange[2] / 100, plotrange[3] / 100))
        plt.colorbar
        plt.grid(True)
        fig.colorbar(cm)
        plt.tick_params(labelsize=20)
        plt.tight_layout()
        plt.savefig('Astrophysical_histo_' + components[0].name + '.png', dpi=300)
        plt.close()

def main():
    # example to run with command-line arguments:
    # python CBC_Foreground.py --pop_file=CBC_pop.hdf5 --detectors ET CE2

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pop_file', type=str, default='./injections/BBH_1e5.hdf5', nargs=1,
        help='Population to run the analysis on.'
             'Runs on BBH_1e5.hdf5 if no argument given.')
    parser.add_argument(
        '--detectors', type=str, default=['ET'], nargs='+',
        help='Detectors to analyze. Uses ET as default if no argument given.')
    parser.add_argument(
        '--outdir', type=str, default='./', 
        help='Output directory.')
    parser.add_argument(
        '--config', type=str, default='GWFish/detectors.yaml',
        help='Configuration file where the detector specifications are stored. Uses GWFish/detectors.yaml as default if no argument given.')

    args = parser.parse_args()
    ConfigDet = args.config

    dT = 60
    N = 7200
    # dT = 24*3600
    # N = 100
    t0 = 1104105616

    threshold_SNR = 5000  # min. network SNR for detection
    duty_cycle = False  # whether to consider the duty cycle of detectors

    pop_file = args.pop_file

    detectors_ids = args.detectors

    parameters = pd.read_hdf(pop_file)

    ns = len(parameters)

    network = gw.detection.Network(detectors_ids, detection_SNR=threshold_SNR, parameters=parameters,
                                   fisher_parameters=None, config=ConfigDet)

    waveform_model = 'lalbbh_IMRPhenomD'
    #waveform_model = 'gwfish_TaylorF2'
    # waveform_model = 'lalbbh_TaylorF2'

    frequencyvector = network.detectors[0].frequencyvector

    h_of_f = np.zeros((len(frequencyvector), len(network.detectors), N), dtype=complex)
    cnt = np.zeros((N,))

    print('Processing CBC population')
    for k in tqdm(np.arange(ns)):
        parameter_values = parameters.iloc[k]
        tc = parameter_values['geocent_time']

        # make a precut on the signals; note that this depends on how long signals stay in band (here not more than 3 days)
        if ((tc>t0) & (tc-3*86400<t0+N*dT)):
            signals = np.zeros((len(frequencyvector), len(network.detectors)), dtype=complex)  # contains only 1 of 3 streams in case of ET
            for d in np.arange(len(network.detectors)):
                wave, t_of_f = gw.waveforms.hphc_amplitudes(waveform_model, parameter_values,
                                                            network.detectors[d].frequencyvector)

                det_signals = gw.detection.projection(parameter_values, network.detectors[d], wave, t_of_f)
                signals[:,d] = det_signals[:,0]

                SNRs = gw.detection.SNR(network.detectors[d], det_signals, duty_cycle=duty_cycle)
                network.detectors[d].SNR = np.sqrt(np.sum(SNRs ** 2))

            SNRsq = 0
            for detector in network.detectors:
                SNRsq += detector.SNR ** 2

            if (np.sqrt(SNRsq) < threshold_SNR):
                for n in np.arange(N):
                    t1 = t0+n*dT
                    t2 = t1+dT
                    ii = np.argwhere((t_of_f[:,0] < t1) | (t_of_f[:,0] > t2))
                    signals_ii = np.copy(signals)

                    if (len(ii) < len(t_of_f)):
                        #print("Signal {0} contributes to segment {1}.".format(k,n))
                        cnt[n] += 1
                        signals_ii[ii,:] = 0
                        h_of_f[:,:,n] += signals_ii

    analyzeForeground(network, h_of_f, dT)

    print('Out of {0} signals, {1} are in average undetected binaries falling in a {2}s time window.'.format(ns, np.mean(cnt), dT))

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
