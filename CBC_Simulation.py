#!/usr/bin/env python

import numpy as np
import pandas as pd

from numpy.random import default_rng

import time
import json
import progressbar

from tqdm import tqdm

from scipy.interpolate import interp1d
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

import argparse

cosmo = FlatLambdaCDM(H0=69.6, Om0=0.286)

rng = default_rng()

Mpc = 3.086e22
Msol = 1.9885e30
R_earth = 6.37e6
AU = 1.5e11
sidereal_day = 23.9344696
lunar_sidereal_period = 655.7198333
ecliptic = 23.45 * np.pi / 180.

c = 299792458.
G = 6.674e-11
h = 6.626e-34



class Interferometer:

    def __init__(self, name='ET', interferometer='', plot=False):
        self.plot = plot
        self.ifo_id = interferometer
        self.name = name + str(interferometer)

        self.setProperties()

    def setProperties(self):

        k = self.ifo_id

        if self.name[0:3] == 'ET0':
            print('ET0')
            # the lat/lon/azimuth values are just approximations (for Sardinia site)
            self.lat = (43 + 37. / 60 + 53.0921 / 3600)*np.pi / 180.
            self.lon = (10 + 30. / 60 + 16.1878 / 3600) * np.pi / 180.
            self.opening_angle = np.pi / 3.
            self.arm_azimuth = 70.5674 * np.pi / 180. + 2. * k * np.pi / 3.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_B_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]

        elif self.name[0:3] == 'ET1':
            print('ET1')
            # the lat/lon/azimuth values are just approximations (for Sardinia site)
            self.lat = (40 + 31.0 / 60) * np.pi / 180.
            self.lon = (9 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 3.
            self.arm_azimuth = 87. * np.pi / 180. + 2. * k * np.pi / 3.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_10_HF_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name[0:3] == 'ET2': 
            print('ET2')
            self.lat = (40 + 31.0 / 60) * np.pi / 180.
            self.lon = (9 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 3.
            self.arm_azimuth = 87. * np.pi / 180. + 2. * k * np.pi / 3.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_15_HF_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name[0:3] == 'ET3': 
            print('ET3')
            self.lat = (40 + 31.0 / 60) * np.pi / 180.
            self.lon = (9 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 3.
            self.arm_azimuth = 87. * np.pi / 180. + 2. * k * np.pi / 3.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_10_full_cryo_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name[0:3] == 'ET4': 
            print('ET4')
            self.lat = (40 + 31.0 / 60) * np.pi / 180.
            self.lon = (9 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 3.
            self.arm_azimuth = 87. * np.pi / 180. + 2. * k * np.pi / 3.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_15_full_cryo_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name[0:3] == 'ET5': 
            print('ET5')
            self.lat = (40 + 31.0 / 60) * np.pi / 180.
            self.lon = (9 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 3.
            self.arm_azimuth = 87. * np.pi / 180. + 2. * k * np.pi / 3.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_10_full_room_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name[0:3] == 'ET6': 
            print('ET6')
            self.lat = (40 + 31.0 / 60) * np.pi / 180.
            self.lon = (9 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 3.
            self.arm_azimuth = 87. * np.pi / 180. + 2. * k * np.pi / 3.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_15_full_room_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]

        elif self.name == 'S2L_10kmHF_arm90_00deg_ET':
            print('S2L_10kmHF_arm90_00deg_ET')
            self.lat = (40 + 31.0 / 60) * np.pi / 180.
            self.lon = (9 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 87. * np.pi / 180. 

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_10_HF_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'N2L_10kmHF_arm90_00deg_ET':
            print('N2L_10kmHF_arm90_00deg_ET')
            self.lat = (52 + 31.0 / 60) * np.pi / 180.
            self.lon = (5 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 87. * np.pi / 180. 

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_10_HF_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'N2L_10kmHF_arm90_45deg_ET':
            print('N2L_10kmHF_arm90_45deg_ET')
            self.lat = (52 + 31.0 / 60) * np.pi / 180.
            self.lon = (5 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = (87. + 45.) * np.pi / 180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_10_HF_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'S2L_15kmHF_arm90_00deg_ET':
            print('S2L_15kmHF_arm90_00deg_ET')
            self.lat = (40 + 31.0 / 60) * np.pi / 180.
            self.lon = (9 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 87. * np.pi / 180. 

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_15_HF_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'N2L_15kmHF_arm90_00deg_ET':
            print('N2L_15kmHF_arm90_00deg_ET')
            self.lat = (52 + 31.0 / 60) * np.pi / 180.
            self.lon = (5 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 87. * np.pi / 180. 

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_15_HF_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'N2L_15kmHF_arm90_45deg_ET':
            print('N2L_15kmHF_arm90_45deg_ET')
            self.lat = (52 + 31.0 / 60) * np.pi / 180.
            self.lon = (5 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = (87. + 45.) * np.pi / 180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_15_HF_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'S2L_20kmHF_arm90_00deg_ET':
            print('S2L_20kmHF_arm90_00deg_ET')
            self.lat = (40 + 31.0 / 60) * np.pi / 180.
            self.lon = (9 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 87. * np.pi / 180. 

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_20_HF_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'N2L_20kmHF_arm90_00deg_ET':
            print('N2L_20kmHF_arm90_00deg_ET')
            self.lat = (52 + 31.0 / 60) * np.pi / 180.
            self.lon = (5 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 87. * np.pi / 180. 

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_20_HF_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'N2L_20kmHF_arm90_45deg_ET':
            print('N2L_20kmHF_arm90_45deg_ET')
            self.lat = (52 + 31.0 / 60) * np.pi / 180.
            self.lon = (5 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = (87. + 45.) * np.pi / 180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_20_HF_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'S2L_10kmfull_arm90_00deg_ET':
            print('S2L_10kmfull_arm90_00deg_ET')
            self.lat = (40 + 31.0 / 60) * np.pi / 180.
            self.lon = (9 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 87. * np.pi / 180. 

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_10_full_cryo_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'N2L_10kmfull_arm90_00deg_ET':
            print('N2L_10kmfull_arm90_00deg_ET')
            self.lat = (52 + 31.0 / 60) * np.pi / 180.
            self.lon = (5 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 87. * np.pi / 180. 

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_10_full_cryo_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'N2L_10kmfull_arm90_45deg_ET':
            print('N2L_10kmfull_arm90_45deg_ET')
            self.lat = (52 + 31.0 / 60) * np.pi / 180.
            self.lon = (5 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = (87. + 45.) * np.pi / 180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_10_full_cryo_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'S2L_15kmfull_arm90_00deg_ET':
            print('S2L_15kmfull_arm90_00deg_ET')
            self.lat = (40 + 31.0 / 60) * np.pi / 180.
            self.lon = (9 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 87. * np.pi / 180. 

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_15_full_cryo_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'N2L_15kmfull_arm90_00deg_ET':
            print('N2L_15kmfull_arm90_00deg_ET')
            self.lat = (52 + 31.0 / 60) * np.pi / 180.
            self.lon = (5 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 87. * np.pi / 180. 

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_15_full_cryo_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'N2L_15kmfull_arm90_45deg_ET':
            print('N2L_15kmfull_arm90_45deg_ET')
            self.lat = (52 + 31.0 / 60) * np.pi / 180.
            self.lon = (5 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = (87. + 45.) * np.pi / 180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_15_full_cryo_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'S2L_20kmfull_arm90_00deg_ET':
            print('S2L_20kmfull_arm90_00deg_ET')
            self.lat = (40 + 31.0 / 60) * np.pi / 180.
            self.lon = (9 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 87. * np.pi / 180. 

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_20_full_cryo_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'N2L_20kmfull_arm90_00deg_ET':
            print('N2L_20kmfull_arm90_00deg_ET')
            self.lat = (52 + 31.0 / 60) * np.pi / 180.
            self.lon = (5 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 87. * np.pi / 180. 

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_20_full_cryo_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'N2L_20kmfull_arm90_45deg_ET':
            print('N2L_20kmfull_arm90_45deg_ET')
            self.lat = (52 + 31.0 / 60) * np.pi / 180.
            self.lon = (5 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = (87. + 45.) * np.pi / 180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_20_full_cryo_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'S2L_10kmfull_arm90_00deg_ET_room':
            print('S2L_10kmfull_arm90_00deg_ET_room')
            self.lat = (40 + 31.0 / 60) * np.pi / 180.
            self.lon = (9 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 87. * np.pi / 180. 

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_10_full_room_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'N2L_10kmfull_arm90_00deg_ET_room':
            print('N2L_10kmfull_arm90_00deg_ET_room')
            self.lat = (52 + 31.0 / 60) * np.pi / 180.
            self.lon = (5 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 87. * np.pi / 180. 

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_10_full_room_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'N2L_10kmfull_arm90_45deg_ET_room':
            print('N2L_10kmfull_arm90_45deg_ET_room')
            self.lat = (52 + 31.0 / 60) * np.pi / 180.
            self.lon = (5 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = (87. + 45.) * np.pi / 180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_10_full_room_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'S2L_15kmfull_arm90_00deg_ET_room':
            print('S2L_15kmfull_arm90_00deg_ET_room')
            self.lat = (40 + 31.0 / 60) * np.pi / 180.
            self.lon = (9 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 87. * np.pi / 180. 

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_15_full_room_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'N2L_15kmfull_arm90_00deg_ET_room':
            print('N2L_15kmfull_arm90_00deg_ET_room')
            self.lat = (52 + 31.0 / 60) * np.pi / 180.
            self.lon = (5 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 87. * np.pi / 180. 

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_15_full_room_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'N2L_15kmfull_arm90_45deg_ET_room':
            print('N2L_15kmfull_arm90_45deg_ET_room')
            self.lat = (52 + 31.0 / 60) * np.pi / 180.
            self.lon = (5 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = (87. + 45.) * np.pi / 180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_15_full_room_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'S2L_20kmfull_arm90_00deg_ET_room':
            print('S2L_20kmfull_arm90_00deg_ET_room')
            self.lat = (40 + 31.0 / 60) * np.pi / 180.
            self.lon = (9 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 87. * np.pi / 180. 

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_20_full_room_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'N2L_20kmfull_arm90_00deg_ET_room':
            print('N2L_20kmfull_arm90_00deg_ET_room')
            self.lat = (52 + 31.0 / 60) * np.pi / 180.
            self.lon = (5 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 87. * np.pi / 180. 

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_20_full_room_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'N2L_20kmfull_arm90_45deg_ET_room':
            print('N2L_20kmfull_arm90_45deg_ET_room')
            self.lat = (52 + 31.0 / 60) * np.pi / 180.
            self.lon = (5 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = (87. + 45.) * np.pi / 180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * e_long + np.sin(self.arm_azimuth + self.opening_angle) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_20_full_room_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'CE':
            print('CE')
            self.lat = 46.5 * np.pi / 180.
            self.lon = -119.4 * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 126. * np.pi / 180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + np.pi / 2.) * e_long + np.sin(self.arm_azimuth + np.pi / 2.) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/CE2_40km_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [8, 3000, 1e-25, 1e-20]

        elif self.name[0:3] == 'aET':
            # the lat/lon/azimuth values are just approximations (for Sardinia site)
            self.lat = (40 + 31.0 / 60) * np.pi / 180.
            self.lon = (9 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 3.
            self.arm_azimuth = 87. * np.pi / 180. + 2. * k * np.pi / 3.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + np.pi / 3.) * e_long + np.sin(self.arm_azimuth + np.pi / 3.) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_D_evol_psd.txt')
            self.psd_data = self.psd_data[:, [0, 1]]

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]

        
        elif self.name[0:3] == 'bET':
            # the lat/lon/azimuth values are just approximations (for Sardinia site)
            self.lat = (40 + 31.0 / 60) * np.pi / 180.
            self.lon = (9 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 3.
            self.arm_azimuth = 87. * np.pi / 180. + 2. * k * np.pi / 3.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + np.pi / 3.) * e_long + np.sin(self.arm_azimuth + np.pi / 3.) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_D_evol_psd.txt')
            self.psd_data = self.psd_data[:, [0, 2]]

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name[0:3] == 'cET':
            # the lat/lon/azimuth values are just approximations (for Sardinia site)
            self.lat = (40 + 31.0 / 60) * np.pi / 180.
            self.lon = (9 + 25.0 / 60) * np.pi / 180.
            self.opening_angle = np.pi / 3.
            self.arm_azimuth = 87. * np.pi / 180. + 2. * k * np.pi / 3.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + np.pi / 3.) * e_long + np.sin(self.arm_azimuth + np.pi / 3.) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/ET_D_evol_psd.txt')
            self.psd_data = self.psd_data[:, [0, 3]]

            self.duty_factor = 0.85

            self.plotrange = [3, 3000, 1e-25, 1e-20]
        elif self.name == 'VOH':
            self.lat = 46.5 * np.pi / 180.
            self.lon = -119.4 * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 126. * np.pi / 180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + np.pi / 2.) * e_long + np.sin(self.arm_azimuth + np.pi / 2.) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/Voyager_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [10, 1000, 1e-25, 1e-20]
        elif self.name == 'VOL':
            self.lat = 30.56 * np.pi / 180.
            self.lon = -90.77 * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = -197.7 * np.pi / 180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + np.pi / 2.) * e_long + np.sin(self.arm_azimuth + np.pi / 2.) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/Voyager_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [10, 1000, 1e-25, 1e-20]
        elif self.name == 'VOI':
            self.lat = 19.61 * np.pi / 180.
            self.lon = 77.03 * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 100. * np.pi / 180.  # this value is guessed from wikipage

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + np.pi / 2.) * e_long + np.sin(self.arm_azimuth + np.pi / 2.) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/Voyager_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [10, 1000, 1e-25, 1e-20]
        elif self.name == 'CE1':
            self.lat = 46.5 * np.pi / 180.
            self.lon = -119.4 * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 126. * np.pi / 180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + np.pi / 2.) * e_long + np.sin(self.arm_azimuth + np.pi / 2.) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/CE1_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [8, 3000, 1e-25, 1e-20]
        elif self.name == 'CE2':
            self.lat = 46.5 * np.pi / 180.
            self.lon = -119.4 * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 126. * np.pi / 180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + np.pi / 2.) * e_long + np.sin(self.arm_azimuth + np.pi / 2.) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/CE2_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [8, 3000, 1e-25, 1e-20]
        elif self.name == 'CEA':  # hypothetical CE1 in Australia
            self.lat = -20.517
            self.lon = 131.061
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 126. * np.pi / 180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + np.pi / 2.) * e_long + np.sin(self.arm_azimuth + np.pi / 2.) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/CE1_psd.txt')

            self.duty_factor = 0.85
            self.plotrange = [8, 3000, 1e-25, 1e-20]
        elif self.name == 'LLO':
            self.lat = 30.56 * np.pi / 180.
            self.lon = -90.77 * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = -197.7 * np.pi / 180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + np.pi / 2.) * e_long + np.sin(self.arm_azimuth + np.pi / 2.) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/LIGO_O5_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [10, 1000, 1e-25, 1e-20]
        elif self.name == 'LHO':
            self.lat = 46.46 * np.pi / 180.
            self.lon = -119.4 * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 126. * np.pi / 180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + np.pi / 2.) * e_long + np.sin(self.arm_azimuth + np.pi / 2.) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/LIGO_O5_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [10, 1000, 1e-25, 1e-20]
        elif self.name == 'Vir':
            self.lat = 43.6 * np.pi / 180.
            self.lon = 10.5 * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 71. * np.pi / 180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + np.pi / 2.) * e_long + np.sin(self.arm_azimuth + np.pi / 2.) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/Virgo_O5_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [10, 1000, 1e-25, 1e-20]
        elif self.name == 'KAG':
            self.lat = 36.41 * np.pi / 180.
            self.lon = 137.3 * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 29.6 * np.pi / 180.

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + np.pi / 2.) * e_long + np.sin(self.arm_azimuth + np.pi / 2.) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/Kagra_128Mpc_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [10, 1000, 1e-25, 1e-20]
        elif self.name == 'LIN':
            self.lat = 19.61 * np.pi / 180.
            self.lon = 77.03 * np.pi / 180.
            self.opening_angle = np.pi / 2.
            self.arm_azimuth = 100. * np.pi / 180.  # this value is guessed from wikipage

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])

            self.position = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * e_long + np.sin(self.arm_azimuth) * e_lat
            self.e2 = np.cos(self.arm_azimuth + np.pi / 2.) * e_long + np.sin(self.arm_azimuth + np.pi / 2.) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/LIGO_O5_psd.txt')

            self.duty_factor = 0.85

            self.plotrange = [10, 1000, 1e-25, 1e-20]
        elif self.name[0:4] == 'LISA':
            self.opening_angle = np.pi / 3.
            self.lat = 43.6 * np.pi / 180.
            self.lon = 10.5 * np.pi / 180.
            self.arm_azimuth = 71. * np.pi / 180.

            ff = np.logspace(-4, 0, 1000)

            # see LISA 2017 mission document
            P_rec = 700e-12
            S_acc = 9e-30 * (1 + (4e-4 / ff) ** 2) * (1 + (ff / 8e-3) ** 4)
            self.L = 2.5e9
            f0 = c / 1064e-9
            self.eps = self.L / AU / (2 * np.sqrt(3))

            self.psd_data = np.zeros((len(ff), 2))
            self.psd_data[:, 0] = ff
            # noises in units of GW strain
            # S_opt = (c/(2*np.pi*f0*self.L))**2*h*f0/P_rec #pure quantum noise
            S_opt = 1e-22 * (1 + (2e-3 / ff) ** 4) / self.L ** 2
            S_pm = (2 / self.L) ** 2 * S_acc / (2 * np.pi * ff) ** 4

            if k < 2:
                # instrument noise of A,E channels
                self.psd_data[:, 1] = 16 * np.sin(np.pi * ff * self.L / c) ** 2 * (
                            3 + 2 * np.cos(2 * np.pi * ff * self.L / c) + np.cos(4 * np.pi * ff * self.L / c)) * S_pm \
                                      + 8 * np.sin(np.pi * ff * self.L / c) ** 2 * (
                                                  2 + np.cos(2 * np.pi * ff * self.L / c)) * S_opt
            else:
                # instrument noise of T channel
                self.psd_data[:, 1] = 2 * (1 + 2 * np.cos(2 * np.pi * ff * self.L / c)) ** 2 * (
                            4 * np.sin(np.pi * ff * self.L / c) ** 2 * S_pm + S_opt)

            # the sensitivity model is based on a sky-averaged GW response. In the long-wavelength regime, it can be
            # converted into a simple noise PSD by multiplying with 3/10 (arXiv:1803.01944)
            self.duty_factor = 1.

            self.plotrange = [1e-3, 0.3, 1e-22, 1e-19]
        elif self.name[0:4] == 'LGWA':
            if k == 0: #Shackleton
                self.lat = -89.9 * np.pi / 180.
                self.lon = 0
                self.hor_direction = np.pi
            elif k == 1: #de Garlache
                self.lat = -88.5 * np.pi / 180.
                self.lon = -87.1 * np.pi / 180.
                self.hor_direction = np.pi
            elif k == 2: #Shoemaker
                self.lat = -88.1 * np.pi / 180.
                self.lon = 44.9 * np.pi / 180.
                self.hor_direction = np.pi
            elif k == 3: #Faustini
                self.lat = -87.3 * np.pi / 180.
                self.lon = 77 * np.pi / 180.
                self.hor_direction = np.pi

            e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),
                              -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])
            e_rad = np.array([np.cos(self.lat) * np.cos(self.lon),
                                      np.cos(self.lat) * np.sin(self.lon),
                                      np.sin(self.lat)])

            self.e1 = e_rad
            self.e2 = np.cos(self.hor_direction) * e_long + np.sin(self.hor_direction) * e_lat

            self.psd_data = np.loadtxt('./detector_psd/LGWA_psd.txt')

            self.duty_factor = 0.7
        else:
            print('Detector ' + self.name + ' invalid!')
            exit()

        self.Sn = interp1d(self.psd_data[:, 0], self.psd_data[:, 1], bounds_error=False, fill_value=1.)

        if self.plot:
            plt.figure()
            plt.loglog(self.psd_data[:, 0], np.sqrt(self.psd_data[:, 1]))
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Strain noise')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('Sensitivity_' + self.name + '.png')
            plt.close()


class Detector:

    def __init__(self, name='ET', number_of_signals=1, parameters=None, plot=False):
        self.interferometers = []
        self.fisher_matrix = np.zeros((len(parameters), 9, 9))
        self.name = name
        self.SNR = np.zeros(len(parameters))

        if name == 'LISA':
            self.coordinates = 'barycentric'
            self.mission_lifetime = 4 * 3.16e7
        elif name == 'LGWA':
            self.coordinates = 'selenographic'
            self.mission_lifetime = 10 * 3.16e7
        else:
            self.coordinates = 'earthbased'

        if name[0:2] == 'ET' or name[1:3] == 'ET' or name == 'LISA':
            for k in np.arange(3):
                self.interferometers.append(
                    Interferometer(name=name, interferometer=k, plot=plot))
        elif name[0:4] == 'LGWA':
            for k in np.arange(4):
                self.interferometers.append(
                    Interferometer(name=name, interferometer=k, plot=plot))
        else:
            self.interferometers.append(
                Interferometer(name=name, plot=plot))


class Network:

    def __init__(self, detector_ids=['ET'], number_of_signals=1, detection_SNR=8., parameters=None, plot=False):
        self.name = detector_ids[0]
        for id in detector_ids[1:]:
            self.name += '_' + id

        self.detection_SNR = detection_SNR
        self.SNR = np.zeros(len(parameters))

        self.detectors = []
        for d in np.arange(len(detector_ids)):
            detectors = Detector(name=detector_ids[d], number_of_signals=number_of_signals, parameters=parameters, plot=plot)
            self.detectors.append(detectors)


def invertSVD(matrix):
    dm = np.sqrt(np.diag(matrix))
    normalizer = np.outer(dm, dm)
    matrix_norm = matrix / normalizer

    [U, S, Vh] = np.linalg.svd(matrix_norm)
    thresh = 1e-10
    kVal = sum(S > thresh)
    matrix_inverse_norm = U[:, 0:kVal] @ np.diag(1. / S[0:kVal]) @ Vh[0:kVal, :]
    # print(matrix @ (matrix_inverse_norm / normalizer))

    return matrix_inverse_norm / normalizer


def fisco(parameters):
    M = (parameters['mass_1'] + parameters['mass_2']) * Msol * (1 + parameters['redshift'])

    return 1 / (np.pi) * c ** 3 / (G * M) / 6 ** 1.5  # frequency of innermost stable circular orbit


def TaylorF2(parameters, frequencyvector, maxn=8, plot=None):
    ff = frequencyvector
    ones = np.ones((len(ff), 1))

    phic = parameters['phase']
    tc = parameters['geocent_time']
    z = parameters['redshift']
    r = parameters['luminosity_distance'] * Mpc
    iota = parameters['iota']
    M1 = parameters['mass_1'] * (1 + z) * Msol
    M2 = parameters['mass_2'] * (1 + z) * Msol

    M = M1 + M2
    mu = M1 * M2 / M

    Mc = G * mu ** 0.6 * M ** 0.4 / c ** 3

    # compute GW amplitudes (https://arxiv.org/pdf/2012.01350.pdf)
    hp = c / (2. * r) * np.sqrt(5. * np.pi / 24.) * Mc ** (5. / 6.) / (np.pi * ff) ** (7. / 6.) * (
                1. + np.cos(iota) ** 2.)
    hc = c / (2. * r) * np.sqrt(5. * np.pi / 24.) * Mc ** (5. / 6.) / (np.pi * ff) ** (7. / 6.) * 2. * np.cos(iota)

    C = 0.57721566  # Euler constant
    eta = mu / M

    f_isco = fisco(parameters)

    v = (np.pi * G * M / c ** 3 * ff) ** (1. / 3.)
    v_isco = (np.pi * G * M / c ** 3 * f_isco) ** (1. / 3.)

    # coefficients of the PN expansion (https://arxiv.org/pdf/0907.0700.pdf)
    pp = np.hstack((1. * ones, 0. * ones, 20. / 9. * (743. / 336. + eta * 11. / 4.) * ones, -16 * np.pi * ones,
                    10. * (3058673. / 1016064. + 5429. / 1008. * eta + 617. / 144. * eta ** 2) * ones,
                    np.pi * (38645. / 756. - 65. / 9. * eta) * (1 + 3. * np.log(v / v_isco)),
                    11583231236531. / 4694215680. - 640. / 3. * np.pi ** 2 - 6848. / 21. * (C + np.log(4 * v))
                    + (
                                -15737765635. / 3048192. + 2255. / 12. * np.pi ** 2) * eta + 76055. / 1728. * eta ** 2 - 127825. / 1296. * eta ** 3,
                    np.pi * (77096675. / 254016. + 378515. / 1512. * eta - 74045. / 756. * eta ** 2) * ones))
    #print('pp = ',pp[:,0])

    psi = 0.

    for k in np.arange(maxn):
        PNc = pp[:, k]
        psi += PNc[:, np.newaxis] * v ** k

    psi *= 3. / (128. * eta * v ** 5)

    # t(f) is required to calculate slowly varying antenna pattern as function of instantaneous frequency.
    # This FD approach follows Marsat/Baker arXiv:1806.10734v1; equation (22) neglecting the phase term, which does not
    # matter for SNR calculations.
    t_of_f = np.diff(psi, axis=0) / (2. * np.pi * (ff[1] - ff[0]))
    #print('t_of_f', t_of_f)
    t_of_f = tc + np.vstack((t_of_f, [t_of_f[-1]]))
    #print('t_of_f', t_of_f)

    psi += 2. * np.pi * ff * tc - phic - np.pi / 4.
    phase = np.exp(1.j * psi)
    #print('phase = ',phase)
    #print('phase*j = ',1.j * phase)
    polarizations = np.hstack((hp * phase, hc * 1.j * phase))
    polarizations[np.where(ff > 2 * f_isco), :] = 0.j  # very crude high-f cut-off

    if plot != None:
        plt.figure()
        plt.semilogx(ff, t_of_f - tc)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('t(f)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('t_of_f.png')
        plt.close()

    return polarizations, t_of_f


def GreenwichMeanSiderealTime(gps):
    # calculate the Greenwhich mean sidereal time
    return np.mod(9.533088395981618 + (gps - 1126260000.) / 3600. * 24. / sidereal_day, 24) * np.pi / 12.

def LunarMeanSiderealTime(gps):
    # calculate the Lunar mean sidereal time
    return np.mod((gps - 1126260000.) / (3600. * lunar_sidereal_period), 1) * 2. * np.pi

def lisalike(tt, R, eps, a0, b0):
    w0 = np.sqrt(G * Msol / R ** 3)  # w0 has a 1% error when using this equation for Earth orbit

    a = w0 * tt + a0
    b = b0 + 2 * np.pi / 3. * np.arange(3)
    pp = np.zeros((len(tt), 3, 3))
    # the trajectories describe th cartwheel motion, but not the breathing motion proportional to eps^2
    for k in np.arange(3):
        pp[:, k, :] = R * np.hstack((np.cos(a), np.sin(a), np.zeros_like(a))) \
                      + R * eps * np.hstack((0.5 * np.cos(2 * a - b[k]) - 1.5 * np.cos(b[k]),
                                             0.5 * np.sin(2 * a - b[k]) - 1.5 * np.sin(b[k]),
                                             -np.sqrt(3) * np.cos(a - b[k])))

    return pp


def yGW(i, j, polarizations, eij, theta, ra, psi, L, ff):
    ek = -np.array([np.sin(theta) * np.cos(ra), np.sin(theta) * np.sin(ra), np.cos(theta)])
    u = np.array([np.cos(theta) * np.cos(ra), np.cos(theta) * np.sin(ra), -np.sin(theta)])
    v = np.array([-np.sin(ra), np.cos(ra), 0])

    m = -u * np.sin(psi) - v * np.cos(psi)
    n = -u * np.cos(psi) + v * np.sin(psi)

    hxx = polarizations[:, 0] * (m[0] * m[0] - n[0] * n[0]) + polarizations[:, 1] * (m[0] * n[0] + n[0] * m[0])
    hxy = polarizations[:, 0] * (m[0] * m[1] - n[0] * n[1]) + polarizations[:, 1] * (m[0] * n[1] + n[0] * m[1])
    hxz = polarizations[:, 0] * (m[0] * m[2] - n[0] * n[2]) + polarizations[:, 1] * (m[0] * n[2] + n[0] * m[2])
    hyy = polarizations[:, 0] * (m[1] * m[1] - n[1] * n[1]) + polarizations[:, 1] * (m[1] * n[1] + n[1] * m[1])
    hyz = polarizations[:, 0] * (m[1] * m[2] - n[1] * n[2]) + polarizations[:, 1] * (m[1] * n[2] + n[1] * m[2])
    hzz = polarizations[:, 0] * (m[2] * m[2] - n[2] * n[2]) + polarizations[:, 1] * (m[2] * n[2] + n[2] * m[2])

    if i == np.mod(j + 1, 3):
        sgn = -1
    else:
        sgn = 1
    k = np.delete(np.array([0, 1, 2]), [i, j])[0]

    muk = (eij[:, np.mod(k + 1, 3), :] - eij[:, np.mod(k - 1, 3), :]) @ ek
    muk = muk[:, np.newaxis]
    muj = (eij[:, np.mod(j + 1, 3), :] - eij[:, np.mod(j - 1, 3), :]) @ ek
    muj = muj[:, np.newaxis]

    proj = eij[:, i, :] @ ek

    h_ifo = 0.5 * (eij[:, i, 0] ** 2 * hxx + eij[:, i, 1] ** 2 * hyy + eij[:, i, 2] ** 2 * hzz) \
            + eij[:, i, 0] * eij[:, i, 1] * hxy + eij[:, i, 0] * eij[:, i, 2] * hxz + eij[:, i, 1] * eij[:, i, 2] * hyz

    y = 0.5 / (1 + sgn * proj[:, np.newaxis]) * (
                np.exp(2j * np.pi * ff * L / c * (muk / 3. + 1.)) - np.exp(2j * np.pi * ff * L / c * muj / 3.)) * h_ifo[
                                                                                                                  :,
                                                                                                                  np.newaxis]

    return y


def alpha(i, yGWij, L, ff):
    dL = np.exp(2j * np.pi * ff[:, 0] * L / c)
    dL2 = np.exp(4j * np.pi * ff[:, 0] * L / c)

    return yGWij[:, np.mod(i + 1, 3), i] - yGWij[:, np.mod(i - 1, 3), i] \
           + dL * (yGWij[:, i, np.mod(i - 1, 3)] - yGWij[:, i, np.mod(i + 1, 3)]) \
           + dL2 * (yGWij[:, np.mod(i - 1, 3), np.mod(i + 1, 3)] - yGWij[:, np.mod(i + 1, 3), np.mod(i - 1, 3)])


def AET(polarizations, eij, theta, ra, psi, L, ff):
    yGWij = np.zeros((len(ff), 3, 3), dtype='complex')
    for k1 in np.arange(3):
        for k2 in np.arange(k1 + 1, 3):
            yGWij[:, k1, k2] = yGW(k1, k2, polarizations, eij, theta, ra, psi, L, ff)[:, 0]
            yGWij[:, k2, k1] = yGW(k2, k1, polarizations, eij, theta, ra, psi, L, ff)[:, 0]

    a0 = alpha(0, yGWij, L, ff)
    a1 = alpha(1, yGWij, L, ff)
    a2 = alpha(2, yGWij, L, ff)

    A = (a2 - a0) / np.sqrt(2)
    E = (a0 - 2 * a1 + a2) / np.sqrt(6)
    T = (a0 + a1 + a2) / np.sqrt(3)

    return np.hstack((A[:, np.newaxis], E[:, np.newaxis], T[:, np.newaxis]))


def projection(parameters, detector, polarizations, timevector, frequencyvector, max_time_until_merger):
    # rudimentary:
    # coords = SkyCoord(ra=ra, dec=dec, frame='icrs', unit='rad')
    # angles = coords.transform_to('barycentricmeanecliptic')

    if detector.coordinates == 'earthbased':
        proj = projection_longwavelength(parameters, detector, polarizations, timevector)
    elif detector.coordinates == 'selenographic':
        proj = projection_moon(parameters, detector, polarizations, timevector, max_time_until_merger)
    else:
        nt = len(polarizations[:, 0])
        ff = frequencyvector

        interferometers = detector.interferometers

        if timevector.ndim == 1:
            timevector = timevector[:, np.newaxis]

        ra = parameters['ra']
        dec = parameters['dec']
        psi = parameters['psi']

        theta = np.pi / 2. - dec

        pp = lisalike(timevector, AU, interferometers[0].eps, 0., 0.)
        eij = (pp[:, [1, 2, 0], :] - pp[:, [2, 0, 1], :]) / interferometers[0].L

        # start_time = time.time()
        doppler_to_strain = c / (interferometers[0].L * 2 * np.pi * ff)
        proj = doppler_to_strain * AET(polarizations, eij, theta, ra, psi, interferometers[0].L, ff)
        # print("Calculation of projection: %s seconds" % (time.time() - start_time))

        # define LISA observation window
        max_observation_time = detector.mission_lifetime
        tc = parameters['geocent_time']
        proj[np.where(timevector < tc - max_time_until_merger), :] = 0.j
        proj[np.where(timevector > tc - max_time_until_merger + max_observation_time), :] = 0.j

        i0 = np.argmin(np.abs(timevector - (tc - max_time_until_merger)))
        i1 = np.argmin(np.abs(timevector - (tc - max_time_until_merger + max_observation_time)))

        if 'id' in parameters:
            print('{} observed between {:.3f}Hz to {:.3f}Hz'.format(parameters['id'], ff[i0, 0], ff[i1, 0]))

    return proj


def projection_longwavelength(parameters, detector, polarizations, timevector):
    """
    See Nishizawa et al. (2009) arXiv:0903.0528 for definitions of the polarisation tensors.
    [u, v, w] represent the Earth-frame
    [m, n, omega] represent the wave-frame
    Note1: there is a typo in the definition of the wave-frame in Nishizawa et al.
    Note2: it is computationally more expensive to use numpy.einsum instead of working with several vector quantities
    """

    # timevector = parameters['geocent_time'] * np.ones_like(timevector)  # switch off Earth's rotation

    nt = len(polarizations[:, 0])

    interferometers = detector.interferometers
    proj = np.zeros((nt, len(interferometers)), dtype=complex)

    if timevector.ndim == 1:
        timevector = timevector[:, np.newaxis]

    ra = parameters['ra']
    dec = parameters['dec']
    psi = parameters['psi']

    theta = np.pi / 2. - dec
    gmst = GreenwichMeanSiderealTime(timevector)
    phi = ra - gmst

    # wave vector components
    kx = -np.sin(theta)*np.cos(phi)
    ky = -np.sin(theta)*np.sin(phi)
    kz = -np.cos(theta)

    # start_time = time.time()
    # u = np.array([np.cos(theta) * np.cos(phi[:,0]), np.cos(theta) * np.sin(phi[:,0]), -np.sin(theta)*np.ones_like(phi[:,0])])
    ux = np.cos(theta) * np.cos(phi[:, 0])
    uy = np.cos(theta) * np.sin(phi[:, 0])
    uz = -np.sin(theta)
    # v = np.array([-np.sin(phi[:,0]), np.cos(phi[:,0]), np.zeros_like(phi[:,0])])
    vx = -np.sin(phi[:, 0])
    vy = np.cos(phi[:, 0])
    vz = 0
    # print("Creating vectors u,v: %s seconds" % (time.time() - start_time))

    # start_time = time.time()
    # m = -u * np.sin(psi) - v * np.cos(psi)
    mx = -ux * np.sin(psi) - vx * np.cos(psi)
    my = -uy * np.sin(psi) - vy * np.cos(psi)
    mz = -uz * np.sin(psi) - vz * np.cos(psi)
    # n = -u * np.cos(psi) + v * np.sin(psi)
    nx = -ux * np.cos(psi) + vx * np.sin(psi)
    ny = -uy * np.cos(psi) + vy * np.sin(psi)
    nz = -uz * np.cos(psi) + vz * np.sin(psi)
    # print("Creating vectors m, n: %s seconds" % (time.time() - start_time))

    # start_time = time.time()
    # hpij = np.einsum('ij,kj->jik', m, m) - np.einsum('ij,kj->jik', n, n)
    # hcij = np.einsum('ij,kj->jik', m, n) + np.einsum('ij,kj->jik', n, m)
    # hij = np.einsum('i,ijk->ijk', polarizations[:, 0], hpij) + np.einsum('i,ijk->ijk', polarizations[:, 1], hcij)
    hxx = polarizations[:, 0] * (mx * mx - nx * nx) + polarizations[:, 1] * (mx * nx + nx * mx)
    hxy = polarizations[:, 0] * (mx * my - nx * ny) + polarizations[:, 1] * (mx * ny + nx * my)
    hxz = polarizations[:, 0] * (mx * mz - nx * nz) + polarizations[:, 1] * (mx * nz + nx * mz)
    hyy = polarizations[:, 0] * (my * my - ny * ny) + polarizations[:, 1] * (my * ny + ny * my)
    hyz = polarizations[:, 0] * (my * mz - ny * nz) + polarizations[:, 1] * (my * nz + ny * mz)
    hzz = polarizations[:, 0] * (mz * mz - nz * nz) + polarizations[:, 1] * (mz * nz + nz * mz)
    # print("Calculation GW tensor: %s seconds" % (time.time() - start_time))

    # start_time = time.time()
    for k in np.arange(len(interferometers)):
        e1 = interferometers[k].e1
        e2 = interferometers[k].e2

        # interferometer position
        x_det = interferometers[k].position[0]
        y_det = interferometers[k].position[1]
        z_det = interferometers[k].position[2]
        phase_shift = -1.j *np.squeeze(x_det*kx + y_det*ky + z_det*kz)

        # proj[:, k] = 0.5*(np.einsum('i,jik,k->j', e1, hij, e1) - np.einsum('i,jik,k->j', e2, hij, e2))
        proj[:, k] = 0.5 * (e1[0] ** 2 - e2[0] ** 2) * hxx \
                     + 0.5 * (e1[1] ** 2 - e2[1] ** 2) * hyy \
                     + 0.5 * (e1[2] ** 2 - e2[2] ** 2) * hzz \
                     + (e1[0] * e1[1] - e2[0] * e2[1]) * hxy \
                     + (e1[0] * e1[2] - e2[0] * e2[2]) * hxz \
                     + (e1[1] * e1[2] - e2[1] * e2[2]) * hyz
        proj[:, k] *= phase_shift
    # print("Calculation of projection: %s seconds" % (time.time() - start_time))

    return proj

def projection_moon(parameters, detector, polarizations, timevector, max_time_until_merger):
    """
    See Nishizawa et al. (2009) arXiv:0903.0528 for definitions of the polarisation tensors.
    [u, v, w] represent the Earth-frame
    [m, n, omega] represent the wave-frame
    Note1: there is a typo in the definition of the wave-frame in Nishizawa et al.
    Note2: it is computationally more expensive to use numpy.einsum instead of working with several vector quantities
    """

    # timevector = parameters['geocent_time'] * np.ones_like(timevector)  # switch off Earth's rotation

    nt = len(polarizations[:, 0])

    interferometers = detector.interferometers
    proj = np.zeros((nt, len(interferometers)), dtype=complex)

    if timevector.ndim == 1:
        timevector = timevector[:, np.newaxis]

    ra = parameters['ra']
    dec = parameters['dec']
    psi = parameters['psi']

    theta = np.pi / 2. - dec
    lmst = LunarMeanSiderealTime(timevector)
    phi = ra - lmst

    # saving timevector and lmst for plotting
    #np.save('timevector.npy', timevector)
    #np.save('lmst.npy', lmst)

    # start_time = time.time()
    # u = np.array([np.cos(theta) * np.cos(phi[:,0]), np.cos(theta) * np.sin(phi[:,0]), -np.sin(theta)*np.ones_like(phi[:,0])])
    ux = np.cos(theta) * np.cos(phi[:, 0])
    uy = np.cos(theta) * np.sin(phi[:, 0])
    uz = -np.sin(theta)
    # v = np.array([-np.sin(phi[:,0]), np.cos(phi[:,0]), np.zeros_like(phi[:,0])])
    vx = -np.sin(phi[:, 0])
    vy = np.cos(phi[:, 0])
    vz = 0
    # print("Creating vectors u,v: %s seconds" % (time.time() - start_time))

    # start_time = time.time()
    # m = -u * np.sin(psi) - v * np.cos(psi)
    mx = -ux * np.sin(psi) - vx * np.cos(psi)
    my = -uy * np.sin(psi) - vy * np.cos(psi)
    mz = -uz * np.sin(psi) - vz * np.cos(psi)
    # n = -u * np.cos(psi) + v * np.sin(psi)
    nx = -ux * np.cos(psi) + vx * np.sin(psi)
    ny = -uy * np.cos(psi) + vy * np.sin(psi)
    nz = -uz * np.cos(psi) + vz * np.sin(psi)
    # print("Creating vectors m, n: %s seconds" % (time.time() - start_time))

    # start_time = time.time()
    # hpij = np.einsum('ij,kj->jik', m, m) - np.einsum('ij,kj->jik', n, n)
    # hcij = np.einsum('ij,kj->jik', m, n) + np.einsum('ij,kj->jik', n, m)
    # hij = np.einsum('i,ijk->ijk', polarizations[:, 0], hpij) + np.einsum('i,ijk->ijk', polarizations[:, 1], hcij)
    hxx = polarizations[:, 0] * (mx * mx - nx * nx) + polarizations[:, 1] * (mx * nx + nx * mx)
    hxy = polarizations[:, 0] * (mx * my - nx * ny) + polarizations[:, 1] * (mx * ny + nx * my)
    hxz = polarizations[:, 0] * (mx * mz - nx * nz) + polarizations[:, 1] * (mx * nz + nx * mz)
    hyy = polarizations[:, 0] * (my * my - ny * ny) + polarizations[:, 1] * (my * ny + ny * my)
    hyz = polarizations[:, 0] * (my * mz - ny * nz) + polarizations[:, 1] * (my * nz + ny * mz)
    hzz = polarizations[:, 0] * (mz * mz - nz * nz) + polarizations[:, 1] * (mz * nz + nz * mz)
    # print("Calculation GW tensor: %s seconds" % (time.time() - start_time))

    # start_time = time.time()
    for k in np.arange(len(interferometers)):
        e1 = interferometers[k].e1
        e2 = interferometers[k].e2
        # proj[:, k] = 0.5*(np.einsum('i,jik,k->j', e1, hij, e1) - np.einsum('i,jik,k->j', e2, hij, e2))
        proj[:, k] = 0.5 * (e1[0] ** 2 - e2[0] ** 2) * hxx \
                     + 0.5 * (e1[1] ** 2 - e2[1] ** 2) * hyy \
                     + 0.5 * (e1[2] ** 2 - e2[2] ** 2) * hzz \
                     + (e1[0] * e1[1] - e2[0] * e2[1]) * hxy \
                     + (e1[0] * e1[2] - e2[0] * e2[2]) * hxz \
                     + (e1[1] * e1[2] - e2[1] * e2[2]) * hyz
    # print("Calculation of projection: %s seconds" % (time.time() - start_time))

    max_observation_time = detector.mission_lifetime
    tc = parameters['geocent_time']
    proj[np.where(timevector < tc - max_observation_time), :] = 0.j

    return proj

def lisaGWresponse(detector, frequencyvector):
    ff = frequencyvector
    nf = len(ff)

    polarizations = np.ones((nf, 2))
    timevector = np.zeros((nf, 1))

    interferometers = detector.interferometers

    ra = 0
    dec = np.pi / 2.
    psi = 0

    theta = np.pi / 2. - dec

    pp = lisalike(timevector, AU, interferometers[0].eps, 0., 0.)
    eij = (pp[:, [1, 2, 0], :] - pp[:, [2, 0, 1], :]) / interferometers[0].L

    doppler_to_strain = c / (interferometers[0].L * 2 * np.pi * ff)
    proj = doppler_to_strain * AET(polarizations, eij, theta, ra, psi, interferometers[0].L, ff)

    plt.figure()
    plt.loglog(ff, np.abs(proj[:, 0]))
    plt.loglog(ff, np.abs(proj[:, 1]))
    plt.loglog(ff, np.abs(proj[:, 2]))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('GW response')
    plt.xlim((ff[0], ff[-1]))
    plt.ylim((1e-5, 10))
    plt.grid(True)
    plt.legend(['A', 'E', 'T'])
    plt.tight_layout()
    plt.savefig('ResponseGW_' + detector.name + '.png')
    plt.close()

    N = 10000
    for k in range(N):
        ra = 2 * np.pi * np.random.rand()
        costheta = 2 * np.random.rand() - 1
        psi = 2 * np.pi * np.random.rand()

        pp = lisalike(timevector, AU, interferometers[0].eps, 0., 0.)
        eij = (pp[:, [1, 2, 0], :] - pp[:, [2, 0, 1], :]) / interferometers[0].L

        doppler_to_strain = c / (interferometers[0].L * 2 * np.pi * ff)
        proj += (doppler_to_strain * AET(polarizations, eij, np.arccos(costheta), ra, psi, interferometers[0].L,
                                         ff)) ** 2

    proj /= N

    psds = np.zeros((nf, 3))
    for k in range(3):
        psds[:, k] = interferometers[k].Sn(ff[:, 0])

    plt.figure()
    plt.loglog(ff, np.sqrt(psds[:, 0] / np.abs(proj[:, 0])))
    plt.loglog(ff, np.sqrt(psds[:, 1] / np.abs(proj[:, 1])))
    plt.loglog(ff, np.sqrt(psds[:, 2] / np.abs(proj[:, 2])))
    plt.loglog(ff, 1 / np.sqrt(
        np.abs(proj[:, 0]) / psds[:, 0] + np.abs(proj[:, 1]) / psds[:, 1] + np.abs(proj[:, 2]) / psds[:, 2]))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Sensitivity [Hz^{-1/2}]')
    plt.xlim((ff[0], ff[-1]))
    # plt.ylim((1e-5, 10))
    plt.grid(True)
    plt.legend(['A', 'E', 'T', 'combined'])
    plt.tight_layout()
    plt.savefig('Sensitivity_Skyav_GW_' + detector.name + '.png')
    plt.close()


def derivative(parameters, p, detector, frequencyvector, max_time_until_merger):
    """
    Calculates derivatives with respect to geocent_time and phase analytically, of the other parameters numerically.
    """

    fisher_parameters = ['ra', 'dec', 'psi', 'iota', 'luminosity_distance', 'mass_1', 'mass_2', 'geocent_time', 'phase']
    # fisher_parameters = ['ra', 'dec', 'psi', 'iota', 'redshift', 'mass_1', 'mass_2', 'geocent_time', 'phase']

    local_params = parameters.copy()

    tc = local_params['geocent_time']

    if p == 4:  # derivative wrt luminosity distance
        wave, t_of_f = TaylorF2(local_params, frequencyvector, maxn=8)
        r = local_params['luminosity_distance'] * Mpc
        derivative = -1. / r * projection(local_params, detector, wave, t_of_f, frequencyvector, max_time_until_merger)
    elif p == 7:    # derivative wrt merger time
        wave, t_of_f = TaylorF2(local_params, frequencyvector, maxn=8)

        derivative = 2j * np.pi * frequencyvector * projection(local_params, detector, wave, t_of_f, frequencyvector,
                                                               max_time_until_merger)
    elif p == 8:    # derivative wrt phase parameter
        wave, t_of_f = TaylorF2(local_params, frequencyvector, maxn=8)

        derivative = -1j * projection(local_params, detector, wave, t_of_f, frequencyvector, max_time_until_merger)
    else:
        pv = local_params[fisher_parameters[p]]
        eps = 1e-5  # this follows the simple "cube root of numerical precision" recommendation, which is 1e-16 for double
        dp = np.maximum(eps, eps * pv)

        pv_set1 = parameters.copy()
        pv_set2 = parameters.copy()

        pv_set1[fisher_parameters[p]] = pv - dp / 2.
        pv_set2[fisher_parameters[p]] = pv + dp / 2.

        if p < 3:  # these parameters do not influence the waveform
            wave, t_of_f = TaylorF2(local_params, frequencyvector, maxn=8)

            signal1 = projection(pv_set1, detector, wave, t_of_f, frequencyvector, max_time_until_merger)
            signal2 = projection(pv_set2, detector, wave, t_of_f, frequencyvector, max_time_until_merger)

            derivative = (signal2 - signal1) / dp
        else:
            pv_set1['geocent_time'] = 0.  # to improve precision of numerical differentiation
            pv_set2['geocent_time'] = 0.
            wave1, t_of_f1 = TaylorF2(pv_set1, frequencyvector, maxn=8)
            wave2, t_of_f2 = TaylorF2(pv_set2, frequencyvector, maxn=8)

            pv_set1['geocent_time'] = tc
            pv_set2['geocent_time'] = tc
            signal1 = projection(pv_set1, detector, wave1, t_of_f1+tc, frequencyvector, max_time_until_merger)
            signal2 = projection(pv_set2, detector, wave2, t_of_f2+tc, frequencyvector, max_time_until_merger)

            derivative = np.exp(2j * np.pi * frequencyvector * tc) * (signal2 - signal1) / dp

    # print(fisher_parameters[p] + ': ' + str(derivative))
    return derivative


def scalar_product(deriv1, deriv2, interferometers, ff):
    if deriv1.ndim == 1:
        deriv1 = deriv1[:, np.newaxis]
        deriv2 = deriv2[:, np.newaxis]

    if ff.ndim == 1:
        ff = ff[:, np.newaxis]

    df = ff[1, 0] - ff[0, 0]

    scalar_prods = np.zeros(len(interferometers))
    for k in np.arange(len(interferometers)):
        scalar_prods[k] = 4 * df * np.sum(
            np.real(deriv1[:, k] * np.conjugate(deriv2[:, k])) / interferometers[k].Sn(ff[:, 0]), axis=0)

    return scalar_prods


def FisherMatrix(parameters, detector, frequencyvector, max_time_until_merger):
    num_p = 9
    fm = np.zeros((num_p, num_p))

    for p1 in np.arange(num_p):
        deriv1 = derivative(parameters, p1, detector, frequencyvector, max_time_until_merger)
        # sum Fisher matrices from different interferometers of same detector (e.g., in the case of ET)
        fm[p1, p1] = np.sum(scalar_product(deriv1, deriv1, detector.interferometers, frequencyvector), axis=0)
        for p2 in np.arange(p1 + 1, num_p):
            deriv2 = derivative(parameters, p2, detector, frequencyvector, max_time_until_merger)
            fm[p1, p2] = np.sum(scalar_product(deriv1, deriv2, detector.interferometers, frequencyvector), axis=0)
            fm[p2, p1] = fm[p1, p2]

    return fm


def SNR(interferometers, signals, frequencyvector, duty_cycle=False, plot=None):
    if signals.ndim == 1:
        signals = signals[:, np.newaxis]

    ff = frequencyvector
    df = ff[1, 0] - ff[0, 0]

    SNRs = np.zeros(len(interferometers))
    for k in np.arange(len(interferometers)):

        SNRs[k] = np.sqrt(4 * df * np.sum(np.abs(signals[:, k]) ** 2 / interferometers[k].Sn(ff[:, 0]), axis=0))
        #print(interferometers[k].name + ': ' + str(SNRs[k]))
        if plot != None:
            plotrange = interferometers[k].plotrange
            plt.figure()
            plt.loglog(ff, 2 * np.sqrt(np.abs(signals[:, k]) ** 2 * df))
            plt.loglog(ff, np.sqrt(interferometers[k].Sn(ff)))
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Strain spectra')
            plt.xlim((plotrange[0], plotrange[1]))
            plt.ylim((plotrange[2], plotrange[3]))
            plt.grid(True)
            plt.tight_layout()
            #plt.savefig('SignalNoise_' + str(interferometers[k].name) + '_' + plot + '.png')
            plt.savefig('SignalNoise_' + str(interferometers[k].name) + '.png')
            plt.close()

            plt.figure()
            plt.semilogx(ff, 2 * np.sqrt(np.abs(signals[:, k]) ** 2 / interferometers[k].Sn(ff[:, 0]) * df))
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('SNR spectrum')
            plt.xlim((plotrange[0], plotrange[1]))
            plt.grid(True)
            plt.tight_layout()
            #plt.savefig('SNR_density_' + str(interferometers[k].name) + '_' + plot + '.png')
            plt.savefig('SNR_density_' + str(interferometers[k].name) + '.png')
            plt.close()

        # set SNRs to zero if interferometer is not operating (according to its duty factor [0,1])
        if duty_cycle:
            operating = np.random.rand()
            #print('operating = ',operating)
            if interferometers[k].duty_factor < operating:
                SNRs[k] = 0.

    return SNRs


def horizon(network, parameters, frequencyvector, detSNR, T, fmax):
    ff = frequencyvector

    def dSNR(z, detector, SNR0):
        z = np.max([0.05, z[0]])

        r = cosmo.luminosity_distance(z).value * Mpc

        # define necessary variables, multiplied with solar mass, parsec, etc.
        M = (parameters['mass_1'] + parameters['mass_2']) * Msol * (1 + z)
        mu = (parameters['mass_1'] * parameters['mass_2'] / (
                parameters['mass_1'] + parameters['mass_2'])) * Msol * (1 + z)

        parameters['redshift'] = z
        f_isco_z = fisco(parameters)

        Mc = G * mu ** 0.6 * M ** 0.4 / c ** 3

        # compute GW amplitudes (https://arxiv.org/pdf/2012.01350.pdf) with optimal orientation
        hp = c / r * np.sqrt(5. * np.pi / 24.) * Mc ** (5. / 6.) / (np.pi * ff) ** (7. / 6.)
        hp[
            ff > 5 * f_isco_z] = 0  # very crude, but reasonable high-f cut-off; matches roughly IMR spectra (in qadrupole order)
        print(5 * f_isco_z)

        hc = 1.j * hp

        hpij = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
        hcij = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])

        interferometers = detector.interferometers

        # project signal onto the detector
        proj = np.zeros((len(hp), len(interferometers)), dtype=complex)

        for k in np.arange(len(interferometers)):
            if detector.name == 'ET':
                n = interferometers[k].ifo_id
                az = n * np.pi * 2. / 3.
                opening_angle = np.pi / 3.
                e1 = np.array([np.cos(az), np.sin(az), 0.])
                e2 = np.array([np.cos(az + opening_angle), np.sin(az + opening_angle), 0.])
            else:
                e1 = np.array([1., 0., 0.])
                e2 = np.array([0., 1., 0.])

            proj[:, k] = 0.5 * hp[:, 0] * (e1 @ hpij @ e1 - e2 @ hpij @ e2) \
                         + 0.5 * hc[:, 0] * (e1 @ hcij @ e1 - e2 @ hcij @ e2)

        SNRs = SNR(interferometers, proj, ff)
        SNRtot = np.sqrt(np.sum(SNRs ** 2))

        # print('z = ' + str(z) + ', r = ' + str(cosmo.luminosity_distance(z).value) + 'Mpc, SNR = '+str(SNRtot))

        return SNRtot - SNR0

    for d in np.arange(len(network.detectors)):
        zmax = optimize.root(lambda x: dSNR(x, network.detectors[d], detSNR[1]), 5).x[0]

        print(network.detectors[d].name + ' horizon (time-invariant antenna pattern; M={:.3f}; SNR>{:.2f}): z={:.3f}'
              .format(parameters['mass_1'] + parameters['mass_2'], detSNR[1], zmax))

def analyzeDetections(network, parameters, population, networks_ids):
    param_names = ['ra', 'dec', 'psi', 'iota', 'luminosity_distance', 'mass_1', 'mass_2', 'geocent_time', 'phase']

    detSNR = network.detection_SNR

    ns = len(network.SNR)
    N = len(networks_ids)

    param_pick = param_names + ['redshift']
    save_data = parameters[param_pick]

    for n in np.arange(N):
        maxz = 0
        threshold = np.zeros((len(parameters),))

        network_ids = networks_ids[n]
        network_name = '_'.join([network.detectors[k].name for k in network_ids])

        print('Network: ' + network_name)

        SNR = 0
        for d in network_ids:
            SNR += network.detectors[d].SNR ** 2
        SNR = np.sqrt(SNR)

        save_data = np.c_[save_data, SNR]

        threshold = SNR > detSNR[1]
        print('threshold',threshold[0])
        ndet = len(np.where(threshold)[0])
        print('ndet', ndet)
        if ndet > 0:
            if 'id' in parameters.columns:
                print(parameters['id'][np.where(threshold)] + ' was detected.')
            maxz = np.max(parameters['redshift'].iloc[np.where(threshold)].to_numpy())
        print(
            'Detected signals with SNR>{:.3f}: {:.3f} ({:} out of {:}); z<{:.3f}'.format(detSNR[1], ndet / ns, ndet, ns,
                                                                                         maxz))

        print('SNR: {:.3f} (min) , {:.3f} (max) '.format(np.min(SNR), np.max(SNR)))

    np.savetxt('Signals_' + population + '.txt', save_data, delimiter=' ', fmt='%.3f')


def analyzeFisherErrors(network, parameters, population, networks_ids):
    """
    Analyze parameter errors with respect to the following list of waveform parameters:
        ['ra', 'dec', 'psi', 'iota', 'redshift', 'mass_1', 'mass_2', 'geocent_time', 'phase']
    """
    # param_names = ['ra', 'dec', 'psi', 'iota', 'redshift', 'mass_1', 'mass_2', 'geocent_time', 'phase']
    # units = ['rad', 'rad', 'rad', 'rad', '', 'M_sol', 'M_sol', 's', 'rad']
    param_names = ['ra', 'dec', 'psi', 'iota', 'luminosity_distance', 'mass_1', 'mass_2', 'geocent_time', 'phase']
    units = ['rad', 'rad', 'rad', 'rad', 'Mpc', 'M_sol', 'M_sol', 's', 'rad']

    npar = len(param_names)
    ns = len(network.detectors[0].fisher_matrix[:, 0, 0])  # number of signals
    print('ns = ', ns)
    N = len(networks_ids)

    detect_SNR = network.detection_SNR

    network_names = []
    for n in np.arange(N):
        network_names.append('_'.join([network.detectors[k].name for k in networks_ids[n]]))

    for n in np.arange(N):
        parameter_errors = np.zeros((ns, npar))
        sky_localization = np.zeros((ns,))
        networkSNR = np.zeros((ns,))
        for d in networks_ids[n]:
            networkSNR += network.detectors[d].SNR ** 2
        networkSNR = np.sqrt(networkSNR)

        for k in np.arange(ns):
            print('k = ',k)
            network_fisher_matrix = np.zeros((npar, npar))

            if networkSNR[k] > detect_SNR[1]:
                for d in networks_ids[n]:
                    if network.detectors[d].SNR[k] > detect_SNR[0]:
                        network_fisher_matrix += np.squeeze(network.detectors[d].fisher_matrix[k, :, :])

            if network_fisher_matrix[0, 0] > 0:
                network_fisher_matrix[4, :] *= Mpc  # changing to D_lum error unit to Mpc
                network_fisher_matrix[:, 4] *= Mpc
                network_fisher_inverse = invertSVD(network_fisher_matrix)
                parameter_errors[k, :] = np.sqrt(np.diagonal(network_fisher_inverse))
                sky_localization[k] = 2. * np.pi * np.abs(np.cos(parameters['dec'].iloc[k])) \
                                      * np.sqrt(
                    network_fisher_inverse[0, 0] * network_fisher_inverse[1, 1] - network_fisher_inverse[0, 1] ** 2)
                if(network_fisher_inverse[0, 0] * network_fisher_inverse[1, 1] - network_fisher_inverse[0, 1] ** 2<=0.0):
                    print('ops')

        #ii = np.array(np.where(networkSNR > detect_SNR[1]))
        ii = np.where(networkSNR > detect_SNR[1])[0]
        save_data = np.c_[networkSNR[ii], parameters['redshift'].iloc[ii], parameters[param_names].iloc[ii],
                          sky_localization[ii], parameter_errors[ii, :]]
        np.savetxt('Errors_' + network_names[n] + '_' + population + '_SNR' + str(detect_SNR[1]) + '.txt', save_data,
                   delimiter=' ')  # ,
        # fmt='%.3f')
        np.savetxt('Fisher_' + network_names[n] + '_' + population + '_SNR' + str(detect_SNR[1]) + '.txt', network_fisher_matrix,
                   delimiter=' ')
        np.savetxt('Inverse_Fisher_' + network_names[n] + '_' + population + '_SNR' + str(detect_SNR[1]) + '.txt', network_fisher_inverse,
                   delimiter=' ')

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
    
    if ('ET' in d) or ('aET' in d) or ('bET' in d) or ('cET' in d) or ('ET0' in d) or ('ET1' in d) or\
    ('ET2' in d) or ('ET3' in d) or ('ET4' in d) or ('ET5' in d) or ('ET6' in d) or\
	('S2L_10kmHF_arm90_00deg_ET' in d) or ('N2L_10kmHF_arm90_00deg_ET' in d) or ('N2L_10kmHF_arm90_45deg_ET' in d) or\
	('S2L_15kmHF_arm90_00deg_ET' in d) or ('N2L_15kmHF_arm90_00deg_ET' in d) or ('N2L_15kmHF_arm90_45deg_ET' in d) or\
	('S2L_20kmHF_arm90_00deg_ET' in d) or ('N2L_20kmHF_arm90_00deg_ET' in d) or ('N2L_20kmHF_arm90_45deg_ET' in d) or\
	('S2L_10kmfull_arm90_00deg_ET' in d) or ('N2L_10kmfull_arm90_00deg_ET' in d) or ('N2L_10kmfull_arm90_45deg_ET' in d) or\
	('S2L_15kmfull_arm90_00deg_ET' in d) or ('N2L_15kmfull_arm90_00deg_ET' in d) or ('N2L_15kmfull_arm90_45deg_ET' in d) or\
	('S2L_20kmfull_arm90_00deg_ET' in d) or ('N2L_20kmfull_arm90_00deg_ET' in d) or ('N2L_20kmfull_arm90_45deg_ET' in d) or\
	('S2L_10kmfull_arm90_00deg_ET_room' in d) or ('N2L_10kmfull_arm90_00deg_ET_room' in d) or ('N2L_10kmfull_arm90_45deg_ET_room' in d) or\
	('S2L_15kmfull_arm90_00deg_ET_room' in d) or ('N2L_15kmfull_arm90_00deg_ET_room' in d) or ('N2L_15kmfull_arm90_45deg_ET_room' in d) or\
	('S2L_20kmfull_arm90_00deg_ET_room' in d) or ('N2L_20kmfull_arm90_00deg_ET_room' in d) or ('N2L_20kmfull_arm90_45deg_ET_room' in d) :
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

    network = Network(detectors_ids, number_of_signals=ns, detection_SNR=threshold_SNR, parameters=parameters)

    # lisaGWresponse(network.detectors[0], frequencyvector)
    # exit()

    # horizon(network, parameters.iloc[0], frequencyvector, threshold_SNR, 1./df, fmax)
    # exit()

    #print(parameters.iloc[0])
    print('Processing CBC population')
    #bar = progressbar.ProgressBar(max_value=len(parameters))

    for k in tqdm(np.arange(len(parameters))):
        one_parameters = parameters.iloc[k]
        wave, t_of_f = TaylorF2(one_parameters, frequencyvector, maxn=8)

        networkSNR_sq = 0
        for d in np.arange(len(network.detectors)):
            signal = projection(one_parameters, network.detectors[d], wave, t_of_f, frequencyvector,
                                max_time_until_merger)

            SNRs = SNR(network.detectors[d].interferometers, signal, frequencyvector, duty_cycle=duty_cycle)
            networkSNR_sq += np.sum(SNRs ** 2)
            network.detectors[d].SNR[k] = np.sqrt(np.sum(SNRs ** 2))
            if calculate_errors:
                network.detectors[d].fisher_matrix[k, :, :] = \
                    FisherMatrix(one_parameters, network.detectors[d], frequencyvector, max_time_until_merger)

        network.SNR[k] = np.sqrt(networkSNR_sq)

        #bar.update(k)

    #bar.finish()

    analyzeDetections(network, parameters, population, networks_ids)

    if calculate_errors:
        analyzeFisherErrors(network, parameters, population, networks_ids)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
