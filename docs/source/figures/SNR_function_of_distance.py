import numpy as np
from astropy.cosmology import Planck18

from GWFish.modules.detection import SNR, Detector, projection
from GWFish.modules.waveforms import hphc_amplitudes

import matplotlib.pyplot as plt

redshifts = np.linspace(1e-3, 40, num=200)

params = {
    'mass_1': 1000.0,
    'mass_2': 1000.0,
    'theta_jn': 0.06243186,
    'dec': 0.49576695,
    'ra': 5.33470406,
    'psi': 3.80561946,
    'phase': 5.06447171,
    'geocent_time': 1.75513532e+09,
}

detector = Detector('LGWA', parameters= [None], fisher_parameters= [None])

SNRs = []

for redshift in redshifts:

    distance = Planck18.luminosity_distance(redshift)
    params = params | {'redshift': redshift, 'luminosity_distance': distance}

    polarizations, timevector = hphc_amplitudes(
        'gwfish_TaylorF2', 
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
    SNRs.append(this_SNR)
    
distances = Planck18.luminosity_distance(redshifts)
plt.loglog(distances, SNRs, c='black', label='True SNR')
plt.loglog(distances, SNRs[0]*distances[0]/distances, c='black', ls='--', label='1 / distance scaling')
plt.xlabel('Luminosity distance [Mpc]')
plt.ylabel('SNR [dimensionless]')
plt.legend()
plt.savefig('SNR_against_distance.png')