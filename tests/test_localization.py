from GWFish.modules.fishermatrix import compute_network_errors, compute_detector_fisher, sky_localization_percentile_factor
from GWFish.modules.detection import Network, Detector
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import ticker


BASE_PATH = Path(__file__).parent.parent

def plot_correlations(cov, symbols):
    n = len(symbols)

    sigmas = np.sqrt(np.diagonal(cov))
    correlations = cov / np.kron(sigmas, sigmas).reshape(n, n)

    cmap = plt.get_cmap('RdBu')
    normalization = Normalize(-1, 1)
    fig, axes = plt.subplots(figsize=(10, 6))
    axes.matshow(correlations, cmap=cmap, norm=normalization)
    for axis in [axes.xaxis, axes.yaxis]:
        axis.set_major_locator(ticker.FixedLocator(np.arange(n)))
        axis.set_major_formatter(ticker.FixedFormatter(symbols))
    plt.colorbar(ScalarMappable(cmap=cmap, norm=normalization), ax=axes)
    plt.show()


def test_gw170817_localization():
    
    params = {
        'mass_1': 1.4957673, 
        'mass_2': 1.24276395, 
        'redshift': 0.00980,
        'luminosity_distance': 43.74755446,
        'theta_jn': 2.545065595974997,
        'ra': 3.4461599999999994,
        'dec': -0.4080839999999999,
        'psi': 0.,
        'phase': 0.,
        'geocent_time': 1187008882.4,
        'a_1':0.005136138323169717, 
        'a_2':0.003235146993487445, 
        'lambda_1':368.17802383555687, 
        'lambda_2':586.5487031450857,
    }
    df = pd.DataFrame.from_dict({
        param: np.array([value])
        for param, value in params.items()}
    )
    
    network = Network(['LGWA'])
    
    network_snr, parameter_errors, sky_localization = compute_network_errors(
        network,
        df,
        fisher_parameters=list(params.keys()),
        waveform_model='TaylorF2'
    )

    skyloc_arcmin_square = sky_localization * sky_localization_percentile_factor(90.) * 60**2
    
    assert np.isclose(skyloc_arcmin_square, 3, rtol=0.2)
    
