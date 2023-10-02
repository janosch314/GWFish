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


def test_gw170817_localization(gw170817_params):
    
    network = Network(['LGWA'])
    
    network_snr, parameter_errors, sky_localization = compute_network_errors(
        network,
        gw170817_params,
        fisher_parameters=list(gw170817_params.keys()),
        waveform_model='TaylorF2'
    )

    skyloc_arcmin_square = sky_localization * sky_localization_percentile_factor(90.) * 60**2
    
    assert np.isclose(skyloc_arcmin_square, 3, rtol=0.2)
    
