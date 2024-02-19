import numpy as np
import pandas as pd
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--plot", action="store_true", default=False, help="produce plots for the tests that support them"
    )


@pytest.fixture
def plot(request):
    return request.config.getoption("--plot")

@pytest.fixture
def gw170817_params():
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
    return pd.DataFrame.from_dict({
        param: np.array([value])
        for param, value in params.items()}
    )