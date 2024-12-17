from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import GWFish.modules.waveforms as waveforms
from GWFish.modules.detection import Detector, Network
from GWFish.modules.fishermatrix import (analyze_and_save_to_txt,
                                         compute_detector_fisher,
                                         compute_network_errors)

BASE_PATH = Path(__file__).parent.parent

# @pytest.mark.skip('Takes a long time and still does not check anything')
def test_gwtc3_catalog_results(plot):
    params = pd.read_hdf(BASE_PATH / 'injections/GWTC3_cosmo_median.hdf5')
    
    z = params['redshift'].copy()
    params = params.drop(['event_ID'], axis=1)
    params.loc[:, 'mass_1'] = params['mass_1'].to_numpy()# * (1+z)
    params.loc[:, 'mass_2'] = params['mass_2'].to_numpy()# * (1+z)
    
    fisher_params = [
        'mass_1', 
        'mass_2',
        'luminosity_distance', 
        'dec', 
        'ra',
        'theta_jn', 
        'psi', 
        'geocent_time',
        'phase', 
        'a_1',
        'a_2', 
    ]

    network = Network(['LLO', 'LHO', 'VIR'], detection_SNR=(0., 25.))

    # network_snr, parameter_errors, sky_localization = compute_network_errors(
    #     network,
    #     params.iloc[:5],
    #     fisher_parameters=fisher_params, 
    #     waveform_model='IMRPhenomXPHM'
    # )
    
    analyze_and_save_to_txt(network, params.iloc[:5], fisher_params, [[0, 1, 2]], 'GWTC3', save_matrices=False, waveform_model='IMRPhenomXPHM')
    
    # TODO: assert correctness based on catalog results

def test_gw190521_full_fisher(plot):
    df = pd.read_hdf(BASE_PATH / 'injections/GWTC3_cosmo_median.hdf5')
    
    params = df[df['event_ID']=='GW190521_074359']
    # do not perform the Fisher analysis over z
    z = params['redshift'].copy()
    params.drop(columns='redshift')
    params.loc[:, 'mass_1'] = params['mass_1'].to_numpy()# * (1+z)
    params.loc[:, 'mass_2'] = params['mass_2'].to_numpy()# * (1+z)
    
    # the first parameter is the event ID
    # fisher_params = params.columns.tolist()[1:]
    
    fisher_params = [
        # 'event_ID', 
        'mass_1', 
        'mass_2',
        'luminosity_distance', 
        'dec', 
        'ra',
        'theta_jn', 
        'psi', 
        'geocent_time',
        'phase', 
        # 'redshift', 
        'a_1',
        'a_2', 
        # 'tilt_1', 
        # 'tilt_2',
        # 'phi_12', 
        # 'phi_jl'
    ]
    
    network = Network(['LGWA'], detection_SNR=(0., 1.))
    
    detected, network_snr, parameter_errors, sky_localization = compute_network_errors(
        network,
        params,
        fisher_parameters=fisher_params, 
        waveform_model='IMRPhenomXPHM'
    )
    
    fisher, square_snr = compute_detector_fisher(Detector('LGWA'), params.iloc[0], fisher_params, waveform_model='IMRPhenomXPHM')
    
    assert np.isclose(np.sqrt(square_snr), network_snr[0])

    if plot:
        plot_correlations(np.linalg.inv(fisher), fisher_params)


def test_fisher_analysis_output(mocker):
    params = {
        "mass_1_source": 1.4,
        "mass_2_source": 1.4,
        "redshift": 0.01,
        "luminosity_distance": 40,
        "theta_jn": 5 / 6 * np.pi,
        "ra": 3.45,
        "dec": -0.41,
        "psi": 1.6,
        "phase": 0,
        "geocent_time": 1187008882,
    }

    parameter_values = pd.DataFrame()
    for key, item in params.items():
        parameter_values[key] = np.full((1,), item)

    fisher_parameters = list(params.keys())
    fisher_parameters.remove('redshift')

    network = Network(
        detector_ids=["ET"],
    )

    mocker.patch("numpy.savetxt")

    analyze_and_save_to_txt(
        network=network,
        parameter_values=parameter_values,
        fisher_parameters=fisher_parameters,
        sub_network_ids_list=[[0]],
        population_name="test",
        waveform_class=waveforms.TaylorF2,
        waveform_model='TaylorF2',
    )

    header = (
        "network_SNR mass_1_source mass_2_source redshift luminosity_distance "
        "theta_jn ra dec psi phase geocent_time err_mass_1_source err_mass_2_source "
        "err_luminosity_distance err_theta_jn err_ra "
        "err_dec err_psi err_phase err_geocent_time err_sky_location"
    )

    # data = [
    #     751.6,
    #     1.39999999999e00,
    #     1.39999999999e00,
    #     1.00000000000e-02,
    #     4.00000000000e01,
    #     2.61799387799e00,
    #     3.45000000000e00,
    #     -4.09999999999e-01,
    #     1.60000000000e00,
    #     0.00000000000e00,
    #     1.18700888200e09,
    #     4.14585880e-08,
    #     4.14585880e-08,
    #     2.09920985e+00,
    #     9.25054358e-02,
    #     3.23901261e-03,
    #     2.76717809e-03,
    #     1.74624505e-01,
    #     3.50802588e-01,
    #     5.83902445e-05,
    #     2.57008004e-05,
    # ]
    data = [761.6,
            1.40000000e+00,
            1.40000000e+00,
            1.00000000e-02,
            4.00000000e+01,
            2.61799388e+00,
            3.45000000e+00,
            -4.10000000e-01,
            1.60000000e+00,
            0.00000000e+00,
            1.18700888e+09,
            3.71590828e-08,
            3.71590828e-08,
            2.34748599e+00,
            1.04523607e-01,
            3.83515682e-03,
            3.00391666e-03,
            1.79771752e-01,
            3.61496005e-01,
            6.24007774e-05,
            3.31926067e-05
    ]
    
    assert np.savetxt.call_args.args[0].name == "Errors_ET_test_SNR10.txt"
    assert np.allclose(np.savetxt.call_args.args[1], data, rtol=0.15)

    assert np.savetxt.call_args.kwargs == {
        "delimiter": " ",
        "header": header,
        "comments": "",
        "fmt": (
            "%s %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E "
            "%.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E"
        ),
    }

def test_fisher_analysis_output_nosky(mocker):
    params = {
        "mass_1_source": 1.4,
        "mass_2_source": 1.4,
        "redshift": 0.01,
        "luminosity_distance": 40,
        "theta_jn": 5 / 6 * np.pi,
        "ra": 3.45,
        "dec": -0.41,
        "psi": 1.6,
        "phase": 0,
        "geocent_time": 1187008882,
    }

    parameter_values = pd.DataFrame()
    for key, item in params.items():
        parameter_values[key] = np.full((1,), item)

    fisher_parameters = list(params.keys())
    fisher_parameters.remove('redshift')
    fisher_parameters.remove('dec')

    network = Network(
        detector_ids=["ET"],
    )


    mocker.patch("numpy.savetxt")

    analyze_and_save_to_txt(
        network=network,
        parameter_values=parameter_values,
        fisher_parameters=fisher_parameters,
        sub_network_ids_list=[[0]],
        population_name="test",
        waveform_class=waveforms.TaylorF2,
        waveform_model='TaylorF2',
    )

    header = (
        "network_SNR mass_1_source mass_2_source redshift luminosity_distance "
        "theta_jn ra dec psi phase geocent_time err_mass_1_source err_mass_2_source "
        "err_luminosity_distance err_theta_jn err_ra "
        "err_psi err_phase err_geocent_time"
    )

    data = [
        751.6,
        1.400E+00,
        1.400E+00,
        1.000E-02,
        4.000E+01,
        2.618E+00,
        3.450E+00,
        -4.100E-01,
        1.600E+00,
        0.000E+00,
        1.187E+09,
        4.035E-08,
        4.035E-08,
        2.068E+00,
        9.210E-02,
        3.223E-03,
        7.521E-02,
        1.504E-01,
        2.529E-05,
    ]

    assert np.savetxt.call_args.args[0].name == "Errors_ET_test_SNR10.txt"
    assert np.allclose(np.savetxt.call_args.args[1], data, rtol=2e-1)

    assert np.savetxt.call_args.kwargs == {
        "delimiter": " ",
        "header": header,
        "comments": "",
        "fmt": (
            "%s %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E "
            "%.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E"
        ),
    }

def test_saving_matrices(mocker):
    params = {
        "mass_1_source": 1.4,
        "mass_2_source": 1.4,
        "redshift": 0.01,
        "luminosity_distance": 40,
        "theta_jn": 5 / 6 * np.pi,
        "ra": 3.45,
        "dec": -0.41,
        "psi": 1.6,
        "phase": 0,
        "geocent_time": 1187008882,
    }

    parameter_values = pd.DataFrame()
    for key, item in params.items():
        parameter_values[key] = np.full((1,), item)

    fisher_parameters = list(params.keys())

    network = Network(
        detector_ids=["ET"],
    )

    mocker.patch("numpy.save")

    analyze_and_save_to_txt(
        network=network,
        parameter_values=parameter_values,
        fisher_parameters=fisher_parameters,
        sub_network_ids_list=[[0]],
        population_name="test",
        waveform_class=waveforms.TaylorF2,
        waveform_model='TaylorF2',
        save_matrices=True
    )
    
    assert np.save.call_args_list[0].args[0].name == "fisher_matrices_ET_test_SNR10.npy"
    assert np.save.call_args_list[1].args[0].name == "inv_fisher_matrices_ET_test_SNR10.npy"