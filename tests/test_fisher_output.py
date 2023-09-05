from GWFish.modules.fishermatrix import analyze_and_save_to_txt
import GWFish.modules.fishermatrix as fishermatrix
import GWFish.modules.waveforms as waveforms
from GWFish.modules.detection import Network
import pandas as pd
import numpy as np


def test_fisher_analysis_output(mocker):
    params = {
        "mass_1": 1.4,
        "mass_2": 1.4,
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
        parameters=parameter_values,
        fisher_parameters=fisher_parameters,
        config="detectors.yaml",
    )

    network.detectors[0].fisher_matrix[0, :, :] = fishermatrix.FisherMatrix(
        "gwfish_TaylorF2",
        parameter_values.iloc[0],
        fisher_parameters,
        network.detectors[0],
    )

    network.detectors[0].SNR[0] = 100

    mocker.patch("numpy.savetxt")

    analyze_and_save_to_txt(
        network=network,
        parameter_values=parameter_values,
        fisher_parameters=fisher_parameters,
        sub_network_ids_list=[[0]],
        population_name="test",
    )

    header = (
        "network_SNR mass_1 mass_2 redshift luminosity_distance "
        "theta_jn ra dec psi phase geocent_time err_mass_1 err_mass_2 "
        "err_redshift err_luminosity_distance err_theta_jn err_ra "
        "err_dec err_psi err_phase err_geocent_time err_sky_location"
    )

    data = [
        1.00000000000e02,
        1.39999999999e00,
        1.39999999999e00,
        1.00000000000e-02,
        4.00000000000e01,
        2.61799387799e00,
        3.45000000000e00,
        -4.09999999999e-01,
        1.60000000000e00,
        0.00000000000e00,
        1.18700888200e09,
        1.01791427671e-07,
        1.01791427689e-07,
        8.96883449508e-08,
        2.32204133549e00,
        1.04213847237e-01,
        3.12695677565e-03,
        2.69412953826e-03,
        2.04240222976e-01,
        4.09349000642e-01,
        5.63911212310e-05,
        2.42285325663e-05,
    ]

    assert np.savetxt.call_args.args[0] == "Errors_ET_test_SNR8.0.txt"
    assert np.allclose(np.savetxt.call_args.args[1], data)

    assert np.savetxt.call_args.kwargs == {
        "delimiter": " ",
        "header": header,
        "comments": "",
        "fmt": (
            "%s %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E "
            "%.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E"
        ),
    }

def test_fisher_analysis_output_nosky(mocker):
    params = {
        "mass_1": 1.4,
        "mass_2": 1.4,
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
    fisher_parameters.remove('dec')

    network = Network(
        detector_ids=["ET"],
        parameters=parameter_values,
        fisher_parameters=fisher_parameters,
        config="detectors.yaml",
    )

    network.detectors[0].fisher_matrix[0, :, :] = fishermatrix.FisherMatrix(
        "gwfish_TaylorF2",
        parameter_values.iloc[0],
        fisher_parameters,
        network.detectors[0],
    )

    network.detectors[0].SNR[0] = 100

    mocker.patch("numpy.savetxt")

    analyze_and_save_to_txt(
        network=network,
        parameter_values=parameter_values,
        fisher_parameters=fisher_parameters,
        sub_network_ids_list=[[0]],
        population_name="test",
    )

    header = (
        "network_SNR mass_1 mass_2 redshift luminosity_distance "
        "theta_jn ra dec psi phase geocent_time err_mass_1 err_mass_2 "
        "err_redshift err_luminosity_distance err_theta_jn err_ra "
        "err_psi err_phase err_geocent_time"
    )

    data = [
        100.0,
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
        1.009E-07,
        1.009E-07,
        8.648E-08,
        2.321E+00,
        1.040E-01,
        3.121E-03,
        8.905E-02,
        1.771E-01,
        2.251E-05,
    ]

    assert np.savetxt.call_args.args[0] == "Errors_ET_test_SNR8.0.txt"
    assert np.allclose(np.savetxt.call_args.args[1], data, rtol=2e-3)

    assert np.savetxt.call_args.kwargs == {
        "delimiter": " ",
        "header": header,
        "comments": "",
        "fmt": (
            "%s %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E "
            "%.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E"
        ),
    }