import numpy as np

import GWFish.modules.constants as cst

def convert_args_list_to_float(*args_list):
    """ Converts inputs to floats, returns a list in the same order as the input"""
    try:
        args_list = [float(arg) for arg in args_list]
    except ValueError:
        raise ValueError("Unable to convert inputs to floats")
    return args_list

def lalsim_GetApproximantFromString(waveform_approximant):
    from lalsimulation import GetApproximantFromString
    if isinstance(waveform_approximant, str):
        return GetApproximantFromString(waveform_approximant)
    else:
        raise ValueError("waveform_approximant must be of type str")


def lalsim_SimInspiralFD(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, luminosity_distance, iota, phase,
        longitude_ascending_nodes, eccentricity, mean_per_ano, delta_frequency,
        minimum_frequency, maximum_frequency, reference_frequency,
        waveform_dictionary, approximant):
    """
    Safely call lalsimulation.SimInspiralFD

    Parameters
    ==========
    phase: float, int
    mass_1: float, int
    mass_2: float, int
    spin_1x: float, int
    spin_1y: float, int
    spin_1z: float, int
    spin_2x: float, int
    spin_2y: float, int
    spin_2z: float, int
    reference_frequency: float, int
    luminosity_distance: float, int
    iota: float, int
    longitude_ascending_nodes: float, int
    eccentricity: float, int
    mean_per_ano: float, int
    delta_frequency: float, int
    minimum_frequency: float, int
    maximum_frequency: float, int
    waveform_dictionary: None, lal.Dict
    approximant: int, str
    """
    from lalsimulation import SimInspiralFD

    args = convert_args_list_to_float(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
        luminosity_distance, iota, phase, longitude_ascending_nodes,
        eccentricity, mean_per_ano, delta_frequency, minimum_frequency,
        maximum_frequency, reference_frequency)

    approximant = _get_lalsim_approximant(approximant)

    return SimInspiralFD(*args, waveform_dictionary, approximant)


def lalsim_SimInspiralChooseFDWaveform(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, luminosity_distance, iota, phase,
        longitude_ascending_nodes, eccentricity, mean_per_ano, delta_frequency,
        minimum_frequency, maximum_frequency, reference_frequency,
        waveform_dictionary, approximant):
    """
    Safely call lalsimulation.SimInspiralChooseFDWaveform

    Parameters
    ==========
    phase: float, int
    mass_1: float, int
    mass_2: float, int
    spin_1x: float, int
    spin_1y: float, int
    spin_1z: float, int
    spin_2x: float, int
    spin_2y: float, int
    spin_2z: float, int
    reference_frequency: float, int
    luminosity_distance: float, int
    iota: float, int
    longitude_ascending_nodes: float, int
    eccentricity: float, int
    mean_per_ano: float, int
    delta_frequency: float, int
    minimum_frequency: float, int
    maximum_frequency: float, int
    waveform_dictionary: None, lal.Dict
    approximant: int, str
    """
    from lalsimulation import SimInspiralChooseFDWaveform

    args = convert_args_list_to_float(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
        luminosity_distance, iota, phase, longitude_ascending_nodes,
        eccentricity, mean_per_ano, delta_frequency, minimum_frequency,
        maximum_frequency, reference_frequency)

    approximant = _get_lalsim_approximant(approximant)

    return SimInspiralChooseFDWaveform(*args, waveform_dictionary, approximant)


def _get_lalsim_approximant(approximant):
    if isinstance(approximant, int):
        pass
    elif isinstance(approximant, str):
        approximant = lalsim_GetApproximantFromString(approximant)
    else:
        raise ValueError("approximant not an int")
    return approximant


def lalsim_SimInspiralChooseFDWaveformSequence(
        phase, mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, reference_frequency, luminosity_distance, iota,
        waveform_dictionary, approximant, frequency_array):
    """
    Safely call lalsimulation.SimInspiralChooseFDWaveformSequence

    Parameters
    ==========
    phase: float, int
    mass_1: float, int
    mass_2: float, int
    spin_1x: float, int
    spin_1y: float, int
    spin_1z: float, int
    spin_2x: float, int
    spin_2y: float, int
    spin_2z: float, int
    reference_frequency: float, int
    luminosity_distance: float, int
    iota: float, int
    waveform_dictionary: None, lal.Dict
    approximant: int, str
    frequency_array: np.ndarray, lal.REAL8Vector
    """
    from lal import REAL8Vector, CreateREAL8Vector
    from lalsimulation import SimInspiralChooseFDWaveformSequence

    [mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
     luminosity_distance, iota, phase, reference_frequency] = convert_args_list_to_float(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
        luminosity_distance, iota, phase, reference_frequency)

    if isinstance(approximant, int):
        pass
    elif isinstance(approximant, str):
        approximant = lalsim_GetApproximantFromString(approximant)
    else:
        raise ValueError("approximant not an int")

    if not isinstance(frequency_array, REAL8Vector):
        old_frequency_array = frequency_array.copy()
        frequency_array = CreateREAL8Vector(len(old_frequency_array))
        frequency_array.data = old_frequency_array

    return SimInspiralChooseFDWaveformSequence(
        phase, mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, reference_frequency, luminosity_distance, iota,
        waveform_dictionary, approximant, frequency_array)


def lalsim_SimInspiralWaveformParamsInsertTidalLambda1(
        waveform_dictionary, lambda_1):
    from lalsimulation import SimInspiralWaveformParamsInsertTidalLambda1
    try:
        lambda_1 = float(lambda_1)
    except ValueError:
        raise ValueError("Unable to convert lambda_1 to float")

    return SimInspiralWaveformParamsInsertTidalLambda1(
        waveform_dictionary, lambda_1)


def lalsim_SimInspiralWaveformParamsInsertTidalLambda2(
        waveform_dictionary, lambda_2):
    from lalsimulation import SimInspiralWaveformParamsInsertTidalLambda2
    try:
        lambda_2 = float(lambda_2)
    except ValueError:
        raise ValueError("Unable to convert lambda_2 to float")

    return SimInspiralWaveformParamsInsertTidalLambda2(
        waveform_dictionary, lambda_2)

def lalsim_SimInspiralTransformPrecessingNewInitialConditions(
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
        reference_frequency, phase):
    from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions

    args_list = convert_args_list_to_float(
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
        reference_frequency, phase)

    return SimInspiralTransformPrecessingNewInitialConditions(*args_list)

@np.vectorize
def transform_precessing_spins(theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1,
                               a_2, mass_1, mass_2, reference_frequency, phase):

    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = (
        lalsim_SimInspiralTransformPrecessingNewInitialConditions(
            theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
            reference_frequency, phase))

    return iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z

def bilby_to_lalsimulation_spins(
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
        reference_frequency, phase):
    if (a_1 == 0 or tilt_1 in [0, np.pi]) and (a_2 == 0 or tilt_2 in [0, np.pi]):
        spin_1x = 0
        spin_1y = 0
        spin_1z = a_1 * np.cos(tilt_1)
        spin_2x = 0
        spin_2y = 0
        spin_2z = a_2 * np.cos(tilt_2)
        iota = theta_jn
    else:
        iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = \
            transform_precessing_spins(
                theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1,
                mass_2, reference_frequency, phase)
    return iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z

def lal_binary_black_hole(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs):
    """ A Binary Black Hole waveform model using lalsimulation

    Parameters
    ==========
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the two component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total binary angular momentum and the
        orbital angular momentum
    theta_jn: float
        Angle between the total binary angular momentum and the line of sight
    phase: float
        The phase at coalescence
    kwargs: dict
        Optional keyword arguments
        Supported arguments:

        - waveform_approximant
        - reference_frequency
        - minimum_frequency
        - maximum_frequency
        - catch_waveform_errors
        - pn_spin_order
        - pn_tidal_order
        - pn_phase_order
        - pn_amplitude_order
        - mode_array:
          Activate a specific mode array and evaluate the model using those
          modes only.  e.g. waveform_arguments =
          dict(waveform_approximant='IMRPhenomHM', mode_array=[[2,2],[2,-2])
          returns the 22 and 2-2 modes only of IMRPhenomHM.  You can only
          specify modes that are included in that particular model.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[2,-2],[5,5],[5,-5]]) is not allowed because the
          55 modes are not included in this model.  Be aware that some models
          only take positive modes and return the positive and the negative
          mode together, while others need to call both.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[4,-4]]) returns the 22 and 2-2 of IMRPhenomHM.
          However, waveform_arguments =
          dict(waveform_approximant='IMRPhenomXHM', mode_array=[[2,2],[4,-4]])
          returns the 22 and 4-4 of IMRPhenomXHM.
        - lal_waveform_dictionary:
          A dictionary (lal.Dict) of arguments passed to the lalsimulation
          waveform generator. The arguments are specific to the waveform used.

    Returns
    =======
    dict: A dictionary with the plus and cross polarisation strain modes
    """
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    return _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, **waveform_kwargs)


def lal_binary_neutron_star(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, lambda_1, lambda_2,
        **kwargs):
    """ A Binary Neutron Star waveform model using lalsimulation

    Parameters
    ==========
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the two component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total binary angular momentum and the
        orbital angular momentum
    theta_jn: float
        Orbital inclination
    phase: float
        The phase at coalescence
    lambda_1: float
        Dimensionless tidal deformability of mass_1
    lambda_2: float
        Dimensionless tidal deformability of mass_2
    kwargs: dict
        Optional keyword arguments
        Supported arguments:

        - waveform_approximant
        - reference_frequency
        - minimum_frequency
        - maximum_frequency
        - catch_waveform_errors
        - pn_spin_order
        - pn_tidal_order
        - pn_phase_order
        - pn_amplitude_order
        - mode_array:
          Activate a specific mode array and evaluate the model using those
          modes only.  e.g. waveform_arguments =
          dict(waveform_approximant='IMRPhenomHM', mode_array=[[2,2],[2,-2])
          returns the 22 and 2-2 modes only of IMRPhenomHM.  You can only
          specify modes that are included in that particular model.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[2,-2],[5,5],[5,-5]]) is not allowed because the
          55 modes are not included in this model.  Be aware that some models
          only take positive modes and return the positive and the negative
          mode together, while others need to call both.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[4,-4]]) returns the 22 and 2-2 of IMRPhenomHM.
          However, waveform_arguments =
          dict(waveform_approximant='IMRPhenomXHM', mode_array=[[2,2],[4,-4]])
          returns the 22 and 4-4 of IMRPhenomXHM.

    Returns
    =======
    dict: A dictionary with the plus and cross polarisation strain modes
    """
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2_NRTidal', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    return _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, lambda_1=lambda_1, lambda_2=lambda_2, **waveform_kwargs)

def _base_lal_cbc_fd_waveform(
        frequency_array, mass_1, mass_2, luminosity_distance, theta_jn, phase,
        a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0,
        lambda_1=0.0, lambda_2=0.0, eccentricity=0.0, **waveform_kwargs):
    """ Generate a cbc waveform model using lalsimulation

    Parameters
    ==========
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total and orbital angular momenta
    theta_jn: float
        Orbital inclination
    phase: float
        The phase at coalescence
    eccentricity: float
        Binary eccentricity
    lambda_1: float
        Tidal deformability of the more massive object
    lambda_2: float
        Tidal deformability of the less massive object
    kwargs: dict
        Optional keyword arguments

    Returns
    =======
    dict: A dictionary with the plus and cross polarisation strain modes
    """
    import lal
    import lalsimulation as lalsim

    waveform_approximant = waveform_kwargs['waveform_approximant']
    reference_frequency = waveform_kwargs['reference_frequency']
    minimum_frequency = waveform_kwargs['minimum_frequency']
    maximum_frequency = waveform_kwargs['maximum_frequency']
    catch_waveform_errors = waveform_kwargs['catch_waveform_errors']
    pn_spin_order = waveform_kwargs['pn_spin_order']
    pn_tidal_order = waveform_kwargs['pn_tidal_order']
    pn_phase_order = waveform_kwargs['pn_phase_order']
    pn_amplitude_order = waveform_kwargs['pn_amplitude_order']
    waveform_dictionary = waveform_kwargs.get(
        'lal_waveform_dictionary', lal.CreateDict()
    )

    approximant = lalsim_GetApproximantFromString(waveform_approximant)

    if pn_amplitude_order != 0:
        start_frequency = lalsim.SimInspiralfLow2fStart(
            minimum_frequency, int(pn_amplitude_order), approximant)
    else:
        start_frequency = minimum_frequency

    delta_frequency = frequency_array[1] - frequency_array[0]

    frequency_bounds = ((frequency_array >= minimum_frequency) *
                        (frequency_array <= maximum_frequency))

    luminosity_distance = luminosity_distance * cst.Mpc
    mass_1 = mass_1 * cst.Msol
    mass_2 = mass_2 * cst.Msol

    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby_to_lalsimulation_spins(
        theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=mass_1, mass_2=mass_2,
        reference_frequency=reference_frequency, phase=phase)

    longitude_ascending_nodes = 0.0
    mean_per_ano = 0.0

    lalsim.SimInspiralWaveformParamsInsertPNSpinOrder(
        waveform_dictionary, int(pn_spin_order))
    lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(
        waveform_dictionary, int(pn_tidal_order))
    lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(
        waveform_dictionary, int(pn_phase_order))
    lalsim.SimInspiralWaveformParamsInsertPNAmplitudeOrder(
        waveform_dictionary, int(pn_amplitude_order))
    lalsim_SimInspiralWaveformParamsInsertTidalLambda1(
        waveform_dictionary, lambda_1)
    lalsim_SimInspiralWaveformParamsInsertTidalLambda2(
        waveform_dictionary, lambda_2)

    for key, value in waveform_kwargs.items():
        func = getattr(lalsim, "SimInspiralWaveformParamsInsert" + key, None)
        if func is not None:
            func(waveform_dictionary, value)

    if waveform_kwargs.get('numerical_relativity_file', None) is not None:
        lalsim.SimInspiralWaveformParamsInsertNumRelData(
            waveform_dictionary, waveform_kwargs['numerical_relativity_file'])

    if ('mode_array' in waveform_kwargs) and waveform_kwargs['mode_array'] is not None:
        mode_array = waveform_kwargs['mode_array']
        mode_array_lal = lalsim.SimInspiralCreateModeArray()
        for mode in mode_array:
            lalsim.SimInspiralModeArrayActivateMode(mode_array_lal, mode[0], mode[1])
        lalsim.SimInspiralWaveformParamsInsertModeArray(waveform_dictionary, mode_array_lal)

    if lalsim.SimInspiralImplementedFDApproximants(approximant):
        wf_func = lalsim_SimInspiralChooseFDWaveform
    else:
        wf_func = lalsim_SimInspiralFD
    try:
        hplus, hcross = wf_func(
            mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
            spin_2z, luminosity_distance, iota, phase,
            longitude_ascending_nodes, eccentricity, mean_per_ano, delta_frequency,
            start_frequency, maximum_frequency, reference_frequency,
            waveform_dictionary, approximant)
    except Exception as e:
        if not catch_waveform_errors:
            raise
        else:
            EDOM = (e.args[0] == 'Internal function call failed: Input domain error')
            if EDOM:
                failed_parameters = dict(mass_1=mass_1, mass_2=mass_2,
                                         spin_1=(spin_1x, spin_2y, spin_1z),
                                         spin_2=(spin_2x, spin_2y, spin_2z),
                                         luminosity_distance=luminosity_distance,
                                         iota=iota, phase=phase,
                                         eccentricity=eccentricity,
                                         start_frequency=start_frequency)
                print("Evaluating the waveform failed with error: {}\n".format(e) +
                               "The parameters were {}\n".format(failed_parameters) +
                               "Likelihood will be set to -inf.")
                return None
            else:
                raise

    h_plus = np.zeros_like(frequency_array, dtype=complex)
    h_cross = np.zeros_like(frequency_array, dtype=complex)

    if len(hplus.data.data) > len(frequency_array):
        print("LALsim waveform longer than bilby's `frequency_array`" +
                     "({} vs {}), ".format(len(hplus.data.data), len(frequency_array)) +
                     "probably because padded with zeros up to the next power of two length." +
                     " Truncating lalsim array.")
        h_plus = hplus.data.data[:len(h_plus)]
        h_cross = hcross.data.data[:len(h_cross)]
    else:
        h_plus[:len(hplus.data.data)] = hplus.data.data
        h_cross[:len(hcross.data.data)] = hcross.data.data

    h_plus *= frequency_bounds
    h_cross *= frequency_bounds

    if wf_func == lalsim_SimInspiralFD:
        dt = 1 / hplus.deltaF + (hplus.epoch.gpsSeconds + hplus.epoch.gpsNanoSeconds * 1e-9)
        time_shift = np.exp(-1j * 2 * np.pi * dt * frequency_array[frequency_bounds])
        h_plus[frequency_bounds] *= time_shift
        h_cross[frequency_bounds] *= time_shift

    return dict(plus=h_plus, cross=h_cross)
