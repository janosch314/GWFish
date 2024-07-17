from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path
import copy
import GWFish.modules.constants as cst
import GWFish.modules.ephemeris as ephem
from astropy.coordinates import EarthLocation
import warnings
from GWFish.modules.waveforms import t_of_f_PN
from astropy.utils.exceptions import AstropyWarning

DEFAULT_CONFIG = Path(__file__).parent.parent / 'detectors.yaml'
PSD_PATH = Path(__file__).parent.parent / 'detector_psd'

# used when redefining the time and frequency vectors
N_FREQUENCY_POINTS = 1000

class DetectorComponent:

    def __init__(self, name, component, detector_def):
        self.id = component
        self.name = name
        self.detector_def = detector_def
        self.setProperties()

    def setProperties(self):
        detector_def = self.detector_def

        self.duty_factor = eval(str(detector_def['duty_factor']))
        if 'psd_path' in detector_def:
            self.psd_path = eval(detector_def['psd_path'])
        else:
            self.psd_path = PSD_PATH

        if (detector_def['detector_class'] == 'earthDelta') or (detector_def['detector_class'] == 'earthL'):

            self.lat = eval(str(detector_def['lat']))
            self.lon = eval(str(detector_def['lon']))
            
            self.ephem = ephem.EarthLocationGCRSEphemeris(
                EarthLocation.from_geodetic(
                    np.rad2deg(self.lon), 
                    np.rad2deg(self.lat)
            ))
            
            self.arm_azimuth = eval(str(detector_def['azimuth']))

            self.opening_angle = eval(str(detector_def['opening_angle']))

            if (detector_def['detector_class'] == 'earthDelta'):
                self.arm_azimuth += 2.*self.id*np.pi/3.

            self.e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            self.e_lat = np.array(
                [-np.sin(self.lat) * np.cos(self.lon), -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])
            self.position = np.array(
                [np.cos(self.lat) * np.cos(self.lon), np.cos(self.lat) * np.sin(self.lon), np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * self.e_long + np.sin(self.arm_azimuth) * self.e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * self.e_long + np.sin(
                self.arm_azimuth + self.opening_angle) * self.e_lat

            self.psd_data = np.loadtxt(self.psd_path / detector_def['psd_data'])
            
            

        elif detector_def['detector_class'] == 'lunararray':

            self.lat = eval(str(detector_def['lat']))
            self.lon = eval(str(detector_def['lon']))
            self.ephem = ephem.MoonEphemeris()
            
            self.azimuth = eval(str(detector_def['azimuth']))

            self.e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            self.e_lat = np.array(
                [-np.sin(self.lat) * np.cos(self.lon), -np.sin(self.lat) * np.sin(self.lon),
                 np.cos(self.lat)])
            self.e1 = np.array([np.cos(self.lat) * np.cos(self.lon), np.cos(self.lat) * np.sin(self.lon), np.sin(self.lat)])
            self.e2 = np.cos(self.azimuth) * self.e_long + np.sin(self.azimuth) * self.e_lat

            self.psd_data = np.loadtxt(self.psd_path / detector_def['psd_data'])
            self.psd_data[:, 1] = self.psd_data[:, 1]/eval(str(detector_def['number_stations']))
        elif detector_def['detector_class'] == 'satellitesolarorbit':
            self.L = eval(str(detector_def['arm_length']))
            self.eps = self.L / cst.AU / (2 * np.sqrt(3))

            
            # psd_data contains proof-mass (PM) and optical-metrology-subsystem (OMS) noise as Doppler noise (y)
            raw_data = np.loadtxt(PSD_PATH / detector_def['psd_data'])
            ff = raw_data[:,0]
            self.psd_data = np.zeros((len(ff), 2))
            S_pm = raw_data[:,1]
            S_oms = raw_data[:,2]

            self.psd_data[:, 0] = ff

            if self.id < 2:
                # instrument noise of A,E channels
                self.psd_data[:, 1] = 16 * np.sin(np.pi * ff * self.L / cst.c) ** 2 * (
                        3 + 2 * np.cos(2 * np.pi * ff * self.L / cst.c) + np.cos(
                    4 * np.pi * ff * self.L / cst.c)) * S_pm \
                                      + 8 * np.sin(np.pi * ff * self.L / cst.c) ** 2 * (
                                              2 + np.cos(2 * np.pi * ff * self.L / cst.c)) * S_oms
            else:
                # instrument noise of T channel
                self.psd_data[:, 1] = (2 + 4 * np.cos(2 * np.pi * ff * self.L / cst.c)**2) * (
                        4 * np.sin(np.pi * ff * self.L / cst.c) ** 2 * S_pm + S_oms)

        self.Sn = interp1d(self.psd_data[:, 0], self.psd_data[:, 1], bounds_error=False, fill_value=1.)

    def plot_psd(self):
        plt.loglog(self.psd_data[:, 0], np.sqrt(self.psd_data[:, 1]), label=f'Component {self.id}')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Strain noise [Hz$^{1/2}$]')
        plt.grid(True)
        plt.tight_layout()

class Detector:
    
    def __init__(self, name: str, config=DEFAULT_CONFIG):
        """"""
        self.components = []

        self.name = name
        self.config = config

        with open(config) as f:
            doc = yaml.load(f, Loader=yaml.FullLoader)

        detectors = []
        for key in doc.keys():
            detectors.append(key)
        if self.name not in detectors:
            raise ValueError('Detector ' + self.name + ' invalid!')

        detector_def = doc[self.name]

        self.plotrange = np.fromstring(detector_def['plotrange'], dtype=float, sep=',')

        fmin = eval(str(detector_def['fmin']))
        fmax = eval(str(detector_def['fmax']))
        spacing = str(detector_def['spacing'])

        self.L = eval(str(detector_def['arm_length']))
        

        if spacing == 'linear':
            df = eval(str(detector_def['df']))
            self.frequencyvector = np.linspace(fmin, fmax, int((fmax - fmin) / df) + 1)
        elif spacing == 'geometric':
            npoints = eval(str(detector_def['npoints']))
            self.frequencyvector = np.geomspace(fmin, fmax, num=int(npoints))

        self.frequencyvector = self.frequencyvector[:, np.newaxis]

        if detector_def['detector_class'] == 'lunararray':
            self.location = 'moon'
            self.mission_lifetime = eval(str(detector_def['mission_lifetime']))
        elif (detector_def['detector_class'] == 'earthDelta') or (detector_def['detector_class'] == 'earthL'):
            self.location = 'earth'
        elif detector_def['detector_class'] == 'satellitesolarorbit':
            self.location = 'solarorbit'
            self.mission_lifetime = eval(str(detector_def['mission_lifetime']))

        if (detector_def['detector_class'] == 'earthDelta') or (detector_def['detector_class'] == 'satellitesolarorbit'):
            for k in np.arange(3):
                self.components.append(DetectorComponent(name=name, component=k, detector_def=detector_def))
        elif detector_def['detector_class'] == 'lunararray':
            if detector_def['azimuth']==None:
                detector_def['azimuth'] = '0'
                self.components.append(DetectorComponent(name=name, component=0, detector_def=detector_def))
                detector_def['azimuth'] = 'np.pi/2.'
                self.components.append(DetectorComponent(name=name, component=1, detector_def=detector_def))
            else:
                self.components.append(DetectorComponent(name=name, component=0, detector_def=detector_def))
        else:
            self.components.append(DetectorComponent(name=name, component=0, detector_def=detector_def))


class Network:
    """Class for a network of detectors.
    
    Example initialization:
    
    ```
    >>> network = Network(['ET', 'CE1'])
    >>> print(network.name)
    ET_CE1
    
    ```
    
    
    :attr detectors: list of `Detector` objects 
    """

    def __init__(self, detector_ids: list[str], detection_SNR: tuple[float, float]=(0., 10.), config: Path=DEFAULT_CONFIG):
        """
        :param detector_ids: list of detector names
        :param detection_SNR: tuple of single-detector and network detection threshold SNR
            - the first value is single-detector SNR threshold for that detector
            to be included in the Fisher matrix analysis
            - the second value is the network SNR threshold for the signal to be 
            processed at all
        :param config: configuration yaml file, defaults to the one described in the [included detectors](#included-detectors) section

        """
        self.name = detector_ids[0]
        for id in detector_ids[1:]:
            self.name += '_' + id

        self.detection_SNR = detection_SNR
        self.config = config

        self.detectors = [
            Detector(name=identifier, config=config)
            for identifier in detector_ids
        ]

    def partial(self, sub_network_ids: list[int]):
        
        new_network = copy.deepcopy(self)
        
        new_network.detectors = [
            self.detectors[i] for i in sub_network_ids
        ]
        
        return new_network

def GreenwichMeanSiderealTime(gps):
    # calculate the Greenwhich mean sidereal time
    return np.mod(9.533088395981618 + (gps - 1126260000.) / 3600. * 24. / cst.sidereal_day, 24) * np.pi / 12.


def LunarMeanSiderealTime(gps):
    # calculate the Lunar mean sidereal time
    return np.mod((gps - 1126260000.) / (3600. * cst.lunar_sidereal_period), 1) * 2. * np.pi


def solarorbit(tt, R, eps, a0, b0):
    w0 = np.sqrt(cst.G * cst.Msol / R ** 3)  # w0 has a 1% error when using this equation for Earth orbit

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
            np.exp(2j * np.pi * ff * L / cst.c * (muk / 3. + 1.)) - np.exp(2j * np.pi * ff * L / cst.c * muj / 3.)) * h_ifo[:,np.newaxis]

    return y


def alpha(i, yGWij, L, ff):
    dL = np.exp(2j * np.pi * ff[:, 0] * L / cst.c)
    dL2 = np.exp(4j * np.pi * ff[:, 0] * L / cst.c)

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


def projection(parameters, detector, polarizations, timevector, redefine_tf_vectors=False, long_wavelength_approx = True):

    f_max = parameters.get('max_frequency_cutoff', None)
    detector_lifetime = getattr(detector, 'mission_lifetime', None)

    in_band_slice, new_timevector = in_band_window(
        np.squeeze(timevector), 
        np.squeeze(detector.frequencyvector), 
        detector_lifetime, 
        f_max, 
        redefine_timevector=redefine_tf_vectors
    )
    
    if redefine_tf_vectors:
        new_fmin = detector.frequencyvector[in_band_slice.start - 1, 0]
        new_fmax = detector.frequencyvector[in_band_slice.stop + 1, 0]
        new_frequencyvector = np.geomspace(new_fmin, new_fmax, num=N_FREQUENCY_POINTS)[:, None]
        temp_timevector = t_of_f_PN(parameters, new_frequencyvector)
        in_band_slice, new_timevector = in_band_window(
            np.squeeze(temp_timevector), 
            np.squeeze(new_frequencyvector), 
            detector_lifetime, 
            f_max,
            redefine_timevector=True,
            final_time=parameters['geocent_time']
        )
    
    proj = np.zeros_like(new_timevector)
    if is_null_slice(in_band_slice):
        return proj
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyWarning)
        if detector.location == 'earth':
            proj = projection_earth(parameters, detector, polarizations, new_timevector, in_band_slice, long_wavelength_approx = long_wavelength_approx)
        elif detector.location == 'moon':
            proj = projection_moon(parameters, detector, polarizations, new_timevector, in_band_slice)
        elif detector.location == 'solarorbit':
            proj = projection_solarorbit(parameters, detector, polarizations, new_timevector, in_band_slice)
        else:
            print('Unknown detector location')
            exit(0)

    if redefine_tf_vectors:
        return proj, new_timevector, new_frequencyvector
    return proj


def in_band_window(
    timevector, 
    frequencyvector, 
    detector_lifetime, 
    max_frequency_cutoff,
    redefine_timevector=False,
    final_time=None
    ):
    """Truncate the evolution of the source to the detector lifetime.
    
    If there is no maximum frequency cutoff, then 
    this just amounts to going back in time
    from the highest frequency available.
    
    If there is a maximum frequency cutoff, then 
    we should:
    - shift the timevector so that the cutoff is placed at the given gps time
    - truncate the projection so that the temporal duration of the nonzero section
        corresponds to the detector lifetime 
    """
    
    if final_time is None:
        final_time = timevector[-1]
    
    if max_frequency_cutoff is None:
        i_final = len(timevector)
        fmax_time = final_time
        new_timevector = timevector
    else:
        if max_frequency_cutoff <= frequencyvector[0]:
            warnings.warn("The max_frequency given is lower than the lowest frequency for the detector."
                          "Returning a zero projection.")
            return slice(0, 0), timevector
        elif max_frequency_cutoff >= frequencyvector[-1]:
            i_final = -1
        else:
            i_final = np.searchsorted(frequencyvector, max_frequency_cutoff)

        fmax_time = timevector[i_final]

    if redefine_timevector:
        # potentially dangerous loss of precision here...
        # shift the timevector so that the cutoff is placed at the given gps time
        new_timevector = timevector + final_time - fmax_time
    else:
        new_timevector = timevector

    if detector_lifetime is None:
        i_initial = 0
    elif final_time - detector_lifetime < new_timevector[0]:
        i_initial = 0
    else:
        if redefine_timevector:
            i_initial = np.searchsorted(new_timevector, final_time - detector_lifetime)
        else:
            i_initial = np.searchsorted(new_timevector, fmax_time - detector_lifetime)

    return slice(i_initial, i_final), new_timevector

def projection_solarorbit(parameters, detector, polarizations, timevector, in_band_slice=slice(None)):
    ff = detector.frequencyvector[in_band_slice]
    components = detector.components

    proj = np.zeros(
        shape=(len(timevector), len(components)),
        dtype=complex
    )
    
    if timevector.ndim == 1:
        timevector = timevector[:, np.newaxis]

    # note that RA/DEC are not translated into solar-centered coordinates! TO BE FIXED
    ra = parameters['ra']
    dec = parameters['dec']
    psi = parameters['psi']

    theta = np.pi / 2. - dec

    pp = solarorbit(timevector[in_band_slice], cst.AU, components[0].eps, 0., 0.)
    eij = (pp[:, [1, 2, 0], :] - pp[:, [2, 0, 1], :]) / components[0].L

    # start_time = time.time()
    proj[in_band_slice, :] = AET(polarizations[in_band_slice, :], eij, theta, ra, psi, components[0].L, ff)    # print("Calculation of projection: %s seconds" % (time.time() - start_time))

    return proj

def sinc(x):
    return np.sin(x)/x

def Michelson_transfer_function(x, x_c, proj_arm):

    norm_f = x/(2*x_c)

    term1 = sinc(norm_f*(1-proj_arm)) * np.exp(-1.j * norm_f *(3+proj_arm))
    term2 = sinc(norm_f*(1+proj_arm)) * np.exp(-1.j * norm_f *(1+proj_arm))

    return 0.5 * (term1 + term2)


def projection_earth(parameters, detector, polarizations, timevector, in_band_slice=slice(None), long_wavelength_approx = True):
    """
    See Nishizawa et al. (2009) arXiv:0903.0528 for definitions of the polarisation tensors.
    [u, v, w] represent the Earth-frame
    [m, n, omega] represent the wave-frame
    Note1: there is a typo in the definition of the wave-frame in Nishizawa et al.
    Note2: it is computationally more expensive to use numpy.einsum instead of working with several vector quantities
    """

    # timevector = parameters['geocent_time'] * np.ones_like(timevector)  # switch off Earth's rotation

    nf = len(polarizations[:, 0])
    ff = detector.frequencyvector[in_band_slice, :]

    components = detector.components
    proj = np.zeros((nf, len(components)), dtype=complex)

    if timevector.ndim == 1:
        timevector = timevector[:, np.newaxis]

    ra = parameters['ra']
    dec = parameters['dec']
    psi = parameters['psi']

    # generally the long wavelenght approximation is True
    # when long_wl it is set to 1 inside the parmaeters passed, you automatically set long_wavelenght_approx = False

    theta = np.pi / 2. - dec
    gmst = GreenwichMeanSiderealTime(timevector[in_band_slice])
    phi = ra - gmst

    # wave vector components
    kx = -np.sin(theta) * np.cos(phi)
    ky = -np.sin(theta) * np.sin(phi)
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
    hxx = polarizations[in_band_slice, 0] * (mx * mx - nx * nx) + polarizations[in_band_slice, 1] * (mx * nx + nx * mx)
    hxy = polarizations[in_band_slice, 0] * (mx * my - nx * ny) + polarizations[in_band_slice, 1] * (mx * ny + nx * my)
    hxz = polarizations[in_band_slice, 0] * (mx * mz - nx * nz) + polarizations[in_band_slice, 1] * (mx * nz + nx * mz)
    hyy = polarizations[in_band_slice, 0] * (my * my - ny * ny) + polarizations[in_band_slice, 1] * (my * ny + ny * my)
    hyz = polarizations[in_band_slice, 0] * (my * mz - ny * nz) + polarizations[in_band_slice, 1] * (my * nz + ny * mz)
    hzz = polarizations[in_band_slice, 0] * (mz * mz - nz * nz) + polarizations[in_band_slice, 1] * (mz * nz + nz * mz)
    # print("Calculation GW tensor: %s seconds" % (time.time() - start_time))

    # start_time = time.time()
    for k in np.arange(len(components)):
        e1 = components[k].e1
        e2 = components[k].e2

        # interferometer position
        # x_det = components[k].position[0] * cst.R_earth
        # y_det = components[k].position[1] * cst.R_earth
        # z_det = components[k].position[2] * cst.R_earth
        # phase_shift = np.squeeze(x_det * kx + y_det * ky + z_det * kz) * 2 * np.pi / cst.c * np.squeeze(ff)
        
        phase_shift = components[k].ephem.phase_term(ra, dec, np.squeeze(timevector)[in_band_slice], np.squeeze(ff))

        if long_wavelength_approx:
            
            proj[in_band_slice, k] = 0.5 * (e1[0] ** 2 - e2[0] ** 2) * hxx \
                        + 0.5 * (e1[1] ** 2 - e2[1] ** 2) * hyy \
                        + 0.5 * (e1[2] ** 2 - e2[2] ** 2) * hzz \
                        + (e1[0] * e1[1] - e2[0] * e2[1]) * hxy \
                        + (e1[0] * e1[2] - e2[0] * e2[2]) * hxz \
                        + (e1[1] * e1[2] - e2[1] * e2[2]) * hyz

            proj[in_band_slice, k] *= np.exp(-1.j * phase_shift)
        
        else:
            # the detailed calculation can be found at this link
            # https://thesis.unipd.it/handle/20.500.12608/1/browse?filter_type=authority&authority=ist48184&filter_value=ist48184&filter_value_display=Amalberti%2C+Loris&type=author&sort_by=ASC&order=&rpp=20
            # in section 2.2
            
            proj_arm1 = kx*e1[0] + ky*e1[1] + kz*e1[2]
            proj_arm2 = kx*e2[0] + ky*e2[1] + kz*e2[2]

            f_c = cst.c / (2*np.pi*detector.L)

            T1 = Michelson_transfer_function(np.squeeze(ff), f_c, proj_arm1[:,0])
            T2 = Michelson_transfer_function(np.squeeze(ff), f_c, proj_arm2[:,0])
                        
            proj[in_band_slice, k] = 0.5 * (T1 * e1[0] ** 2  - T2 * e2[0] ** 2) * hxx \
            + 0.5 * (T1 * e1[1] ** 2 - T2 * e2[1] ** 2) * hyy \
            + 0.5 * (T1 * e1[2] ** 2 - T2 * e2[2] ** 2) * hzz \
            + (T1 * e1[0] * e1[1] - T2 * e2[0] * e2[1]) * hxy \
            + (T1 * e1[0] * e1[2] - T2 * e2[0] * e2[2]) * hxz \
            + (T1 * e1[1] * e1[2] - T2 * e2[1] * e2[2]) * hyz

            proj[in_band_slice, k] *= np.exp(-1.j * phase_shift)
        
    #print("Calculation of projection: %s seconds" % (time.time() - start_time))

    return proj


def projection_moon(parameters, detector, polarizations, timevector, in_band_slice=slice(None)):
    """
    See Nishizawa et al. (2009) arXiv:0903.0528 for definitions of the polarisation tensors.
    [u, v, w] represent the Earth-frame
    [m, n, omega] represent the wave-frame
    Note1: there is a typo in the definition of the wave-frame in Nishizawa et al.
    Note2: it is computationally more expensive to use numpy.einsum instead of working with several vector quantities
    """

    # timevector = parameters['geocent_time'] * np.ones_like(timevector)  # switch off Earth's rotation

    nt = len(polarizations[:, 0])

    components = detector.components
    proj = np.zeros((nt, len(components)), dtype=complex)

    if timevector.ndim == 1:
        timevector = timevector[:, np.newaxis]

    # note that RA/DEC are not translated into lunar-centered coordinates! TO BE FIXED
    ra = parameters['ra']
    dec = parameters['dec']
    psi = parameters['psi']

    theta = np.pi / 2. - dec
    lmst = LunarMeanSiderealTime(timevector[in_band_slice])
    phi = ra - lmst

    # saving timevector and lmst for plotting
    # np.save('timevector.npy', timevector)
    # np.save('lmst.npy', lmst)

    #start_time = time.time()
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
    hxx = polarizations[in_band_slice, 0] * (mx * mx - nx * nx) + polarizations[in_band_slice, 1] * (mx * nx + nx * mx)
    hxy = polarizations[in_band_slice, 0] * (mx * my - nx * ny) + polarizations[in_band_slice, 1] * (mx * ny + nx * my)
    hxz = polarizations[in_band_slice, 0] * (mx * mz - nx * nz) + polarizations[in_band_slice, 1] * (mx * nz + nx * mz)
    hyy = polarizations[in_band_slice, 0] * (my * my - ny * ny) + polarizations[in_band_slice, 1] * (my * ny + ny * my)
    hyz = polarizations[in_band_slice, 0] * (my * mz - ny * nz) + polarizations[in_band_slice, 1] * (my * nz + ny * mz)
    hzz = polarizations[in_band_slice, 0] * (mz * mz - nz * nz) + polarizations[in_band_slice, 1] * (mz * nz + nz * mz)
    #print("Calculation GW tensor: %s seconds" % (time.time() - start_time))

    # start_time = time.time()
    for k in np.arange(len(components)):
        e1 = components[k].e1
        e2 = components[k].e2
        
        phase_shift = components[k].ephem.phase_term(ra, dec, np.squeeze(timevector)[in_band_slice], np.squeeze(detector.frequencyvector)[in_band_slice])

        # proj[:, k] = np.einsum('i,jik,k->j', e1, hij, e2)
        proj[in_band_slice, k] = e1[0] * e2[0] * hxx \
                     + e1[1] * e2[1] * hyy \
                     + e1[2] * e2[2] * hzz \
                     + (e1[0] * e2[1] + e2[0] * e1[1]) * hxy \
                     + (e1[0] * e2[2] + e2[0] * e1[2]) * hxz \
                     + (e1[1] * e2[2] + e2[1] * e1[2]) * hyz
                     
        proj[in_band_slice, k] *= np.exp(-1.j * phase_shift)

    #print("Calculation of projection: %s seconds" % (time.time() - start_time))

    return proj


def lisaGWresponse(detector):
    ff = detector.frequencyvector
    nf = len(ff)

    polarizations = np.ones((nf, 2))
    timevector = np.zeros((nf, 1))

    components = detector.components

    ra = 0
    dec = np.pi / 2.
    psi = 0

    theta = np.pi / 2. - dec

    pp = solarorbit(timevector, cst.AU, components[0].eps, 0., 0.)
    eij = (pp[:, [1, 2, 0], :] - pp[:, [2, 0, 1], :]) / components[0].L

    doppler_to_strain = cst.c / (components[0].L * 2 * np.pi * ff)
    proj = doppler_to_strain * AET(polarizations, eij, theta, ra, psi, components[0].L, ff)

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

        pp = solarorbit(timevector, cst.AU, components[0].eps, 0., 0.)
        eij = (pp[:, [1, 2, 0], :] - pp[:, [2, 0, 1], :]) / components[0].L

        doppler_to_strain = cst.c / (components[0].L * 2 * np.pi * ff)
        proj += (doppler_to_strain * AET(polarizations, eij, np.arccos(costheta), ra, psi, components[0].L,
                                         ff)) ** 2

    proj /= N

    psds = np.zeros((nf, 3))
    for k in range(3):
        psds[:, k] = components[k].Sn(ff[:, 0])

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


def SNR(detector, signals, use_duty_cycle: bool = False, frequencyvector = None):
    if signals.ndim == 1:
        signals = signals[:, np.newaxis]

    if frequencyvector is None:
        frequencyvector = detector.frequencyvector[:, 0]
    components = detector.components

    SNRs = np.zeros(len(components))
    for k, component in enumerate(components):
        integrand = np.abs(signals[:, k]) ** 2 / component.Sn(frequencyvector)
        SNRs[k] = np.sqrt(4 * np.trapz(integrand, frequencyvector, axis=0))

        # set SNRs to zero if interferometer is not operating (according to its duty factor [0,1])
        if use_duty_cycle:
            operating = np.random.rand()
            #print('operating = ',operating)
            if components[k].duty_factor < operating:
                SNRs[k] = 0.

    return SNRs


def analyzeDetections(network, parameters, population, networks_ids):

    detSNR = network.detection_SNR

    ns = len(network.SNR)
    N = len(networks_ids)

    save_data = parameters

    delim = " "
    header = delim.join(parameters.keys())

    for n in np.arange(N):
        maxz = 0

        network_ids = networks_ids[n]
        network_name = '_'.join([network.detectors[k].name for k in network_ids])

        print('Network: ' + network_name)

        SNR = 0
        for d in network_ids:
            SNR += network.detectors[d].SNR ** 2
        SNR = np.sqrt(SNR)

        save_data = np.c_[save_data, SNR]
        header += " " + network_name + "_SNR"

        threshold = SNR > detSNR[1]

        ndet = len(np.where(threshold)[0])

        if ndet > 0:
            maxz = np.max(parameters['redshift'].iloc[np.where(threshold)].to_numpy())
        print(
            'Detected signals with SNR>{:.3f}: {:.3f} ({:} out of {:}); z<{:.3f}'.format(detSNR[1], ndet / ns, ndet, ns,
                                                                                         maxz))

        print('SNR: {:.3f} (min) , {:.3f} (max) '.format(np.min(SNR), np.max(SNR)))

    if 'id' in parameters.columns:
        np.savetxt('Signals_' + population + '.txt', save_data, delimiter=' ', fmt='%s '+"%.3f "*(len(save_data[0,:])-1),
                   header=header, comments='')
    else:
        np.savetxt('Signals_' + population + '.txt', save_data, delimiter=' ', fmt='%.3f', header=header, comments='')

def is_null_slice(s):
    if s.stop is None:
        return False
    if s.start == s.stop == 0:
        return True
    return False
