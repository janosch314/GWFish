import numpy as np
from GWFish.modules.waveforms import TaylorF2, LALFD_Waveform
from GWFish.modules.detection import Detector, projection, create_moon_position_interp, get_moon_coordinates
from GWFish.modules.fishermatrix import FisherMatrix, invertSVD
import matplotlib.pyplot as plt

detector = Detector('LGWA', parameters = [None], fisher_parameters = [None])
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Serif"
})

BNS_PARAMS = {
    'mass_1': 1.4957673, 
    'mass_2': 1.24276395,
    # 'mass_1': 1., 
    # 'mass_2': 1., 
    'redshift': 0.00980,
    # 'luminosity_distance': 100.,
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
    'lambda_2':586.5487031450857
}

data_params = {
    'frequencyvector': detector.frequencyvector,
    'f_ref': 50.
}
waveform_obj = LALFD_Waveform('IMRPhenomD_NRTidalv2', BNS_PARAMS, data_params)
polarizations = waveform_obj()
timevector = waveform_obj.t_of_f

# signal = projection(
#     BNS_PARAMS,
#     detector,
#     polarizations,
#     timevector
# )


# in the scalar_product function, add the lines
        # np.save(f'integrand_{k}.npy', np.real(deriv1[:, k] * np.conjugate(deriv2[:, k])))
        # np.save(f'normalized_integrand_{k}.npy', np.real(deriv1[:, k] * np.conjugate(deriv2[:, k])/ components[k].Sn(ff[:, 0])))

fig, axs = plt.subplots(2, 1, sharex=True)
channels = {
    0: 'channel $x$',
    1: 'channel $y$',
}

fm = FisherMatrix('IMRPhenomD_NRTidalv2', BNS_PARAMS, [
    'dec', 
    'ra',
    'geocent_time',
    'phase',
    'psi',
    'mass_1',
    'mass_2',
    'luminosity_distance',
    'theta_jn',
    'a_1',
    'a_2',
    ], detector, waveform_class=LALFD_Waveform)
inverse, _ = invertSVD(fm.fm)

def sky_localization_area(
    network_fisher_inverse: np.ndarray,
    declination_angle: np.ndarray,
    right_ascension_index: int,
    declination_index: int,
) -> float:
    """
    Compute the 1-sigma sky localization ellipse area starting
    from the full network Fisher matrix inverse and the inclination.
    """
    return (
        np.pi
        * np.abs(np.cos(declination_angle))
        * np.sqrt(
            network_fisher_inverse[right_ascension_index, right_ascension_index]
            * network_fisher_inverse[declination_index, declination_index]
            - network_fisher_inverse[right_ascension_index, declination_index] ** 2
        )
    )

print(np.sqrt(inverse[0, 0]) * (180/np.pi) * 60)
print(np.sqrt(inverse[1, 1]) * (180/np.pi) * 60)
print(sky_localization_area(inverse, BNS_PARAMS['dec'], 1, 0) * 2 * (-np.log(0.1) ) * (180/np.pi)**2 * 3600)

cmap = plt.get_cmap('Paired')
colors = [cmap(index) for index in np.linspace(0, 1, num=8)]
for ax in axs:
    ax.set_prop_cycle(color=colors)
    
ra = BNS_PARAMS['ra']
dec = BNS_PARAMS['dec']
psi = BNS_PARAMS['psi']
theta = np.pi / 2. - dec

kx_icrs = -np.sin(theta) * np.cos(ra)
ky_icrs = -np.sin(theta) * np.sin(ra)
kz_icrs = -np.cos(theta)

f = np.squeeze(detector.frequencyvector)
x, y, z = get_moon_coordinates(np.squeeze(timevector))
phase = (x*kx_icrs + y*ky_icrs + z*kz_icrs) * 2 * np.pi / 299792458. * f

for k in (0, 1):
    # integrand_corr = np.load(f'integrand_corr_{k}.npy')
    integrand = np.load(f'integrand_{k}.npy')
    # normalized_integrand = np.load(f'normalized_integrand_{k}.npy')
    normalized_integrand = integrand / detector.components[k].Sn(f)
    axs[0].semilogx(f, 4*integrand*f, label=channels[k], lw=1.)
    # axs[1].semilogx(f, 4*integrand_corr*f, label=channels[k], lw=1.)
    axs[1].semilogx(f, 4*normalized_integrand*f, label=channels[k], lw=1.)
# axs[1].semilogx(f, phase - phase[-1])
axs[1].set_xlabel('Frequency [Hz]')
axs[0].set_ylabel(r'$4 f|\partial h / \partial \mathrm{dec}|^2$ [rad$^{-2}$ Hz$^{-1}$]') 
axs[1].set_ylabel(r'$4 f|\partial h / \partial \mathrm{dec}|^2 / S_n(f)$ [rad$^{-2}]$')
# axs[0].set_ylim(0, 5e-35)
# axs[1].set_ylim(0, 5e8)

axs[0].set_title('Unnormalized Fisher contribution')
# axs[0].set_title(r'$F_{\mathrm{dec, dec}}$')
# axs[1].set_title(r'$F_{\mathrm{dec, t}}$')
axs[1].set_title('Normalized to LGWA noise')

plt.xlim(8e-2,3)
axs[0].legend()
plt.savefig(f'fisher_integrand.pdf')
plt.show()
plt.close()