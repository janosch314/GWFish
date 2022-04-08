#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


from scipy.interpolate import interp1d
from numpy.random import default_rng
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=69.6, Om0=0.286)

rng = default_rng()

def samples(filename, ns, nx=1000, plot=False):
    pofx = pd.read_csv(filename, names=['x', 'p'], delimiter='\t')
    pofx_interp = interp1d(x=pofx['x'], y=pofx['p'], bounds_error=False, fill_value=0)

    x = np.linspace(min(pofx['x'].to_numpy()), max(pofx['x'].to_numpy()), nx)
    pofx_dense = pofx_interp(x)

    CDF = np.cumsum(pofx_dense)
    CDF = CDF / CDF[-1]

    a = np.random.uniform(0, 1, size=(ns,))
    samples = np.zeros_like(a)
    for k in range(ns):
        samples[k] = x[np.argmax(CDF >= a[k]) - 1]

    if plot:
        plt.plot(pofx['x'], pofx['p'], label='p(x) from file')
        plt.plot(x, pofx_dense, label='p(x) interpolated')
        hist, xx = np.histogram(samples, x)
        plt.bar(0.5 * (xx[0:-1] + xx[1:]), hist / ns, align='center', width=0.9 * (xx[1] - xx[0]),
                label='histogram of samples')
        plt.tight_layout()
        plt.legend()
        plt.savefig(plot, dpi=300)
        plt.close()

    return samples

def redshiftPBH(ns=30000, plot=False):
    z = np.linspace(0, 100, 1000)
    dz = z[1] - z[0]
    t = cosmo.age(z).value
    t0 = cosmo.age(0).value
    R0 = 1.86e-9  # Mpc^-3 yr^-1
    R = R0 * (t / t0) ** (-34. / 37.)  # merger rate density of PBH binaries
    Rz = R / (1 + z) * 4 * np.pi * cosmo.differential_comoving_volume(z).value  # merger rate of PBH binaries
    N = int(np.round(np.sum(Rz * dz)))  # how many PBH mergers
    if ns > N:
        ns = N

    CDF = np.cumsum(Rz)
    CDF = CDF / CDF[-1]

    a = np.random.uniform(0, 1, size=(ns,))
    z_samples = np.zeros_like(a)
    for k in range(ns):
        z_samples[k] = z[np.argmax(CDF >= a[k]) - 1]

    if plot:
        plt.figure(figsize=(9, 6))
        zz = np.linspace(0, 100, 20)
        hist, zz = np.histogram(z_samples, zz)
        plt.bar(0.5 * (zz[0:-1] + zz[1:]), np.cumsum(hist), align='center', width=0.9 * (zz[1] - zz[0]),
                label='histogram of redshifts', color='orange')
        plt.semilogx(z, np.cumsum(Rz * dz), label='Model rate', linewidth=3)
        plt.grid(True)
        plt.xlabel('Redshift', fontsize=20)
        plt.ylabel(r"Rate up to redshift z [yr$^{-1}$]", fontsize=20)
        plt.tick_params(labelsize=20)
        plt.tight_layout()
        plt.savefig('PBH_Rate_Riotti.png', dpi=300)
        plt.show()
        plt.close()

    return z_samples

def main():
    # example to run with command-line argument:
    # python CBC_Population.py --ns=10000

    folder = './injections/'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pop_file_out', type=str, default=['CBC_pop.hdf5'], nargs=1,
        help='Name of output file containing the population to be simulated.'
             'Uses CBC_pop.hdf5 if no argument given.')
    parser.add_argument(
        '--population', type=str, default=['BBH'], nargs=1,
        help='Population to run the analysis on. Options: '
             'BBH, BNS, GWTC-3, PBH, or IMBH. Runs on BBH '
             'if no argument given.')
    parser.add_argument(
        '--z_max', default=['0.1'], type=float, nargs=1,
        help='Maximum simulated redshift.'
             'Uses z_max=10 if no argument given.')
    parser.add_argument(
        '--ns', default=['1'], nargs=1,
        help='Number of signals to simulate (at most, since it can be constrained by redshift selection).'
             'Uses 10000 as default if no argument given.')
    args = parser.parse_args()

    ns = int(args.ns[0])
    z_max = float(args.z_max[0])
    population = args.population[0]
    pop_file_out = args.pop_file_out[0]

    if population == 'BBH':
        parameters = pd.read_hdf(folder+'BBH_1e5.hdf5')
        #print(parameters['redshift'])

        #plt.hist(parameters['redshift'][1000:2000])
        #plt.show()

        ii = np.where(parameters['redshift'] < z_max)[0]

        print('There are ' + str(len(ii)) + ' BBH mergers up to z=' + str(z_max) + '.')
        ii = np.array(rng.choice(ii, size=(ns,), replace=False))
        parameters = parameters.iloc[ii, :]

        if ns < len(parameters):
            parameters = parameters.iloc[0:ns]

        ns = len(parameters)
        
        print("%d events included in this simulation"%ns)

        parameters['theta_jn'] = np.arccos(np.random.uniform(-1., 1., size=(ns,)))
        parameters['dec'] = np.arccos(np.random.uniform(-1., 1., size=(ns,))) - np.pi / 2.
        parameters['ra'] = np.random.uniform(0, 2. * np.pi, size=(ns,))
        parameters['psi'] = np.random.uniform(0, 2. * np.pi, size=(ns,))
        parameters['phase'] = np.random.uniform(0, 2. * np.pi, size=(ns,))
        parameters['geocent_time'] = np.random.uniform(1104105616, 1135641616, size=(ns,))  # full year 2015

        #parameters['mass_1'] = 10. * np.ones((ns,))
        #parameters['mass_2'] = 5. * np.ones((ns,))
        #z = np.random.uniform(0., 30., size=(ns,))
        #parameters['redshift'] = 0.1* np.ones((ns,))

        z = parameters['redshift'].to_numpy()
        parameters['luminosity_distance'] = cosmo.luminosity_distance(z).value

        #parameters = pd.DataFrame([{'redshift': 0.098, 'luminosity_distance': 453, 'mass_1': 10., 'mass_2': 5., 'theta_jn': 1.645,
        #              'dec': -0.363, 'ra': 0.375, 'psi': 1.738, 'phase': 3.472, 'geocent_time': 1120381489.604}])  # signal 1 from paper

    if population == 'IMBH':
        z = np.random.uniform(0., 10., size=(ns,))  # not reasonable of course
        parameters = pd.DataFrame.from_dict({'redshift': z, 'luminosity_distance': cosmo.luminosity_distance(z).value,
                                    'mass_1': 150. * np.ones_like(z), 'mass_2': 150. * np.ones_like(z),
                                    'theta_jn': np.arccos(np.random.uniform(-1., 1., size=(ns,))),
                                    'dec': np.arccos(np.random.uniform(-1., 1., size=(ns,))) - np.pi / 2.,
                                    'ra': np.random.uniform(0, 2. * np.pi, size=(ns,)),
                                    'psi': np.random.uniform(0, 2. * np.pi, size=(ns,)),
                                    'phase': np.random.uniform(0, 2. * np.pi, size=(ns,)),
                                    'geocent_time': np.random.uniform(1104105616, 1135641616, size=(ns,))}) # full year 2015

    if population == 'PBH':
        # primordial BBH according to https://arxiv.org/pdf/2102.03809.pdf
        # parameters['redshift'] = np.random.uniform(10., 30., size=(ns,))
        M1 = samples(folder+'PBH_p_M1.txt', ns)
        q = samples(folder+'PBH_p_q.txt', ns)
        M2 = M1 * q
        z = redshiftPBH(ns=ns)

        parameters = pd.DataFrame.from_dict({'mass_1': M1, 'mass_2': M2, 'redshift': z,
                                        'luminosity_distance': cosmo.luminosity_distance(z).value,
                                        'theta_jn': np.arccos(np.random.uniform(-1., 1., size=(ns,))),
                                        'dec': np.arccos(np.random.uniform(-1., 1., size=(ns,))) - np.pi / 2.,
                                        'ra': np.random.uniform(0, 2. * np.pi, size=(ns,)),
                                        'psi': np.random.uniform(0, 2. * np.pi, size=(ns,)),
                                        'phase': np.random.uniform(0, 2. * np.pi, size=(ns,)),
                                        'geocent_time': np.random.uniform(1104105616, 1135641616, size=(ns,))})  # full year 2015

    if population == 'BNS':
        #parameters = pd.read_csv('/Users/anacarolinaoliveira/Documents/GSSI/Code/injections/BBH_1e5.txt',
        #                             names=['mass_1', 'mass_2', 'redshift', 'luminosity_distance'],
        #                             delimiter=' ')
        parameters = pd.read_csv(folder+'BNS_8e5.txt',
                                     names=['mass_1', 'mass_2', 'redshift', 'luminosity_distance'],
                                     delimiter=' ')

        ii = np.where(parameters['redshift'] < z_max)[0]
        print('There are ' + str(len(ii)) + ' BNS mergers up to z=' + str(z_max) + '.')
        ii = np.array(rng.choice(ii, size=(ns,), replace=False))

        parameters = parameters.iloc[ii, :]
        if ns < len(parameters):
            parameters = parameters.iloc[0:ns]

        ns = len(parameters)

        parameters['theta_jn'] = np.arccos(np.random.uniform(-1., 1., size=(ns,)))
        parameters['dec'] = np.arccos(np.random.uniform(-1., 1., size=(ns,))) - np.pi / 2.
        parameters['ra'] = np.random.uniform(0, 2. * np.pi, size=(ns,))
        parameters['psi'] = np.random.uniform(0, 2. * np.pi, size=(ns,))
        parameters['phase'] = np.random.uniform(0, 2. * np.pi, size=(ns,))
        parameters['geocent_time'] = np.random.uniform(1104105616, 1135641616, size=(ns,))  # full year 2015

        # parameters['mass_1'] = np.random.uniform(1., 2.5, size=(ns,))
        # parameters['mass_2'] = np.random.uniform(1., 2.5, size=(ns,))
        parameters['mass_1'] = 1.4
        parameters['mass_2'] = 1.4

    if population == 'GWTC-3':
        parameters = pd.read_csv('./injections/GWTC-3.txt',
                                names=['id', 'geocent_time', 'mass_1', 'mass_2', 'luminosity_distance', 'chi_eff', 'redshift'],
                                delimiter=',')
        ns = len(parameters)

        parameters['theta_jn'] = np.arccos(np.random.uniform(-1., 1., size=(ns,)))
        parameters['dec'] = np.arccos(np.random.uniform(-1., 1., size=(ns,))) - np.pi / 2.
        parameters['ra'] = np.random.uniform(0, 2. * np.pi, size=(ns,))
        parameters['psi'] = np.random.uniform(0, 2. * np.pi, size=(ns,))
        parameters['phase'] = np.random.uniform(0, 2. * np.pi, size=(ns,))

    parameters.to_hdf(folder+pop_file_out, mode='w', key='root')

if __name__ == '__main__':
    main()
