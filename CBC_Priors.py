#!/usr/bin/env python

import numpy as np
import pandas as pd
from tqdm import tqdm
import GWFish.modules as gw



def main():

	np.random.seed(42)

	##########################################
	lbs_means_file = ['network_SNR', 'mass_1', 'mass_2', 'luminosity_distance', 'dec', 'ra', 
	  'theta_jn', 'psi', 'geocent_time', 'phase', 'redshift', 'err_mass_1',
	  'err_mass_2', 'err_luminosity_distance', 'err_dec', 'err_ra', 'err_theta_jn',
	  'err_psi', 'err_geocent_time', 'err_phase', 'err_sky_location']

	### Load mean values###
	path_to_mean_values = 'Errors_ET_BBH_SNR8.0.txt'
	means_file = pd.read_csv(path_to_mean_values, names = lbs_means_file, delimiter = ' ',
							low_memory = False, skiprows = 1)
	##########################################

	##########################################
	# Pay attention to the ORDER! It's the same as the errors order in the Errors.txt file!
	var_in_cov_matrix = ['mass_1', 'mass_2', 'luminosity_distance', 'dec', 'ra', 
	  					 'theta_jn', 'psi', 'geocent_time', 'phase']
	# Load Covariance matrix
	path_to_cov_matrices = 'Inv_Fishers_ET_BBH_SNR8.0.npy'
	cov_data = np.load(path_to_cov_matrices)
	##########################################

	# Choose variables to process with priors
	mns = means_file[var_in_cov_matrix]

	ns = len(mns)
	npar = len(var_in_cov_matrix)
	new_cov = np.zeros((ns, npar, npar))
	new_parameter_errors = np.zeros((ns, npar))
	sky_localization = np.zeros((ns,))

	idx_ra = var_in_cov_matrix.index('ra')
	idx_dec = var_in_cov_matrix.index('dec')


	number_of_samples = 1e6

	for k in tqdm(np.arange(ns)):
		mns_ev = np.array(mns.iloc[k])
		cov_ev = cov_data[k, :, :]
		samples = np.random.multivariate_normal(mns_ev, np.squeeze(cov_ev), int(number_of_samples))

		data_samples = {'mass_1':samples[:, 0],
				'mass_2':samples[:, 1],
				'luminosity_distance':samples[:, 2],
				'dec':samples[:, 3],
				'ra':samples[:, 4],
				'theta_jn':samples[:, 5],
				'psi':samples[:, 6],
				'geocent_time':samples[:, 7],
				'phase':samples[:, 8]}
		data = pd.DataFrame(data = data_samples)
		mask = gw.priors.uniform_priors(data)

		# Filter data samples
		data_filtered = data.loc[mask]
		new_cov[k, :, :] = np.cov((data_filtered.to_numpy()).T)
		new_parameter_errors[k, :] = np.sqrt(np.diagonal(new_cov[k, :, :]))

		if ('ra' in var_in_cov_matrix) and ('dec' in var_in_cov_matrix):
			sky_localization[k] = np.pi * np.abs(np.cos(mns_ev[idx_dec])) *\
								np.sqrt(new_cov[k, idx_ra, idx_ra] * new_cov[k, idx_dec, idx_dec]\
								- new_cov[k, idx_ra, idx_dec]**2)

	np.save('New_Cov_Matrices.npy', new_cov)

	if ('ra' in var_in_cov_matrix) and ('dec' in var_in_cov_matrix):
		data_to_save = np.c_[mns, new_parameter_errors, sky_localization]
		header = 'mass_1, mass_2, luminosity_distance, dec, ra, theta_jn, psi, geocent_time, phase,err_mass_1, err_mass_2, err_luminosity_distance, err_dec, err_ra, err_theta_jn, err_psi, err_geocent_time, err_phase, err_sky_loc'
		np.savetxt('New_Errors.txt', data_to_save, delimiter=' ', fmt='%.8f', header=header, comments='')
	else:
		data_to_save = np.c_[mns, new_parameter_errors]
		header = 'mass_1, mass_2, luminosity_distance, dec, ra, theta_jn, psi, geocent_time, phase,err_mass_1, err_mass_2, err_luminosity_distance, err_dec, err_ra, err_theta_jn, err_psi, err_geocent_time, err_phase'
		np.savetxt('New_Errors.txt', data_to_save, delimiter=' ', fmt='%.8f', header=header, comments='')





if __name__ == '__main__':
    main()
