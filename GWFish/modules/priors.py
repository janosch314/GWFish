import numpy as np

def mass_prior(samples, min_value = 0.):
	mask_m1 = samples['mass_1'] > min_value
	mask_m2 = samples['mass_2'] > min_value
	return np.logical_and(mask_m1, mask_m2)

def distance_prior(samples, min_value = 0.):
	return samples['luminosity_distance'] > min_value

def ra_prior(samples, min_value = 0., max_value = 2 * np.pi):
	lower = samples['ra'] > min_value
	upper = samples['ra'] < max_value
	return np.logical_and(lower, upper)

def dec_prior_uniform(samples, min_value = -np.pi / 2., max_value = np.pi / 2.):
	lower = samples['dec'] > min_value
	upper = samples['dec'] < max_value
	return np.logical_and(lower, upper)

def iota_prior_uniform(samples, min_value = 0., max_value = np.pi):
	lower = samples['theta_jn'] > min_value
	upper = samples['theta_jn'] < max_value
	return np.logical_and(lower, upper)

def phase_prior(samples, min_value = 0., max_value = 2 * np.pi):
	lower = samples['phase'] > min_value
	upper = samples['phase'] < max_value
	return np.logical_and(lower, upper)

def psi_prior(samples, min_value = 0., max_value = 2 * np.pi):
	lower = samples['psi'] > min_value
	upper = samples['psi'] < max_value
	return np.logical_and(lower, upper)

def time_prior(samples, min_value = 0.):
	return samples['geocent_time'] > min_value

def spin_magnitude_prior(samples, min_value = -1., max_value = 1.):
	a1_min = samples['a_1'] > min_value
	a1_max = samples['a_1'] < max_value
	a2_min = samples['a_2'] > min_value
	a2_max = samples['a_2'] < max_value
	return np.logical_and(np.logical_and(a1_min, a1_max), np.logical_and(a2_min, a2_max))


def uniform_priors(samples):
	filter_mass = mass_prior(samples)
	filter_dist = distance_prior(samples)
	filter_ra = ra_prior(samples)
	filter_dec = dec_prior_uniform(samples)
	mask1 = np.logical_and(np.logical_and(filter_mass, filter_dist), np.logical_and(filter_ra, filter_dec))

	filter_iota = iota_prior_uniform(samples)
	filter_phase = phase_prior(samples)
	filter_psi = psi_prior(samples)
	filter_time = time_prior(samples)
	mask2 = np.logical_and(np.logical_and(filter_iota, filter_phase), np.logical_and(filter_psi, filter_time))

	return np.logical_and(mask1, mask2)




def uniform_priors_spins(samples):
	mask1 = uniform_priors(samples)
	mask2 = spin_magnitude_prior(samples)

	return np.logical_and(mask1, mask2)





