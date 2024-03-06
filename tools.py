import numpy as np
import pandas as pd
import h5py
import os
import pesummary
from pesummary.io import read
import pickle
import yaml
import GWFish.modules as gw
import json
from tqdm import tqdm
from minimax_tilting_sampler import *
import priors



def keys(f):
    return [key for key in f.keys()]


def create_injections_from_gwtc(PATH_TO_DATA, PATH_TO_RESULTS, waveform, params, estimator):
    """
    Load the GWTC catalogs and create a list of events for which the estimator is not None
    The list of events is saved in a txt file (as well as the discareded ones)
    The injections are saved in a hdf5 file
        --> Specify the PATH_TO_DATA and PATH_TO_RESULTS
        --> Specify the waveform, the parameters and the estimator
    The DATA are assumed to be in the LVK format as can be downloaded from Zenodo
    """

    event_list = []
    no_waveform_list = []
    discarded_events_list = []
    for file in os.listdir(PATH_TO_DATA):
        data_pesum = read(PATH_TO_DATA + file, package = 'core')
        if 'C01:' + waveform not in data_pesum.samples_dict.keys():
            no_waveform_list.append(file[:-3])
        else:
            if data_pesum.samples_dict['C01:' + waveform].key_data[params[0]][estimator] != None:
                event_list.append(file[:-3])
                # Create the injections
                estimator_dict = {}
                for param in params:
                    estimator_dict[param] = data_pesum.samples_dict['C01:' + waveform].key_data[param][estimator]
                PATH_TO_INJECTIONS = PATH_TO_RESULTS + 'injections/' + file[:-3]
                if not os.path.exists(PATH_TO_INJECTIONS):
                    os.makedirs(PATH_TO_INJECTIONS)
                estimator_df = pd.DataFrame([estimator_dict], columns = params)
                estimator_df.to_hdf(PATH_TO_INJECTIONS + '/%s_%s_%s.hdf5' %(file[:-3], waveform, estimator), key = 'data', mode = 'w')
            else:
                discarded_events_list.append(file[:-3])

    np.savetxt(PATH_TO_RESULTS + 'info/' + 'event_list_%s_%s.txt' %(waveform, estimator), event_list, fmt = '%s')
    np.savetxt(PATH_TO_RESULTS + 'info/' + '%s_not_in_list_for_%s.txt' %(estimator, waveform), discarded_events_list, fmt = '%s')
    np.savetxt(PATH_TO_RESULTS + 'info/' + 'waveform_not_in_list_%s.txt' %waveform, no_waveform_list, fmt = '%s')


def check_and_store_priors(PATH_TO_DATA, PATH_TO_RESULTS, events_list, waveform):
    events_with_priors = []
    events_with_no_priors = []
    chirp_mass_priors = {}
    for event in events_list:
        data = h5py.File(PATH_TO_DATA + event + '.h5', 'r') 
        if 'analytic' in data['C01:' + waveform]['priors'].keys():
            events_with_priors.append(event)
            string_ov = data['C01:' + waveform]['priors']['analytic']['chirp_mass'][0].decode('utf-8')
            new_string = string_ov.replace('=', ',').split(',')
            min_chirp_mass = new_string[1]
            max_chirp_mass = new_string[3]
            chirp_mass_priors[event] = [min_chirp_mass, max_chirp_mass]
        else:
            events_with_no_priors.append(event)

    np.savetxt(PATH_TO_RESULTS + 'info/' + 'events_with_priors_%s.txt' %waveform, events_with_priors, fmt = '%s')
    np.savetxt(PATH_TO_RESULTS + 'info/' + 'events_with_no_priors_%s.txt' %waveform, events_with_no_priors, fmt = '%s')
    with open(PATH_TO_RESULTS + 'info/' + 'chirp_mass_priors_%s.pkl' %waveform, 'wb') as f:
        pickle.dump(chirp_mass_priors, f)

def detectors_and_yaml_files(PATH_TO_DATA, PATH_TO_RESULTS, PATH_TO_YAML, PATH_TO_PSD, events_list, waveform):

    dict_template = {'L1':{'lat':30.56 * np.pi / 180.,
                    'lon':-90.77 * np.pi / 180.,
                    'opening_angle':np.pi / 2.,
                    'azimuth':197.7 * np.pi / 180.,
                    'duty_factor':0.85,
                    'detector_class':'earthL',
                    'plotrange':'10, 1000, 1e-25, 1e-20',
                    'fmin':8,
                    'fmax':1024,
                    'spacing':'geometric',
                    'df':1/4,
                    'npoints':5000
                    },
               'H1':{'lat':46.45 * np.pi / 180.,
                    'lon':-119.41 * np.pi / 180.,
                    'opening_angle':np.pi / 2.,
                    'azimuth':171.8 * np.pi / 180.,
                    'duty_factor':0.85,
                    'detector_class':'earthL',
                    'plotrange':'10, 1000, 1e-25, 1e-20',
                    'fmin':8,
                    'fmax':1024,
                    'spacing':'geometric',
                    'df':1/4,
                    'npoints':5000
                    },
                'V1':{'lat':43.63 * np.pi / 180.,
                    'lon':10.51 * np.pi / 180.,
                    'opening_angle':np.pi / 2.,
                    'azimuth':116.5 * np.pi / 180.,
                    'duty_factor':0.85,
                    'detector_class':'earthL',
                    'plotrange':'10, 1000, 1e-25, 1e-20',
                    'fmin':8,
                    'fmax':1024,
                    'spacing':'geometric',
                    'df':1/4,
                    'npoints':5000
                    }
            }

    detectors = {}
    for event in events_list:

        local_dictionary = dict_template.copy()

        data = h5py.File(PATH_TO_DATA + event + '.h5', 'r')
        detectors[event] = keys(data['C01:' + waveform]['psds'])

        for j in range(len(detectors[event])):
            local_dictionary[detectors[event][j]].update({'psd_data':PATH_TO_PSD + 'psd_%s_%s_%s.txt' %(waveform, event, detectors[event][j])})

            np.savetxt(PATH_TO_PSD + 'psd_%s_%s_%s.txt' %(waveform, event, detectors[event][j]), 
                       np.c_[data['C01:' + waveform]['psds'][detectors[event][j]][:, 0], 
                             data['C01:' + waveform]['psds'][detectors[event][j]][:, 1]])
                                                               
        with open(PATH_TO_YAML + '%s.yaml' %event, 'w') as my_yaml_file:
            yaml.dump(local_dictionary, my_yaml_file)

    
    with open(PATH_TO_RESULTS + 'info/' + 'detectors_dictionary.pkl', 'wb') as f:
        pickle.dump(detectors, f)



def gwfish_analysis(PATH_TO_YAML, PATH_TO_INJECTIONS, events_list, waveform, estimator,
                    detectors, fisher_parameters, PATH_TO_RESULTS):

    for event in events_list:
        population = '%s_BBH_%s' %(estimator, event)

        detectors_list = detectors[event]
        detectors_event = []
        for j in range(len(detectors_list)):
            detectors_event.append(detectors_list[j])
        networks = np.linspace(0, len(detectors_event) - 1, len(detectors_event), dtype=int)
        networks = str([networks.tolist()])

        detectors_ids = np.array(detectors_event)
        networks_ids = json.loads(networks)
        ConfigDet = os.path.join(PATH_TO_YAML + event + '.yaml')


        waveform_model = waveform
        my_params = pd.read_hdf(PATH_TO_INJECTIONS + event +  '/%s_%s_%s.hdf5' %(event, waveform, estimator))
        gw_parameters = my_params[fisher_parameters]
 
        #gw_parameters['mass1_lvk'] = gw_parameters['mass_1']
        #gw_parameters['mass2_lvk'] = gw_parameters['mass_2']
        #gw_parameters['mass_1'], gw_parameters['mass_2'] = from_mChirp_q_to_m1_m2(gw_parameters['chirp_mass'], gw_parameters['mass_ratio'])
        #print(gw_parameters)

        network = gw.detection.Network(detectors_ids, detection_SNR=(0., 1.), config=ConfigDet)
        gw.fishermatrix.analyze_and_save_to_txt(network = network,
                                        parameter_values  = gw_parameters,
                                        fisher_parameters = fisher_parameters, 
                                        sub_network_ids_list = networks_ids,
                                        population_name = population,
                                        waveform_model = waveform_model,
                                        save_path = PATH_TO_RESULTS,
                                        save_matrices = True)



def from_m1_m2_to_mChirp_q(m1, m2):
    """
    Compute the transformation from m1, m2 to mChirp, q
    """
    mChirp = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
    q = m2 / m1
    return mChirp, q

def from_mChirp_q_to_m1_m2(mChirp, q):
    """
    Compute the transformation from mChirp, q to m1, m2
    """
    m1 = mChirp * (1 + q)**(1/5) * q**(-3/5)
    m2 = mChirp * (1 + q)**(1/5) * q**(2/5)
    return m1, m2

def derivative_m1_m2_dmChirp_dq(m1, m2, mChirp, q):
    """
    Compute the derivative of m1, m2 with respect to mChirp, q
    """
    dm1_dmChirp = (1 + q)**(1/5) * q**(-3/5)
    dm1_dq = mChirp * (1 + q)**(1/5) * (-3/5) * q**(-8/5) + mChirp * (1/5) * (1 + q)**(-4/5) * q**(-3/5)
    dm2_dmChirp = (1 + q)**(1/5) * q**(2/5)
    dm2_dq = mChirp * (1 + q)**(1/5) * (2/5) * q**(-3/5) + mChirp * (1/5) * (1 + q)**(-4/5) * q**(2/5)
    return dm1_dmChirp, dm1_dq, dm2_dmChirp, dm2_dq


def jacobian_for_derivative_from_m1_m2_to_mChirp_q(m1, m2, fisher_matrix):
    """
    Compute the Jacobian for the transformation from m1, m2 to mChirp, q
    """
    mChirp, q = from_m1_m2_to_mChirp_q(m1, m2)
    dm1_dmChirp, dm1_dq, dm2_dmChirp, dm2_dq = derivative_m1_m2_dmChirp_dq(m1, m2, mChirp, q)
    rotated_fisher = fisher_matrix.copy()
    jacobian_matrix = np.zeros_like(fisher_matrix)
    nparams = len(fisher_matrix[0, 0, :])
    for i in range(nparams):
        jacobian_matrix[0, i, i] = 1
    jacobian_matrix[0, 0, 0] = dm1_dmChirp
    jacobian_matrix[0, 0, 1] = dm1_dq
    jacobian_matrix[0, 1, 0] = dm2_dmChirp
    jacobian_matrix[0, 1, 1] = dm2_dq

    rotated_fisher = jacobian_matrix[0, :, :].T @ rotated_fisher[0, :, :] @ jacobian_matrix[0, :, :]

    #nparams = len(fisher_parameters)
    #jacobian = np.identity((nparams))

    #jacobian[np.ix_([fisher_parameters.index('mass_1'), fisher_parameters.index('mass_1')], [fisher_parameters.index('mass_1'), fisher_parameters.index('mass_1')])] = derivative_m1_m2_dmChirp_dq(m1, m2, mChirp, q)

    # Write the jacobian matrix to pass from the fisher matrix in m1 and m2 to fisher in mChirp and q
    #rotated_fisher = jacobian.T@old_fisher@jacobian
    return rotated_fisher[np.newaxis, :, :]

def get_rotated_fisher_matrix(PATH_TO_RESULTS, events_list, detectors_list, estimator, lbs_errs, new_fisher_parameters):
   
    for event in events_list:

        label = get_label(detectors_list, event, estimator, 'errors')

        signals = pd.read_csv(PATH_TO_RESULTS + 'results/gwfish_m1_m2/' +
                            get_label(detectors_list, event, estimator, 'errors'), names = lbs_errs, skiprows = 1,
                            delimiter = ' ')
        
        fishers = np.load(PATH_TO_RESULTS + 'results/gwfish_m1_m2/' + 
                         get_label(detectors_list, event, estimator, 'fishers'))
        m1, m2 = signals[['mass_1', 'mass_2']].iloc[0]
        rotated_fisher = jacobian_for_derivative_from_m1_m2_to_mChirp_q(m1, m2, fishers)
        np.save(PATH_TO_RESULTS + 'results/gwfish_rotated/' + 
                get_label(detectors_list, event, estimator, 'fishers'), rotated_fisher)

        inv_rotated_fisher, _ = gw.fishermatrix.invertSVD(rotated_fisher[0, :, :])
        np.save(PATH_TO_RESULTS + 'results/gwfish_rotated/' + 
                get_label(detectors_list, event, estimator, 'inv_fishers'), inv_rotated_fisher)
        
        new_errors = signals.copy()

        err_params = []
        for l in range(len(new_fisher_parameters)):
            err_params.append('err_' + new_fisher_parameters[l])
        new_errors[err_params] = np.sqrt(np.diag(inv_rotated_fisher))
        np.savetxt(PATH_TO_RESULTS + 'results/gwfish_rotated/' +
                get_label(detectors_list, event, estimator, 'errors'), new_errors, delimiter = ' ', 
                fmt = '%.15f', header = '# ' + ' '.join(new_errors.keys()), comments = '')


def get_label(detectors_list, event, estimator, name_tag):
    detectors_labels = list(detectors_list[event])
    connector = '_'
    network_lbs = detectors_labels[0]
    for j in range(1, len(detectors_labels)):
        network_lbs += connector + detectors_labels[j]
    if name_tag == 'errors':
        label = 'Errors_%s_%s_BBH_%s_SNR1.txt' %(network_lbs, estimator, event)
    elif name_tag == 'fishers':
        label = 'fisher_matrices_%s_%s_BBH_%s_SNR1.npy' %(network_lbs, estimator, event)
    elif name_tag == 'inv_fishers':
        label = 'inv_fisher_matrices_%s_%s_BBH_%s_SNR1.npy' %(network_lbs, estimator, event)
    return label

def get_samples_from_TMVN(min_array, max_array, means, cov, N):
    """
    Draw samples from a truncated multivariate normal distribution
    """
    tmvn = TruncatedMVN(means, cov, min_array, max_array)
    return tmvn.sample(N)

def get_posteriors(samples, priors_dict, N):
    """
    Draw samples from a multivariate normal distribution with priors
    """
    samples['priors'] = priors.uniform_pdf(samples['chirp_mass'].to_numpy(), priors_dict['chirp_mass'][0], priors_dict['chirp_mass'][1])*\
                        priors.uniform_pdf(samples['mass_ratio'].to_numpy(), priors_dict['mass_ratio'][0], priors_dict['mass_ratio'][1])*\
                        priors.uniform_in_distance_squared_pdf(samples['luminosity_distance'].to_numpy(), priors_dict['luminosity_distance'][0], priors_dict['luminosity_distance'][1])*\
                        priors.uniform_in_cosine_pdf(samples['dec'].to_numpy(), priors_dict['dec'][0], priors_dict['dec'][1])*\
                        priors.uniform_pdf(samples['ra'].to_numpy(), priors_dict['ra'][0], priors_dict['ra'][1])*\
                        priors.uniform_in_sine_pdf(samples['theta_jn'].to_numpy(), priors_dict['theta_jn'][0], priors_dict['theta_jn'][1])*\
                        priors.uniform_pdf(samples['psi'].to_numpy(), priors_dict['psi'][0], priors_dict['psi'][1])*\
                        priors.uniform_pdf(samples['phase'].to_numpy(), priors_dict['phase'][0], priors_dict['phase'][1])*\
                        priors.uniform_pdf(samples['geocent_time'].to_numpy(), priors_dict['geocent_time'][0], priors_dict['geocent_time'][1])*\
                        priors.uniform_pdf(samples['a_1'].to_numpy(), priors_dict['a_1'][0], priors_dict['a_1'][1])*\
                        priors.uniform_pdf(samples['a_2'].to_numpy(), priors_dict['a_2'][0], priors_dict['a_2'][1])*\
                        priors.uniform_in_sine_pdf(samples['tilt_1'].to_numpy(), priors_dict['tilt_1'][0], priors_dict['tilt_1'][1])*\
                        priors.uniform_in_sine_pdf(samples['tilt_2'].to_numpy(), priors_dict['tilt_2'][0], priors_dict['tilt_2'][1])*\
                        priors.uniform_pdf(samples['phi_12'].to_numpy(), priors_dict['phi_12'][0], priors_dict['phi_12'][1])*\
                        priors.uniform_pdf(samples['phi_jl'].to_numpy(), priors_dict['phi_jl'][0], priors_dict['phi_jl'][1])
    '''
    priors_results = {
        'chirp_mass': priors.uniform_pdf(samples['chirp_mass'].to_numpy(), priors_dict['chirp_mass'][0], priors_dict['chirp_mass'][1]),
        'mass_ratio': priors.uniform_pdf(samples['mass_ratio'].to_numpy(), priors_dict['mass_ratio'][0], priors_dict['mass_ratio'][1]),
        'luminosity_distance': priors.uniform_in_distance_squared_pdf(samples['luminosity_distance'].to_numpy(), priors_dict['luminosity_distance'][0], priors_dict['luminosity_distance'][1]),
        'dec': priors.uniform_in_cosine_pdf(samples['dec'].to_numpy(), priors_dict['dec'][0], priors_dict['dec'][1]),
        'ra': priors.uniform_pdf(samples['ra'].to_numpy(), priors_dict['ra'][0], priors_dict['ra'][1]),
        'theta_jn': priors.uniform_in_sine_pdf(samples['theta_jn'].to_numpy(), priors_dict['theta_jn'][0], priors_dict['theta_jn'][1]),
        'psi': priors.uniform_pdf(samples['psi'].to_numpy(), priors_dict['psi'][0], priors_dict['psi'][1]),
        'phase': priors.uniform_pdf(samples['phase'].to_numpy(), priors_dict['phase'][0], priors_dict['phase'][1]),
        'geocent_time': priors.uniform_pdf(samples['geocent_time'].to_numpy(), priors_dict['geocent_time'][0], priors_dict['geocent_time'][1]),
        'a_1': priors.uniform_pdf(samples['a_1'].to_numpy(), priors_dict['a_1'][0], priors_dict['a_1'][1]),
        'a_2': priors.uniform_pdf(samples['a_2'].to_numpy(), priors_dict['a_2'][0], priors_dict['a_2'][1]),
        'tilt_1': priors.uniform_in_sine_pdf(samples['tilt_1'].to_numpy(), priors_dict['tilt_1'][0], priors_dict['tilt_1'][1]),
        'tilt_2': priors.uniform_in_sine_pdf(samples['tilt_2'].to_numpy(), priors_dict['tilt_2'][0], priors_dict['tilt_2'][1]),
        'phi_12': priors.uniform_pdf(samples['phi_12'].to_numpy(), priors_dict['phi_12'][0], priors_dict['phi_12'][1]),
        'phi_jl': priors.uniform_pdf(samples['phi_jl'].to_numpy(), priors_dict['phi_jl'][0], priors_dict['phi_jl'][1])
    }
    '''
    samples['weights'] = samples['priors'] / np.sum(samples['priors'])
    prob = samples['weights'].to_numpy()
    index = np.random.choice(np.arange(N), size = N, replace = True, p = prob)
    posteriors = samples.iloc[index]
    
    return posteriors

def get_lvk_samples(PATH_TO_LVK_DATA, event, params):
    """
    Get the LVK samples
    """
    data = h5py.File(PATH_TO_LVK_DATA + event + '.h5', 'r')
    samples_lvk = {}
    for l in range(len(params)):
        samples_lvk[params[l]] = data['C01:IMRPhenomXPHM']['posterior_samples'][params[l]]

    return samples_lvk

def get_confidence_interval(samples, params, confidence_level):
    """
    Compute the confidence intervals
    """
    confidence_level /= 100
    conf_int = {}
    for param in params:
        conf_int[param] = np.percentile(samples[param], [100 * (1 - confidence_level) / 2, 100 * (1 + confidence_level) / 2])
    return conf_int

