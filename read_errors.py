import pandas as pd 
import matplotlib.pyplot as plt

par_5plus5_Fisher = pd.read_csv('Errors_ET_BBH_SNR12.0.txt',     
                                names=['network_SNR','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl',\
                                'dec','ra','iota','psi','phase','redshift','mass_1','mass_2','geocent_time','luminosity_distance',\
                                'err_ra','err_dec','err_psi','err_iota','err_luminosity_distance','err_mass_1','err_mass_2',\
                                'err_geocent_time','err_phase','err_sky_location'],
                                delimiter=' ')

lev_90 = 1.645




'''
print('####################################################')
print('5plus10_z01_GWFish_PhenoD_ET_D')
print('####################################################')
print('RA = ', par_5plus5_Fisher['ra'])
print('err_RA = ', lev_90*par_5plus5_Fisher['err_ra'])
print('----------------------------------------------------')
print('DEC = ', par_5plus5_Fisher['dec'])
print('err_DEC = ', lev_90*par_5plus5_Fisher['err_dec'])
print('----------------------------------------------------')
print('PSI = ', par_5plus5_Fisher['psi'])
print('err_PSI = ', lev_90*par_5plus5_Fisher['err_psi'])
print('----------------------------------------------------')
print('iota = ', par_5plus5_Fisher['iota'])
print('err_iota = ', lev_90*par_5plus5_Fisher['err_iota'])
print('----------------------------------------------------')
print('d_L = ', par_5plus5_Fisher['luminosity_distance'])
print('err_d_L = ', lev_90*par_5plus5_Fisher['err_luminosity_distance'])
print('----------------------------------------------------')
print('mass_1 = ', par_5plus5_Fisher['mass_1'])
print('err_mass_1 = ', lev_90*par_5plus5_Fisher['err_mass_1'])
print('----------------------------------------------------')
print('mass_2 = ', par_5plus5_Fisher['mass_2'])
print('err_mass_2 = ', lev_90*par_5plus5_Fisher['err_mass_2'])
print('####################################################')
'''