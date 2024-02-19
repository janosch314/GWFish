#!/usr/bin/env python
"""Idea inspired by this answer: https://stackoverflow.com/a/7259267/11234313
"""

from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt

item_prefix = '    - '
BASE_PATH = Path(__file__).resolve().parent.parent
IMG_FOLDER = BASE_PATH / 'docs' / 'source' / 'figures'

def format_number(number):
    if number < 1e-10:
        return '$0$'
    if number < 1e-2 or number > 1e2:
        exp = np.floor(np.log10(number))
        fractional_part = number / 10**exp
        if np.isclose(fractional_part, 1, rtol=1e-2):
            return f'$10^{{{exp:.0f}}}$'
        elif np.isclose(fractional_part, np.round(fractional_part), rtol=1e-2):
            return f'${fractional_part:.0f} \\times 10^{{{exp:.0f}}}$'
        return f'${fractional_part:.2f} \\times 10^{{{exp:.0f}}}$'
    else:
        if np.isclose(number, np.round(number), rtol=1e-2):
            return f'${number:.0f}$'
        else:
            return f'${number:.2f}$'

def frequencyvector_description(detector):
    
    fmin = float(eval(str(detector["fmin"])))
    fmax = float(eval(str(detector["fmax"])))
    if detector['spacing'] == 'geometric':
        return item_prefix + f'geometric frequency vector with {detector["npoints"]} points between {format_number(fmin)}Hz and {format_number(fmax)}Hz;\n'
    elif detector['spacing'] == 'linear':
        df = float(eval(str(detector["df"])))
        npoints = (fmax - fmin) / df
        return item_prefix + f'linear frequency vector from {format_number(fmin)}Hz to {format_number(fmax)}Hz with spacing {format_number(df)}Hz ({npoints:.0f} points);\n'
    else:
        raise(ValueError(f'Invalid spacing `{detector["spacing"]}`!'))

def dutyfactor_description(detector):
    df = float(eval(str(detector["duty_factor"])))
    return item_prefix + f'duty factor {df:.0%};\n'

def save_psd_plot(detector, psd_path):
    
    psd = np.loadtxt(psd_path)
    freq = psd[:, 0]
    psd = psd[:, 1]
    plt.loglog(freq, np.sqrt(psd))
    plotrange = np.fromstring(detector['plotrange'], dtype=float, sep=',')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Strain noise [Hz$^{1/2}$]')
    plt.grid(visible=True, which='both')
    plt.tight_layout()
    save_path = IMG_FOLDER / f'psd_{detector["name"]}.png'
    plt.savefig(save_path)
    plt.close()
    return save_path

def psd_description(detector):
    psd_path = BASE_PATH / 'GWFish' / 'detector_psd' / detector["psd_data"]
    if psd_path.is_file():
        image_path = save_psd_plot(detector, psd_path)
        psd_path_description = item_prefix + f'power spectral density data file: `{psd_path.relative_to(BASE_PATH)}`;\n'
        img_include = ('\n'
            f'```{{image}} ../figures/{image_path.relative_to(IMG_FOLDER)}\n'
            ':width: 30%\n'
            ':align: right\n'
            '```\n\n'
        )
        
        return psd_path_description, img_include
    else:
        raise(FileNotFoundError(f'PSD file `{psd_path}` not found!'))

def location_description(detector, location='earth'):
    lat = float(eval(str(detector["lat"])))
    lon = float(eval(str(detector["lon"])))
    
    if location == 'earth':
        return item_prefix + f'location: [{lat:.7f} radians N, {lon:.7f} radians E](https://www.google.com/maps/place/{np.rad2deg(lat):.7f},{np.rad2deg(lon):.7f}) on the Earth;\n'
    else:
        return item_prefix + f'location: {lat:.7f} radians N, {lon:.7f} radians E on {location};\n'

def shape_class_description(detector):
    
    if detector["detector_class"] == 'earthDelta':
        arm_azimuth = float(eval(str(detector["azimuth"])))
        assert np.isclose(float(eval(str(detector["opening_angle"]))), np.pi/3)
        return item_prefix + f'Triangle-shaped detector on the Earth, with an opening angle of $\\pi/3$ radians and an arm azimuth of {format_number(arm_azimuth)}rad;\n'

    elif detector["detector_class"] == 'earthL':
        arm_azimuth = float(eval(str(detector["azimuth"])))
        assert np.isclose(float(eval(str(detector["opening_angle"]))), np.pi/2)
        return item_prefix + f'L-shaped detector on the Earth, with an opening angle of $\\pi/2$ radians and an arm azimuth of {format_number(arm_azimuth)}rad;\n'
    elif detector["detector_class"] == 'lunararray':
        try:
            arm_azimuth = float(eval(str(detector["azimuth"])))
        except TypeError:
            arm_azimuth = 0.
        return item_prefix + f'Detector on the Moon with a seismometer azimuth of {format_number(arm_azimuth)}rad;\n'
    elif detector["detector_class"] == 'satellitesolarorbit':
        return item_prefix + f'Space-based detector;\n'
    else:
        raise(ValueError(f'Invalid detector class `{detector["detector_class"]}`!'))

def lifetime_description(detector):
    lifetime = float(eval(str(detector["mission_lifetime"])))
    
    seconds_per_year = 31557600.
    seconds_per_month = 2629800.
    
    if lifetime / seconds_per_year > 1:
        return item_prefix + f'Detector lifetime: {format_number(lifetime / seconds_per_year)} years;\n'
    elif lifetime / seconds_per_month > 1:
        return item_prefix + f'Detector lifetime: {format_number(lifetime / seconds_per_month)} months;\n'
    else:
        return item_prefix + f'Detector lifetime: {format_number(lifetime)} seconds;\n'

def main():
    autogen_path = Path(__file__).resolve().parent / 'source' / "detectors_autogen.inc"
    yaml_path = Path(__file__).resolve().parent.parent / 'GWFish' / "detectors.yaml"
    with open(autogen_path, 'w') as f:
        with open(yaml_path, 'r') as y:
            yaml_data = yaml.load(y, Loader=yaml.FullLoader)
            for detector_name, detector in yaml_data.items():
                
                detector['name'] = detector_name
                
                psd_path_description, img_include = psd_description(detector)
                if detector_name != 'LISA':
                    f.write(img_include)
                f.write(f'- {detector_name}\n')
                f.write(shape_class_description(detector))
                
                if detector['detector_class'] in ['lunararray', 'satellitesolarorbit']:
                    f.write(lifetime_description(detector))
                
                if detector['detector_class'] in ['earthDelta', 'earthL']:
                    f.write(location_description(detector, location='earth'))
                elif detector['detector_class'] == 'lunararray':
                    f.write(location_description(detector, location='the Moon'))
                f.write(dutyfactor_description(detector))
                f.write(frequencyvector_description(detector))
                f.write(psd_path_description)

if __name__ == '__main__':
    main()