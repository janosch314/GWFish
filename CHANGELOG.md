# Changelog

All notable changes to `GWFish` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- #99: fix in the `check_and_convert_to_mass_1_mass_2` function
- #95: bugfix in spin conversions
- #94: GWFish paper reproducibility files
- #92: minor fixes
- #91: new `utilities` module and updated tutorial
- #90: add a more precise treatment of the high frequency response of ground-based detectors
- #82 and #83: new possibilities for mass input (detector frame and chirp mass-q)
- #76: new GWFish tutorial notebook
- #75: bugfixes related to `max_frequency_cutoff`
- #73: fix: ensure that the time step is small enough in ephemeris caching
- #70: fix: make it possible to import the module with partial functionality when LAL is not present
- #68: correct the definition of `theta_jn` in the documentation
- #67: changed the signature of `compute_fisher_errors` to also return the indices of detected sources

### [1.0.0]

Changes in PRs up to #65.

- Add Solar-Centered ephemeris computed through astropy/jplephem for Earth and Moon-based detectors: `modules.ephemeris`
    - Phase shift is now computed from there during the projection
- Refactor temporal truncation with detector lifetime, now shared across detector classes
    - The `projection` function now has an optional flag, `redefine_tf_vectors`: if enabled, 
        this will redefine the time and frequency vectors as required to shift the time corresponding to the
        given `maximum_frequency_cutoff` to the `geocent_time` --- this way, the signal is guaranteed to lie 
        in the few years before the `geocent_time`, where the ephemeris do not give errors. 
- Remove old Fisher errors computation
    - Removed `SNR` attribute of the `Detector` object, 
    as well as the practice of dynamically saving Fisher matrices as detector object attributes 
    - Remove `analyzeFisherErrors`
- Add new restructured Fisher errors computation
    - `sky_localization_area`, utility function to get sky localization from the Fisher inverse
    - `sky_localization_percentile_factor`, for the conversion from sigmas to percentile contours
    - `compute_detector_fisher`:
        - compute the signal
        - project the signal onto the detector
        - compute SNR and Fisher matrix
    - `compute_network_errors`:
        - loop through signals
        - compute network SNR/Fisher and invert
        - return errors and sky localizations for signals above threshold 
- Add I/O functions for the Fisher errors, reproducing the old functionality
    - `errors_file_name`
    - `output_to_txt_file`
    - `analyze_and_save_to_txt`
- Add the possibility to have PSDs saved in locations other than the GWFish folder 
    - this is accomplished by adding a `psd_path` entry to the detector definition
- Remove `plot` argument to detector definition, which generated a plot at initialization
    - Reintroduced the same functionality as a `self.plot` method
- Make `detector_ids` argument mandatory for network initialization
    - Before, it defaulted to a network of ET only
- Rename `duty_factor` argument of `SNR` function to `use_duty_factor` and cosmetic refactor
- Allow for the computation of horizon as a function of detector-frame mass
- Use dual annealing in the computation of the optimum sky position (max SNR)
- Many new tests and improvements to the test suite

[unreleased]: https://github.com/janosch314/GWFish/compare/main...9c8b9e7873

[1.0.0]: https://github.com/janosch314/GWFish/compare/9c8b9e787...934a9a90fa