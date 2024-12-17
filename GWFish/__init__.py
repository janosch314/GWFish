"""
GWFish
=====

GWFish: a GW detector network simulation tool with Fisher-matrix analysis

"""


from __future__ import absolute_import
import sys

from . import modules

from .modules import auxiliary, constants, detection, ephemeris, fft, fishermatrix, horizon, minimax_tilting_sampler, priors, utilities, waveforms

if sys.version_info < (3,):
    raise ImportError("You need Python 3")
