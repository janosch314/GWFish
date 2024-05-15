"""
GWFish
=====

GWFish: a GW detector network simulation tool with Fisher-matrix analysis

"""


from __future__ import absolute_import
import sys

from . import modules
from .modules import auxiliary, detection, fishermatrix, waveforms, constants

from . import priors_modules
from .priors_modules import priors, minimax_tilting_sampler

if sys.version_info < (3,):
    raise ImportError("You need Python 3")
