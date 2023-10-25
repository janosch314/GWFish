"""Tests regarding the inversion of matrices, which
is a pain points for Fisher matrix codes.

The invertSVD function really is computing the _pseudo_ inverse of 
the given matrix
"""

import numpy as np
import pytest
from hypothesis import given, reject, seed
from hypothesis import strategies as st
from hypothesis import target
from hypothesis.extra.numpy import arrays

from GWFish.modules.fishermatrix import invertSVD

MATRIX_DIMENSION = 4
ABS_TOLERANCE = 1e-1
REL_TOLERANCE = 1e-2
MIN_NORM = 1e-5
MAX_NORM = 1e5

def assert_matrix_pseudo_inverse_correctness(matrix, pseudo_inverse):
    
    product_1 = matrix @ pseudo_inverse @ matrix
    product_2 = pseudo_inverse @ matrix @ pseudo_inverse
    
    assert np.allclose(product_1, matrix, atol=ABS_TOLERANCE, rtol=REL_TOLERANCE)
    assert np.allclose(product_2, pseudo_inverse, atol=ABS_TOLERANCE, rtol=REL_TOLERANCE)


@seed(1)
@given(
    vector_norms=arrays(
        np.float64,
        (MATRIX_DIMENSION,),
        elements=st.floats(
            min_value=MIN_NORM,
            max_value=MAX_NORM,
        ),
        unique=True,
    ),
    cosines=arrays(
        np.float64,
        (MATRIX_DIMENSION, MATRIX_DIMENSION),
        elements=st.floats(
            min_value=-1.0,
            max_value=1.0,
        ),
        unique=True,
    ),
)
def test_matrix_inversion_hypothesis(vector_norms, cosines):

    cosines[np.arange(MATRIX_DIMENSION), np.arange(MATRIX_DIMENSION)] = 1
    cosines = np.maximum(cosines, cosines.T)

    matrix = np.outer(vector_norms, vector_norms) * cosines
    pseudo_inverse, _ = invertSVD(matrix)

    assert_matrix_pseudo_inverse_correctness(matrix, pseudo_inverse)