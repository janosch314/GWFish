"""Tests regarding the inversion of matrices, which
is a pain points for Fisher matrix codes.

For now, the invertSVD function is only tested
with matrices having

- all entries distributed according to a uniform distribution on [0, 1];
- a matrix constructed as before and then multiplied with its transpose
    to yield a symmetric, positive definite matrix;
- a matrix constructed as before and then modified by forcing 
the (i, j) entry to be the same as the (j, i) one.

Only the first of these fails, suggesting that the algorithm 
does not work with non-symmetric matrices but is fine otherwise.

It is currently marked with "xfail" (expected-to-fail),
remove it to see the traceback of the error.

"""

import pytest
import numpy as np
from GWFish.modules.fishermatrix import invertSVD

# Arrays used for the parametrization 
SEEDS = [1]
MATRIX_SIZES = [10, 20]

def assert_matrix_inverse_correctness(matrix):
    
    # swap the line for this one to check that the test passes
    # when using standard numpy inversion
    # inverse = np.linalg.inv(matrix)
    inverse = invertSVD(matrix)
    should_be_identity = matrix @ inverse
    identity = np.eye(*matrix.shape)
    
    assert np.allclose(identity, should_be_identity)

@pytest.mark.xfail
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("matrix_size", MATRIX_SIZES)
def test_matrix_inversion_all_entries_uniform(seed, matrix_size):

    rng = np.random.default_rng(seed=seed)
    matrix = rng.uniform(low=0, high=1, size=(matrix_size, matrix_size))
    
    assert_matrix_inverse_correctness(matrix)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("matrix_size", MATRIX_SIZES)
def test_matrix_inversion_symmetric_positive_definite(seed, matrix_size):

    rng = np.random.default_rng(seed=seed)
    
    normal_matrix = rng.normal(size=(matrix_size, matrix_size))
    # create a positive definite matrix
    matrix = normal_matrix @ normal_matrix.T
    
    assert_matrix_inverse_correctness(matrix)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("matrix_size", MATRIX_SIZES)
def test_matrix_inversion_all_entries_uniform_symmetric(seed, matrix_size):

    rng = np.random.default_rng(seed=seed)
    
    matrix = rng.uniform(low=0, high=1, size=(matrix_size, matrix_size))
    
    for (i, j), entry in np.ndenumerate(matrix):
        if j > i:
            matrix[j, i] = entry
    
    assert np.allclose(matrix, matrix.T)
    
    assert_matrix_inverse_correctness(matrix)
