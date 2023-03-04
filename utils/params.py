"""
Utility functions for handling data and parameters.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

import jax
import jax.numpy as jnp

import numpy as np


def epoch(key, data, batch_size, batch_axis=0, ragged=False):
    """Generate batches to form an epoch over a data set."""
    # Check for consistent dimensions along `batch_axis`
    flat_data, _ = jax.tree_util.tree_flatten(data)
    num_samples = jnp.array(jax.tree_util.tree_map(
        lambda x: jnp.shape(x)[batch_axis],
        flat_data
    ))
    if not jnp.all(num_samples == num_samples[0]):
        raise ValueError('Batch dimensions not equal!')
    num_samples = num_samples[0]

    # Compute the number of batches
    if ragged:
        num_batches = -(-num_samples // batch_size)  # ceiling division
    else:
        num_batches = num_samples // batch_size  # floor division

    # Loop through batches (with pre-shuffling)
    shuffled_idx = jax.random.permutation(key, num_samples)
    for i in range(num_batches):
        batch_idx = shuffled_idx[i*batch_size:(i+1)*batch_size]
        batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, batch_idx, batch_axis),
            data
        )
        yield batch


def mat_to_svec_dim(n):
    """Compute the number of unique entries in a symmetric matrix."""
    d = (n * (n + 1)) // 2
    return d


def svec_to_mat_dim(d):
    """Compute the symmetric matrix dimension with `d` unique elements."""
    n = (int(np.sqrt(8 * d + 1)) - 1) // 2
    if d != mat_to_svec_dim(n):
        raise ValueError('Invalid vector length `d = %d` for filling the '
                         'triangular of a symmetric matrix!' % d)
    return n


def svec_diag_indices(n):
    """Compute indices of `svec(A)` corresponding to diagonal elements.

    Example for `n = 3`:
    [ 0       ]
    [ 1  3    ]  => [0, 3, 5]
    [ 2  4  5 ]

    For general `n`, indices of `svec` corresponding to the diagonal are:
      [0, n, n + (n-1), ..., n*(n+1)/2 - 1]
      = n*(n+1)/2 - [n*(n+1)/2, (n-1)*n/2, ..., 1]
    """
    d = mat_to_svec_dim(n)
    idx = d - mat_to_svec_dim(np.arange(1, n+1)[::-1])
    return idx


def svec(X, scale=True):
    """Compute the symmetric vectorization of symmetric matrix `X`."""
    shape = jnp.shape(X)
    if len(shape) < 2:
        raise ValueError('Argument `X` must be at least 2D!')
    if shape[-2] != shape[-1]:
        raise ValueError('Last two dimensions of `X` must be equal!')
    n = shape[-1]

    if scale:
        # Scale elements corresponding to the off-diagonal, lower-triangular
        # part of `X` by `sqrt(2)` to preserve the inner product
        rows, cols = jnp.tril_indices(n, -1)
        X = X.at[..., rows, cols].mul(jnp.sqrt(2))

    # Vectorize the lower-triangular part of `X` in row-major order
    rows, cols = jnp.tril_indices(n)
    svec_X = X[..., rows, cols]
    return svec_X


def smat(svec_X, scale=True):
    """Compute the symmetric matrix `X` given `svec(X)`."""
    svec_X = jnp.atleast_1d(svec_X)
    d = svec_X.shape[-1]
    n = svec_to_mat_dim(d)  # corresponding symmetric matrix dimension

    # Fill the lower triangular of `X` in row-major order with the elements
    # of `svec_X`
    rows, cols = jnp.tril_indices(n)
    X = jnp.zeros((*svec_X.shape[:-1], n, n))
    X = X.at[..., rows, cols].set(svec_X)
    if scale:
        # Scale elements corresponding to the off-diagonal, lower-triangular
        # elements of `X` by `1 / sqrt(2)` to preserve the inner product
        rows, cols = jnp.tril_indices(n, -1)
        X = X.at[..., rows, cols].mul(1 / jnp.sqrt(2))

    # Make `X` symmetric
    rows, cols = jnp.triu_indices(n, 1)
    X = X.at[..., rows, cols].set(X[..., cols, rows])
    return X


def cholesky_to_params(L):
    """Uniquely parameterize a positive-definite Cholesky factor."""
    shape = jnp.shape(L)
    if len(shape) < 2:
        raise ValueError('Argument `L` must be at least 2D!')
    if shape[-2] != shape[-1]:
        raise ValueError('Last two dimensions of `L` must be equal!')
    n = shape[-1]
    rows, cols = jnp.diag_indices(n)
    log_L = L.at[..., rows, cols].set(jnp.log(L[..., rows, cols]))
    params = svec(log_L, scale=False)
    return params


def params_to_cholesky(params):
    """Map free parameters to a positive-definite Cholesky factor."""
    params = jnp.atleast_1d(params)
    d = params.shape[-1]
    n = svec_to_mat_dim(d)  # corresponding symmetric matrix dimension
    rows, cols = jnp.tril_indices(n)
    log_L = jnp.zeros((*params.shape[:-1], n, n)).at[...,
                                                     rows, cols].set(params)
    rows, cols = jnp.diag_indices(n)
    L = log_L.at[..., rows, cols].set(jnp.exp(log_L[..., rows, cols]))
    return L


def params_to_posdef(params):
    """Map free parameters to a positive-definite matrix."""
    L = params_to_cholesky(params)
    LT = jnp.swapaxes(L, -2, -1)
    X = L @ LT
    return X
