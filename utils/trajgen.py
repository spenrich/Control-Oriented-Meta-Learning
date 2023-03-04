"""
Utilities for trajectory generation, particularly via spline interpolation.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag

import numpy as np


@partial(jax.jit, static_argnums=(2, 3))
def _scalar_smooth_trajectory(x_knots, t_knots, poly_order, deriv_order):
    """Construct a smooth trajectory through given points.

    References
    ----------
    .. [1] Charles Richter, Adam Bry, and Nicholas Roy,
           "Polynomial trajectory planning for aggressive quadrotor flight in
           dense indoor environments", ISRR 2013.
    .. [2] Daniel Mellinger and Vijay Kumar,
           "Minimum snap trajectory generation and control for quadrotors",
           ICRA 2011.
    .. [3] Declan Burke, Airlie Chapman, and Iman Shames,
           "Generating minimum-snap quadrotor trajectories really fast",
           IROS 2020.
    """
    num_coefs = poly_order + 1          # number of coefficients per polynomial
    num_knots = x_knots.size            # number of interpolating points
    num_polys = num_knots - 1           # number of polynomials
    primal_dim = num_coefs * num_polys  # number of unknown coefficients

    T = jnp.diff(t_knots)                # polynomial lengths in time
    powers = jnp.arange(poly_order + 1)  # exponents defining each monomial
    D = jnp.diag(powers[1:], -1)         # maps monomials to their derivatives

    c0 = jnp.zeros((deriv_order + 1, num_coefs)).at[0, 0].set(1.)
    c1 = jnp.zeros((deriv_order + 1, num_coefs)).at[0, :].set(1.)
    for n in range(1, deriv_order + 1):
        c0 = c0.at[n].set(D @ c0[n - 1])
        c1 = c1.at[n].set(D @ c1[n - 1])

    # Assemble constraints in the form `A @ x = b`, where `x` is the vector of
    # stacked polynomial coefficients

    # Knots
    b_knots = jnp.concatenate((x_knots[:-1], x_knots[1:]))
    A_knots = jnp.vstack([
        block_diag(*jnp.tile(c0[0], (num_polys, 1))),
        block_diag(*jnp.tile(c1[0], (num_polys, 1)))
    ])

    # Zero initial conditions (velocity, acceleration, jerk)
    b_init = jnp.zeros(deriv_order - 1)
    A_init = jnp.zeros((deriv_order - 1, primal_dim))
    A_init = A_init.at[:deriv_order - 1, :num_coefs].set(c0[1:deriv_order])

    # Zero final conditions (velocity, acceleration, jerk)
    b_fin = jnp.zeros(deriv_order - 1)
    A_fin = jnp.zeros((deriv_order - 1, primal_dim))
    A_fin = A_fin.at[:deriv_order - 1, -num_coefs:].set(c1[1:deriv_order])

    # Continuity (velocity, acceleration, jerk, snap)
    b_cont = jnp.zeros(deriv_order * (num_polys - 1))
    As = []
    zero_pad = jnp.zeros((num_polys - 1, num_coefs))
    Tn = jnp.ones_like(T)
    for n in range(1, deriv_order + 1):
        Tn = T * Tn
        diag_c0 = block_diag(*(c0[n] / Tn[1:].reshape([-1, 1])))
        diag_c1 = block_diag(*(c1[n] / Tn[:-1].reshape([-1, 1])))
        As.append(jnp.hstack((diag_c1, zero_pad))
                  - jnp.hstack((zero_pad, diag_c0)))
    A_cont = jnp.vstack(As)

    # Assemble
    A = jnp.vstack((A_knots, A_init, A_fin, A_cont))
    b = jnp.concatenate((b_knots, b_init, b_fin, b_cont))
    dual_dim = b.size

    # Compute the cost Hessian `Q(T)` as a function of the length `T` for each
    # polynomial, and stack them into the full block-diagonal Hessian
    ij_1 = powers.reshape([-1, 1]) + powers + 1
    D_snap = jnp.linalg.matrix_power(D, deriv_order)
    Q_snap = D_snap @ (1 / ij_1) @ D_snap.T
    Q_poly = lambda T: Q_snap / (T**(2*deriv_order - 1))  # noqa: E731
    Q = block_diag(*jax.vmap(Q_poly)(T))

    # Assemble KKT system and solve for coefficients
    K = jnp.block([
        [Q, A.T],
        [A, jnp.zeros((dual_dim, dual_dim))]
    ])
    soln = jnp.linalg.solve(K, jnp.concatenate((jnp.zeros(primal_dim), b)))
    primal, dual = soln[:primal_dim], soln[-dual_dim:]
    coefs = primal.reshape((num_polys, -1))
    r_primal = A@primal - b
    r_dual = Q@primal + A.T@dual
    return coefs, r_primal, r_dual


@partial(jax.jit, static_argnums=(2, 3))
def smooth_trajectory(x_knots, t_knots, poly_order, deriv_order):
    """Compute the coefficients of a smooth spline through the given knots."""
    num_knots = x_knots.shape[0]
    knot_shape = x_knots.shape[1:]
    flat_x_knots = jnp.reshape(x_knots, (num_knots, -1))
    in_axes = (1, None, None, None)
    out_axes = (2, 1, 1)
    flat_coefs, _, _ = jax.vmap(_scalar_smooth_trajectory,
                                in_axes, out_axes)(flat_x_knots, t_knots,
                                                   poly_order, deriv_order)
    num_polys = num_knots - 1
    coefs = jnp.reshape(flat_coefs, (num_polys, poly_order + 1, *knot_shape))
    return coefs


@jax.jit
def spline(t, t_knots, coefs):
    """Compute the value of a polynomial spline at time `t`."""
    num_polys = coefs.shape[0]
    poly_order = coefs.shape[1] - 1
    powers = jnp.arange(poly_order + 1)

    # Identify which polynomial segment to use for time `t`
    i = jnp.clip(jnp.searchsorted(t_knots, t, 'left') - 1, 0, num_polys - 1)

    # Evaluate the polynomial (with scaled time)
    tau = (t - t_knots[i]) / (t_knots[i+1] - t_knots[i])
    x = jnp.sum(coefs[i] * (tau**powers))
    return x


def spline_factory(t_knots, *coefs):
    """Define and return a polynomial spline function."""
    spline_func = (lambda t, t_knots=t_knots, coefs=coefs:
                   jnp.array([spline(t, t_knots, C) for C in coefs]))
    return spline_func


def uniform_random_walk(key, num_steps, shape=(), min_step=0., max_step=1.):
    """Sample a random walk of points.

    The step size is sampled uniformly from a closed interval.
    """
    minvals = jnp.broadcast_to(min_step, shape)
    maxvals = jnp.broadcast_to(max_step, shape)
    noise = minvals + (maxvals - minvals)*jax.random.uniform(key, (num_steps,
                                                                   *shape))
    points = jnp.concatenate((jnp.zeros((1, *shape)),
                              jnp.cumsum(noise, axis=0)), axis=0)
    return points


def random_spline(key, T_total, num_knots, poly_order, deriv_order,
                  shape=(), min_step=0., max_step=1.):
    """Sample a random walk and fit a spline to it."""
    knots = uniform_random_walk(key, num_knots - 1, shape, min_step, max_step)
    flat_knots = jnp.reshape(knots, (num_knots, -1))
    diffs = jnp.linalg.norm(jnp.diff(flat_knots, axis=0), axis=1)
    T = T_total * (diffs / jnp.sum(diffs))
    t_knots = jnp.concatenate((jnp.array([0., ]),
                               jnp.cumsum(T))).at[-1].set(T_total)
    coefs = smooth_trajectory(knots, t_knots, poly_order, deriv_order)
    return knots, t_knots, coefs


def random_ragged_spline(key, T_total, num_knots, poly_orders, deriv_orders,
                         min_step, max_step, min_knot=-jnp.inf,
                         max_knot=jnp.inf):
    """Sample a random walk and fit a spline to it.

    Different polynomial and smoothness orders can be used for each dimension.
    If this is done, the spline coefficient arrays will be of different shapes
    for each dimension, i.e., "ragged".
    """
    poly_orders = np.array(poly_orders).ravel().astype(int)
    deriv_orders = np.array(deriv_orders).ravel().astype(int)
    num_dims = poly_orders.size
    assert deriv_orders.size == num_dims
    shape = (num_dims,)
    knots = uniform_random_walk(key, num_knots - 1, shape, min_step, max_step)
    knots = jnp.clip(knots, min_knot, max_knot)
    flat_knots = jnp.reshape(knots, (num_knots, -1))
    diffs = jnp.linalg.norm(jnp.diff(flat_knots, axis=0), axis=1)
    T = T_total * (diffs / jnp.sum(diffs))
    t_knots = jnp.concatenate((jnp.array([0., ]),
                               jnp.cumsum(T))).at[-1].set(T_total)
    coefs = []
    for i, (p, d) in enumerate(zip(poly_orders, deriv_orders)):
        coefs.append(smooth_trajectory(knots[:, i], t_knots, p, d))
    coefs = tuple(coefs)
    knots = tuple(knots[:, i] for i in range(num_dims))
    return t_knots, knots, coefs
