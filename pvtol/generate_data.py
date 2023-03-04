"""
Generate training data for the PVTOL system.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

import os
import pickle
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

from utils.pvtol import (closed_loop, dynamics, feedback, reference_flat,
                         reference_state)
from utils.trajgen import random_ragged_spline, spline


if __name__ == '__main__':
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update('jax_enable_x64', True)

    # Set physical, controller, and simulation parameters
    T = 30.                         # simulation time horizon
    dt = 0.01                       # simulation time step
    num_traj = 500                  # number of generated trajectories
    g = 9.81                        # gravity
    β = jnp.array([0.01, 1., 1.])   # drag coefficients
    r = 1.                          # arm length
    w_min, w_max = 0., 3.           # min. and max. wind velocities
    beta_shape = (5., 9.)           # shape parameters for beta distribution
    ka = jnp.array([0.])            # no adaptation
    kx = jnp.array([1., 1.])
    cx = jnp.array([1., 1.])
    ky = jnp.array([1., 1.])
    cy = jnp.array([1., 1.])
    kϕ = jnp.array([1., 1.])
    assert cy[0] + cy[1] < g

    # Seed random numbers
    seed = 0
    key = jax.random.PRNGKey(seed)

    # Generate knots and coefficients for smooth splines fit to random walks in
    # `(x, y)`-space
    key, *subkeys = jax.random.split(key, 1 + num_traj)
    subkeys = jnp.vstack(subkeys)
    kwargs = {
        'num_knots':    6,
        'poly_orders':  (9, 9),
        'deriv_orders': (3, 3),
        'min_step':     jnp.array([-2., -2.]),
        'max_step':     jnp.array([2., 2.]),
    }
    spline_generator = partial(random_ragged_spline, **kwargs)
    t_knots, r_knots, coefs = jax.vmap(spline_generator, (0, None))(subkeys, T)
    r_knots = jnp.dstack(r_knots)

    # Sample wind velocities from the training distribution
    key, subkey = jax.random.split(key, 2)
    x_beta = jax.random.beta(subkey, *beta_shape, (num_traj,))
    w = w_min + (w_max - w_min)*x_beta

    # Construct a tracking simulator for the PVTOL system subject to wind
    def simulate(t, t_knots, coefs, w):
        """Simulate the PVTOL system in closed-loop with tracking control."""
        flat_func = lambda t: jnp.array([spline(t, t_knots, C)  # noqa: E731
                                        for C in coefs])

        # Dummy features (with no adaptation)
        feature_func = lambda s: jnp.zeros((6, 1))              # noqa: E731

        def controller(s, a, t):
            z, dz, ddz, d3z, d4z = reference_flat(t, flat_func)
            u, *_ = feedback(s, kx, ky, kϕ, cx, cy, z, dz, ddz, d3z, d4z)
            _, _, B_pseudo = dynamics(s)
            u -= B_pseudo @ feature_func(s) @ a
            return u

        def ode(sa, t):
            s, a = sa
            ds, da = closed_loop(t, s, a, w, r, β, kx, ky, kϕ, cx, cy, ka,
                                 feature_func, flat_func)
            return (ds, da)

        s_ref, u_ref = jax.vmap(reference_state, (0, None))(t, flat_func)
        s0 = s_ref[0]
        a0 = jnp.zeros(1)
        s, a = odeint(ode, (s0, a0), t)
        u = jax.vmap(controller)(s, a, t)

        return s_ref, u_ref, s, u, a

    # Simulate tracking for each sampled wind velocity `w`
    t = jnp.arange(0, T + dt, dt)
    simulate_parallel = jax.vmap(simulate, (None, 0, 0, 0))
    s_ref, u_ref, s, u, a = simulate_parallel(t, t_knots, coefs, w)

    # Record and save simulations as training data
    data = {
        'seed':         seed,
        'prng_key':     key,
        't':            t,
        'x':            s,
        'u':            u,
        'x_ref':        s_ref,
        'u_ref':        u_ref,
        't_knots':      t_knots,
        'r_knots':      r_knots,
        'w':            w,
        'w_min':        w_min,
        'w_max':        w_max,
        'beta_shape':   beta_shape,
        'control_gains': {
            'kx': kx,
            'ky': ky,
            'kϕ': kϕ,
            'cx': cx,
            'cy': cy,
        },
    }
    path = os.join('pvtol', 'training_data.pkl')
    with open(path, 'wb') as file:
        pickle.dump(data, file)
