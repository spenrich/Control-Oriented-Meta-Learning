"""
Utility functions for integrating ODEs.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import numpy as np


@partial(jax.jit, static_argnums=(0,))
def rk38_step(func, h, x, t, *args):
    """Do a single step of the RK-38 integration scheme."""
    # RK38 Butcher tableau
    s = 4
    A = jnp.array([
        [0,    0, 0, 0],
        [1/3,  0, 0, 0],
        [-1/3, 1, 0, 0],
        [1,   -1, 1, 0],
    ])
    b = jnp.array([1/8, 3/8, 3/8, 1/8])
    c = jnp.array([0,   1/3, 2/3, 1])

    def scan_fun(carry, cut):
        i, ai, bi, ci = cut
        x, t, h, K, *args = carry
        ti = t + h*ci
        xi = x + h*(K.T @ ai)
        ki = func(xi, ti, *args)
        K = K.at[i].set(ki)
        carry = (x, t, h, K, *args)
        return carry, ki

    init_carry = (x, t, h, jnp.squeeze(jnp.zeros((s, x.size))), *args)
    carry, K = jax.lax.scan(scan_fun, init_carry, (jnp.arange(s), A, b, c))
    xf = x + h*(K.T @ b)
    return xf


@partial(jax.jit, static_argnums=(0,))
def _odeint_ckpt(func, x0, ts, *args):

    def scan_fun(carry, t1):
        x0, t0, *args = carry
        x1 = rk38_step(func, t1 - t0, x0, t0, *args)
        carry = (x1, t1, *args)
        return carry, x1

    ts = jnp.atleast_1d(ts)
    init_carry = (x0, ts[0], *args)  # dummy state at same time as `t0`
    carry, xs = jax.lax.scan(scan_fun, init_carry, ts)
    return xs


@partial(jax.jit, static_argnums=(0,))
def odeint_ckpt(func, x0, ts, *args):
    """Integrate an ODE forward in time."""
    flat_x0, unravel = ravel_pytree(x0)

    def flat_func(flat_x, t, *args):
        x = unravel(flat_x)
        dx = func(x, t, *args)
        flat_dx, _ = ravel_pytree(dx)
        return flat_dx

    # Solve in flat form
    flat_xs = _odeint_ckpt(flat_func, flat_x0, ts, *args)
    xs = jax.vmap(unravel)(flat_xs)
    return xs


@partial(jax.jit, static_argnums=(0, 2, 3, 4))
def odeint_fixed_step(func, x0, t0, t1, step_size, *args):
    """Integrate an ODE forward in time with a fixed step and end time."""
    # Use `numpy` for purely static operations on static arguments
    # (see: https://github.com/google/jax/issues/5208)
    num_steps = int(np.maximum(np.abs((t1 - t0)/step_size), 1))

    ts = jnp.linspace(t0, t1, num_steps + 1)
    xs = odeint_ckpt(func, x0, ts, *args)
    return xs, ts
