"""
Simulation functions for the PVTOL system.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

import jax
import jax.numpy as jnp

import numpy as np

from .ode import odeint_ckpt as odeint
from .params import params_to_posdef
from .pvtol import corrected_feedback, flat_to_state, params_to_gains
from .trajgen import spline_factory


def simulate(T, dt, dynamics_func, control_func, reference_func,
             feature_func=None, adapt_func=None, model_func=None,
             cost_func=None):
    """Simulate an adaptive closed-loop system in continuous time."""
    if model_func is None:
        model_func = dynamics_func
    if cost_func is None:
        cost_func = lambda x, u, x_ref, u_ref: 0.  # noqa: E731

    # Use `numpy` for purely static operations on static arguments
    # (see: https://github.com/google/jax/issues/5208)
    num_steps = int(np.maximum(T/dt, 1))
    t = jnp.linspace(0., T, num_steps + 1)

    # Infer some shapes
    x_ref, u_ref, *aux_ref = jax.vmap(reference_func)(t)
    m = 1 if u_ref.ndim == 1 else u_ref.shape[1]
    if feature_func is None or adapt_func is None:
        feature_func = lambda x: jnp.zeros(m)            # noqa: E731
        adapt_func = lambda y, *args: jnp.zeros((m, m))  # noqa: E731
        p = m
    else:
        y0_ref = jnp.asarray(feature_func(x_ref[0]))
        p = 1 if y0_ref.ndim == 0 else y0_ref.shape[0]

    def ode(xAc, t):
        x, A, c = xAc
        y = feature_func(x)
        x_ref, u_ref, *aux_ref = reference_func(t)
        u, *aux_ctrl = control_func(x, x_ref, u_ref, *aux_ref)
        u -= A @ jnp.atleast_1d(y)
        f, B = model_func(t, x, u)
        dA = adapt_func(y, f, B, *aux_ctrl)
        f_true, B_true = dynamics_func(t, x, u)
        dx = f_true + B_true@u
        dc = cost_func(x, u, x_ref, u_ref)
        return dx, dA, dc

    x0 = x_ref[0]
    A0 = jnp.zeros((m, p))
    c0 = 0.
    x, A, c = odeint(ode, (x0, A0, c0), t)
    u, *aux_ctrl = jax.vmap(control_func)(x, x_ref, u_ref, *aux_ref)
    return t, x, u, A, c, x_ref, u_ref, aux_ref, aux_ctrl


def simulate_pvtol(T, dt, kx, ky, kϕ, cx, cy, P, drag, w, t_knots, coefs,
                   dynamics_func=None, feature_func=None, cost_func=None):
    """Simulate the PVTOL system."""
    g = 9.81
    ε = 0.
    r = 1.

    flat_pos = spline_factory(t_knots, *coefs)
    flat_vel = jax.jacfwd(flat_pos)
    flat_acc = jax.jacfwd(flat_vel)
    flat_jerk = jax.jacfwd(flat_acc)
    flat_snap = jax.jacfwd(flat_jerk)

    def model_func(t, s, u, g=g, ε=ε):
        """Compute the state derivative for the PVTOL system."""
        x, y, ϕ, dx, dy, dϕ = s
        sinϕ, cosϕ = jnp.sin(ϕ), jnp.cos(ϕ)
        f = jnp.array([dx, dy, dϕ, 0., -g, 0.])
        B = jnp.array([
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [-sinϕ, ε*cosϕ],
            [cosϕ, ε*sinϕ],
            [0., 1.]
        ])
        return f, B

    if dynamics_func is None:
        def dynamics_func(t, s, u, w=w, r=r, β=drag):
            """Compute the state derivative for the disturbed PVTOL system."""
            f, B = model_func(t, s, u)
            x, y, ϕ, dx, dy, dϕ = s

            # Rotation matrix of body w.r.t. inertial frame
            sinϕ, cosϕ = jnp.sin(ϕ), jnp.cos(ϕ)
            R = jnp.array([[cosϕ, -sinϕ],
                           [sinϕ, cosϕ]])

            # Linear drag
            vi = jnp.array([dx - w, dy])  # relative air velocity (inertial)
            vb = R.T @ vi                 # relative air velocity (body frame)
            db = -β[:2]*vb*jnp.abs(vb)    # drag force (body frame)
            di = R @ db                   # drag force (inertial frame)

            # Rotational drag
            vR = r*dϕ - vb[1]   # relative air velocity (body frame, right arm)
            vL = -r*dϕ - vb[1]  # relative air velocity (body frame, left arm)
            dr = -β[2]*(vR*jnp.abs(vR) - vL*jnp.abs(vL))

            # Assemble
            f_ext = jnp.array([0., 0., 0., di[0], di[1], dr])
            return f + f_ext, B

    def reference_func(t):
        z = flat_pos(t)
        dz = flat_vel(t)
        ddz = flat_acc(t)
        d3z = flat_jerk(t)
        d4z = flat_snap(t)
        s_ref, ds_ref, u_ref = flat_to_state(z, dz, ddz, d3z, d4z)
        return s_ref, u_ref, ds_ref, z, dz, ddz, d3z, d4z

    def lyapunov_func(*args, **kwargs):
        u, V, V_components = corrected_feedback(*args, **kwargs)
        return V

    def control_func(s, s_ref, u_ref, ds_ref, z, dz, ddz, d3z, d4z):
        u, V, _ = corrected_feedback(s, kx, ky, kϕ, cx, cy, g, ε,
                                     z, dz, ddz, d3z, d4z)
        dVdx = jax.grad(lyapunov_func, argnums=0)(s, kx, ky, kϕ, cx, cy, g, ε,
                                                  z, dz, ddz, d3z, d4z)
        return u, V, dVdx

    def adapt_func(y, f, B, V, dVdx, P=P):
        dA = jnp.outer(P @ B.T @ dVdx, y)
        return dA

    t, x, u, A, c, x_ref, u_ref, *_ = simulate(T, dt, dynamics_func,
                                               control_func, reference_func,
                                               feature_func, adapt_func,
                                               model_func, cost_func)

    return t, x, u, A, c, x_ref, u_ref


def simulate_pvtol_parametric(params, T, dt, drag, w, t_knots, coefs,
                              dynamics_func=None, cost_func=None):
    """Simulate the PVTOL system with parameterized features and gains."""
    g = 9.81
    kx, ky, kϕ, cx, cy = params_to_gains(params['control_gains'], g)
    P = params_to_posdef(params['adaptation_gain'])
    Ws = params['W']
    bs = params['b']

    def feature_func(x, Ws=Ws, bs=bs):
        y = x
        for W, b in zip(Ws, bs):
            y = jnp.tanh(W@y + b)
        return y

    t, x, u, A, c, x_ref, u_ref = simulate_pvtol(T, dt, kx, ky, kϕ, cx, cy, P,
                                                 drag, w, t_knots, coefs,
                                                 dynamics_func, feature_func,
                                                 cost_func)
    return t, x, u, A, c, x_ref, u_ref


def simulate_pvtol_timevarying(params, T, dt, drag, w_func, flat_pos,
                               cost_func=None):
    """Simulate the PVTOL system."""
    g = 9.81
    ε = 0.
    r = 1.
    flat_vel = jax.jacfwd(flat_pos)
    flat_acc = jax.jacfwd(flat_vel)
    flat_jerk = jax.jacfwd(flat_acc)
    flat_snap = jax.jacfwd(flat_jerk)

    kx, ky, kϕ, cx, cy = params_to_gains(params['control_gains'], g)
    P = params_to_posdef(params['adaptation_gain'])
    Ws = params['W']
    bs = params['b']

    def feature_func(x, Ws=Ws, bs=bs):
        y = x
        for W, b in zip(Ws, bs):
            y = jnp.tanh(W@y + b)
        return y

    def model_func(t, s, u, g=g, ε=ε):
        """Compute the state derivative for the PVTOL system."""
        x, y, ϕ, dx, dy, dϕ = s
        sinϕ, cosϕ = jnp.sin(ϕ), jnp.cos(ϕ)
        f = jnp.array([dx, dy, dϕ, 0., -g, 0.])
        B = jnp.array([
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [-sinϕ, ε*cosϕ],
            [cosϕ, ε*sinϕ],
            [0., 1.]
        ])
        return f, B

    def dynamics_func(t, s, u, r=r, β=drag):
        """Compute the state derivative for the disturbed PVTOL system."""
        f, B = model_func(t, s, u)
        x, y, ϕ, dx, dy, dϕ = s

        # Rotation matrix of body w.r.t. inertial frame
        sinϕ, cosϕ = jnp.sin(ϕ), jnp.cos(ϕ)
        R = jnp.array([[cosϕ, -sinϕ],
                       [sinϕ, cosϕ]])

        # Linear drag
        w = w_func(t)
        vi = jnp.array([dx - w, dy])  # relative air velocity (inertial frame)
        vb = R.T @ vi                 # relative air velocity (body frame)
        db = -β[:2]*vb*jnp.abs(vb)    # drag force (body frame)
        di = R @ db                   # drag force (inertial frame)

        # Rotational drag
        vR = r*dϕ - vb[1]   # relative air velocity (body frame, right arm)
        vL = -r*dϕ - vb[1]  # relative air velocity (body frame, left arm)
        dr = -β[2]*(vR*jnp.abs(vR) - vL*jnp.abs(vL))

        # Assemble
        f_ext = jnp.array([0., 0., 0., di[0], di[1], dr])
        return f + f_ext, B

    def reference_func(t):
        z = flat_pos(t)
        dz = flat_vel(t)
        ddz = flat_acc(t)
        d3z = flat_jerk(t)
        d4z = flat_snap(t)
        s_ref, ds_ref, u_ref = flat_to_state(z, dz, ddz, d3z, d4z)
        return s_ref, u_ref, ds_ref, z, dz, ddz, d3z, d4z

    def lyapunov_func(*args, **kwargs):
        u, V, V_components = corrected_feedback(*args, **kwargs)
        return V

    def control_func(s, s_ref, u_ref, ds_ref, z, dz, ddz, d3z, d4z):
        u, V, _ = corrected_feedback(s, kx, ky, kϕ, cx, cy, g, ε,
                                     z, dz, ddz, d3z, d4z)
        dVdx = jax.grad(lyapunov_func, argnums=0)(s, kx, ky, kϕ, cx, cy, g, ε,
                                                  z, dz, ddz, d3z, d4z)
        return u, V, dVdx

    def adapt_func(y, f, B, V, dVdx, P=P):
        dA = jnp.outer(P @ B.T @ dVdx, y)
        return dA

    t, s, u, A, c, s_ref, u_ref, aux_ref, aux_ctrl = simulate(
        T, dt, dynamics_func, control_func, reference_func, feature_func,
        adapt_func, model_func, cost_func
    )

    V, dVdx = aux_ctrl
    ds_ref, z, dz, ddz, d3z, d4z = aux_ref
    fs, Bs = jax.vmap(dynamics_func)(t, s, u)
    ds = fs + jnp.squeeze(Bs@jnp.expand_dims(u, -1))
    V_dot = jnp.sum(dVdx * (ds - ds_ref), axis=-1)

    def estimator(t, s, u, A):
        f_true, B_true = dynamics_func(t, s, u)
        f, B = model_func(t, s, u)
        y = feature_func(s)
        f_ext_est = B @ A @ y
        f_ext = f_true - f
        return f_ext_est, f_ext

    f_ext_est, f_ext = jax.vmap(estimator)(t, s, u, A)
    aux = {'V': V, 'c': c, 'f_ext': f_ext, 'f_ext_est': f_ext_est,
           'V_dot': V_dot}
    return t, s, u, s_ref, u_ref, aux
