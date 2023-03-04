"""
Dynamics, trajectory generation, and feedback control for the PVTOL system.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""


import jax
from jax import numpy as jnp

# Define an invertible function that maps reals to strictly positive reals
posfunc = jnp.exp
posfunc_inv = jnp.log


def dynamics(s):
    """Evaluate the control-affine terms of the PVTOL dynamics."""
    x, y, ϕ, dx, dy, dϕ = s
    sinϕ, cosϕ = jnp.sin(ϕ), jnp.cos(ϕ)
    g = 9.81
    ε = 0.
    f = jnp.array([dx, dy, dϕ, 0., -g, 0.])
    B = jnp.array([
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [-sinϕ, ε*cosϕ],
        [cosϕ, ε*sinϕ],
        [0., 1.]
    ])
    # B_pseudo = jax.scipy.linalg.solve(B.T @ B, B.T, assume_a='pos')
    B_pseudo = B.T
    return f, B, B_pseudo


def disturbance(s, w, r, β):
    """Compute the disturbance force due to wind on the PVTOL system."""
    x, y, ϕ, dx, dy, dϕ = s

    # Rotation matrix of body w.r.t. inertial frame
    sinϕ, cosϕ = jnp.sin(ϕ), jnp.cos(ϕ)
    R = jnp.array([[cosϕ, -sinϕ],
                   [sinϕ, cosϕ]])

    # Linear drag
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
    return f_ext


def params_to_gains(θ, g=9.81):
    """Map an unconstrained parameter vector to valid PVTOL control gains."""
    θ_kx, θ_ky, θ_kϕ, θ_cx, θ_cy = jnp.split(θ, 5)
    kx = posfunc(θ_kx)
    ky = posfunc(θ_ky)
    kϕ = posfunc(θ_kϕ)
    cx = posfunc(θ_cx)
    cy = posfunc(θ_cy)
    cy = g * cy / (1 + cy[0] + cy[1])  # cy[0] + cy[1] < g
    return kx, ky, kϕ, cx, cy


def gains_to_params(kx, ky, kϕ, cx, cy, g=9.81):
    """Map valid PVTOL control gains to an unconstrained parameter vector."""
    θ_kx = posfunc_inv(kx)
    θ_ky = posfunc_inv(ky)
    θ_kϕ = posfunc_inv(kϕ)
    θ_cx = posfunc_inv(cx)
    θ_cy = posfunc_inv(cy / (g - cy[0] - cy[1]))  # cy[0] + cy[1] < g
    θ = jnp.concatenate((θ_kx, θ_ky, θ_kϕ, θ_cx, θ_cy))
    return θ


def corrected_feedback(s, kx, ky, kϕ, cx, cy, g=9.81, ε=0., z=(0., 0.),
                       dz=(0., 0.), ddz=(0., 0.), d3z=(0., 0.), d4z=(0., 0.)):
    """Compute a tracking control feedback signal and Lyapunov function."""
    x, y, ϕ, dx, dy, dϕ = s
    sinϕ, cosϕ = jnp.sin(ϕ), jnp.cos(ϕ)

    # Change of variables
    x = x - ε*sinϕ
    y = y + ε*(cosϕ - 1)
    dx = dx - ε*dϕ*cosϕ
    dy = dy - ε*dϕ*sinϕ

    # Element-wise sigmoid function and its derivatives
    σ = jnp.tanh
    dσ = lambda x: 1 / (jnp.cosh(x)**2)                # noqa: E731
    ddσ = lambda x: -2*jnp.tanh(x) / (jnp.cosh(x)**2)  # noqa: E731

    Kx = jnp.array([[kx[0], kx[1]],
                    [0., kx[1]]])
    Ky = jnp.array([[ky[0], ky[1]],
                    [0., ky[1]]])

    wx = Kx @ jnp.array([x - z[0], dx - dz[0]])
    wy = Ky @ jnp.array([y - z[1], dy - dz[1]])
    vx = ddz[0] - jnp.dot(cx, σ(wx))
    vy = ddz[1] - jnp.dot(cy, σ(wy))

    u1 = jnp.sqrt(vx**2 + (vy+g)**2)
    ϕ_des = jnp.arctan(-vx / (vy+g))
    ddx = -u1*sinϕ
    ddy = u1*cosϕ - g

    dwx = Kx @ jnp.array([dx - dz[0], ddx - ddz[0]])
    dwy = Ky @ jnp.array([dy - dz[1], ddy - ddz[1]])
    dvx = d3z[0] - jnp.dot(cx, dσ(wx)*dwx)
    dvy = d3z[1] - jnp.dot(cy, dσ(wy)*dwy)

    du1 = (vx*dvx + (vy + g)*dvy) / u1
    dϕ_des = (vx*dvy - dvx*(vy + g)) / (u1**2)
    d3x = -du1*sinϕ - u1*dϕ*cosϕ
    d3y = du1*cosϕ - u1*dϕ*sinϕ

    ddwx = Kx @ jnp.array([ddx - ddz[0], d3x - d3z[0]])
    ddwy = Ky @ jnp.array([ddy - ddz[1], d3y - d3z[1]])
    ddvx = d4z[0] - jnp.dot(cx, ddσ(wx)*(dwx**2) + dσ(wx)*ddwx)
    ddvy = d4z[1] - jnp.dot(cy, ddσ(wy)*(dwy**2) + dσ(wy)*ddwy)

    ddϕ_des = (vx*ddvy - ddvx*(vy+g) - 2*dϕ*(vx*dvx + (vy+g)*dvy)) / (u1**2)
    u2 = ddϕ_des - kϕ[0]*(ϕ - ϕ_des) - kϕ[1]*(dϕ - dϕ_des)
    u = jnp.array([
        u1 + ε*(dϕ**2),
        u2
    ])

    # Lyapunov function components
    Vx = jnp.dot(cx, jnp.log(jnp.cosh(wx))) + 0.5*kx[0]*((dx - dz[0])**2)
    Vy = jnp.dot(cy, jnp.log(jnp.cosh(wy))) + 0.5*ky[0]*((dy - dz[1])**2)

    Q = jnp.array([
        [kϕ[0]*(kϕ[0] + 1) + kϕ[1]**2, kϕ[1]],
        [kϕ[1],                        kϕ[0] + 1]
    ])
    wϕ = jnp.array([ϕ - ϕ_des, dϕ - dϕ_des])
    Vϕ = 0.5 * jnp.dot(wϕ, Q @ wϕ)

    V = jnp.array([Vx, Vy, Vϕ])

    return u, jnp.sum(V), V


def double_integrator_feedback(x, dx, c, k, σ=jnp.tanh,
                               r=0., dr=0., ddr=0., d3r=0., d4r=0.):
    """Compute a globally stabilizing feedback law for a double integrator."""
    # Sigmoid function and its derivatives
    dσ = jax.grad(σ)
    ddσ = jax.grad(dσ)

    # Error signal
    e = x - r
    de = dx - dr

    # Control input
    z = k[1]*de
    w = k[0]*e + z
    dde = -c[0]*σ(w) - c[1]*σ(z)
    u = ddr + dde

    # First derivative of control input
    dz = k[1]*dde
    dw = k[0]*de + dz
    d3e = -c[0]*dσ(w)*dw - c[1]*dσ(z)*dz
    du = d3r + d3e

    # Second derivative of control input
    ddz = k[1]*d3e
    ddw = k[0]*dde + ddz
    d4e = (-c[0]*(ddσ(w)*(dw**2) + dσ(w)*ddw)
           - c[1]*(ddσ(z)*(dz**2) + dσ(z)*ddz))
    ddu = d4r + d4e

    return u, du, ddu


def flat_to_rot(z, dz, ddz, d3z, d4z, g=9.81):
    """Map flat outputs to rotational states and inputs."""
    wx, dwx, ddwx = -ddz[0], -d3z[0], -d4z[0]
    wy, dwy, ddwy = ddz[1] + g, d3z[1], d4z[1]
    u0 = jnp.sqrt(wx**2 + wy**2)
    ϕ = jnp.arctan(wx / wy)
    dϕ = (dwx*wy - wx*dwy) / (u0**2)
    ddϕ = (ddwx*wy - wx*ddwy - 2*dϕ*(wx*dwx + wy*dwy)) / (u0**2)
    return u0, ϕ, dϕ, ddϕ


def flat_to_state(z, dz, ddz, d3z, d4z):
    """Map flat outputs to states and inputs."""
    u0, ϕ, dϕ, ddϕ = flat_to_rot(z, dz, ddz, d3z, d4z)
    s_ref = jnp.array([z[0], z[1], ϕ, dz[0], dz[1], dϕ])
    ds_ref = jnp.array([dz[0], dz[1], dϕ, ddz[0], ddz[1], ddϕ])
    u_ref = jnp.array([u0, ddϕ])
    return s_ref, ds_ref, u_ref


def reference_flat(t, flat_func):
    """Compute the required derivatives of the flat output."""
    flat_vel = jax.jacfwd(flat_func)
    flat_acc = jax.jacfwd(flat_vel)
    flat_jerk = jax.jacfwd(flat_acc)
    flat_snap = jax.jacfwd(flat_jerk)
    z = flat_func(t)
    dz = flat_vel(t)
    ddz = flat_acc(t)
    d3z = flat_jerk(t)
    d4z = flat_snap(t)
    return z, dz, ddz, d3z, d4z


def reference_state(t, flat_func):
    """Compute the state and input from a flat output signal."""
    z, dz, ddz, d3z, d4z = reference_flat(t, flat_func)
    s_ref, _, u_ref = flat_to_state(z, dz, ddz, d3z, d4z)
    return s_ref, u_ref


def feedback(s, kx, ky, kϕ, cx, cy, z=(0., 0.), dz=(0., 0.), ddz=(0., 0.),
             d3z=(0., 0.), d4z=(0., 0.)):
    """Compute the PVTOL tracking feedback signal given the flat output."""
    ε = 0.
    x, y, ϕ, dx, dy, dϕ = s
    sinϕ, cosϕ = jnp.sin(ϕ), jnp.cos(ϕ)

    # Change of variables
    x = x - ε*sinϕ
    y = y + ε*(cosϕ - 1)
    dx = dx - ε*dϕ*cosϕ
    dy = dy - ε*dϕ*sinϕ

    # Sigmoid functions
    σx = jnp.tanh
    σy = jnp.tanh

    # Virtual control inputs for (x,y)-subsystem
    vx, dvx, ddvx = double_integrator_feedback(x, dx, cx, kx, σx, z[0], dz[0],
                                               ddz[0], d3z[0], d4z[0])
    vy, dvy, ddvy = double_integrator_feedback(y, dy, cy, ky, σy, z[1], dz[1],
                                               ddz[1], d3z[1], d4z[1])
    v = jnp.array([vx, vy])
    dv = jnp.array([dvx, dvy])
    ddv = jnp.array([ddvx, ddvy])

    # Control inputs
    u0, ϕ_des, dϕ_des, ddϕ_des = flat_to_rot(z, dz, v, dv, ddv)
    u = jnp.array([
        u0 + ε*(dϕ**2),
        ddϕ_des - kϕ[0]*(ϕ - ϕ_des) - kϕ[1]*(dϕ - dϕ_des)
    ])
    return u, u0, ϕ_des, dϕ_des, ddϕ_des


def lyapunov(s, kx, ky, kϕ, cx, cy,
             hx=lambda x: jnp.log(jnp.cosh(x)),
             hy=lambda y: jnp.log(jnp.cosh(y))):
    """Compute the control Lyapunov function for the PVTOL system."""
    x, y, ϕ, dx, dy, dϕ = s

    Vx = cx[0]*hx(kx[0]*x + kx[1]*dx) + cx[1]*hx(kx[1]*dx) + (kx[0]/2)*(dx**2)
    Vy = cy[0]*hy(ky[0]*y + ky[1]*dy) + cy[1]*hy(ky[1]*dy) + (ky[0]/2)*(dy**2)

    P = jnp.array([
        [kϕ[0]*(kϕ[0] + 1) + kϕ[1]**2, kϕ[1]],
        [kϕ[1],                        kϕ[0] + 1]
    ])
    sϕ = jnp.array([ϕ, dϕ])
    Vϕ = 0.5 * jnp.dot(sϕ, P @ sϕ)

    V = Vx + Vy + Vϕ
    return V


def control_lyapunov(s, t, kx, ky, kϕ, cx, cy, flat_func):
    """Compute the control Lyapunov function for the PVTOL system."""
    # Reference trajectory (flat output space)
    z, dz, ddz, d3z, d4z = reference_flat(t, flat_func)

    # Closed-loop feedback for nominal system
    u, u0, ϕ_des, dϕ_des, ddϕ_des = feedback(s, kx, ky, kϕ, cx, cy,
                                             z, dz, ddz, d3z, d4z)

    # Lyapunov function gradient for error dynamics
    r = jnp.array([z[0], z[1], ϕ_des, dz[0], dz[1], dϕ_des])
    e = s - r
    dV = jax.grad(lyapunov, argnums=0)(e, kx, ky, kϕ, cx, cy)

    return u, e, dV


def adaptation(s, Ψ, Γ, kx, ky, kϕ, cx, cy):
    """Evaluate the adaptation dynamics for the closed-loop PVTOL system."""
    _, B, B_pseudo = dynamics(s)
    dVds = jax.grad(lyapunov, argnums=(0,))(s, kx, ky, kϕ, cx, cy)
    da = Γ @ Ψ.T @ B_pseudo.T @ B.T @ dVds
    return da


def closed_loop(t, s, a, w, r, β, kx, ky, kϕ, cx, cy, ka,
                feature_func, flat_func):
    """Evaluate the total dynamics for the adaptive closed-loop PVTOL."""
    # Reference trajectory (flat output space)
    z, dz, ddz, d3z, d4z = reference_flat(t, flat_func)

    # Closed-loop feedback for nominal system + certainty-equivalent term
    u, *_ = feedback(s, kx, ky, kϕ, cx, cy, z, dz, ddz, d3z, d4z)
    f, B, B_pseudo = dynamics(s)
    Ψ = feature_func(s)
    u -= B_pseudo @ Ψ @ a

    # Adaptation
    dV = jax.grad(lyapunov, argnums=0)(s, kx, ky, kϕ, cx, cy)
    da = ka * (Ψ.T @ B_pseudo.T @ B.T @ dV)

    # Dynamics
    f_ext = disturbance(s, w, r, β)
    ds = f + B@u + f_ext

    return ds, da
