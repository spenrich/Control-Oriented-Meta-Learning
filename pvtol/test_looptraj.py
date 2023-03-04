"""
Test learned models of the PVTOL system on a double loop-the-loop trajectory.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

import os
import pickle
from copy import deepcopy

import jax
import jax.numpy as jnp

from tqdm import tqdm

from utils.simulation import simulate_pvtol_timevarying

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')

# Hyperparameters
hparams = {
    'seed':     1,      # training seed
    'M':        10,     # number of trajectories in training set
    'T':        20.,    # simulation time horizon
    'dt':       0.01,   # simulation time discretization
    't_max':    15.,    # time that wind velocity reaches its peak
    'w_max':    8.,     # maximum wind velocity
    'drag':     jnp.array([0.01, 1., 1.]),  # drag coefficients
}


# Define the double loop-the-loop trajectory
def loop_traj(t):
    """Compute `(x, y)`-coordinates along a double loop-the-loop trajectory."""
    T = 10.     # loop period
    d = 5       # displacement along `x` from `t=0` to `t=T`
    w = 3.      # loop width (upper bound)
    h = 5.      # loop height
    x = w*jnp.sin(2*jnp.pi * t/T) + d*(t/T)
    y = (h/2)*(1 - jnp.cos(2*jnp.pi * t/T))
    r = jnp.squeeze(jnp.column_stack((x, y)))
    return r


# Define a time-varying wind disturbance
def wind_func(t, t_max, w_max, T):
    """Evaluate a time-varying wind velocity.

    The wind velocity peaks at `t_max` with a value of `w_max`.
    """
    w_min = 0.
    spread = 0.5
    a = 1/spread
    b = a*(T/t_max - 1)
    s = (t/t_max)**a * (jnp.abs(T - t)/(T - t_max))**b
    support = jnp.logical_and(t > 0., t < T)
    w = jnp.where(support, w_min + s*(w_max - w_min), w_min)
    return w


# Construct a tracking simulator
@jax.jit
def simulator(params, drag, t_max, w_max):
    """Simulate the PVTOL system with a time-varying wind disturbance."""
    T, dt = hparams['T'], hparams['dt']
    t, x, u, x_ref, u_ref, aux = simulate_pvtol_timevarying(
        params, T, dt, drag,
        lambda t: wind_func(t, t_max, w_max, T),
        loop_traj
    )
    V = aux['V']
    return t, x, u, x_ref, u_ref, V


# Load learned model configurations
params = {}
prefix = 'seed={}_M={}'.format(hparams['seed'], hparams['M'])

filename = os.path.join('pvtol', 'models', 'ours', prefix + '.pkl')
with open(filename, 'rb') as file:
    data_ours = pickle.load(file)
params['ours'] = data_ours['meta_params']
params_init = data_ours['meta_params_init']

filename = os.path.join('pvtol', 'models', 'mrr', prefix + '.pkl')
with open(filename, 'rb') as file:
    data_mrr = pickle.load(file)
params['mrr'] = deepcopy(data_mrr['meta_params'])
params['mrr']['control_gains'] = params_init['control_gains']
params['mrr']['adaptation_gain'] = params_init['adaptation_gain']
params['mrr, new gains'] = deepcopy(data_mrr['meta_params'])
params['mrr, new gains']['control_gains'] = params['ours']['control_gains']
params['mrr, new gains']['adaptation_gain'] = params['ours']['adaptation_gain']

params['no adapt'] = {
    'W': [0. * W for W in params['ours']['W']],
    'b': [0. * b for b in params['ours']['b']],
    'control_gains': params_init['control_gains'],
    'adaptation_gain': params['ours']['adaptation_gain'],
}
params['no adapt, new gains'] = {
    'W': [0. * W for W in params['ours']['W']],
    'b': [0. * b for b in params['ours']['b']],
    'control_gains': params['ours']['control_gains'],
    'adaptation_gain': params['ours']['adaptation_gain'],
}

# Simulate each model configuration and save the results
results = {}
methods = ('no adapt', 'no adapt, new gains', 'mrr', 'mrr, new gains', 'ours')
for method in tqdm(methods):
    t, x, u, x_ref, u_ref, V = simulator(
        params[method], hparams['drag'], hparams['t_max'], hparams['w_max']
    )
    results[method] = {
        't': t, 'x': x, 'u': u, 'x_ref': x_ref, 'u_ref': u_ref, 'V': V,
    }
results['hparams'] = hparams
results['w'] = wind_func(t, hparams['t_max'], hparams['w_max'], hparams['T'])
path = os.path.join('pvtol', 'results_looptraj.pkl')
with open(path, 'wb') as file:
    pickle.dump(results, file)
