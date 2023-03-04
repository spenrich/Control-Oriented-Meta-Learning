"""
Test learned models of the PVTOL system with adaptive closed-loop feedback.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

import argparse
import os
import pickle
import time
from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp

from tqdm.auto import tqdm

from utils.pvtol import gains_to_params
from utils.simulation import simulate_pvtol_parametric
from utils.trajgen import random_ragged_spline

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('seed', help='seed for pseudo-random number generation',
                    type=int)
parser.add_argument('M', help='number of trajectories to sub-sample',
                    type=int)
parser.add_argument('--use_x64', help='use 64-bit precision',
                    action='store_true')
parser.add_argument('--use_cpu', help='use CPU only',
                    action='store_true')
args = parser.parse_args()

# Set precision and device
if args.use_x64:
    jax.config.update('jax_enable_x64', True)
if args.use_cpu:
    jax.config.update('jax_platform_name', 'cpu')

methods = ('ours', 'no adapt, new gains', 'mrr', 'mrr, new gains')

# Seed random numbers (with offset from original seeds to make sure we do not
# sample the same reference trajectories as those in the training set)
seed_test = 19
key = jax.random.PRNGKey(seed_test)

hparams = {
    'use_x64':      args.use_x64,
    'drag':         jnp.array([0.01, 1., 1.]),  # drag coefficients
    'w_min':        0.,      # minimum wind velocity in inertial `x`-direction
    'w_max':        10.,     # maximum wind velocity in inertial `x`-direction
    'beta_shape':   (5, 7),  # shape parameters for beta distribution
    'T':            10.,     # time horizon for each reference trajectory
    'dt':           1e-2,    # numerical integration time step
    'num_traj':     200,     # number of generated reference trajectories
}

if __name__ == '__main__':
    print('Testing ... ', flush=True)
    start = time.time()

    # Generate knots and coefficients for smooth splines fit to random walks in
    # `(x, y)`-space
    key, *subkeys = jax.random.split(key, 1 + hparams['num_traj'])
    subkeys = jnp.vstack(subkeys)
    kwargs = {
        'num_knots':    6,
        'poly_orders':  (9, 9),
        'deriv_orders': (3, 3),
        'min_step':     jnp.array([-2., -2.]),
        'max_step':     jnp.array([2., 2.]),
    }
    spline_generator = partial(random_ragged_spline, **kwargs)
    t_knots, r_knots, coefs = jax.vmap(spline_generator,
                                       (0, None))(subkeys, hparams['T'])
    r_knots = jnp.dstack(r_knots)

    # Sample a wind velocity for each reference trajectory from the test
    # distribution
    key, subkey = jax.random.split(key, 2)
    w_beta = jax.random.beta(subkey, *hparams['beta_shape'],
                             (hparams['num_traj'],))
    w = hparams['w_min'] + (hparams['w_max'] - hparams['w_min'])*w_beta

    # Construct a tracking simulator parallelized across reference and wind
    # velocity pairs
    in_axes = (None, None, None, None, 0, 0, 0)
    simulator = jax.jit(
        lambda p: jax.vmap(simulate_pvtol_parametric, in_axes)(
            p, hparams['T'], hparams['dt'], hparams['drag'], w, t_knots, coefs
        )
    )

    # Setup and execute tests, starting with the initial control gains and
    # no adaptation (which do not vary across seeds or `M`s)
    path = os.path.join('pvtol', 'training_data.pkl')
    with open(path, 'rb') as file:
        old_control_gains = pickle.load(file)['control_gains']
    n, m, d = 6, 2, 2
    params_no_adapt = {
        'W': [jnp.zeros((d, n)), ],
        'b': [jnp.zeros(d)],
        'control_gains': gains_to_params(old_control_gains['kx'],
                                         old_control_gains['ky'],
                                         old_control_gains['kϕ'],
                                         old_control_gains['cx'],
                                         old_control_gains['cy']),
        'adaptation_gain': jnp.zeros(d * (d + 1) // 2),
    }
    t, x, u, A, c, x_ref, u_ref = simulator(params_no_adapt)
    assert jnp.sum(A) == 0.
    results_no_adapt = {'t': t, 'x': x, 'u': u, 'A': A, 'c': c,
                        'x_ref': x_ref, 'u_ref': u_ref}

    # Initialize dictionary to store test results
    tests = {
        'hparams': deepcopy(hparams),
        'results': {method: {} for method in methods},
    }
    tests['results']['no adapt'] = deepcopy(results_no_adapt)

    # Load meta-learned dynamics model features
    prefix = 'seed={}_M={}'.format(args.seed, args.M)
    path = os.path.join('pvtol', 'models', 'ours', prefix + '.pkl')
    with open(path, 'rb') as file:
        model_ours = pickle.load(file)
    path = os.path.join('pvtol', 'models', 'mrr', prefix + '.pkl')
    with open(path, 'rb') as file:
        model_mrr = pickle.load(file)

    # Loop over all model/gain configurations
    for method in tqdm(methods):
        if method == 'ours':
            params = model_ours['meta_params']
        elif method == 'no adapt, new gains':
            params = {
                'W': [0. * W for W in model_ours['meta_params']['W']],
                'b': [0. * b for b in model_ours['meta_params']['b']],
                'control_gains':
                    model_ours['meta_params']['control_gains'],
                'adaptation_gain':
                    model_ours['meta_params']['adaptation_gain'],
            }
        elif method == 'mrr':
            params = {
                'W': model_mrr['meta_params']['W'],
                'b': model_mrr['meta_params']['b'],
                'control_gains':
                    model_ours['meta_params_init']['control_gains'],
                'adaptation_gain':
                    model_ours['meta_params_init']['adaptation_gain'],
            }
        elif method == 'mrr, new gains':
            params = {
                'W': model_mrr['meta_params']['W'],
                'b': model_mrr['meta_params']['b'],
                'control_gains':
                    model_ours['meta_params']['control_gains'],
                'adaptation_gain':
                    model_ours['meta_params']['adaptation_gain'],
            }
        else:
            raise ValueError('Unrecognized method.')

        # Simulate
        t, x, u, A, c, x_ref, u_ref = simulator(params)
        tests['results'][method]['t'] = t
        tests['results'][method]['x'] = x
        tests['results'][method]['u'] = u
        tests['results'][method]['A'] = A
        tests['results'][method]['c'] = c
        tests['results'][method]['x_ref'] = x_ref
        tests['results'][method]['u_ref'] = u_ref

    # Save test results
    directory = os.path.join('pvtol', 'results')
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, prefix + '.pkl')
    with open(path, 'wb') as file:
        pickle.dump(tests, file)

end = time.time()
print('done!', flush=True)
