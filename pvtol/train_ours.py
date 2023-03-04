"""
Train an adaptable PVTOL dynamics model via control-oriented meta-learning.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

import argparse
import os
import pickle
import time
import warnings
from functools import partial

import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers

from matplotlib import pyplot as plt

from tqdm.auto import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('seed', help='seed for pseudo-random number generation',
                    type=int)
parser.add_argument('M', help='number of trajectories to sub-sample',
                    type=int)
parser.add_argument('num_steps', help='maximum number of gradient steps',
                    type=int)
parser.add_argument('--use_x64', help='use 64-bit precision',
                    action='store_true')
parser.add_argument('--use_cpu', help='use CPU only',
                    action='store_true')
parser.add_argument('--plot', help='plot useful information for debugging',
                    action='store_true')
args = parser.parse_args()

do_plots = args.plot

# Set precision and device
if args.use_x64:
    jax.config.update('jax_enable_x64', True)
if args.use_cpu:
    jax.config.update('jax_platform_name', 'cpu')

from utils.ode import rk38_step                                 # noqa: E402
from utils.params import epoch                                  # noqa: E402
from utils.pvtol import dynamics as prior, gains_to_params      # noqa: E402
from utils.pytree import tree_normsq                            # noqa: E402
from utils.simulation import simulate_pvtol_parametric          # noqa: E402
from utils.traj import random_ragged_spline                     # noqa: E402

# Initialize PRNG key
key = jax.random.PRNGKey(args.seed)

# Hyperparameters
hparams = {
    'seed':        args.seed,     #
    'use_x64':     args.use_x64,  #
    'num_subtraj': args.M,        # number of trajectories to sub-sample
    'filename':    'pvtol_training_data.pkl',

    # For training the model ensemble
    'ensemble': {
        'num_hlayers':    2,     # number of hidden layers in each model
        'hdim':           32,    # number of hidden units per layer
        'train_frac':     0.75,  # fraction of each trajectory for training
        'batch_frac':     0.25,  # fraction of training data per batch
        'regularizer_l2': 1e-4,  # coefficient for L2-regularization
        'learning_rate':  1e-2,  # step size for gradient optimization
        'num_epochs':     1000,  # number of epochs
    },
    # For meta-training
    'meta': {
        'num_hlayers':       2,             # number of hidden layers
        'hdim':              32,            # number of hidden units per layer
        'train_frac':        0.75,          #
        'learning_rate':     1e-2,          # gradient descent step size
        'num_steps':         args.num_steps,    # max. number of gradient steps
        'regularizer_l2':    1e-4,          # coefficient for L2-regularization
        'regularizer_ctrl':  1e-3,          #
        'T':                 10.,           # time horizon for each reference
        'dt':                1e-2,          # numerical integration time step
        'num_refs':          10,            # number of generated trajectories
        'num_knots':         6,             # knot points per reference spline
        'poly_orders':       (9, 9),        # spline orders for each DOF
        'deriv_orders':      (3, 3),        # smoothness objective for each DOF
        'min_step':          (-2., -2.),    #
        'max_step':          (2., 2.),      #
    },
}

if __name__ == '__main__':
    # DATA PROCESSING ########################################################
    # Load raw data and infer some key dimensions
    with open(hparams['filename'], 'rb') as file:
        raw = pickle.load(file)
    num_traj = raw['x'].shape[0]         # total number of raw trajectories
    num_samples = raw['x'].shape[1] - 1  # number of transitions per trajectory
    num_states = raw['x'].shape[2]       #
    num_inputs = raw['u'].shape[2]       #
    num_dof = num_states // 2            #
    assert 2*num_dof == num_states       #

    # Fixed control gains
    old_control_gains = raw['control_gains']

    # Arrange data in samples of the form `(t, x, u, t_next, x_next)`
    # for each trajectory
    t = raw['t'][:, :-1]
    t_next = raw['t'][:, 1:]
    x = raw['x'][:, :-1]
    x_next = raw['x'][:, 1:]
    u = raw['u'][:, :-1]
    data = {'t': t, 'x': x, 'u': u, 't_next': t_next, 'x_next': x_next}

    # Shuffle and sub-sample trajectories
    if hparams['num_subtraj'] > num_traj:
        warnings.warn('Cannot sub-sample {:d} trajectories! '
                      'Capping at {:d}.'.format(hparams['num_subtraj'],
                                                num_traj))
        hparams['num_subtraj'] = num_traj

    key, subkey = jax.random.split(key, 2)
    shuffled_idx = jax.random.permutation(subkey, num_traj)
    hparams['subtraj_idx'] = shuffled_idx[:hparams['num_subtraj']]
    data = jax.tree_util.tree_map(
        lambda a: jnp.take(a, hparams['subtraj_idx'], axis=0),
        data
    )

    # MODEL ENSEMBLE TRAINING ################################################
    # Controlled ODE
    def ode(x, t, u, params, prior=prior):
        """TODO: docstring."""
        # Each model in the ensemble is a feed-forward neural network with zero
        # output bias; only changes to the model accelerations are considered
        # (model velocities are governed by kinematics)
        h = x
        for W, b in zip(params['W'], params['b']):
            h = jnp.tanh(W@h + b)
        f_ext = jnp.concatenate((jnp.zeros(num_dof), params['A'] @ h))

        # Evaluate dynamics model
        f, B, _ = prior(x)
        dx = f + B@u + f_ext
        return dx

    # Loss function along a trajectory
    def loss(params, regularizer, t, x, u, t_next, x_next, ode=ode):
        """TODO: docstring."""
        num_samples = t.size
        dt = t_next - t
        x_next_est = jax.vmap(rk38_step, (None, 0, 0, 0, 0, None))(
            ode, dt, x, t, u, params
        )
        loss = (jnp.sum((x_next_est - x_next)**2)
                + regularizer*tree_normsq(params)) / num_samples
        return loss

    # Parallel updates for each model in the ensemble
    @partial(jax.jit, static_argnums=(4, 5))
    @partial(jax.vmap, in_axes=(None, 0, None, 0, None, None))
    def step(idx, opt_state, regularizer, batch, get_params, update_opt,
             loss=loss):
        """TODO: docstring."""
        params = get_params(opt_state)
        grads = jax.grad(loss, argnums=0)(params, regularizer, **batch)
        opt_state = update_opt(idx, grads, opt_state)
        return opt_state

    @jax.jit
    @jax.vmap
    def update_ensemble(old_params, old_loss, new_params, batch):
        """TODO: docstring."""
        new_loss = loss(new_params, 0., **batch)  # do not regularize
        best_params = jax.tree_util.tree_multimap(
            lambda x, y: jnp.where(new_loss < old_loss, x, y),
            new_params,
            old_params
        )
        best_loss = jnp.where(new_loss < old_loss, new_loss, old_loss)
        return best_params, best_loss, new_loss

    # Initialize model parameters
    num_models = hparams['num_subtraj']  # one model per trajectory
    num_hlayers = hparams['ensemble']['num_hlayers']
    hdim = hparams['ensemble']['hdim']
    if num_hlayers >= 1:
        shapes = [(hdim, num_states), ] + (num_hlayers-1)*[(hdim, hdim), ]
    else:
        shapes = []
    key, *subkeys = jax.random.split(key, 1 + 2*num_hlayers + 1)
    keys_W = subkeys[:num_hlayers]
    keys_b = subkeys[num_hlayers:-1]
    key_A = subkeys[-1]
    ensemble = {
        # hidden layer weights
        'W': [0.1*jax.random.normal(keys_W[i], (num_models, *shapes[i]))
              for i in range(num_hlayers)],
        # hidden layer biases
        'b': [0.1*jax.random.normal(keys_b[i], (num_models, shapes[i][0]))
              for i in range(num_hlayers)],
        # last layer weights
        'A': 0.1*jax.random.normal(key_A, (num_models, num_dof, hdim))
    }

    # Shuffle samples in time along each trajectory, then split each
    # trajectory into training and validation sets (i.e., for each model)
    key, *subkeys = jax.random.split(key, 1 + num_models)
    subkeys = jnp.asarray(subkeys)
    shuffled_data = jax.tree_util.tree_map(
        lambda a: jax.vmap(jax.random.permutation)(subkeys, a),
        data
    )
    num_train_samples = int(hparams['ensemble']['train_frac'] * num_samples)
    ensemble_train_data = jax.tree_util.tree_map(
        lambda a: a[:, :num_train_samples],
        shuffled_data
    )
    ensemble_valid_data = jax.tree_util.tree_map(
        lambda a: a[:, num_train_samples:],
        shuffled_data
    )

    # Initialize gradient-based optimizer (ADAM)
    learning_rate = hparams['ensemble']['learning_rate']
    batch_size = int(hparams['ensemble']['batch_frac'] * num_train_samples)
    num_batches = num_train_samples // batch_size
    init_opt, update_opt, get_params = optimizers.adam(learning_rate)
    opt_states = jax.vmap(init_opt)(ensemble)
    get_ensemble = jax.jit(jax.vmap(get_params))
    step_idx = 0
    best_idx = jnp.zeros(num_models)

    # Pre-compile before training
    print('ENSEMBLE TRAINING: Pre-compiling ... ', end='', flush=True)
    start = time.time()
    batch = next(epoch(key, ensemble_train_data, batch_size,
                       batch_axis=1, ragged=False))
    _ = step(step_idx, opt_states, hparams['ensemble']['regularizer_l2'],
             batch, get_params, update_opt)
    inf_losses = jnp.broadcast_to(jnp.inf, (num_models,))
    best_ensemble, best_losses, _ = update_ensemble(ensemble, inf_losses,
                                                    ensemble,
                                                    ensemble_valid_data)
    _ = get_ensemble(opt_states)
    end = time.time()
    print('done ({:.2f} s)!'.format(end - start))

    # Do gradient descent
    valid_losses_hist = jnp.zeros((
        hparams['ensemble']['num_epochs'],
        num_models
    ))
    with tqdm(range(hparams['ensemble']['num_epochs'])) as progress_bar:
        for k in progress_bar:
            key, subkey = jax.random.split(key, 2)
            for batch in epoch(subkey, ensemble_train_data, batch_size,
                               batch_axis=1, ragged=False):
                opt_states = step(step_idx, opt_states,
                                  hparams['ensemble']['regularizer_l2'],
                                  batch, get_params, update_opt)
                new_ensemble = get_ensemble(opt_states)
                old_losses = best_losses
                best_ensemble, best_losses, valid_losses = update_ensemble(
                    best_ensemble, best_losses, new_ensemble, batch
                )
                step_idx += 1
                best_idx = jnp.where(old_losses == best_losses,
                                     best_idx, step_idx)
            valid_losses_hist = valid_losses_hist.at[k, :].set(valid_losses)

            # Customize progress bar printed to terminal
            str_format = lambda x: '{:.3e}'.format(x)  # noqa: E731
            progress_bar.set_postfix(
                valid_mean=str_format(jnp.mean(valid_losses_hist[k, :])),
                valid_std=str_format(jnp.std(valid_losses_hist[k, :])),
                best_idx_mean=int(jnp.mean(best_idx)),
                best_idx_std=int(jnp.std(best_idx)),
                best_valid_mean=str_format(jnp.mean(best_losses)),
                best_valid_std=str_format(jnp.std(best_losses)),
            )

    # PLOTS
    if do_plots:
        fig, ax = plt.subplots()
        ax.semilogy(valid_losses_hist)
        # ax.semilogy(best_idx, best_losses, 'ro', markersize=2)
        fig.tight_layout()
        plt.show()

    # META-TRAINING ##########################################################
    # def loss(meta_params, ensemble_params, t_knots, coefs, T, dt,
    #          regularizer_l2, regularizer_ctrl, regularizer_error):

    # Simulate the adaptive control loop for each model in the ensemble and
    # each reference trajectory (i.e., spline coefficients)
    @partial(jax.vmap, in_axes=(None, None, 0, 0, None, None, None))
    @partial(jax.vmap, in_axes=(None, 0, None, None, None, None, None))
    def simulate_all(meta_params, ensemble_params, t_knots, coefs, T, dt,
                     μ_ctrl):
        """TODO: docstring."""
        def cost_func(x, u, x_ref, u_ref):
            dc = jnp.sum((x - x_ref)**2) + μ_ctrl*jnp.sum((u - u_ref)**2)
            return dc

        def dynamics_func(t, x, u, prior=prior, params=ensemble_params):
            f, B, _ = prior(x)
            h = x
            for W, b in zip(params['W'], params['b']):
                h = jnp.tanh(W@h + b)
            f_ext = jnp.concatenate((jnp.zeros(num_dof), params['A'] @ h))
            return f + f_ext, B

        drag = jnp.zeros(3)
        w = 0.

        t, x, u, A, c, x_ref, u_ref = simulate_pvtol_parametric(
            meta_params, T, dt, drag, w, t_knots, coefs,
            dynamics_func=dynamics_func, cost_func=cost_func
        )
        return t, x, u, A, c, x_ref, u_ref

    @partial(jax.jit, static_argnums=(4, 5))
    def loss(meta_params, ensemble_params, t_knots, coefs, T, dt,
             μ_ctrl, μ_l2):
        """TODO: docstring."""
        # Simulate on each model for each reference trajectory
        t, x, u, A, c, x_ref, u_ref = simulate_all(
            meta_params, ensemble_params, t_knots, coefs, T, dt, μ_ctrl
        )
        normalizer = c.shape[0] * c.shape[1] * T
        cost_tracking = jnp.nanmean(c[:, :, -1]) / T
        cost_l2 = tree_normsq((meta_params['W'], meta_params['b']))
        loss = cost_tracking + μ_l2*cost_l2/normalizer
        aux = {
            't': t,
            'x': x,
            'u': u,
            'A': A,
            'c': c,
            'x_ref': x_ref,
            'u_ref': u_ref,
        }
        return loss, aux

    # Initialize meta-model parameters
    num_hlayers = hparams['meta']['num_hlayers']
    hdim = hparams['meta']['hdim']
    if num_hlayers >= 1:
        shapes = [(hdim, num_states), ] + (num_hlayers-1)*[(hdim, hdim), ]
    else:
        shapes = []
    key, *subkeys = jax.random.split(key, 1 + 2*num_hlayers + 2)
    subkeys_W = subkeys[:num_hlayers]
    subkeys_b = subkeys[num_hlayers:-1]
    subkey_control = subkeys[-2]
    subkey_adapt = subkeys[-1]
    meta_params = {
        # hidden layer weights
        'W': [0.1*jax.random.normal(subkeys_W[i], shapes[i])
              for i in range(num_hlayers)],
        # hidden layer biases
        'b': [0.1*jax.random.normal(subkeys_b[i], (shapes[i][0],))
              for i in range(num_hlayers)],
        'control_gains': gains_to_params(old_control_gains['kx'],
                                         old_control_gains['ky'],
                                         old_control_gains['kϕ'],
                                         old_control_gains['cx'],
                                         old_control_gains['cy']),
        'adaptation_gain': 0.1*jax.random.normal(
            subkey_adapt, ((num_inputs*(num_inputs + 1)) // 2,)
        ),
    }

    # Initialize spline coefficients for each reference trajectory
    num_refs = hparams['meta']['num_refs']
    key, *subkeys = jax.random.split(key, 1 + num_refs)
    subkeys = jnp.vstack(subkeys)
    in_axes = (0, None, None, None, None, None, None)
    t_knots, knots, coefs = jax.vmap(random_ragged_spline, in_axes)(
        subkeys,
        hparams['meta']['T'],
        hparams['meta']['num_knots'],
        hparams['meta']['poly_orders'],
        hparams['meta']['deriv_orders'],
        jnp.asarray(hparams['meta']['min_step']),
        jnp.asarray(hparams['meta']['max_step']),
    )
    r_knots = jnp.dstack(knots)

    # Shuffle and split ensemble into training and validation sets
    train_frac = hparams['meta']['train_frac']
    num_train_models = int(train_frac * num_models)
    key, subkey = jax.random.split(key, 2)
    model_idx = jax.random.permutation(subkey, num_models)
    train_model_idx = model_idx[:num_train_models]
    valid_model_idx = model_idx[num_train_models:]
    train_ensemble = jax.tree_util.tree_map(lambda x: x[train_model_idx],
                                            best_ensemble)
    valid_ensemble = jax.tree_util.tree_map(lambda x: x[valid_model_idx],
                                            best_ensemble)

    # Split reference trajectories into training and validation sets
    num_train_refs = int(train_frac * num_refs)
    train_t_knots = jax.tree_util.tree_map(lambda a: a[:num_train_refs],
                                           t_knots)
    train_coefs = jax.tree_util.tree_map(lambda a: a[:num_train_refs], coefs)
    valid_t_knots = jax.tree_util.tree_map(lambda a: a[num_train_refs:],
                                           t_knots)
    valid_coefs = jax.tree_util.tree_map(lambda a: a[num_train_refs:], coefs)

    # Initialize gradient-based optimizer (ADAM)
    init_opt, update_opt, get_params = optimizers.adam(
        hparams['meta']['learning_rate']
    )
    opt_state = init_opt(meta_params)
    step_idx = 0
    best_idx = 0
    best_loss = jnp.inf
    best_meta_params = meta_params

    @partial(jax.jit, static_argnums=(5, 6))
    def step(idx, opt_state, ensemble_params, t_knots, coefs, T, dt,
             μ_ctrl, μ_l2):
        """TODO: docstring."""
        meta_params = get_params(opt_state)
        grads, aux = jax.grad(loss, argnums=0, has_aux=True)(
            meta_params, ensemble_params, t_knots, coefs, T, dt, μ_ctrl, μ_l2
        )
        opt_state = update_opt(idx, grads, opt_state)
        return opt_state, aux

    # Pre-compile before training
    print('META-TRAINING: Pre-compiling ... ', end='', flush=True)
    dt = hparams['meta']['dt']
    T = hparams['meta']['T']
    μ_ctrl = hparams['meta']['regularizer_ctrl']
    μ_l2 = hparams['meta']['regularizer_l2']
    start = time.time()
    _ = step(0, opt_state, train_ensemble, train_t_knots, train_coefs, T, dt,
             μ_ctrl, μ_l2)
    _ = loss(meta_params, valid_ensemble, valid_t_knots, valid_coefs, T, dt,
             0., 0.)
    end = time.time()
    print('done ({:.2f} s)! Now training ...'.format(
          end - start))
    start = time.time()

    # Do gradient descent
    train_loss_hist = jnp.zeros(hparams['meta']['num_steps'])
    valid_loss_hist = jnp.zeros(hparams['meta']['num_steps'])

    with tqdm(range(hparams['meta']['num_steps'])) as progress_bar:
        for _ in tqdm(range(hparams['meta']['num_steps'])):
            opt_state, train_aux = step(
                step_idx, opt_state, train_ensemble, train_t_knots,
                train_coefs, T, dt, μ_ctrl, μ_l2
            )
            new_meta_params = get_params(opt_state)
            train_loss, train_aux = loss(
                new_meta_params, train_ensemble, train_t_knots, train_coefs,
                T, dt, 0., 0.
            )
            valid_loss, valid_aux = loss(
                new_meta_params, valid_ensemble, valid_t_knots, valid_coefs,
                T, dt, 0., 0.
            )
            if valid_loss < best_loss:
                best_meta_params = new_meta_params
                best_loss = valid_loss
                best_idx = step_idx
            train_loss_hist = train_loss_hist.at[step_idx].set(train_loss)
            valid_loss_hist = valid_loss_hist.at[step_idx].set(valid_loss)
            step_idx += 1

            # Customize progress bar printed to terminal
            str_format = lambda x: '{:.3e}'.format(x)  # noqa: E731
            progress_bar.set_postfix(
                train_loss=str_format(train_loss),
                valid_loss=str_format(valid_loss),
                best_idx=best_idx,
                best_loss=str_format(best_loss),
            )

    # Save hyperparameters, ensemble, model, and controller
    output_name = 'seed={:d}_M={:d}'.format(hparams['seed'], num_models)
    results = {
        'train_loss_hist': train_loss_hist,
        'valid_loss_hist': valid_loss_hist,
        'best_step_idx': best_idx,
        'hparams': hparams,
        'ensemble': best_ensemble,
        'meta_params': {
            'W': best_meta_params['W'],
            'b': best_meta_params['b'],
            'control_gains': best_meta_params['control_gains'],
            'adaptation_gain': best_meta_params['adaptation_gain'],
        },
        'meta_params_init': {
            'W': meta_params['W'],
            'b': meta_params['b'],
            'control_gains': meta_params['control_gains'],
            'adaptation_gain': meta_params['adaptation_gain'],
        }
    }
    output_path = os.path.join('pvtol', 'models', 'ours', output_name + '.pkl')
    with open(output_path, 'wb') as file:
        pickle.dump(results, file)

    end = time.time()
    print('done ({:.2f} s)! Best step index: {}'.format(end - start, best_idx))

    # PLOTS
    if do_plots:
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax[0].plot(train_loss_hist[:step_idx])
        ax[0].plot(valid_loss_hist[:step_idx])
        ax[1].semilogy(train_loss_hist[:step_idx])
        ax[1].semilogy(valid_loss_hist[:step_idx])
        fig.tight_layout()
        plt.show()
