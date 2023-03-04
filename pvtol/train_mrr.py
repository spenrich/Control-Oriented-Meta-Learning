"""
Train an adaptable PVTOL dynamics model via meta-ridge regression (MRR).

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
from jax.example_libraries.optimizers import adam

from tqdm.auto import tqdm

from utils.pvtol import dynamics
from utils.pytree import tree_normsq

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
args = parser.parse_args()


# Set precision and device
if args.use_x64:
    jax.config.update('jax_enable_x64', True)
if args.use_cpu:
    jax.config.update('jax_platform_name', 'cpu')


# Initialize PRNG key
key = jax.random.PRNGKey(args.seed)

# Hyperparameters
system_name = 'pvtol'
hparams = {
    'seed':        args.seed,     #
    'use_x64':     args.use_x64,  #
    'num_subtraj': args.M,        # number of trajectories to sub-sample
    'filename':    'training_data.pkl',

    'num_hlayers':        2,      # number of hidden layers
    'hdim':               32,     # number of hidden units per layer
    'train_frac':         0.75,   # fraction per trajectory for training
    'ridge_frac':         0.25,   # (fraction of samples used in the ridge
                                  #  regression solution per trajectory)
    'regularizer_l2':     1e-4,   # coefficient for L2-regularization
    'regularizer_ridge':  1e-6,   # (coefficient for L2-regularization of
                                  #  least-squares solution)
    'learning_rate':      1e-2,   # step size for gradient optimization
    'num_steps':          args.num_steps,   # number of epochs
}


if __name__ == '__main__':
    # DATA PROCESSING #########################################################
    # Load raw data and infer some key dimensions.
    path = os.path.join(system_name, hparams['filename'])
    with open(path, 'rb') as file:
        raw = pickle.load(file)
    num_traj = raw['x'].shape[0]         # total number of raw trajectories
    num_samples = raw['x'].shape[1] - 1  # number of transitions per trajectory
    num_states = raw['x'].shape[2]       #
    num_inputs = raw['u'].shape[2]       #
    num_dof = num_states // 2            #
    assert 2*num_dof == num_states       #

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

    # META-TRAIN MODEL ########################################################
    def feedforward(x, Ws, bs):
        """TODO: docstring."""
        h = x
        for W, b in zip(Ws[:-1], bs[:-1]):
            h = jnp.tanh(W@h + b)
        y = Ws[-1]@h + bs[-1]
        return y

    def euler_residual_coefficients(params, t, x, u, t_next, x_next):
        """TODO: docstring."""
        dt = t_next - t
        dx = x_next - x
        f, B, _ = dynamics(x)
        y = jnp.tanh(feedforward(x, params['W'], params['b']))

        # Form coefficients of residual `W @ A @ y - b` w.r.t. last layer `A`
        # resulting from Euler integration of the dynamics from `t` to `t_next`
        W = dt * B
        b = dx - dt*(f + B@u)
        return W, y, b

    # Map over trajectory index
    @partial(jax.vmap, in_axes=(None, 0, None, None, 0, 0, 0, 0, 0))
    def trajectory_loss(params, key, num_ridge_samples, regularizer_ridge,
                        ts, xs, us, ts_next, xs_next):
        """TODO: docstring."""
        # Compute least-squares coefficients and shuffle them
        in_axes = (None, 0, 0, 0, 0, 0)
        Ws, ys, bs = jax.vmap(euler_residual_coefficients, in_axes)(
            params, ts, xs, us, ts_next, xs_next
        )
        shape_A = (Ws.shape[2], ys.shape[1])
        num_samples = ys.shape[0]
        idx = jax.random.permutation(key, num_samples)
        Ws, ys, bs = Ws[idx], ys[idx], bs[idx]
        Ys = jnp.expand_dims(ys, -1)
        Bs = jnp.expand_dims(bs, -1)

        # Solve for the last layer as the regularized least-squares solution
        # on a subset of the data
        Ws_sub = Ws[:num_ridge_samples]
        Ys_sub = Ys[:num_ridge_samples]
        Bs_sub = Bs[:num_ridge_samples]
        Zs = jax.vmap(lambda W, Y: jnp.kron(W.T@W, Y@Y.T))(Ws_sub, Ys_sub)
        Rs = jax.vmap(lambda W, Y, B: W.T @ B @ Y.T)(Ws_sub, Ys_sub, Bs_sub)
        Z = jnp.sum(Zs / num_ridge_samples, axis=0)
        Z_reg = Z.at[jnp.diag_indices(Z.shape[0])].add(regularizer_ridge /
                                                       num_ridge_samples)
        vec_R = jnp.sum(Rs / num_ridge_samples, axis=0).ravel()
        vec_A = jax.scipy.linalg.solve(Z_reg, vec_R, assume_a='pos')
        A = jnp.reshape(vec_A, shape_A)

        # Compute loss on ALL of the data
        loss = jnp.sum((Ws @ A @ Ys - Bs)**2)
        return loss

    @partial(jax.jit, static_argnums=(3,))
    def loss(params, regularizer_l2, keys, num_ridge_samples,
             regularizer_ridge, t, x, u, t_next, x_next):
        """TODO: docstring."""
        num_traj, num_samples = t.shape
        normalizer = num_traj * num_samples
        traj_losses = trajectory_loss(params, keys, num_ridge_samples,
                                      regularizer_ridge,
                                      t, x, u, t_next, x_next)
        loss = (jnp.sum(traj_losses)
                + regularizer_l2*tree_normsq(params)) / normalizer
        return loss

    # Initialize model parameters
    num_hlayers = hparams['num_hlayers']
    hdim = hparams['hdim']
    if num_hlayers >= 1:
        shapes = [(hdim, num_states), ] + (num_hlayers-1)*[(hdim, hdim), ]
    else:
        shapes = []
    key, *subkeys = jax.random.split(key, 1 + 2*num_hlayers)
    keys_W = subkeys[:num_hlayers]
    keys_b = subkeys[num_hlayers:]
    params = {
        # hidden layer weights
        'W': [0.1*jax.random.normal(keys_W[i], shapes[i])
              for i in range(num_hlayers)],
        # hidden layer biases
        'b': [0.1*jax.random.normal(keys_b[i], (shapes[i][0],))
              for i in range(num_hlayers)],
    }

    # Shuffle samples in time along each trajectory, then split each
    # trajectory into training and validation sets
    key, *subkeys = jax.random.split(key, 1 + hparams['num_subtraj'])
    subkeys = jnp.asarray(subkeys)
    shuffled_data = jax.tree_util.tree_map(
        lambda a: jax.vmap(jax.random.permutation)(subkeys, a),
        data
    )
    num_train_samples = int(hparams['train_frac'] * num_samples)
    num_valid_samples = num_samples - num_train_samples
    train_data = jax.tree_util.tree_map(lambda a: a[:, :num_train_samples],
                                        shuffled_data)
    valid_data = jax.tree_util.tree_map(lambda a: a[:, num_train_samples:],
                                        shuffled_data)

    # Initialize gradient-based optimizer (ADAM)
    num_ridge_samples = int(hparams['ridge_frac']*num_train_samples)
    learning_rate = hparams['learning_rate']
    init_opt, update_opt, get_params = adam(learning_rate)
    opt_state = init_opt(params)
    step_idx = 0
    best_idx = 0
    best_loss = jnp.inf
    best_params = params

    @partial(jax.jit, static_argnums=(4,))
    def step(idx, opt_state, regularizer_l2, keys, num_ridge_samples,
             regularizer_ridge, batch):
        """TODO: docstring."""
        params = get_params(opt_state)
        l, grads = jax.value_and_grad(loss, argnums=0)(params, regularizer_l2,
                                                       keys, num_ridge_samples,
                                                       regularizer_ridge,
                                                       **batch)
        opt_state = update_opt(idx, grads, opt_state)
        return opt_state, l

    # Pre-compile before training
    print('MODEL META-TRAINING: Pre-compiling ... ', end='', flush=True)
    start = time.time()
    _ = step(step_idx, opt_state, hparams['regularizer_l2'],
             subkeys, num_ridge_samples,
             hparams['regularizer_ridge'], train_data)
    _ = loss(params, 0., subkeys, num_valid_samples,
             hparams['regularizer_ridge'], **valid_data)
    end = time.time()
    print('done ({:.2f} s)! Now training ...'.format(end - start))
    start = time.time()

    # Do gradient descent
    train_loss_hist = jnp.zeros(hparams['num_steps'])
    valid_loss_hist = jnp.zeros(hparams['num_steps'])

    for _ in tqdm(range(hparams['num_steps'])):
        key, *subkeys = jax.random.split(key, 1 + hparams['num_subtraj'])
        subkeys = jnp.asarray(subkeys)
        opt_state, train_loss = step(step_idx, opt_state,
                                     hparams['regularizer_l2'],
                                     subkeys, num_ridge_samples,
                                     hparams['regularizer_ridge'], train_data)
        new_params = get_params(opt_state)
        valid_loss = loss(new_params, 0., subkeys, num_valid_samples,
                          hparams['regularizer_ridge'], **valid_data)
        train_loss_hist = train_loss_hist.at[step_idx].set(train_loss)
        valid_loss_hist = valid_loss_hist.at[step_idx].set(valid_loss)
        step_idx += 1
        if valid_loss < best_loss:
            best_idx = step_idx
            best_loss = valid_loss
            best_params = new_params

    # Save hyperparameters, loss curves, and model
    results = {
        'train_loss_hist': train_loss_hist,
        'valid_loss_hist': valid_loss_hist,
        'best_step_idx': best_idx,
        'hparams': hparams,
        'meta_params': {
            'W': best_params['W'],
            'b': best_params['b'],
        },
        'meta_params_init': {
            'W': params['W'],
            'b': params['b'],
        },
    }
    directory = os.path.join(system_name, 'models', 'mrr')
    os.makedirs(directory, exist_ok=True)
    prefix = 'seed={:d}_M={:d}'.format(
        hparams['seed'], hparams['num_subtraj']
    )
    path = os.path.join(directory, prefix + '.pkl')
    with open(path, 'wb') as file:
        pickle.dump(results, file)

    end = time.time()
    print('done ({:.2f} s)! Best step index: {}'.format(end - start, best_idx))
