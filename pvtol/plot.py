"""
Plot test results for the PVTOL system.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

import os
import pickle

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import numpy as np

from tqdm.auto import tqdm

plt.close('all')
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'CMU Serif'],
    'mathtext.fontset': 'cm',
    'font.size':        18,
    'legend.fontsize':  'medium',
    'axes.titlesize':   'medium',
    'lines.linewidth':  2,
    'lines.markersize': 10,
    'errorbar.capsize': 6,
})

system_name = 'pvtol'
methods = ('no adapt', 'no adapt, new gains', 'mrr', 'mrr, new gains', 'ours')
labels = {
    'ours':
        r'ours',
    'no adapt':
        r'no adaptation, $\kappa_\mathrm{init}$',
    'no adapt, new gains':
        r'no adaptation, $\kappa_\mathrm{ours}$',
    'mrr':
        r'MRR, $(\kappa_\mathrm{init}$, $\Gamma_\mathrm{init})$',
    'mrr, new gains':
        r'MRR, $(\kappa_\mathrm{ours}$, $\Gamma_\mathrm{ours})$',
}
colors = {
    'ours':                 'tab:blue',
    'no adapt':             'tab:pink',
    'no adapt, new gains':  'tab:cyan',
    'mrr':                  'tab:green',
    'mrr, new gains':       'tab:orange',
}
patches = [Patch(color=colors[method], label=labels[method])
           for method in methods]
lines = [Line2D([0], [0], color=colors[method], label=labels[method])
         for method in methods]

seeds = tuple(range(10))
Ms = (2, 5, 10, 20, 30, 40, 50)

prefix = 'seed={:d}_M={:d}'.format(seeds[0], Ms[0])
path = os.path.join(system_name, 'results', prefix + '.pkl')
with open(path, 'rb') as file:
    N = pickle.load(file)['hparams']['num_traj']

# Compute the state and control trajectory RMSE for each seed, `M`, and
# model configuration
ex = {method: np.zeros((len(seeds), len(Ms), N)) for method in methods}
eu = {method: np.zeros((len(seeds), len(Ms), N)) for method in methods}
for i, seed in enumerate(tqdm(seeds)):
    for j, M in enumerate(Ms):
        # Load file
        prefix = 'seed={:d}_M={:d}'.format(seed, M)
        path = os.path.join(system_name, 'results', prefix + '.pkl')
        with open(path, 'rb') as file:
            tests = pickle.load(file)

        # Compute the trajectory RMSEs for each method
        for method in methods:
            x = tests['results'][method]['x']
            x_ref = tests['results'][method]['x_ref']
            u = tests['results'][method]['u']
            u_ref = tests['results'][method]['u_ref']
            ex[method][i, j] = np.sqrt(
                np.nanmean(np.nansum((x - x_ref)**2, axis=-1), axis=-1)
            )
            eu[method][i, j] = np.sqrt(
                np.nanmean(np.nansum((u - u_ref)**2, axis=-1), axis=-1)
            )

###############################################################################
# Plot box-plots of state trajectory RMSE across seeds as a function of `M`
num_cols = len(methods)
fig, ax = plt.subplots(1, num_cols, figsize=(3.25*num_cols, 4), sharey=True)
for k, method in enumerate(methods):
    ex_seeds = np.nanmean(ex[method], axis=-1)
    ax[k].boxplot(
        ex_seeds, labels=Ms, patch_artist=True, notch=False,
        medianprops={'color': 'k', 'lw': 2},
        boxprops={'facecolor': colors[method], 'alpha': 1.},
        flierprops={
            'marker':           'o',
            'markersize':       6,
            'markerfacecolor':  colors[method],
            'alpha':            1.,
        },
    )
    ax[k].plot(1 + np.arange(len(Ms)), np.nanmedian(ex_seeds, axis=0),
               ls='', color='k', marker='D', markersize=6)
    ax[k].set_title(labels[method])
    ax[k].set_xlabel('$M$')
ax[0].set_ylabel(r'$\dfrac{1}{N_\mathrm{test}}'
                 r'\sum_{i=1}^{N_\mathrm{test}}\,\mathrm{RMS}(x_i-x_i^*)$')
fig.tight_layout()
fig.savefig(os.path.join('figures', 'pvtol_boxplot.pdf'), bbox_inches='tight')
plt.show()

###############################################################################
# Plot line-plots of state trajectory RMSE across seeds as a function of `M`
fig, ax = plt.subplots(figsize=(9, 7))
for method in methods:
    ex_seeds = np.nanmean(ex[method], axis=-1)
    ax.errorbar(Ms, np.nanmean(ex_seeds, axis=0), np.nanstd(ex_seeds, axis=0),
                fmt='-o', label=labels[method], color=colors[method])
ax.set_ylabel(r'$\dfrac{1}{N_\mathrm{test}}'
              r'\sum_{i=1}^{N_\mathrm{test}}\,\mathrm{RMS}(x_i-x_i^*)$')
ax.set_xlabel('$M$')
fig.legend(handles=patches, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, 0.))
fig.subplots_adjust(bottom=0.25)
fig.savefig(os.path.join('figures', 'pvtol_lineplot.pdf'), bbox_inches='tight')
plt.show()

###############################################################################
# Load and plot results for the double-loop trajectory
path = os.path.join(system_name, 'results_looptraj.pkl')
with open(path, 'rb') as file:
    results = pickle.load(file)

fig = plt.figure(figsize=(15, 13))
grid = fig.add_gridspec(2, 1, height_ratios=[1.75, 2], hspace=0.2)
subgrid = grid[1].subgridspec(2, 2, hspace=0.15, wspace=0.2)
ax = fig.add_subplot(grid[0])
subax = subgrid.subplots(sharex=True)

for method in methods:
    w = results['w']
    t, V = results[method]['t'], results[method]['V']
    V_dot = np.gradient(V, t[1] - t[0])
    x, x_ref = results[method]['x'], results[method]['x_ref']
    e = np.linalg.norm(x - x_ref, axis=-1)
    px, py = x[:, 0], x[:, 1]
    ax.plot(px, py, color=colors[method])
    subax[0, 1].plot(t, V, color=colors[method])
    subax[1, 0].plot(t, e, color=colors[method])
    subax[1, 1].plot(t, V_dot, color=colors[method])
px_ref, py_ref = x_ref[:, 0], x_ref[:, 1]
ax.plot(px_ref, py_ref, ls='--', lw=3, color='tab:red')
subax[0, 0].plot(t, w, ls='-', color='tab:red')

ax.set_xlabel(r'$x~\mathrm{[m]}$')
ax.set_ylabel(r'$y~\mathrm{[m]}$')
subax[0, 0].set_ylabel(r'$w(t)~\mathrm{[m/s]}$')
subax[0, 1].set_ylabel(r'$\bar{V}\,(x(t),x\!^*\!(t))$')
subax[1, 0].set_ylabel(r'$\|x(t) -  x\!^*\!(t)\|_2$')
subax[1, 1].set_ylabel(r'$\dot{\bar{V}}\,(x(t),x\!^*\!(t))$')
for sax in subax.ravel():
    if sax.get_subplotspec().is_last_row():
        sax.set_xlabel(r'$t~\mathrm{[s]}$')

im_height = 1.5
im_width = 1.5
im_x0, im_y0 = -1.15, 3.
pad = 0.2
ax.text(im_x0 + 0.5, im_y0 + im_height + pad, r'$w(t)$')
ax.imshow(
    plt.imread(os.path.join('figures', 'wind.png')),
    aspect='auto',
    interpolation='none',
    extent=(im_x0, im_x0 + im_width, im_y0, im_y0 + im_height)
)
ax.set_xlim([-1.25, 13.25])
ax.set_ylim([-0.5, 5.5])

handles = lines + [Line2D([0], [0], color='tab:red', label='target',
                   linestyle='--', lw=4.)]
fig.legend(handles=handles, ncol=3, loc='lower center')
fig.subplots_adjust(bottom=0.15)
fig.savefig(os.path.join('figures', 'pvtol_looptraj.pdf'), bbox_inches='tight')
plt.show()
