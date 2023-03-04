"""
Some utilities for dealing with PyTrees of parameters.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

import jax
import jax.numpy as jnp


def tree_scale(x_tree, a):
    """Scale the children of a PyTree by the scalar `a`."""
    return jax.tree_util.tree_map(lambda x: a * x, x_tree)


def tree_add(x_tree, y_tree):
    """Add pairwise the children of two PyTrees."""
    return jax.tree_util.tree_multimap(lambda x, y: x + y, x_tree, y_tree)


def tree_index(x_tree, i):
    """Index child arrays in PyTree."""
    return jax.tree_util.tree_map(lambda x: x[i], x_tree)


def tree_index_update(x_tree, i, y_tree):
    """Update indices of child arrays in PyTree with new values."""
    return jax.tree_util.tree_multimap(lambda x, y:
                                       jax.ops.index_update(x, i, y),
                                       x_tree, y_tree)


def tree_axpy(a, x_tree, y_tree):
    """Compute `a*x + y` for two PyTrees `(x, y)` and a scalar `a`."""
    ax = tree_scale(x_tree, a)
    axpy = jax.tree_util.tree_multimap(lambda x, y: x + y, ax, y_tree)
    return axpy


def tree_dot(x_tree, y_tree):
    """Compute the dot products between children of two PyTrees."""
    xy = jax.tree_util.tree_multimap(lambda x, y: jnp.sum(x*y), x_tree, y_tree)
    return xy


def tree_normsq(x_tree):
    """Compute sum of squared norms across a PyTree."""
    normsq = jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(y**2),
                                       x_tree, 0.)
    return normsq


def tree_anynan(tree):
    """Check if there are any NAN elements in the PyTree."""
    any_isnan_tree = jax.tree_util.tree_map(lambda a: jnp.any(jnp.isnan(a)),
                                            tree)
    any_isnan = jax.tree_util.tree_reduce(lambda x, y: jnp.logical_or(x, y),
                                          any_isnan_tree, False)
    return any_isnan
