#!/usr/bin/env python

import numpy as np
import time
import sys

import copy
import pickle
import torch
from torch import nn
import pyro
from showerSim.pyro_simulator import PyroSimulator
from showerSim.utils import get_logger
import os

logger = get_logger()


class Simulator(PyroSimulator):
    def __init__(self, jet_p=None, pt_cut=1.0, Mw=None, Delta_0=None, num_samples=1):
        super(Simulator, self).__init__()

        self.pt_cut = pt_cut
        self.Mw = Mw
        self.Delta_0 = Delta_0
        self.num_samples = num_samples

        self.jet_p = jet_p

    def forward(self, inputs):

        decay_rate = inputs

        logger.info(f"Num samples: {self.num_samples}")
        logger.info(f"Energy Scales: {self.Delta_0}")

        # Define the pyro distributions for theta and Delta as global variables
        globals()["phi_dist"] = pyro.distributions.Uniform(0, 2 * np.pi)
        globals()["decay_dist"] = pyro.distributions.Exponential(decay_rate)

        jet_list = []
        for i in range(self.num_samples):

            tree, content, deltas, draws, leaves = _traverse(
                self.jet_p,
                delta_P=self.Delta_0,
                cut_off=self.pt_cut,
                rate=decay_rate,
                Mw=self.Mw,
            )

            jet = dict()
            jet["root_id"] = 0
            jet["tree"] = np.asarray(tree).reshape(-1, 2)  # Labels for the nodes in the tree
            jet["content"] = np.array([np.asarray(c) for c in content])
            jet["Lambda"] = decay_rate
            jet["Delta_0"] = self.Delta_0
            jet["pt_cut"] = self.pt_cut
            jet["M_Hard"] = float(self.Mw)
            jet["algorithm"] = "truth"
            jet["deltas"] = np.asarray(deltas)
            jet["draws"] = np.asarray(draws)
            jet["leaves"] = np.array([np.asarray(c) for c in leaves])
            jet_list.append(jet)

            logger.debug(f"Tree: {jet['tree']}")
            logger.debug(f"Content: {jet['content']}")
            logger.debug(f"Deltas: {jet['deltas']}")
            logger.debug(f"Draws: {jet['draws']}")
            logger.debug(f"Leaves: {jet['leaves']}")

        return jet_list

    @staticmethod
    def save(jet_list, outdir, filename):
        out_filename = os.path.join(outdir, filename + ".pkl")
        with open(out_filename, "wb") as f:
            pickle.dump(jet_list, f, protocol=2)


def dir2D(phi):
    return torch.tensor([np.sin(phi), np.cos(phi)])


def _traverse(root, delta_P=None, cut_off=None, rate=None, Mw=None):

    """
    This function call the recursive function _traverse_rec to make the trees starting from the root

    Inputs
    root: numpy array representing the initial jet momentum
    delta_P: Initial value for Delta
    cut_off: Min value of Delta below which evolution stops
    rate: parametrizes exponential distribution
    Mw: if not None and if initial step, the next Delta_P is not drawn but set to Mw/2

    Outputs
    content: a list of numpy array representing the momenta flowing
        through every possible edge of the tree. content[0] is the root momentum
    tree: an array of integers >= -1, such that
        content[tree[2 * i]] and content[tree[2 * i + 1]] represent the momenta
        associated repsectively to the left and right child of content[i].
        If content[i] is a leaf, tree[2 * i] == tree[2 * i + 1] == 1
    deltas: Delta value associated to content[i]
    draws: r value  associated to content[i]
    """

    tree = []
    content = []
    deltas = []
    draws = []

    leaves = []

    _traverse_rec(
        root,
        -1,
        False,
        tree,
        content,
        deltas,
        draws,
        leaves,
        delta_P=delta_P,
        cut_off=cut_off,
        rate=rate,
        Mw=Mw,
    )  # We start from the root=jet 4-vector

    return tree, content, deltas, draws, leaves


def _traverse_rec(
    root,
    parent_idx,
    is_left,
    tree,
    content,
    deltas,
    draws,
    leaves,
    delta_P=None,
    drew=None,
    cut_off=None,
    rate=None,
    Mw=None,
):

    """
    Recursive function to make the tree.
    """

    idx = len(tree) // 2
    if parent_idx >= 0:
        if is_left:
            tree[2 * parent_idx] = idx
        else:
            tree[2 * parent_idx + 1] = idx

    # We insert 2 new nodes to the vector that constitutes the tree.
    # In the next iteration we will replace this 2 values with the location of the parent of the new nodes
    tree.append(-1)
    tree.append(-1)

    # We fill the content vector with the values of the node
    content.append(root)

    if idx == 0:
        # draw_decay_root = pyro.sample("decay" + str(idx) + str(is_left), decay_dist)
        if Mw:
            delta_P = Mw / 2
            drew = (-1 / rate) * np.log(1 / rate)  # So that likelihood=1

        elif Mw is None:

            draw_decay_root = pyro.sample("decay" + str(idx) + str(is_left), decay_dist)

            delta_P = delta_P * draw_decay_root
            drew = draw_decay_root

    if delta_P > cut_off:
        deltas.append(delta_P)
        draws.append(drew)
    else:
        deltas.append(0)
        draws.append(drew)
        leaves.append(root)

    if delta_P > cut_off:

        draw_phi = pyro.sample("phi" + str(idx) + str(is_left), phi_dist)
        ptL = root / 2 - delta_P * dir2D(draw_phi)
        ptR = root / 2 + delta_P * dir2D(draw_phi)

        draw_decay_L = pyro.sample(
            "L_decay" + str(idx) + str(is_left), decay_dist
        )  # We draw a number to get the left child delta
        draw_decay_R = pyro.sample(
            "R_decay" + str(idx) + str(is_left), decay_dist
        )  # We draw a number to get the right child delta

        delta_L = delta_P * draw_decay_L
        delta_R = delta_P * draw_decay_R

        _traverse_rec(
            ptL,
            idx,
            True,
            tree,
            content,
            deltas,
            draws,
            leaves,
            delta_P=delta_L,
            cut_off=cut_off,
            rate=rate,
            drew=draw_decay_L,
        )

        _traverse_rec(
            ptR,
            idx,
            False,
            tree,
            content,
            deltas,
            draws,
            leaves,
            delta_P=delta_R,
            cut_off=cut_off,
            rate=rate,
            drew=draw_decay_R,
        )

