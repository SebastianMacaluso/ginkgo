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


#-----------------------------
class Simulator(PyroSimulator):

    def __init__(self, jet_p=None, pt_cut=1.0, rate=1.0, Mw=None):
        super(Simulator, self).__init__()

        self.pt_cut = pt_cut
        self.rate = rate
        self.Mw = Mw

        if jet_p is None:
            self.jet_p = [0.0, 0.0]
        else:
            self.jet_p = jet_p

    def forward(self, inputs, num_samples=1):
        inputs = inputs.view(-1, 1)
        kt = inputs[:, 0]  # Input kt scale

        logger.info(f"Num samples: {num_samples}")
        logger.info(f"Energy Scales: {kt}")

        py = self.jet_p[0]
        pz = self.jet_p[1]

        # Define the pyro distributions for theta and Delta as global variables
        globals()["phi_dist"] = pyro.distributions.Uniform(0, 2 * np.pi)
        globals()["decay_dist"] = pyro.distributions.Exponential(self.rate)

        jet_list = []
        for i in range(num_samples):

            tree, content, deltas, draws = _traverse(
                py,
                pz,
                delta_P=kt,
                cut_off=self.pt_cut,
                rate=self.rate,
                Mw=self.Mw,
            )

            tree = np.asarray([tree])
            tree = np.asarray([np.asarray(e).reshape(-1, 2) for e in tree])
            content = np.asarray([content])
            content = np.asarray([np.asarray(e).reshape(-1, 2) for e in content])
            deltas=np.asarray(deltas)
            draws = np.asarray(draws)

            logger.debug(f"Tree: {tree}")
            logger.debug(f"Content: {content}")
            logger.debug(f"Deltas: {deltas}")
            logger.debug(f"Draws: {draws}")


            jet = dict()
            jet["root_id"] = 0
            jet["tree"] = tree[0]  # Labels for the nodes in the tree
            jet["content"] = np.reshape(content[0], (-1, 2))
            jet["Lambda"] = self.rate
            jet["Delta_0"] = kt
            jet["pt_cut"] = self.pt_cut
            jet["M_Hard"] = self.Mw
            jet["deltas"] = deltas
            jet["draws"] = draws
            logger.debug(f"Jet dictionary: {jet}")
            jet_list.append(jet)

        return jet_list

    @staticmethod
    def save(jet_list, outdir, filename):
        out_filename = os.path.join(outdir, filename + ".pkl")
        with open(out_filename, "wb") as f:
            pickle.dump(jet_list, f, protocol=2)



# --------------------------------------------------------------------------------------------------------------
# /////////////////////     AUXILIARY FUNCTIONS     //////////////////////////////////////////////
# -------------------------------------------------------------------------------------------------------------

def _traverse_rec(
    root_y,
    root_z,
    parent_id,
    is_left,
    tree,
    content,
    deltas,
    draws,
    delta_P=None,
    drew=None,
    cut_off=None,
    rate=None,
    Mw=None,
):

    '''
    Recursive function to make the tree. We will call this function below in _traverse.
    '''

    id = len(tree) // 2
    if parent_id >= 0:
        if is_left:
            tree[
                2 * parent_id
            ] = (
                id
            )  # We set the location of the lef child in the content array of the 4-vector stored in content[parent_id]. So the left child will be content[tree[2 * parent_id]]
        else:
            tree[
                2 * parent_id + 1
            ] = (
                id
            )  # We set the location of the right child in the content array of the 4-vector stored in content[parent_id]. So the right child will be content[tree[2 * parent_id+1]]
    #  This is correct because with each 4-vector we increase the content array by one element and the tree array by 2 elements. But then we take id=tree.size()//2, so the id increases by 1. The left and right children are added one after the other.

    # -------------------------------
    # We insert 2 new nodes to the vector that constitutes the tree. In the next iteration we will replace this 2 values with the location of the parent of the new nodes
    tree.append(-1)
    tree.append(-1)

    #     We fill the content vector with the values of the node
    content.append(root_y)
    content.append(root_z)


    # ------------------------------
    # Call the function recursively. We move from the root down until we get to the leaves.

    root_y = root_y / 2
    root_z = root_z / 2

    draw_phi = pyro.sample("phi" + str(id) + str(is_left), phi_dist)

    # Fix start delta_P for W jets
    if Mw and id == 0:
        delta_P = Mw / 2
        drew = 1

    # The cut_off gives a kt measure
    if delta_P > cut_off:

        ptL_y = root_y - delta_P * np.sin(draw_phi)
        ptL_z = root_z - delta_P * np.cos(draw_phi)

        ptR_y = root_y + delta_P * np.sin(draw_phi)
        ptR_z = root_z + delta_P * np.cos(draw_phi)


        deltas.append(delta_P)
        draws.append(drew)

        draw_decay_L = pyro.sample(
            "L_decay" + str(id) + str(is_left), decay_dist
        )  # We draw a number to get the left child delta
        draw_decay_R = pyro.sample(
            "R_decay" + str(id) + str(is_left), decay_dist
        )  # We draw a number to get the right child delta

        delta_L = delta_P * draw_decay_L
        delta_R = delta_P * draw_decay_R


        _traverse_rec(
            ptL_y,
            ptL_z,
            id,
            True,
            tree,
            content,
            deltas,
            draws,
            delta_P=delta_L,
            cut_off=cut_off,
            rate=rate,
            drew=draw_decay_L,
        )

        _traverse_rec(
            ptR_y,
            ptR_z,
            id,
            False,
            tree,
            content,
            deltas,
            draws,
            delta_P=delta_R,
            cut_off=cut_off,
            rate=rate,
            drew=draw_decay_R,
        )

    else:

        # print('---' * 10)
        deltas.append(-1)
        draws.append(-1)


# -------------------------------------------------------------------------------------------------------------

def _traverse(
    root_y,
    root_z,
    delta_P=None,
    cut_off=None,
    rate=None,
    Mw=None,
):

    '''
    This function call the recursive function _traverse_rec to make the trees starting from the root
    '''

    tree = []
    content = []
    deltas = []
    draws = []


    _traverse_rec(
        root_y,
        root_z,
        -1,
        False,
        tree,
        content,
        deltas,
        draws,
        delta_P=delta_P,
        cut_off=cut_off,
        rate=rate,
        Mw=Mw,
    )  # We start from the root=jet 4-vector

    return tree, content, deltas, draws



