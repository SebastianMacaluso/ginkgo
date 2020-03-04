#!/usr/bin/env python

import numpy as np
import time
import sys
import os
import copy
import pickle
import torch
from torch import nn
import pyro
from showerSim.pyro_simulator import PyroSimulator
from showerSim.utils import get_logger
from showerSim import likelihood_invM as likelihood
from showerSim import auxFunctions

logger = get_logger()


class Simulator(PyroSimulator):
    def __init__(self, jet_p=None, pt_cut=1.0, M_hard=None, Delta_0=None, num_samples=1):
        super(Simulator, self).__init__()

        self.pt_cut = pt_cut
        self.M_hard = M_hard
        self.Delta_0 = Delta_0
        self.num_samples = num_samples

        self.jet_p = jet_p

    def forward(self, inputs):

        root_rate = inputs[0]
        decay_rate = inputs[1]

        logger.info(f"Num samples: {self.num_samples}")
        logger.info(f"Initial squared mass: {self.Delta_0}")

        """Define pyro distributions as global variables"""
        """Sample a unit vector uniformly over the 2-sphere"""
        globals()["multiNormal_dist"] = pyro.distributions.MultivariateNormal(torch.zeros(3), torch.eye(3))

        globals()["root_dist"] = pyro.distributions.Exponential(root_rate)
        globals()["decay_dist"] = pyro.distributions.Exponential(decay_rate)


        jet_list = []
        for i in range(self.num_samples):

            tree, content, deltas, draws, leaves = _traverse(
                self.jet_p,
                delta_P=self.Delta_0,
                cut_off=self.pt_cut,
                rate=decay_rate,
            )

            jet = dict()
            jet["root_id"] = 0
            jet["tree"] = np.asarray(tree).reshape(-1, 2)  # Labels for the nodes in the tree
            jet["content"] = np.array([np.asarray(c) for c in content])
            jet["LambdaRoot"] = root_rate
            jet["Lambda"] = decay_rate
            jet["Delta_0"] = self.Delta_0
            jet["pt_cut"] = self.pt_cut
            jet["algorithm"] = "truth"
            jet["deltas"] = np.asarray(deltas)
            jet["draws"] = np.asarray(draws)
            jet["leaves"] = np.array([np.asarray(c) for c in leaves])
            if self.M_hard:
                jet["M_Hard"] = float(self.M_hard)

            """Fill jet dictionaries with log likelihood of truth jet"""
            likelihood.enrich_jet_logLH(jet, dij=True)

            """ Angular quantities"""
            ConstPhi, PhiDelta, PhiDeltaListRel = auxFunctions.traversePhi(jet, jet["root_id"], [], [],[])
            jet["ConstPhi"] = ConstPhi
            jet["PhiDelta"] = PhiDelta
            jet["PhiDeltaRel"] = PhiDeltaListRel

            jet_list.append(jet)

            # print(" N const = ",len(jet['leaves']))
            logger.info(f" Leaves  = {jet['leaves']}")
            logger.info(f" N const = {len(jet['leaves'])}")
            logger.debug(f"Tree: {jet['tree']}")
            logger.debug(f"Content: {jet['content']}")
            logger.info(f" Total momentum from root = {jet['content'][0]}")
            logger.info(f" Total momentum from leaves = {np.sum(jet['leaves'],axis=0)}")
            logger.info(f" Jet Mass = {jet['M_Hard']}")
            logger.info(f" Jet likelihood = {jet['logLH']}")


            logger.debug(f"Tree: {jet['tree']}")
            logger.debug(f"Content: {jet['content']}")
            logger.debug(f"Deltas: {jet['deltas']}")
            logger.debug(f"Draws: {jet['draws']}")
            logger.debug(f"Leaves: {jet['leaves']}")

            if i%100==0:
                print("Generated ",i," jets")

        return jet_list

    @staticmethod
    def save(jet_list, outdir, filename):
        out_filename = os.path.join(outdir, filename + ".pkl")
        with open(out_filename, "wb") as f:
            pickle.dump(jet_list, f, protocol=2)


def dir2D(phi):
    return torch.tensor([np.sin(phi), np.cos(phi)])


def _traverse(root, delta_P=None, cut_off=None, rate=None):

    """
    This function call the recursive function _traverse_rec to make the trees starting from the root

    Inputs
    root: numpy array representing the initial jet momentum
    delta_P: Initial value for the parent mass squared
    cut_off: Min value of the mass squared below which evolution stops
    rate: parametrizes the exponential distribution
    M_hard: value for the mass of the jet (root of the binary tree)

    Outputs
    content: a list of numpy array representing the momenta flowing
        through every possible edge of the tree. content[0] is the root momentum
    tree: an array of integers >= -1, such that
        content[tree[2 * i]] and content[tree[2 * i + 1]] represent the momenta
        associated repsectively to the left and right child of content[i].
        If content[i] is a leaf, tree[2 * i] == tree[2 * i + 1] == 1
    deltas: mass squared value associated to content[i]
    draws: r value  associated to content[i]
    """

    tree = []
    content = []
    deltas = []
    draws = []

    leaves = []

    """ Start from the root=jet 4-vector"""
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
    )

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
):

    """
    Recursive function to make the jet tree.
    """

    idx = len(tree) // 2
    if parent_idx >= 0:
        if is_left:
            tree[2 * parent_idx] = idx
        else:
            tree[2 * parent_idx + 1] = idx

    """Insert 2 new nodes to the vector that constitutes the tree. 
    In the next iteration we will replace this 2 values with the location of the parent of the new nodes"""
    tree.append(-1)
    tree.append(-1)

    """Fill the content vector with the values of the node"""
    content.append(root)

    draws.append(drew)

    if delta_P > cut_off:
        deltas.append(delta_P)
    else:
        deltas.append(0)
        leaves.append(root)


    if delta_P > cut_off:

        """ Sample uniformly over the sphere of unit radius a unit vector for the decay products in the CM frame"""
        r_CM = pyro.sample("rCM"+ str(idx) + str(is_left), multiNormal_dist)
        r_CM = r_CM.numpy()
        r_CM = r_CM / np.linalg.norm(r_CM)


        """  Use different distributions to model the root node splitting, e.g. W decay"""
        if idx == 0:  sampling_dist = root_dist
        else: sampling_dist = decay_dist

        logger.debug(f" dist = {sampling_dist}")


        """ Sample new values for the children invariant mass squared"""
        draw_decay_L = np.inf
        draw_decay_R = np.inf
        nL=0
        nR=0
        logger.debug(f"draw_decay_L Before= {draw_decay_L, nL}")
        logger.debug(f"draw_decay_R Before = {draw_decay_R, nR}")

        """ The invariant mass squared should decrease strictly"""
        while draw_decay_L >= (1. - 1e-3):
            draw_decay_L = pyro.sample(
                "L_decay" + str(idx) + str(is_left), sampling_dist
            )  # We draw a number to get the left child delta
            nL+=1

        while draw_decay_R >= (1. - 1e-3):
            draw_decay_R = pyro.sample(
                "R_decay" + str(idx) + str(is_left), sampling_dist
            )  # We draw a number to get the right child delta
            nR+=1

        logger.debug(f"draw_decay_L After= {draw_decay_L, nL}")
        logger.debug(f"draw_decay_R After = {draw_decay_R, nR}")

        t0 = delta_P
        tL = t0 * draw_decay_L
        tR = (np.sqrt(t0) - np.sqrt(tL))**2 * draw_decay_R

        if idx ==0: logger.info(f" Off-shell subjets mass = {np.sqrt(tL),np.sqrt(tR)}")


        """ 2-Body decay in the parent CM frame"""
        EL_cm = CenterofMassE(tp = t0, t_child = tL, t_sib = tR)
        ER_cm = CenterofMassE(tp = t0, t_child = tR, t_sib = tL)

        P_CM = CenterofMassP(tp = t0, t_child = tR, t_sib = tL)
        logger.debug(f"P_CM =  {P_CM}")


        """Boost to the lab frame"""
        P0_lab = np.linalg.norm(root[1::])
        n0= - root[1::]/P0_lab
        logger.debug(f" n0 = {n0}")
        logger.debug(f"norm r_CM = {np.linalg.norm(r_CM)}")

        pL_mu = labEP(tp = t0, Ep_lab = root[0], Pp_lab = P0_lab, n = n0, Echild_CM = EL_cm, Pchild_CM = P_CM, p_versor = r_CM)
        pR_mu = labEP(tp = t0, Ep_lab = root[0], Pp_lab = P0_lab, n = n0, Echild_CM = ER_cm, Pchild_CM = P_CM, p_versor = - r_CM)

        logger.debug(f" Off-shell subjets mass = {np.sqrt(tL), np.sqrt(tR)}")
        logger.debug(f"pL inv mass from p^2 in lab  frame: {np.sqrt(pL_mu[0]**2-np.linalg.norm(pL_mu[1::])**2)}")
        logger.debug(f"pR inv mass from p^2 in lab  frame: {np.sqrt(pR_mu[0] ** 2 - np.linalg.norm(pR_mu[1::]) ** 2)}")
        logger.debug(f"----"*10)


        _traverse_rec(
            pL_mu,
            idx,
            True,
            tree,
            content,
            deltas,
            draws,
            leaves,
            delta_P=tL,
            cut_off=cut_off,
            rate=rate,
            drew=draw_decay_L,
        )

        _traverse_rec(
            pR_mu,
            idx,
            False,
            tree,
            content,
            deltas,
            draws,
            leaves,
            delta_P=tR,
            cut_off=cut_off,
            rate=rate,
            drew=draw_decay_R,
        )



### Auxiliary functions:
def CenterofMassE(tp = None,t_child = None,t_sib= None):
    """ Decay product energies in the parent CM frame"""
    E = np.sqrt(tp)/2 * (1 + t_child/tp - t_sib/tp)
    return E

def CenterofMassP(tp= None, t_child= None, t_sib= None):
    """ Decay product spatial momentum in the parent CM frame"""
    P = np.sqrt(tp)/2 * np.sqrt( 1 - 2 * (t_child+t_sib)/tp + (t_child - t_sib)**2 / tp**2 )

    return P

def labEP(tp= None,Ep_lab= None, Pp_lab= None , n= None, Echild_CM= None, Pchild_CM= None, p_versor= None):
    """ Boost to the lab frame"""
    logger.debug(f"{type(tp)}")
    logger.debug(f"{type(Ep_lab)}")
    # tp = torch.tensor(tp)
    # Ep_lab = torch.tensor(Ep_lab)

    tp = tp.numpy()
    Echild_CM = Echild_CM.numpy()
    Pchild_CM = Pchild_CM.numpy()

    Elab = Ep_lab/np.sqrt(tp)* Echild_CM - Pp_lab/np.sqrt(tp) * Pchild_CM * np.dot(n,p_versor)

    Plab = - Pp_lab/np.sqrt(tp) * Echild_CM * n + Pchild_CM * (p_versor + (Ep_lab/np.sqrt(tp) - 1) * np.dot(p_versor,n) * n)

    if Elab < np.linalg.norm(Plab):
        print("---" * 10)
        logger.debug(f" Elab = {Elab}")
        logger.debug(f" Plab = {np.linalg.norm(Plab)}")
        logger.debug(f" sqrt(tp) = {np.sqrt(tp)}")
        logger.debug(f" Ep_lab = {Ep_lab}")
        logger.debug(f" Pp_lab = {Pp_lab}")
        logger.debug(f" np.dot(n,p_versor) = {np.dot(n,p_versor)}")
        logger.debug(f"Echild CM = {Echild_CM}")
        logger.debug(f"Pchild_CM = {Pchild_CM}")
        logger.debug(f" terms = {Pp_lab/np.sqrt(tp) * Echild_CM * n,+ Pchild_CM * (p_versor ),Pchild_CM * ( (Ep_lab/np.sqrt(tp) - 1) * np.dot(p_versor,n) * n) }")
        logger.debug(f"---" * 10)

    p_mu = np.concatenate(([Elab],Plab))

    return p_mu





