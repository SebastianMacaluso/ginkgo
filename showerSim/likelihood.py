import pickle
import numpy as np
import torch
from showerSim import exp2DShowerTree
from showerSim.utils import get_logger


def get_delta_LR(pL, pR):
    """
    Calculate delta of an edge, given its child momenta
    """
    return np.sqrt(np.sum((pR / 2 - pL / 2) ** 2))


def get_delta_PC(p, pC):
    """
    Calculate delta of an edge, given its momentum and the momentum of a child
    """
    return np.sqrt(np.sum((p / 2 - pC) ** 2))


def split_logLH(pL, delta_L, pR, delta_R, delta_min, lam):
    """
    Takes two edges (p, delta) and
    return the splitting that generated them (p, delta_P, phi)
    with its log likelihood
    """
    p = pR + pL
    delta_vec = (pR - pL) / 2
    phi = np.arctan(delta_vec[0] / delta_vec[1])
    delta_P = get_delta_LR(pL, pR)

    def get_p(delta_P, delta, delta_min, lam):
        if delta > 0:
            if delta < delta_min:
                raise ValueError("Input delta is below cutoff but non-zero")
            r = delta / delta_P
            return np.log(lam * np.exp(-lam * r))
        else:
            r = delta_min / delta_P
            return np.log(1 - np.exp(-lam * r))

    logLH = (
        get_p(delta_P, delta_L, delta_min, lam)
        + get_p(delta_P, delta_R, delta_min, lam)
        + np.log(1 / 2 / np.pi)
    )

    return logLH, p, delta_P, phi


def fill_jet_info(jet, root_id=0, parent_id=None):
    """
    Fill jet["deltas"] amd jet["draws"] given jet["tree"] and jet["content"]
    Assing r = None to the root and the leaves, and assign delta = 0 to the leaves
    """
    deltas = []
    draws = []

    _get_jet_info(
        jet,
        root_id=root_id,
        parent_id=parent_id,
        deltas=deltas,
        draws=draws,
    )

    jet["deltas"] = deltas
    jet["draws"] = draws


def _get_jet_info(jet, root_id=None, parent_id=None, deltas=None, draws=None):
    """
    Recursion to fill jet["deltas"] amd jet["draws"]
    """

    if jet["tree"][root_id][0] != -1 and jet["tree"][root_id][1] != -1:

        idL = jet["tree"][root_id][0]
        idR = jet["tree"][root_id][1]
        pL = jet["content"][idL]
        pR = jet["content"][idR]
        delta = get_delta_LR(pL, pR)
        if parent_id is not None:
            delta_parent = deltas[parent_id]
            r = torch.tensor(delta / delta_parent)
        else:
            r = None

        deltas.append(delta)
        draws.append(r)

        _get_jet_info(jet, root_id=idL, parent_id=root_id, deltas=deltas, draws=draws)
        _get_jet_info(jet, root_id=idR, parent_id=root_id, deltas=deltas, draws=draws)

    else:
        if jet["tree"][root_id][0] * jet["tree"][root_id][1] != 1:
            raise ValueError(f"Invalid jet left and right child are not both -1")
        else:
            deltas.append(0)
            draws.append(None)


def enrich_jet_logLH(jet, root_id=0, Lambda=None, delta_min=None):
    """
    Attach splitting log likelihood to each edge, by calling recursive
    _get_jet_likelihood.
    """
    logLH = []

    if Lambda is None:
        Lambda = jet.get("Lambda")
        if Lambda is None:
            raise ValueError(f"No Lambda specified by the jet.")
    if delta_min is None:
        delta_min = jet.get("pt_cut")
        if delta_min is None:
            raise ValueError(f"No pt_cut specified by the jet.")

    _get_jet_logLH(
        jet,
        root_id=root_id,
        Lambda=Lambda,
        delta_min=delta_min,
        logLH=logLH,
    )

    jet["logLH"] = logLH


def _get_jet_logLH(jet, root_id=None, Lambda=None, delta_min=None, logLH=None):
    """
    Recursively enrich every edge from root_id downward with their log likelihood.
    log likelihood of a leave is 0. Assumes a valid jet.
    """
    if jet["tree"][root_id][0] != -1:
        idL = jet["tree"][root_id][0]
        idR = jet["tree"][root_id][1]
        pL = jet["content"][idL]
        pR = jet["content"][idR]
        delta_L = jet["deltas"][idL]
        delta_R = jet["deltas"][idR]
        llh, _, _, _ = split_logLH(pL, delta_L, pR, delta_R, delta_min, Lambda)
        logLH.append(llh)

        _get_jet_logLH(jet, root_id=idL, Lambda=Lambda, delta_min=delta_min, logLH=logLH)
        _get_jet_logLH(jet, root_id=idR, Lambda=Lambda, delta_min=delta_min, logLH=logLH)
    else:
        logLH.append(0)


if __name__ == "__main__":

    logger = get_logger()

    simulator = exp2DShowerTree.Simulator(
        jet_p=torch.tensor([800.0, 600.0]),
        Mw=torch.tensor(80.0),
        pt_cut=0.04,
        Delta_0=60.0,
        num_samples=1,
    )
    logger.info(f"Generating random jet")
    jet_list = simulator(torch.tensor(4.0))

    jet_dic = jet_list[0]

    jet_dic_0 = {
        "tree": jet_dic['tree'],
        "content": jet_dic['content'],
        "Lambda": jet_dic['Lambda'],
        "pt_cut": jet_dic['pt_cut']
    }

    fill_jet_info(jet_dic_0)
    logger.info(f"Comparing deltas")
    logger.info(f"deltas: {jet_dic['deltas']}")
    logger.info(f"deltas: {jet_dic_0['deltas']}")
    logger.info(f"Comparing draws")
    logger.info(f"draws: {jet_dic['draws']}")
    logger.info(f"draws: {jet_dic_0['draws']}")

    enrich_jet_logLH(jet_dic_0)
    logger.info(f"Top-down likelihood")
    logger.info(f"logLH: {jet_dic_0['logLH']}")


