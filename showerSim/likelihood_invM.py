import pickle
import numpy as np
import torch


def get_delta_LR(pL, pR):
    """
    Calculate invariant mass of a node, given its child momenta
    """
    pP = pR + pL

    """Parent invariant mass squared"""
    tp1 = pP[0] ** 2 - np.linalg.norm(pP[1::]) ** 2

    return tp1

#
# def get_delta_PC(p, pC):
#     """
#     Calculate delta of an edge, given its momentum and the momentum of a child
#     """
#     return np.sqrt(np.sum((p / 2 - pC) ** 2))



def split_logLH(pL, tL, pR, tR, t_cut, lam):
    """
    Take two nodes and return the splitting log likelihood
    """
    pP = pR + pL

    """Parent invariant mass squared"""
    tp1 = pP[0] ** 2 - np.linalg.norm(pP[1::]) ** 2

    tmax = max(tL,tR)
    tmin = min(tL,tR)

    tp2 = (np.sqrt(tp1) - np.sqrt(tmax)) ** 2

    """ We add a normalization factor -np.log(1 - np.exp(- lam)) because we need the mass squared to be strictly decreasing """
    def get_p(tP, t, t_cut, lam):
        if t > 0:
            return -np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(tP) - lam * t / tP

        else: # if t<t_min then we set t=0
            return -np.log(1 - np.exp(- lam)) + np.log(1 - np.exp(-lam * t_cut / tP))

    """We sample a unit vector uniformly over the 2-sphere, so the angular likelihood is 1/(4*pi)"""
    logLH = (
        get_p(tp1, tmax, t_cut, lam)
        + get_p(tp2, tmin, t_cut, lam)
        + np.log(1 / (4 * np.pi))
    )

    "If the pairing is not allowed"
    if tp1 < t_cut:
        logLH = - np.inf

    return logLH




def fill_jet_info(jet, parent_id=None):
    """
    Fill jet["deltas"] amd jet["draws"] given jet["tree"] and jet["content"]
    Assing r = None to the root and the leaves, and assign delta = 0 to the leaves
    """
    deltas = []
    draws = []

    root_id = jet["root_id"]

    _get_jet_info(jet, root_id=root_id, parent_id=parent_id, deltas=deltas, draws=draws)

    jet["deltas"] = deltas
    jet["draws"] = draws

    return jet

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


def enrich_jet_logLH(jet, delta_min=None, dij=False, alpha = None):
    """
    Attach splitting log likelihood to each edge, by calling recursive
    _get_jet_likelihood.
    """
    logLH = []
    dijList = []

    root_id = jet["root_id"]

    if delta_min is None:
        delta_min = jet.get("pt_cut")
        if delta_min is None:
            raise ValueError(f"No pt_cut specified by the jet.")

    _get_jet_logLH(
        jet,
        root_id = root_id,
        delta_min = delta_min,
        logLH = logLH,
        dij = dij,
        dijList = dijList,
        alpha = alpha,
    )

    jet["logLH"] = np.asarray(logLH)
    jet["dij"] = dijList

    return jet


def _get_jet_logLH(
        jet,
        root_id = None,
        delta_min = None,
        logLH = None,
        dij = False,
        dijList = None,
        alpha = None
):
    """
    Recursively enrich every edge from root_id downward with their log likelihood.
    log likelihood of a leaf is 0. Assumes a valid jet.
    """
    if jet["tree"][root_id][0] != -1:


        idL = jet["tree"][root_id][0]
        idR = jet["tree"][root_id][1]
        pL = jet["content"][idL]
        pR = jet["content"][idR]
        tL = jet["deltas"][idL]
        tR = jet["deltas"][idR]

        Lambda = jet["Lambda"]
        if root_id == jet["root_id"]:
            Lambda = jet["LambdaRoot"]


        llh = split_logLH(pL, tL, pR, tR, delta_min, Lambda)
        logLH.append(llh)
        # print('logLH = ', llh)

        if dij:

            """ dij=min(pTi^(2 alpha),pTj^(2 alpha)) * [arccos((pi.pj)/|pi|*|pj|)]^2 """
            dijs= [float(llh)]

            for alpha in [-1,0,1]:

                tempCos = np.dot(pL[1::], pR[1::]) / (np.linalg.norm(pL[1::]) * np.linalg.norm(pR[1::]))
                if abs(tempCos) > 1: tempCos = np.sign(tempCos)

                dijVal = np.sort((np.array([np.linalg.norm(pL[1:3]),np.linalg.norm(pR[1:3])])) ** (2 * alpha))[0]  * \
                         (
                             np.arccos(tempCos)
                          ) ** 2

                dijs.append(dijVal)

            dijList.append(dijs)


        _get_jet_logLH(
            jet,
            root_id = idL,
            delta_min = delta_min,
            logLH = logLH,
            dij = dij,
            dijList = dijList,
            alpha = alpha,
        )
        _get_jet_logLH(
            jet,
            root_id = idR,
            delta_min = delta_min,
            logLH = logLH,
            dij = dij,
            dijList = dijList,
            alpha = alpha,
        )

    else:

        logLH.append(0)


