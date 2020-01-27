import pickle
import numpy as np
import torch


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
    phi = np.arctan2(delta_vec[0], delta_vec[1])
    delta_P = get_delta_LR(pL, pR)

    def get_p(delta_P, delta, delta_min, lam):
        if delta > 0:
            # r = delta / delta_P
            return np.log(lam) - np.log(delta_P) - lam * delta / delta_P
        else:
            # r = delta_min / delta_P
            return np.log(1 - np.exp(-lam * delta_min / delta_P))

    logLH = (
        get_p(delta_P, delta_L, delta_min, lam)
        + get_p(delta_P, delta_R, delta_min, lam)
        + np.log(1 / 2 / np.pi)
    )

    if delta_P < delta_min:
        logLH = - np.inf

    return logLH, p, delta_P, phi




def Basic_split_logLH(pL, delta_L, pR, delta_R, delta_min, lam):
    """
    Takes two edges (p, delta) and
    return the splitting that generated them (p, delta_P, phi)
    with its log likelihood

    Note: Leaves in the Toy Generative Model are assigned Delta=0
    """

    delta_P = get_delta_LR(pL, pR)

    # Get logLH
    def get_p(delta_P, delta, delta_min, lam):
        if delta > 0:
            # r = delta / delta_P
            return np.log(lam) - np.log(delta_P) - lam * delta / delta_P
        else:
            """ We set Delta=0 if the node is a leaf """
            # r = delta_min / delta_P
            return np.log(1 - np.exp(-lam * delta_min / delta_P))

    if delta_P < delta_min:
        logLH = - np.inf

    else:

        logLH = (
            get_p(delta_P, delta_L, delta_min, lam)
            + get_p(delta_P, delta_R, delta_min, lam)
            + np.log(1 / 2 / np.pi)
        )

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


def enrich_jet_logLH(jet, Lambda=None, delta_min=None, dij=False, alpha = None):
    """
    Attach splitting log likelihood to each edge, by calling recursive
    _get_jet_likelihood.
    """
    logLH = []
    dijList = []

    root_id = jet["root_id"]

    if Lambda is None:
        Lambda = float(jet.get("Lambda"))
        if Lambda is None:
            raise ValueError(f"No Lambda specified by the jet.")
    if delta_min is None:
        delta_min = jet.get("pt_cut")
        if delta_min is None:
            raise ValueError(f"No pt_cut specified by the jet.")

    _get_jet_logLH(
        jet,
        root_id = root_id,
        Lambda = Lambda,
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
        Lambda = None,
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
        delta_L = jet["deltas"][idL]
        delta_R = jet["deltas"][idR]


        # p_P =jet["content"][root_id]
        # delta_L = get_delta_PC(p_P, pL)
        # delta_R = get_delta_PC(p_P, pR)
        # print(idL, idR,pL,pR,delta_L,delta_R,  delta_min, Lambda)



        llh, _ , _ , _ = split_logLH(pL, delta_L, pR, delta_R, delta_min, Lambda)
        logLH.append(llh)
        # print('logLH = ', llh)

        if dij:

            """ dij=min(pTi^(2 alpha),pTj^(2 alpha)) * [arccos((pi.pj)/|pi|*|pj|)]^2 """
            # epsilon = 1e-6  # For numerical stability
            dijs= [float(llh)]

            for alpha in [-1,0,1]:

                tempCos = np.dot(pL, pR) / (np.linalg.norm(pL) * np.linalg.norm(pR))
                if abs(tempCos) > 1: tempCos = np.sign(tempCos)

                dijVal = np.sort((np.abs([pL[0],pR[0]])) ** (2 * alpha))[0]  * \
                         (
                             np.arccos(tempCos)
                          ) ** 2

                dijs.append(dijVal)

            dijList.append(dijs)


        _get_jet_logLH(
            jet,
            root_id = idL,
            Lambda = Lambda,
            delta_min = delta_min,
            logLH = logLH,
            dij = dij,
            dijList = dijList,
            alpha = alpha,
        )
        _get_jet_logLH(
            jet,
            root_id = idR,
            Lambda = Lambda,
            delta_min = delta_min,
            logLH = logLH,
            dij = dij,
            dijList = dijList,
            alpha = alpha,
        )

    else:

        logLH.append(0)


