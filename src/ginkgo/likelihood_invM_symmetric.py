import pickle
import numpy as np
import torch
from scipy.special import logsumexp
import copy

import warnings
warnings.simplefilter('default')

def get_delta_LR(pL, pR):
    """
    Calculate invariant mass of a node, given its child momenta
    """
    pP = pR + pL

    """Parent invariant mass squared"""
    tp1 = pP[0] ** 2 - np.linalg.norm(pP[1::]) ** 2

    return tp1





def split_logLH_with_stop_nonstop_prob(pL, pR, t_cut, lam):
    """
    Take two nodes and return the splitting log likelihood
    """
    tL = pL[0] ** 2 - np.linalg.norm(pL[1::]) ** 2
    tR = pR[0] ** 2 - np.linalg.norm(pR[1::]) ** 2


    pP = pR + pL


    """Parent invariant mass squared"""
    tp = pP[0] ** 2 - np.linalg.norm(pP[1::]) ** 2

    if tL < 0 or tR < 0:  print("tL = ", tL, " | tR= ", tR, " | tP = ", tp, " | tcut = ", t_cut)
    # print("lam= ",lam, " | pP = ", pP, " pL = ", pL, " | pR= ", pR)

    """ We add a normalization factor -np.log(1 - np.exp(- lam)) because we need the mass squared to be strictly decreasing. This way the likelihood integrates to 1 for 0<t<t_p. All leaves should have t=0, this is a convention we are taking (instead of keeping their value for t given that it is below the threshold t_cut)"""
    def get_logp(tP_local, t, t_cut, lam):


        if t > t_cut:
            """ Probability of the shower to stop F_s"""
            # F_s = 1 / (1 - np.exp(- lam)) * (1 - np.exp(-lam * t_cut / tP_local))
            # if F_s>=1:
            #     print("Fs = ", F_s, "| tP_local = ", tP_local, "| t_cut = ", t_cut, "| t = ",t)

            # print("Inner - t = ",t," | tL =",tL, " | tR = ",tR," pL = ", pL, " | pR= ", pR, " | pP = ", pP, "logLH = ",-np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(tP_local) - lam * t / tP_local)
            # return -np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(tP_local) - lam * t / tP_local + np.log(1-F_s)
            return -np.log(1 - np.exp(- lam/4)) + np.log(lam) - np.log(tP_local) - lam * t / tP_local

        else: # For leaves we have t<t_cut
            t_upper = min(tP_local,t_cut) #There are cases where tp2 < t_cut
            log_F_s = -np.log(1 - np.exp(- lam)) + np.log(1 - np.exp(-lam * t_upper / tP_local))
            # print("Outer - t = ",t," | tL =",tL, " | tR = ",tR," pL = ", pL, " | pR= ", pR, " | pP = ", pP, "logLH = ", log_F_s)
            return log_F_s


    if tp <= t_cut:
        "If the pairing is not allowed"
        logLH = - np.inf

    else:
        """We sample a unit vector uniformly over the 2-sphere, so the angular likelihood is 1/(4*pi)"""

        logLH = get_logp(tp, tL, t_cut, lam) + get_logp(tp, tR, t_cut, lam)+ np.log(1 / (4 * np.pi))

    return logLH




#
# def get_delta_PC(p, pC):
#     """
#     Calculate delta of an edge, given its momentum and the momentum of a child
#     """
#     return np.sqrt(np.sum((p / 2 - pC) ** 2))
#

# def split_logLH_with_both_stop_nonstop_prob(pL,  pR,  t_cut, lam):
# # def split_logLH_with_both_stop_nonstop_prob(pL, L_is_leaf, pR, R_is_leaf,  t_cut, lam):
#     """
#     Take two nodes and return the splitting log likelihood. We have a stop/non-stop probability of splitting and the correct normalization from t_cut to tp for the internal nodes likelihood density.
#     """
#     tL = pL[0] ** 2 - np.linalg.norm(pL[1::]) ** 2
#     tR = pR[0] ** 2 - np.linalg.norm(pR[1::]) ** 2
#
#
#     pP = pR + pL
#
#     """Parent invariant mass squared"""
#     tp = pP[0] ** 2 - np.linalg.norm(pP[1::]) ** 2
#
#
#     """ We add a normalization factor -np.log(1 - np.exp(- lam)) because we need the mass squared to be strictly decreasing. This way the likelihood integrates to 1 for 0<t<t_p. All leaves should have t=0, this is a convention we are taking (instead of keeping their value for t given that it is below the threshold t_cut)"""
#     def get_logp(tP_local, t, t_cut, lam):
#
#
#         if t > t_cut:
#             """ Probability of the shower to stop F_s"""
#             F_s = 1 / (1 - np.exp(- lam)) * (1 - np.exp(-lam * t_cut / tP_local))
#             # if F_s>=1:
#             #     print("Fs = ", F_s, "| tP_local = ", tP_local, "| t_cut = ", t_cut, "| t = ",t)
#
#             return -np.log(np.exp(-lam * t_cut / tP_local) - np.exp(- lam)) + np.log(lam) - np.log(tP_local) - lam * t / tP_local + np.log(1-F_s)
#             # return -np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(tP_local) - lam * t / tP_local
#
#         else: # For leaves we have t<t_cut
#             t_upper = min(tP_local,t_cut) #There are cases where tp2 < t_cut
#             log_F_s = -np.log(1 - np.exp(- lam)) + np.log(1 - np.exp(-lam * t_upper / tP_local))
#             return log_F_s
#
#
#     if tp <= t_cut:
#         """If the pairing is not allowed"""
#         logLH = - np.inf
#
#     # elif (tL > t_cut and L_is_leaf) or (tR > t_cut and R_is_leaf):
#     #     """If we vary t_cut such that the leaves values for t in the dataset are above t_cut"""
#     #     # print("Leaf value above t_cut, not allowed | ","tL = ", tL, " | tR = ", tR, " | t_cut =", t_cut )
#     #     logLH = - np.inf
#
#
#     else:
#         """We sample a unit vector uniformly over the 2-sphere, so the angular likelihood is 1/(4*pi)"""
#
#         tpLR = (np.sqrt(tp) - np.sqrt(tL)) ** 2
#         tpRL = (np.sqrt(tp) - np.sqrt(tR)) ** 2
#
#         logpLR = np.log(1/2)+ get_logp(tp, tL, t_cut, lam) + get_logp(tpLR, tR, t_cut, lam) #First sample tL
#         logpRL = np.log(1/2)+ get_logp(tp, tR, t_cut, lam) + get_logp(tpRL, tL, t_cut, lam) #First sample tR
#
#         logp_split = logsumexp(np.asarray([logpLR, logpRL]))
#
#         logLH = (logp_split + np.log(1 / (4 * np.pi)) )
#
#     return logLH

#
#
#
# def split_logLH_with_stop_prob(pL, L_is_leaf, pR, R_is_leaf,  t_cut, lam):
#     """
#     Take two nodes and return the splitting log likelihood. We only have the stop probability.
#     """
#     tL = pL[0] ** 2 - np.linalg.norm(pL[1::]) ** 2
#     tR = pR[0] ** 2 - np.linalg.norm(pR[1::]) ** 2
#
#
#     pP = pR + pL
#
#     """Parent invariant mass squared"""
#     tp = pP[0] ** 2 - np.linalg.norm(pP[1::]) ** 2
#
#
#     """ We add a normalization factor -np.log(1 - np.exp(- lam)) because we need the mass squared to be strictly decreasing. This way the likelihood integrates to 1 for 0<t<t_p. All leaves should have t=0, this is a convention we are taking (instead of keeping their value for t given that it is below the threshold t_cut)"""
#     def get_logp(tP_local, t, t_cut, lam):
#
#
#         if t > t_cut:
#             """ Probability of the shower to stop F_s"""
#             # F_s = 1 / (1 - np.exp(- lam)) * (1 - np.exp(-lam * t_cut / tP_local))
#             # if F_s>=1:
#             #     print("Fs = ", F_s, "| tP_local = ", tP_local, "| t_cut = ", t_cut, "| t = ",t)
#
#             # return -np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(tP_local) - lam * t / tP_local + np.log(1-F_s)
#             return -np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(tP_local) - lam * t / tP_local
#
#         else: # For leaves we have t<t_cut
#             t_upper = min(tP_local,t_cut) #There are cases where tp2 < t_cut
#             log_F_s = -np.log(1 - np.exp(- lam)) + np.log(1 - np.exp(-lam * t_upper / tP_local))
#             return log_F_s
#
#
#     if tp <= t_cut:
#         """If the pairing is not allowed"""
#         logLH = - np.inf
#
#     elif (tL > t_cut and L_is_leaf) or (tR > t_cut and R_is_leaf):
#         """If we vary t_cut such that the leaves values for t in the dataset are above t_cut"""
#         # print("Leaf value above t_cut, not allowed | ","tL = ", tL, " | tR = ", tR, " | t_cut =", t_cut )
#         logLH = - np.inf
#
#     # elif (tL < t_cut and not L_is_leaf) or (tR < t_cut and not R_is_leaf):
#     #     """If we vary t_cut such that internal nodesthe leaves values for t in the dataset are above t_cut"""
#     #     logLH = - np.inf
#
#     else:
#         """We sample a unit vector uniformly over the 2-sphere, so the angular likelihood is 1/(4*pi)"""
#
#         tpLR = (np.sqrt(tp) - np.sqrt(tL)) ** 2
#         tpRL = (np.sqrt(tp) - np.sqrt(tR)) ** 2
#
#         logpLR = np.log(1/2)+ get_logp(tp, tL, t_cut, lam) + get_logp(tpLR, tR, t_cut, lam) #First sample tL
#         logpRL = np.log(1/2)+ get_logp(tp, tR, t_cut, lam) + get_logp(tpRL, tL, t_cut, lam) #First sample tR
#
#         logp_split = logsumexp(np.asarray([logpLR, logpRL]))
#
#         logLH = (logp_split + np.log(1 / (4 * np.pi)) )
#
#     return logLH










#
# def split_logLH_with_stop_nonstop_prob(pL, pR, t_cut, lam):
#     """
#     Take two nodes and return the splitting log likelihood
#     """
#     tL = pL[0] ** 2 - np.linalg.norm(pL[1::]) ** 2
#     tR = pR[0] ** 2 - np.linalg.norm(pR[1::]) ** 2
#
#     if -1e-2<tL<0: tL=0
#     if -1e-2<tR<0: tR=0
#
#     pP = pR + pL
#
#
#     # print("pL = ", pL, " | pR= ", pR," | pP = ", pP)
#
#     """Parent invariant mass squared"""
#     tp = pP[0] ** 2 - np.linalg.norm(pP[1::]) ** 2
#     if tL < 0 or tR < 0:  print("tL = ", tL, " | tR= ", tR, " | tP = ", tp, " | tcut = ", t_cut)
#
#     radius = np.sqrt(tL / tp) + np.sqrt(tR / tp)**2
#     t_draw = radius * tp
#
#     """ We add a normalization factor -np.log(1 - np.exp(- lam)) because we need the mass squared to be strictly decreasing. This way the likelihood integrates to 1 for 0<t<t_p. All leaves should have t=0, this is a convention we are taking (instead of keeping their value for t given that it is below the threshold t_cut)"""
#     def get_logp(tP_local, t, t_cut, lam):
#
#
#         if t > t_cut:
#             """ Probability of the shower to stop F_s"""
#             # F_s = 1 / (1 - np.exp(- lam)) * (1 - np.exp(-lam * t_cut / tP_local))
#             # if F_s>=1:
#             #     print("Fs = ", F_s, "| tP_local = ", tP_local, "| t_cut = ", t_cut, "| t = ",t)
#
#             # print("pL = ", pL, " | pR= ", pR, " | pP = ", pP, "logLH = ",-np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(tP_local) - lam * t / tP_local)
#             # print("Inner - t = ", t, " pL = ", pL, " | pR= ", pR, " | pP = ", pP,
#             #       "logLH = ", -np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(tP_local) - lam * t / tP_local)
#
#             # return -np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(tP_local) - lam * t / tP_local + np.log(1-F_s)
#             return -np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(tP_local) - lam * t / tP_local
#
#         else: # For leaves we have t<t_cut
#             log_F_s = -np.log(1 - np.exp(- lam)) + np.log(1 - np.exp(-lam * t_cut / tP_local))
#             # print("pL = ", pL, " | pR= ", pR, " | pP = ", pP, "logLH = ",log_F_s)
#             # print("Outer - t = ",t," pL = ", pL, " | pR= ", pR, " | pP = ", pP, "logLH = ", log_F_s)
#             return log_F_s
#
#
#     if tp <= t_cut:
#         "If the pairing is not allowed"
#         logLH = - np.inf
#
#     else:
#         """We  sample a unit vector uniformly over the 1-sphere, so the angular likelihood is 1/(2*pi).
#         We also sample a unit vector uniformly over the 2-sphere, so the angular likelihood is 1/(4*pi)"""
#
#         # tpLR = tp
#         # tpRL = tp
#         #
#         # logpLR = np.log(1/2)+ get_logp(tp, tL, t_cut, lam) + get_logp(tpLR, tR, t_cut, lam) #First sample tL
#         # logpRL = np.log(1/2)+ get_logp(tp, tR, t_cut, lam) + get_logp(tpRL, tL, t_cut, lam) #First sample tR
#         #
#         # logp_split = logsumexp(np.asarray([logpLR, logpRL]))
#
#         logLH = (get_logp(tp, t_draw, t_cut, lam) + np.log(1 / (2 * np.pi)) + np.log(1 / (4 * np.pi)) )
#
#     return logLH




# def split_logLH_with_stop_nonstop_prob(pL, pR, t_cut, lam):
#     """
#     Take two nodes and return the splitting log likelihood
#     """
#     tL = pL[0] ** 2 - np.linalg.norm(pL[1::]) ** 2
#     tR = pR[0] ** 2 - np.linalg.norm(pR[1::]) ** 2
#
#
#     pP = pR + pL
#
#     """Parent invariant mass squared"""
#     tp1 = pP[0] ** 2 - np.linalg.norm(pP[1::]) ** 2
#
#     tmax = max(tL,tR)
#     tmin = min(tL,tR)
#
#     tp2 = (np.sqrt(tp1) - np.sqrt(tmax)) ** 2
#
#     """ We add a normalization factor -np.log(1 - np.exp(- lam)) because we need the mass squared to be strictly decreasing. This way the likelihood integrates to 1 for 0<t<t_p. All leaves should have t=0, this is a convention we are taking (instead of keeping their value for t given that it is below the threshold t_cut)"""
#     def get_logp(tP_local, t, t_cut, lam):
#
#
#         if t > t_cut:
#             """ Probability of the shower to stop F_s"""
#             F_s = 1 / (1 - np.exp(- lam)) * (1 - np.exp(-lam * t_cut / tP_local))
#             # if F_s>=1:
#             #     print("Fs = ", F_s, "| tP_local = ", tP_local, "| t_cut = ", t_cut, "| t = ",t)
#
#             return -np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(tP_local) - lam * t / tP_local + np.log(1-F_s)
#
#         else: # For leaves we have t<t_cut
#             t_upper = min(tP_local,t_cut) #There are cases where tp2 < t_cut
#             log_F_s = -np.log(1 - np.exp(- lam)) + np.log(1 - np.exp(-lam * t_upper / tP_local))
#             return log_F_s
#
#
#     if tp1 <= t_cut:
#         "If the pairing is not allowed"
#         logLH = - np.inf
#
#     else:
#         """We sample a unit vector uniformly over the 2-sphere, so the angular likelihood is 1/(4*pi)"""
#         logLH = (
#             get_logp(tp1, tmax, t_cut, lam)
#             + get_logp(tp2, tmin, t_cut, lam)
#             + np.log(1 / (4 * np.pi))
#         )
#
#     return logLH







# def split_logLH_stop_nonstop_prob(pL, tL, pR, tR, t_cut, lam):
#     """
#     Take two nodes and return the splitting log likelihood
#     """
#     pP = pR + pL
#
#     """Parent invariant mass squared"""
#     tp1 = pP[0] ** 2 - np.linalg.norm(pP[1::]) ** 2
#
#     tmax = max(tL,tR)
#     tmin = min(tL,tR)
#
#     tp2 = (np.sqrt(tp1) - np.sqrt(tmax)) ** 2
#
#     """ We add a normalization factor -np.log(1 - np.exp(- lam)) because we need the mass squared to be strictly decreasing. This way the likelihood integrates to 1 for 0<t<t_p. All leaves should have t=0, this is a convention we are taking (instead of keeping their value for t given that it is below the threshold t_cut)"""
#     def get_p(tP, t, t_cut, lam):
#
#         if t > 0:
#             return -np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(tP) - lam * t / tP
#
#         else: # For leaves we have t<t_min, then we set t=0
#             t_upper = min(tP,t_cut)
#             log_F_s = -np.log(1 - np.exp(- lam)) + np.log(1 - np.exp(-lam * t_upper / tP))
#             return log_F_s
#
#     "If the pairing is not allowed"
#     if tp1 <= t_cut:
#         logLH = - np.inf
#
#     else:
#         """ Probability of the shower to stop F_s"""
#         F_s = 1 / (1 - np.exp(- lam)) * (1 - np.exp(-lam * t_cut / tp1))
#         # print( "Fs = ",F_s, "tp1 = ", tp1, "| t_cut = ", t_cut)
#
#         """We sample a unit vector uniformly over the 2-sphere, so the angular likelihood is 1/(4*pi)"""
#         logLH = (
#             get_p(tp1, tmax, t_cut, lam)
#             + get_p(tp2, tmin, t_cut, lam)
#             + np.log(1 / (4 * np.pi))
#             + np.log(1-F_s)
#         )
#
#     return logLH


def split_logLH(pL, tL, pR, tR, t_cut, lam):
    warnings.warn(
        "split_logLH is deprecated. Use 'split_logLH_stop_nonstop_prob'. Note: if reproducing results from arXiv:2105.10512 , arXiv:2104.07061, arXiv:2011.08191 or arXiv:2002.11661, use 'split_logLH_without_non_stop_prob' ",
        DeprecationWarning
    )
    # return

def split_logLH_without_non_stop_prob(pL, tL, pR, tR, t_cut, lam):
    """
    Likelihood function used in  arXiv:2105.10512 , arXiv:2104.07061, arXiv:2011.08191 and arXiv:2002.11661. Take two nodes and return the splitting log likelihood
    """
    pP = pR + pL

    """Parent invariant mass squared"""
    tp1 = pP[0] ** 2 - np.linalg.norm(pP[1::]) ** 2

    tmax = max(tL,tR)
    tmin = min(tL,tR)

    tp2 = (np.sqrt(tp1) - np.sqrt(tmax)) ** 2

    """ We add a normalization factor -np.log(1 - np.exp(- lam)) because we need the mass squared to be strictly decreasing. This way the likelihood integrates to 1 for 0<t<t_p. All leaves should have t=0, this is a convention we are taking (instead of keeping their value for t given that it is below the threshold t_cut)"""
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


        # llh = split_logLH(pL, tL, pR, tR, delta_min, Lambda)
        llh = split_logLH_with_stop_nonstop_prob(pL,  pR,  delta_min, Lambda)
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




##############################
def reevaluate_jet_logLH(in_jet, delta_min=None, Lambda = None, LambdaRoot = None, split_log_LH=None):
    """
    Attach splitting log likelihood to each edge, by calling recursive
    _get_jet_likelihood.
    """
    jet = copy.deepcopy(in_jet)
    logLH = []
    root_id = jet["root_id"]

    if delta_min is None:
        raise ValueError(f"No pt_cut parameter specified.")
    if Lambda is None:
        raise ValueError(f"No lambda parameter specified.")


    _evaluate_jet_logLH(
        jet,
        root_id = root_id,
        delta_min = delta_min,
        Lambda=Lambda,
        LambdaRoot =LambdaRoot,
        logLH = logLH,
        split_log_LH=split_log_LH
    )

    jet["logLH"] = np.asarray(logLH)


    return jet


def _evaluate_jet_logLH(
        jet,
        root_id = None,
        delta_min = None,
        Lambda = None,
        LambdaRoot = None,
        logLH = None,
        split_log_LH = None
):
    """
    Recursively enrich every edge from root_id downward with their log likelihood.
    log likelihood of a leaf is 0. Assumes a valid jet.
    """
    if jet["tree"][root_id][0] != -1:

        # print('Lambda = ', Lambda)
        idL = jet["tree"][root_id][0]
        idR = jet["tree"][root_id][1]
        pL = jet["content"][idL]
        pR = jet["content"][idR]

        # llh = split_logLH(pL, tL, pR, tR, delta_min, Lambda)
        if root_id==jet["root_id"]:
            llh = split_log_LH(pL,  pR,  delta_min, LambdaRoot)
        else:
            llh = split_log_LH(pL, pR, delta_min, Lambda)

        # print(llh)

        logLH.append(llh)
        # print('logLH = ', llh)


        _evaluate_jet_logLH(
            jet,
            root_id = idL,
            delta_min = delta_min,
            Lambda =Lambda,
            LambdaRoot =LambdaRoot,
            logLH = logLH,
            split_log_LH = split_log_LH
        )
        _evaluate_jet_logLH(
            jet,
            root_id = idR,
            delta_min = delta_min,
            Lambda=Lambda,
            LambdaRoot = LambdaRoot,
            logLH = logLH,
            split_log_LH= split_log_LH
        )

    else:

        logLH.append(0)





def split_logLH_forceLR(pL, pR, t_cut, lam):
    """
    Take two nodes and return the splitting log likelihood
    """
    tL = pL[0] ** 2 - np.linalg.norm(pL[1::]) ** 2
    tR = pR[0] ** 2 - np.linalg.norm(pR[1::]) ** 2


    pP = pR + pL

    """Parent invariant mass squared"""
    tp = pP[0] ** 2 - np.linalg.norm(pP[1::]) ** 2


    """ We add a normalization factor -np.log(1 - np.exp(- lam)) because we need the mass squared to be strictly decreasing. This way the likelihood integrates to 1 for 0<t<t_p. All leaves should have t=0, this is a convention we are taking (instead of keeping their value for t given that it is below the threshold t_cut)"""
    def get_logp(tP_local, t, t_cut, lam):


        if t > t_cut:
            """ Probability of the shower to stop F_s"""
            # F_s = 1 / (1 - np.exp(- lam)) * (1 - np.exp(-lam * t_cut / tP_local))
            # if F_s>=1:
            #     print("Fs = ", F_s, "| tP_local = ", tP_local, "| t_cut = ", t_cut, "| t = ",t)

            # return -np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(tP_local) - lam * t / tP_local + np.log(1-F_s)
            return -np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(tP_local) - lam * t / tP_local


        else: # For leaves we have t<t_cut
            t_upper = min(tP_local,t_cut) #There are cases where tp2 < t_cut
            log_F_s = -np.log(1 - np.exp(- lam)) + np.log(1 - np.exp(-lam * t_upper / tP_local))
            return log_F_s


    if tp <= t_cut:
        "If the pairing is not allowed"
        logLH = - np.inf

    else:
        """We sample a unit vector uniformly over the 2-sphere, so the angular likelihood is 1/(4*pi)"""

        tpLR = (np.sqrt(tp) - np.sqrt(tL)) ** 2

        logpLR = get_logp(tp, tL, t_cut, lam) + get_logp(tpLR, tR, t_cut, lam) #First sample tL

        logLH = logpLR + np.log(1 / (4 * np.pi))

    return logLH