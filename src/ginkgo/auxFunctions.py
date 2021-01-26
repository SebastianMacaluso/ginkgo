import numpy as np
import pickle
import time





def traversePhi(jet, node_id, constPhiList, PhiDeltaList, PhiDeltaListRel):
    """
    Recursive function that traverses the tree. Gets leaves angle phi, and delta_parent phi angle for all parents in the tree.
    """

    if jet["tree"][node_id, 0] == -1:

        constPhi = np.arctan2(jet["content"][node_id][0], jet["content"][node_id][1])
        constPhiList.append(constPhi)

    else:

        """ Get angle for the splitting value Delta """
        idL = jet["tree"][node_id][0]
        idR = jet["tree"][node_id][1]
        pL = jet["content"][idL]
        pR = jet["content"][idR]

        delta_vec = (pR - pL) / 2

        """ Find subjet angle"""
        PhiPseudoJet = np.arctan2(jet["content"][node_id][0], jet["content"][node_id][1])

        """ arctan2 to find the right quadrant"""
        TempDeltaPhi = np.arctan2(delta_vec[0], delta_vec[1])
        PhiDeltaList.append(TempDeltaPhi)

        PhiDeltaListRel.append( abs(TempDeltaPhi - PhiPseudoJet))

        traversePhi(
            jet,
            jet["tree"][node_id, 0],
            constPhiList,
            PhiDeltaList,
            PhiDeltaListRel,
        )

        traversePhi(
            jet,
            jet["tree"][node_id, 1],
            constPhiList,
            PhiDeltaList,
            PhiDeltaListRel,
        )

    return constPhiList, PhiDeltaList, PhiDeltaListRel
