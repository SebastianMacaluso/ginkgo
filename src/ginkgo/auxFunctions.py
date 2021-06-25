import numpy as np
import pickle
import time
import logging
import pyro

from .utils import get_logger

logger = get_logger(level=logging.INFO)




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




def traverse(
        root,
        jetContent,
        jetTree=None,
        Nleaves=None,
        # flip=False,
):
    """
    This function call the recursive function _traverse_rec to make the trees starting from the root
    :param root: root node id
    :param jetContent: array with the momentum of all the nodes of the jet tree (both leaves and inners).
    :param jetTree: dictionary that has the node id of a parent as a key and a list with the id of the 2 children as the values
    :param Nleaves: Number of constituents (leaves)
    :param dendrogram: bool. If True, then return tree_ancestors list.

    :return:
    - tree: Reclustered tree structure.
    - content: Reclustered tree momentum vectors
    - node_id:   list where leaves idxs are added in the order that they appear when we traverse the reclustered tree (each number indicates the node id that we picked when we did the reclustering.). However, the idx value specifies the order in which the leaf nodes appear when traversing the origianl jet (e.g. truth level jet). The value here is an integer between 0 and Nleaves.
    So if we go from truth to kt algorithm, then in the truth tree the leaves go as [0,1,2,3,4,,...,Nleaves-1]
    - tree_ancestors: List with one entry for each leaf of the tree, where each entry lists all the ancestor node ids when traversing the tree from the root to the leaf node.

    """

    tree = []
    content = []
    node_id = []
    tree_ancestors = []
    leaves = []

    globals()["Bernoulli_dist"] = pyro.distributions.Bernoulli(probs=0.5)

    _traverse_flipLR(
    root,
    -1,
    False,
    tree,
    content,
    jetContent,
    leaves,
    jetTree=jetTree,
    Nleaves=Nleaves,
        # flip=flip,
    )


    return tree, content, leaves, node_id, tree_ancestors






def _traverse_flipLR(
        root,
        parent_id,
        is_left,
        tree,
        content,
        jetContent,
        leaves,
        jetTree=None,
        Nleaves=None,
        # flip = False,
):
    """
	Recursive function to build the jet tree structure.
	:param root: parent node momentum
	:param parent_id: parent node idx
	:param is_left: bool.
	:param tree: List with the tree
	:param content: List with the momentum vectors
	:param jetContent: array with the momentum of all the nodes of the jet tree (both leaves and inners).
	:param jetTree: dictionary that has the node id of a parent as a key and a list with the id of the 2 children as the values
	:param Nleaves: Number of constituents (leaves)
	:param node_id: list where leaves idxs are added in the order they appear when we traverse the reclustered tree (each number indicates the node id
	that we picked when we did the reclustering.). However, the idx value specifies the order in which the leaf nodes appear when traversing the truth level jet . The value here is an integer between 0 and Nleaves.
	So if we went from truth to kt algorithm, then in the truth tree the leaves go as [0,1,2,3,4,,...,Nleaves-1]
	:param ancestors: 1 entry of tree_ancestors (there is one for each leaf of the tree). It is appended to tree_ancestors.
	:param tree_ancestors: List with one entry for each leaf of the tree, where each entry lists all the ancestor node ids when traversing the tree from the root to the leaf node.
	:param dendrogram: bool. If True, append ancestors to tree_ancestors list.
	"""

    """"
	(With each momentum vector we increase the content array by one element and the tree array by 2 elements. 
	But then we take id=tree.size()//2, so the id increases by 1.)
	"""
    id = len(tree) // 2

    if parent_id >= 0:
        if is_left:

            """Insert in the tree list, the location of the lef child in the content array."""
            tree[2 * parent_id] = id
        else:

            """Insert in the tree list, the location of the right child in the content array."""
            tree[2 * parent_id + 1] = id


    """Insert 2 new nodes to the vector that constitutes the tree. If the current node is a parent, then we will replace the -1 with its children idx in the content array"""
    tree.append(-1)
    tree.append(-1)


    """ Append node momentum to content list """
    content.append(jetContent[root])

    # print('Root = ', root)
    # print("Nleaves = ", Nleaves)
    """ Move from the root down recursively until we get to the leaves. """
    # if root <= Nleaves and root>0:
    if jetTree[root][0] != -1:

        # print('Root2 = ', root)
        children = jetTree[root]

        logger.debug(f"Children = {children}")

        flip = pyro.sample("Bernoulli" + str(root), Bernoulli_dist)

        if flip:
            L_idx = children[1]
            R_idx = children[0]

        else:
            L_idx = children[0]
            R_idx = children[1]


        _traverse_flipLR(L_idx,
                         id,
                      True,
                      tree,
                      content,
                      jetContent,
                         leaves,
                      jetTree,
                      Nleaves=Nleaves,
                         # flip = flip,
                      )

        _traverse_flipLR(R_idx,
                      id,
                      False,
                      tree,
                      content,
                      jetContent,
                         leaves,
                      jetTree,
                      Nleaves=Nleaves,
                         # flip=flip,
                      )



    else:
        """ If the node is a leaf, then append idx to node_id and its ancestors as a new row of tree_ancestors """
        leaves.append(jetContent[root])



