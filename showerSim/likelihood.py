import time

# import matplotlib.pyplot as plt
import copy
import pickle
import sys
import numpy as np
import os


def split_logLH(pL, delta_L, pR, delta_R, delta_min, lam):
    """
    Takes two edges (p, delta) and
    return the splitting that generated them (p, delta_P, phi)
    with its log likelihood
    """
    p = pR + pL
    delta_vec = (pR - pL) / 2
    phi = np.arctan(delta_vec[0] / delta_vec[1])
    delta_P = np.sqrt(np.sum(delta_vec ** 2))

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

    return p, delta_P, phi, logLH


# -------------------------------------------------------------------------------------------------------------
###   GET THE SPLITTING LIKELIHOOD
# -------------------------------------------------------------------------------------------------------------


def split_likelihood(in_jet, root_id=None, parent_node_id=None):
    """
  Calculate the log likelihood of a splitting
  :param in_jet: dictionary with the jet info
  :param root_id:  id of the node used to get the split likelihood
  :param parent_node_id: id of the parent of the splitting node. If not provided, the function will search for it.
  :return: log(split likelihood) of the root id node.
  """

    Delta_0 = in_jet["Delta_0"]
    Lambda = in_jet["Lambda"]

    print("Decaying exponential rate Lambda =", Lambda)
    print("Initial scale for the splitting =", Delta_0)

    if root_id is None:
        root_id = in_jet["root_id"]

    if (
        in_jet["tree"][root_id][0] != -1
    ):  # because if it is -1, then there are no children and so it is a leaf (SM)

        # Node momentum
        p = in_jet["content"][root_id]
        print("Node momentum =", p)

        # Children momentum
        left = in_jet["tree"][root_id][
            0
        ]  # root_id left child. This gives the id of the left child (SM)
        #         right = jet["tree"][root_id][1] #root_id right child. This gives the id of the right child (SM)
        # So object["tree"][root_id] contains the position of the left and right children of object in jet["content"] (SM)

        pL = in_jet["content"][left]
        #         pR=jet["content"][right]

        # Parent momentum
        #         p_P=pL+pR

        if parent_node_id == None:
            parent_node_id = np.where(in_jet["tree"] == root_id)[0]

        print("parent_node_id=", parent_node_id)

        # If the node is the root of the tree, we use Delta_p=Delta_0. Delta_0. If the 1st splitting is for Delta=M_hard/2, then the 1st children will have Delta_p=Delta_0
        if root_id == in_jet["root_id"] or (
            in_jet["M_Hard"] != None
            and len(parent_node_id) > 0
            and parent_node_id[0] == in_jet["root_id"]
        ):
            Delta_p = Delta_0
            print("Delta parent=", Delta_p)

        else:

            # Parent momentum
            p_parent = in_jet["content"][parent_node_id]
            print("p_parent = ", p_parent)

            Delta_p = np.sqrt(np.sum((p_parent / 2 - p) ** 2))
            # print('p_parent/2-p =', p_parent / 2 - p)
            # print('(p_parent/2-p)**2) =', (p_parent / 2 - p) ** 2)
            print("Delta parent=", Delta_p)

        Delta = np.sqrt(np.sum((p / 2 - pL) ** 2))
        print("Delta = ", Delta)

        r = Delta / Delta_p
        print(" r =", r)

        split_likelihood = Lambda * np.exp(-Lambda * r)

        if root_id == in_jet["root_id"] and in_jet["M_Hard"] != None:
            # print('jet[M_Hard] =', in_jet['M_Hard'])
            split_likelihood = 1

        return np.log(split_likelihood)

    else:
        print("This is a leaf => There is no splitting")


# -------------------------------------------------------------------------------------------------------------
###   GET THE TREE (OR A BRANCH) LIKELIHOOD
# -------------------------------------------------------------------------------------------------------------


def traverse_likelihood(in_jet, root_id=None, parent_id=None):
    """
  This function calls "branch_likelihood" to traverse the tree (or a branch) and get the tree log likelihood.

  Args:
  :param in_jet: dictionary with the input jet dictionary
  :param root_id: Starting node to traverse the tree down to the leaves. If not the tree root id, then this function outputs the loglikelihood of a branch.
  :param parent_id: parent id of the starting node.
  :return: log likelihood of a branch
  """

    log_likelihood = []

    Delta_0 = in_jet["Delta_0"]
    Lambda = in_jet["Lambda"]

    print("Decaying exponential rate Lambda =", Lambda)
    print("Initial scale for the splitting =", Delta_0)
    print("jet[M_Hard] =", in_jet["M_Hard"])

    deltas = []
    draws = []

    branch_likelihood(
        in_jet,
        root_id=root_id,
        parent_id=parent_id,
        Delta_0=Delta_0,
        Lambda=Lambda,
        log_likelihood=log_likelihood,
        deltas=deltas,
        draws=draws,
    )

    print("---" * 10)
    print("Reconstructed deltas =", deltas)
    print("Reconstructed draws =", draws)
    print("---" * 10)

    print("log_likelihood list =", log_likelihood)

    branch_log_likelihood = np.sum(log_likelihood)

    return branch_log_likelihood


# -------------------------------------------------------------------------------------------------------------


# -----------
def branch_likelihood(
    in_jet,
    root_id=None,
    parent_id=None,
    Lambda=None,
    Delta_0=None,
    log_likelihood=None,
    deltas=None,
    draws=None,
):
    """
  Recursive function to traverse the tree (or a branch) and get the tree log likelihood.
  :param jet: dictionary with the jet info
  :param root_id: Starting node to traverse the tree down to the leaves. If not the tree root id, then this function outputs the loglikelihood of a branch.
  :param parent_id: parent id of the starting node.
  :param Lambda: Decaying exponential rate
  :param Delta_0: Initial scale for the splitting
  :param log_likelihood: List with the log likelihood of all previous splittings

  """

    # print('----' * 10)
    # print('Node id = ', root_id)

    if (
        in_jet["tree"][root_id][0] != -1
    ):  # because if it is -1, then there are no children and so it is a leaf (SM)

        # Node momentum
        p = in_jet["content"][root_id]
        # print('Node momentum =', p)

        # Left child momentum
        left = in_jet["tree"][root_id][
            0
        ]  # root_id left child. This gives the id of the left child (SM)
        #         right = jet["tree"][root_id][1] #root_id right child. This gives the id of the right child (SM)
        # So object["tree"][root_id] contains the position of the left and right children of object in jet["content"] (SM)

        pL = in_jet["content"][left]
        #         pR=jet["content"][right]

        if parent_id < 0 or (parent_id == 0 and in_jet["M_Hard"] != None):

            Delta_p = Delta_0
            # print('Delta parent=', Delta_p)

        else:

            # Parent momentum
            p_parent = in_jet["content"][parent_id]
            # print('p_parent = ', p_parent)

            Delta_p = np.sqrt(np.sum((p_parent / 2 - p) ** 2))
            # print('p_parent/2-p =', p_parent / 2 - p)
            # print('(p_parent/2-p)**2) =', (p_parent / 2 - p) ** 2)
            print("Delta parent=", Delta_p)

        Delta = np.sqrt(np.sum((p / 2 - pL) ** 2))

        r = Delta / Delta_p
        print(" r =", r)

        deltas.append(Delta)
        draws.append(r)

        split_likelihood = Lambda * np.exp(-Lambda * r)

        # If we use a hard scale to force  the 1st splitting the likelihood is 1.
        if parent_id < 0 and in_jet["M_Hard"] != None:
            # print('jet[M_Hard] =', in_jet['M_Hard'])
            split_likelihood = 1
            deltas.pop()
            deltas.append(in_jet["M_Hard"] / 2)
            draws.pop()
            draws.append("heavy")

        log_likelihood.append(np.log(split_likelihood))

        # print('Split_likelihood = ', split_likelihood)
        # print('Log(split likelihood)=', np.log(split_likelihood))

        #         branch_log_likelihood = log_likelihood + np.log(split_likelihood)
        #         print('----'*10)

        branch_likelihood(
            in_jet,
            root_id=in_jet["tree"][root_id][0],
            parent_id=root_id,
            Lambda=Lambda,
            Delta_0=Delta_0,
            log_likelihood=log_likelihood,
            deltas=deltas,
            draws=draws,
        )
        branch_likelihood(
            in_jet,
            root_id=in_jet["tree"][root_id][1],
            parent_id=root_id,
            Lambda=Lambda,
            Delta_0=Delta_0,
            log_likelihood=log_likelihood,
            deltas=deltas,
            draws=draws,
        )


# --------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # -------------------------
    # Load a jet
    input_jet = "tree_18_truth"
    input_dir = "../data/"

    fd = open(input_dir + str(input_jet) + ".pkl", "rb")
    jet_dic = pickle.load(fd, encoding="latin-1")
    # X= pickle.load(fd,encoding='latin-1')
    fd.close()

    print("jet dictionary =", jet_dic)

    # -------------------------
    # Get the log likelihood of a splitting. Choose the splitting node with node_id
    print("---" * 20)
    print("Calculating the log likelihood of a splitting")

    node_id = 4
    node_likelihood = split_likelihood(jet_dic, root_id=node_id)
    # print('+=+='*20)
    # node_likelihood = split_likelihood(jet_dic, root_id=1, parent_node_id=0)
    print("Node " + str(node_id) + " splitting log likelihood =", node_likelihood)

    # -------------------------
    # Get the log likelihood of a branch. Choose the starting node with root_node_id
    print("---" * 20)
    print("Calculating the log likelihood of a tree brach")

    root_node_id = 0
    brach_log_likelihood = traverse_likelihood(
        jet_dic, root_id=root_node_id, parent_id=-1
    )

    print(
        "Branch log likelihood from root id:" + str(root_node_id) + " =",
        brach_log_likelihood,
    )
