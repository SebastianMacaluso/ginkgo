import numpy as np
import torch
import pprint
import matplotlib.pyplot as plt
import sys
import pickle
import argparse
import logging
import os

from ginkgo import invMass_ginkgo
from ginkgo.utils import get_logger


logger = get_logger(level=logging.WARNING)

augmented_data=False

# Nsamples=sys.argv[1]
# start = int(sys.argv[2])
# end = int(sys.argv[3])
#-----------------------
'''
Runs Ginkgo invariant mass toy parton shower
'''
parser = argparse.ArgumentParser(description="Generate synthetic jets")
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Increase output verbosity"
)

parser.add_argument(
    "--Nsamples", type=int, required=True, help="Number of jet samples"
)

parser.add_argument(
    "--id", type=str, default=0, help="dataset id"
)
parser.add_argument(
    "--jetType", type=str, default="W", help="jet type: W or QCD"
)


parser.add_argument(
    "--minLeaves", type=int, help="Minimum number of jet constituents (leaves)"
)
parser.add_argument(
    "--maxLeaves", type=int, help="Maximum number of jet constituents (leaves)"
)
parser.add_argument(
    "--maxNTry", type=int, default=20000, help="Maximum number of times we run the simulator"
)

args = parser.parse_args()


rate2=torch.tensor(8.)

# Parameters to get ~<10 constituents to test the trellis algorithm
# pt_min = torch.tensor(4.**2)
#
pt_min = torch.tensor(0.5**2)

### Physics inspired parameters to get ~ between 20 and 50 constituents
W_rate = 3.
QCD_rate = 1.5
# pt_min = torch.tensor(1.2**2)

QCD_mass = 30.

if args.jetType == "W":
    """ W jets"""
    rate=torch.tensor([W_rate,QCD_rate]) #Entries: [root node, every other node] decaying rates. Choose same values for a QCD jet
    M2start = torch.tensor(80.**2)

elif args.jetType == "QCD":
    """ QCD jets """
    rate=torch.tensor([QCD_rate,QCD_rate]) #Entries: [root node, every other node] decaying rates. Choose same values for a QCD jet
    M2start = torch.tensor(QCD_mass**2)
else:
    raise ValueError("Choose a valid jet type between W or QCD")


jetM = np.sqrt(M2start.numpy())

# pt_min = torch.tensor(1**2)
# pt_min = torch.tensor(10**2)


jetdir = np.array([1,1,1])
jetP = 400.
jetvec = jetP * jetdir / np.linalg.norm(jetdir)

jet4vec = np.concatenate(([np.sqrt(jetP**2 + jetM**2)], jetvec))
logger.debug(f"jet4vec = {jet4vec}")



simulator = invMass_ginkgo.Simulator(jet_p=jet4vec,
                                     pt_cut=float(pt_min),
                                     Delta_0=M2start,
                                     M_hard=jetM ,
                                     num_samples=int(args.Nsamples),
                                     minLeaves =int(args.minLeaves),
                                     maxLeaves = int(args.maxLeaves),
                                     maxNTry = int(args.maxNTry)
                                     )


#---------------
if not augmented_data:
    jet_list = simulator(rate)

    logger.debug(f"---"*10)
    logger.debug(f"jet_list = {jet_list}")

else:

  jet_list, joint_score, joint_log_ratio, joint_log_prob = simulator.augmented_data(rate,
                                                                                    None,
                                                                                    rate2,
                                                                                    exponential=True,
                                                                                    uniform=False)


  logger.info(f"---"*10)
  logger.debug(f"jet_list = {jet_list}")
  logger.info(f"joint_score = {joint_score}")
  logger.info(f"joint_log_likelihood_ratio= {joint_log_ratio}")
  logger.info(f"joint_log_prob= {joint_log_prob}")
  logger.info(f"---"*10)




  def jet_log_likelihood():

    Lambda = jet_list[0]['Lambda']
    log_likelihood = 0

    if Lambda.requires_grad:
      Lambda=Lambda.detach().numpy()

    for entry in jet_list[0]['draws'][1::]:
      #     if entry!=-1:
      if entry.requires_grad:
        entry = entry.detach().numpy()

      log_likelihood += np.log(Lambda * np.exp(-Lambda * entry))

    return log_likelihood


  jet_log_likelihood_cross_check=jet_log_likelihood()
  logger.info(f" jet_log_likelihood cross-chec"
              f"k = {jet_log_likelihood_cross_check}")
  #---------------------------



ToyModelDir = "/scratch/sm4511/ToyJetsShower/data/invMassGinkgo/Trellis"
TreeAlgoDir = "/scratch/sm4511/TreeAlgorithms/data/invMassGinkgo/Trellis/Truth"

# ToyModelDir = "/scratch/sm4511/ToyJetsShower/data/invMassGinkgo/"
# TreeAlgoDir = "/scratch/sm4511/TreeAlgorithms/data/invMassGinkgo/Truth"

# os.system("mkdir -p "+ToyModelDir)
# os.system("mkdir -p "+TreeAlgoDir)
#
#
# simulator.save(jet_list, TreeAlgoDir, "tree_" + str(args.Nsamples) + "_truth_" + str(args.id))
# simulator.save(jet_list, ToyModelDir, "tree_"+str(args.Nsamples)+"_truth_"+str(args.id))




save = False

if save:
    TreeAlgoDataDir = "../TreeAlgorithms/data/invMassGinkgo/Truth"
    # ShowerDatadir = "./data/invMassGinkgo"
    ShowerDatadir = "/Users/sebastianmacaluso/Documents/PrinceData/invMassGinkgo"
    A_star_dir="/Users/sebastianmacaluso/Dropbox/Documents/Physics_projects/simulator/a_star_trellis/data/Ginkgo"
    os.system("mkdir -p "+ShowerDatadir)
    os.system("mkdir -p "+TreeAlgoDataDir)

    # simulator.save(jet_list, ShowerDatadir, "tree_"+ str(args.jetType)+"_leaves_"+str(args.minLeaves)+"_" + str(args.Nsamples)+"_m2min_"+str(float(pt_min))[0:3] +"_rate01_"+str(rate.numpy()[0])+"_"+str(rate.numpy()[1]))
    # simulator.save(jet_list,TreeAlgoDataDir, "tree_"+ str(args.jetType)+"_" +str(args.Nsamples) +"_m2min_"+str(float(pt_min))[0:3]+"_rate01_"+str(rate.numpy()[0])+"_"+str(rate.numpy()[1]))
    simulator.save(jet_list, A_star_dir, "test_" + str(args.minLeaves) + "_jets")

# To run: python run_invMassGinkgo.py --jetType=W --Nsamples=2 --id=0
#         python run_invMassGinkgo.py --jetType=QCD --Nsamples=2 --id=0
#     python run_invMassGinkgo.py --jetType=QCD --Nsamples=100 --minLeaves=9 --maxLeaves=10 --maxNTry=20000