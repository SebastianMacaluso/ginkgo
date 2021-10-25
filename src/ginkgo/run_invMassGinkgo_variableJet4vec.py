import numpy as np
import torch
import pprint
import matplotlib.pyplot as plt
import sys
import pickle
import argparse
import logging
import os
import pyro

from ginkgo import invMass_ginkgo_variableJet4vec as invMass_ginkgo
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

parser.add_argument(
    "--out_dir", type=str, default="../../data/invMassGinkgo", help="Output dataset dir"
)

args = parser.parse_args()


rate2=torch.tensor(8.)

# Parameters to get ~<10 constituents to test the trellis algorithm
pt_min = torch.tensor(6.**2)
# pt_min = torch.tensor(83.) # Marginal likelihood fits values
# pt_min = torch.tensor(2.3**2)

# pt_min = torch.tensor(1.1**2) # For leaves between 5 and 40

# pt_min = torch.tensor(0.02**2)

### Physics inspired parameters to get ~ between 20 and 50 constituents
W_rate = 3.
QCD_rate = 1.5
# QCD_rate = 0.0 # Marginal likelihood fits values
# pt_min = torch.tensor(1.2**2)

# QCD_mass = 30.



if args.jetType == "W":
    """ W jets"""
    rate=torch.tensor([W_rate,QCD_rate]) #Entries: [root node, every other node] decaying rates. Choose same values for a QCD jet
    mean_mass = [torch.tensor(80.)]

elif args.jetType == "QCD":
    """ QCD jets """
    rate=torch.tensor([QCD_rate,QCD_rate]) #Entries: [root node, every other node] decaying rates. Choose same values for a QCD jet
    mean_mass = [torch.tensor(30.)]
else:
    raise ValueError("Choose a valid jet type between W or QCD")




# jetM = np.sqrt(M2start.numpy())

# pt_min = torch.tensor(1**2)
# pt_min = torch.tensor(10**2)


# jetdir = np.array([1,1,1])
# jetP = 400.
# jetvec = jetP * jetdir / np.linalg.norm(jetdir)
#
# jet4vec = np.concatenate(([np.sqrt(jetP**2 + jetM**2)], jetvec))
# logger.debug(f"jet4vec = {jet4vec}")

# mean_mass = [30.]
sigma = [5.]
mean_p_xyz = [200., 200., 200.]
covariance_vec = [100., 100., 100.]


simulator = invMass_ginkgo.Simulator(mean_mass=mean_mass,
                                     sigma=sigma,
                                     mean_p_xyz=mean_p_xyz,
                                     covariance_vec=covariance_vec,
                                     pt_cut=float(pt_min),
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







save = True

if save:

    # output_dir = "../../data/invMassGinkgo"
    # output_dir = "/scratch/sm4511/ginkgo/data/invMassGinkgo"
    os.system("mkdir -p "+args.out_dir)
    print("Output dir = ", args.out_dir)

    simulator.save(jet_list, args.out_dir, "jets_"+str(args.jetType)+"_" + str(args.minLeaves) + "N_"+ str(args.Nsamples)+"trees_"+str(int(10*np.sqrt(pt_min)))+"tcut_"+ str(args.id))

# To run: python run_invMassGinkgo.py --jetType=W --Nsamples=2 --id=0
#         python run_invMassGinkgo.py --jetType=QCD --Nsamples=2 --id=0
#     python run_invMassGinkgo.py --jetType=QCD --Nsamples=100 --minLeaves=9 --maxLeaves=10 --maxNTry=20000