import numpy as np
import torch
import pprint
import matplotlib.pyplot as plt
import sys
import pickle

from showerSim import exp2DShowerTree
from showerSim.utils import get_logger


logger = get_logger()

augmented_data=True

#-----------------------
'''
Gaussian distribution shower
'''
# from benchmark.galton.galtonPyroWrong import Simulator
# from benchmark.galton.galtonPyro import Simulator
# from gaussShower import Simulator


# input_scale = torch.tensor(4*[[100.]])
# Delta_0 = torch.tensor([[60.]])
rate=torch.tensor(10.)
rate2=torch.tensor(8.)

# from gaussShowerTree import Simulator
# simulator = Simulator(jet_pt=0., rate=10., Mw=80., pt_cut=2.)

# import pyro
# import exp2DShowerTree
# from exp2DShowerTree import Simulator

# Lambda=8
# decay_dist = pyro.distributions.Exponential(Lambda)

# Values that give ~ 50 leaves, typically with px,py >0 for all of them.
# simulator = exp2DShowerTree.Simulator(jet_p=[800.,600.], rate=4, Mw=80., pt_cut=0.04)


# Values for tests
simulator = exp2DShowerTree.Simulator(jet_p=torch.tensor([800.,600.]), Mw=torch.tensor(80.), pt_cut=1, Delta_0=60., num_samples=1)
# simulator = exp2DShowerTree.Simulator(jet_p=[800.,600.], rate=10, pt_cut=0.5)
#simulator = Simulator(sensitivities=True)


#

# sys.exit()

# print('Trace nodes =', simulator.trace(theta).nodes)
#
# x, joint_score, joint_log_ratio = simulator.augmented_data(theta,theta, theta_ref)

if not augmented_data:
  jet_list = simulator(rate)

  logger.info(f"---"*10)
  logger.info(f"jet_list = {jet_list}")

else:

  jet_list, joint_score, joint_log_ratio, joint_log_prob = simulator.augmented_data(rate,
                                                                                    None,
                                                                                    rate2,
                                                                                    exponential=True,
                                                                                    uniform=False)

  # jet_list = simulator(Delta_0, num_samples=1)
  filename=37
  simulator.save(jet_list, "./data", "tree_"+str(filename)+"_truth")

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
  logger.info(f" jet_log_likelihood cross-check = {jet_log_likelihood_cross_check}")
  #---------------------------


