import numpy as np
import torch
import pprint
import matplotlib.pyplot as plt
import sys

from showerSim import exp2DShowerTree


#-----------------------
'''
Gaussian distribution shower
'''
# from benchmark.galton.galtonPyroWrong import Simulator
# from benchmark.galton.galtonPyro import Simulator
# from gaussShower import Simulator


# input_scale = torch.tensor(4*[[100.]])
kt_scale = torch.tensor([[60.]])


# from gaussShowerTree import Simulator
# simulator = Simulator(jet_pt=0., rate=10., Mw=80., pt_cut=2.)

# import pyro
# import exp2DShowerTree
# from exp2DShowerTree import Simulator

# Lambda=8
# decay_dist = pyro.distributions.Exponential(Lambda)

# Values that give ~ 50 leaves, typically with px,py >0 for all of them.
simulator = exp2DShowerTree.Simulator(jet_p=[800.,600.], rate=4, Mw=80., pt_cut=0.04)


# Values for tests
# simulator = exp2DShowerTree.Simulator(jet_p=[800.,600.], rate=10, Mw=80., pt_cut=0.04, jet_name='32')

#simulator = Simulator(sensitivities=True)

jet_list = simulator(kt_scale, num_samples=2)
simulator.save(jet_list, "test_jet")
#

sys.exit()

# print('Trace nodes =', simulator.trace(theta).nodes)
#
# x, joint_score, joint_log_ratio = simulator.augmented_data(theta,theta, theta_ref)
x, joint_score, joint_log_ratio, joint_log_prob = simulator.augmented_data(kt_scale, None, None)

# print('x = ', x)
# print('joint_score = ',joint_score)
# print('joint_log_ratio= ',joint_log_ratio)
# print('joint_log_prob= ',joint_log_prob)
print('---'*5)




#---------------------------

# #-----------------------
# '''
# Beta distribution shower
# '''
# # from benchmark.galton.galtonPyroWrong import Simulator
# # from benchmark.galton.galtonPyro import Simulator
# from betaShower import Simulator
#
# input_scale = torch.tensor(4*[[100.]])
#
# simulator = Simulator(start_pt=100.)
# #simulator = Simulator(sensitivities=True)
# output = simulator(input_scale)
#
#
# # print('Trace nodes =', simulator.trace(theta).nodes)
# #
# # x, joint_score, joint_log_ratio = simulator.augmented_data(theta,theta, theta_ref)
# ## x, joint_score, joint_log_ratio = simulator.augmented_data(theta,None, None)
#
# print('x = ', x)
# print('joint_score = ',joint_score)
# print('joint_log_ratio= ',joint_log_ratio)
# print('---'*5)
#
#
#
#
# #---------------------------

# theta = torch.tensor(5000*[[3.]])
# theta_ref = torch.tensor(5000*[[0.7]])

# theta = torch.tensor(5*[[3.]])
# theta_ref = torch.tensor(5*[[0.7]])
#
#
# simulator = Simulator(n_rows=20, n_nails=31)
# #simulator = Simulator(sensitivities=True)
# # output = simulator(theta)
#
#
# # print('Trace nodes =', simulator.trace(theta).nodes)
# #
# x, joint_score, joint_log_ratio = simulator.augmented_data(theta,theta, theta_ref)
# ## x, joint_score, joint_log_ratio = simulator.augmented_data(theta,None, None)
#
# print('x = ', x)
# print('joint_score = ',joint_score)
# print('joint_log_ratio= ',joint_log_ratio)
# print('---'*5)


#
# print(torch.mean(joint_score, dim=0))
# print(torch.mean(torch.exp(joint_log_ratio), dim=0))

#printer = pprint.PrettyPrinter(indent=4)
#printer.pprint(simulator.trace(inputs).nodes)

# plt.hist(output.numpy(), 31,range=[0,31])
# plt.xlabel(r"$x$")
# plt.ylabel("Frequency")
#
# plt.show()
