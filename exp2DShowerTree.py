#!/usr/bin/env python

import numpy as np
import time
import sys
# import matplotlib.pyplot as plt
import copy
import pickle

import sys
import numpy as np
import torch
from torch import nn
import pyro
from pyro_simulator import PyroSimulator
out_dir='trees/'

phi_dist = pyro.distributions.Uniform(0, 2 * np.pi)



class Simulator(PyroSimulator):

    """
    Generalized Galton board example from arXiv:1805.12244.


    Has one parameter:
    theta

    Three hyperparameters:
    n_row = number of rows
    n_nails = number of nails
    start_pos = starting position (default = n_nails / 2)
    """
    def __init__(self, pt_cut=1., jet_p=None, rate=1., Mw=None , jet_name=None):
        super(Simulator, self).__init__()

        self.pt_cut=pt_cut
        self.rate=rate
        self.Mw=Mw
        self.jet_name = jet_name
        # self.deltaMax=deltaMax

        if jet_p == None:
            self.jet_p = [0,0]
        else:
            self.jet_p = jet_p

    def forward(self, inputs):
      inputs = inputs.view(-1, 1)
      num_samples = inputs.shape[0]
      kt = inputs[:, 0]  # We could have input variables

      # alpha=2
      # beta=2
      # alpha = torch.tensor(num_samples * [2.])
      # beta = torch.tensor(num_samples * [2.])
      print('Num samples = ', num_samples)
      print('Energy Scales = ', kt)

      # Define a pyro distribution
      # distribution_u = pyro.distributions.Uniform(0, 1).expand([num_samples])
      # dist_bern = pyro.distributions.Bernoulli(probs=a)

      # print('distribution_u = ', distribution_u)
      print('---' * 3)
      # dist_beta = pyro.distributions.Beta(alpha, beta)
      # dist_beta = pyro.distributions.Normal(alpha, beta)
      i = 0
      # sigma = kt
      # deltaMax=kt
      py = self.jet_p[0]
      pz = self.jet_p[1]



      tree, content, deltas, draws = _traverse(py,pz, extra_info=False, deltaMax=kt, sigma=None, cut_off=self.pt_cut, rate=self.rate, Mw=self.Mw)

      tree = np.asarray([tree])
      tree = np.asarray([np.asarray(e).reshape(-1, 2) for e in tree])
      content = np.asarray([content])
      content = np.asarray([np.asarray(e).reshape(-1, 2) for e in content])

      print('Tree = ', tree)
      print('Content = ', content)
      print('---'*10)
      print('Deltas =', deltas)
      print('---'*10)
      print('draws =', draws)
      print('---' * 10)

      jet = make_dictionary(tree, content)
      print('Jet dictionary =', jet)
      print('===' * 10)


      # SAVE OUTPUT FILE
      out_filename = out_dir+ 'tree_'+str(self.jet_name)+'_truth'+'.pkl'
      print('out_filename=', out_filename)
      with open(out_filename, "wb") as f: pickle.dump(jet, f, protocol=2)

      #
      # # while ptL[0] > self.pt_cut: #This only considers the 1st element of the batch - CHANGE!!!
      # for i in range(6):
      #   # dist_beta= pyro.distributions.Beta(alpha, beta)
      #   # dist_beta = pyro.distributions.Beta(torch.tensor([0.5]), torch.tensor([0.5]))
      #   dist = pyro.distributions.Normal(0, sigma)
      #   # dist2D = pyro.distributions.MultivariateNormal(Qscales,)
      #
      #   draw = pyro.sample("u" + str(i), dist)  # We draw a number for each ball from the distribution_u
      #   # print('Sample =',drawL)
      #
      #   ptL = px + draw
      #   # ptL= drawL.abs()
      #   # print('ptL =',ptL)
      #   ptR = px - draw
      #   sigma = 0.9* sigma
      #   print('sigma= ', sigma)
      #   print('Left particle pT for branching #', i, ' = ', ptL)
      #   print('Right particle pT for branching #', i, '= ', ptR)
      print('---' * 10)

      return content


      # sys.exit()





#--------------------------------------------------------------------------------------------------------------
#/////////////////////     FUNCTIONS     //////////////////////////////////////////////
#-------------------------------------------------------------------------------------------------------------    



#------------------------------------------------------------------------------------------------------------- 
# Recursive function to access fastjet clustering history and make the tree. We will call this function below in _traverse.
def _traverse_rec(root_y, root_z, parent_id, is_left, tree, content, deltas, draws, deltaMax=None, sigma=None, drew=None, cut_off=None,  rate=None, Mw=None, extra_info=True): #root should be a fj.PseudoJet
  
  id=len(tree)//2
  if parent_id>=0:
    if is_left:
       tree[2 * parent_id] = id #We set the location of the lef child in the content array of the 4-vector stored in content[parent_id]. So the left child will be content[tree[2 * parent_id]]
    else:
       tree[2 * parent_id + 1] = id #We set the location of the right child in the content array of the 4-vector stored in content[parent_id]. So the right child will be content[tree[2 * parent_id+1]]
#  This is correct because with each 4-vector we increase the content array by one element and the tree array by 2 elements. But then we take id=tree.size()//2, so the id increases by 1. The left and right children are added one after the other.

  #-------------------------------
  # We insert 2 new nodes to the vector that constitutes the tree. In the next iteration we will replace this 2 values with the location of the parent of the new nodes
  tree.append(-1)
  tree.append(-1)
    
#     We fill the content vector with the values of the node
  content.append(root_y)
  content.append(root_z)

  # draws.append(drew)

#   content.append(root.px())
#   content.append(root.py())
#   content.append(root.pz())
#   content.append(root.e())

  #--------------------------------------
  # We move from the root down until we get to the leaves. We do this recursively


  #------------------------------
  # Call the function recursively

  root_y=root_y/2
  root_z = root_z / 2

  # phi_dist = pyro.distributions.Uniform(0,2*np.pi)
  draw_phi = pyro.sample("phi" + str(id) + str(is_left), phi_dist)

  if Mw and id == 0:
    # ptL = torch.tensor([root - Mw / 2])
    # ptR = torch.tensor([root + Mw / 2])

    ptL_y = root_y - Mw / 2 * np.sin(draw_phi)
    ptL_z = root_z - Mw / 2 * np.cos(draw_phi)

    ptR_y = root_y + Mw / 2 * np.sin(draw_phi)
    ptR_z = root_z + Mw / 2 * np.cos(draw_phi)


    # sigma = sigmaStart
    draw = 'heavy'
    deltas.append('Mw')
    draws.append('heavy')

    _traverse_rec(ptL_y, ptL_z, id, True, tree, content, deltas, draws, deltaMax=deltaMax,  cut_off=cut_off, rate=rate,
                  extra_info=extra_info)  # pieces[0] is the left child
    _traverse_rec(ptR_y, ptR_z, id, False, tree, content, deltas, draws, deltaMax=deltaMax,  cut_off=cut_off,  rate=rate,
                  extra_info=extra_info)  # pieces[1] is the right child


  else:


    # Exponential distribution: Exponential(lambda)= lambda * Exp[- lambda x ]. Then log_prob= Log[Exponential(lambda=draw)]
    decay_dist = pyro.distributions.Exponential(rate)
    draw_decay = pyro.sample("decay" + str(id) + str(is_left),
                             decay_dist)  # We draw a number for each ball from the distribution_u

    # sigmaStart = sigma

    delta = deltaMax * draw_decay
    # print('Draw decay =', draw_decay)

    # print('cut_off = ', cut_off)
    # print('sigma =', sigma)

    # The cut_off gives a kt measure
    if delta> cut_off:

      ptL_y = root_y - delta * np.sin(draw_phi)
      ptL_z = root_z - delta * np.cos(draw_phi)

      ptR_y = root_y + delta * np.sin(draw_phi)
      ptR_z = root_z + delta * np.cos(draw_phi)

      # print('root y =', root_y)
      # print('ptL, id =', id ,' = ', ptL_y)
      # print('ptR, id =', id, ' = ', ptR_y)

      print('+=+=' * 10)

      print('---' * 10)
      deltas.append(delta)
      draws.append(draw_decay)

      _traverse_rec(ptL_y, ptL_z, id, True, tree, content, deltas, draws, deltaMax=delta,   cut_off=cut_off, rate=rate,
                    extra_info=extra_info)  # pieces[0] is the left child
      _traverse_rec(ptR_y, ptR_z, id, False, tree, content, deltas, draws, deltaMax=delta,  cut_off=cut_off, rate=rate,
                    extra_info=extra_info)  # pieces[1] is the right child

    else:

      print('---' * 10)
      deltas.append('outer')
      draws.append('outer')





    #
    # if sigma > cut_off:
    #   print('sigma =', sigma)
    #
    #   # dist_beta= pyro.distributions.Beta(alpha, beta)
    #   # dist_beta = pyro.distributions.Beta(torch.tensor([0.5]), torch.tensor([0.5]))
    #
    #   # NormalDistribution[\[Mu], \[Sigma]], x]. Then log_prob=Log[NormalDistribution[\[Mu], \[Sigma]], draw]. In Mathematica evaluate as  Log[PDF[NormalDistribution[\[Mu], \[Sigma]], x]]
    #   dist = pyro.distributions.Normal(root, sigma)
    #   # dist2D = pyro.distributions.MultivariateNormal(Qscales,)
    #
    #   draw = pyro.sample("px" + str(id), dist)  # We draw a number for each ball from the distribution_u
    #   # print('Sample =',drawL)
    #
    #   ptL = draw
    #   ptR = root + (root - draw)
    #
    #   print('+=+=' * 10)
    #   print('sigma =', sigma)
    #   print('---' * 10)
    #   sigmas.append(sigma)
    #   draws.append(draw)
    #
    #   _traverse_rec(ptL, id, True, tree, content, deltas, draws, sigma=sigma, drew=draw, cut_off=cut_off, decay=decay, rate=rate,
    #                 extra_info=extra_info)  # pieces[0] is the left child
    #   _traverse_rec(ptR, id, False, tree, content, deltas, draws, sigma=sigma, drew=draw, cut_off=cut_off, decay=decay, rate=rate,
    #                 extra_info=extra_info)  # pieces[1] is the right child
    #
    # else:
    #
    #   print('---' * 10)
    #   sigmas.append('outer')
    #   draws.append('outer')





  
  
#------------------------------------------------------------------------------------------------------------- 
# This function call the recursive function to make the trees starting from the root
def _traverse(root_y, root_z, deltaMax=None, extra_info=False, sigma=None , cut_off=None, rate=None, Mw=None):#root should be a fj.PseudoJet



  tree=[]
  content=[]
  deltas=[]
  draws=[]
  # charge=[]
  # abs_charge=[]
  # muon=[]
#   sum_abs_charge=0



  _traverse_rec(root_y, root_z, -1, False, tree, content,deltas, draws, deltaMax=deltaMax, sigma=None, cut_off=cut_off, rate=rate,Mw=Mw, extra_info=extra_info) #We start from the root=jet 4-vector


  return tree, content, deltas, draws

#------------------------------------------------------------------------------------------------------------- 
# #------------------------------------------------------------------------
# #-----------------------------------
# # #Create a dictionary with all the jet tree info (topology, constituents features: eta, phi, pT, E, muon label)
# def make_tree_list(out_jet):
#   # Create the lists with the trees
#   jets_tree = []
#   for i in range(len(out_jet)):
#
#     tree, content, charge, abs_charge, muon = cluster_h._traverse(out_jet[i], extra_info=False)
#     tree = np.asarray([tree])
#     tree = np.asarray([np.asarray(e).reshape(-1, 2) for e in tree])
#     content = np.asarray([content])
#     content = np.asarray([np.asarray(e).reshape(-1, 4) for e in content])
#     #     print('Content =',content)
#
#     mass = out_jet[i].m()
#     pt = out_jet[i].pt()
#
#     jets_tree.append((tree, content, mass, pt))
#
#     if i > 0: print('More than 1 reclustered jet')
#
#     return jets_tree


# #------------------------------------------------------------------------
# Create a dictionary with all the jet tree info (topology, constituents features: eta, phi, pT, E, muon label)
# Keep only the leading jet
def make_dictionary(tree, content, mass=None, pt=None, charge=None, abs_charge=None, muon=None):
  jet = {}

  jet["root_id"] = 0
  jet["tree"] = tree[0]  # Labels for the jet constituents in the tree
  #             jet["content"] = np.reshape(content[i],(-1,4,1)) #Where content[i][0] is the jet 4-momentum, and the other entries are the jets constituents 4 momentum. Use this format if using TensorFlow
  jet["content"] = np.reshape(content[0], (-1, 2))  # Use this format if using Pytorch
  # jet["mass"] = mass
  # jet["pt"] = pt
  # jet["energy"] = content[0][0, 3]
  #
  px = content[0][0]  # The jet is the first entry of content. And then we have (px,py,pz,E)
  # py = content[0][0, 1]
  # pz = content[0][0, 2]
  # p = (content[0][0, 0:3] ** 2).sum() ** 0.5
  #         jet["Calc energy"]=(p**2+mass[i]**2)**0.5
  # eta = 0.5 * (np.log(p + pz) - np.log(p - pz))  # pseudorapidity eta
  # phi = np.arctan2(py, px)
  #
  # jet["eta"] = eta
  # jet["phi"] = phi
  #
  if charge:
    jet["charge"] = charge[0]
    jet["abs_charge"] = abs_charge[0]
  if muon:
    jet["muon"] = muon[0]

  return jet