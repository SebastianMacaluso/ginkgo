import torch
from torch.autograd import grad
import pyro
from pyro import poutine
import inspect
import numpy as np

from showerSim.simulator import Simulator
from showerSim.utils import get_logger
import os


logger = get_logger()

class PyroSimulator(Simulator):
    """ Pyro simulator interface """

    def forward(self, inputs):
        raise NotImplementedError

    def trace(self, inputs):
        # self.forward will be implemented when we create a simulator that inherits from PyroSimulator. And PyroSimulator inherits from Simulator where the __call__ method returns self.forwad (as in PyTorch)
        return pyro.poutine.trace(self.forward).get_trace(inputs)

    # -------------------------------
    def augmented_data(self, inputs, inputs_num, inputs_den, exponential=True, uniform=False):
        """
        Forward pass of the simulator that also calculates the joint likelihood ratio and the joint score, as defined
        in arXiv:1805.12244.

        Args:
            inputs (torch.Tensor): Values of the parameters used for sampling. Have shape (n_batch, n_parameters). The
                                   joint score is also evaluated at these parameters.
            inputs_num (torch.Tensor or None): Values of the parameters used for the numerator of the joint likelihood
                                               ratio. If None, inputs is used instead.
            inputs_den (torch.Tensor or None): Values of the parameters used for the denominator of the joint likelihood
                                               ratio. If None, inputs is used instead.

        Returns:
            outputs (torch.Tensor): Generated data (observables), sampled from `p(outputs | inputs)`.
            joint_score (torch.Tensor): Joint score `grad_inputs log p(outputs, latents | inputs)`.
            joint_log_likelihood_ratio (torch.Tensor): Joint log likelihood ratio
                                            `log (p(outputs, latents | inputs_num) / p(outputs, latents | inputs_den))`.

        """
        inputs.requires_grad = False

        # Get dictionary with all nodes
        trace = self.trace(inputs)

        logger.debug(f"Trace nodes = {trace.nodes}")
        # logger.debug("Trace nodes = "+ trace.nodes)
        logger.debug("====" * 20)

        x = self._calculate_x(trace)

        # --------
        logger.debug("----- JOINT LOG PROB ------")

        joint_log_prob = self._calculate_joint_log_prob(trace, exponential=exponential, uniform=uniform)

        logger.debug(f"joint_log_prob = {joint_log_prob}")
        logger.debug("====" * 20)

        #--------
        logger.debug("----- JOINT SCORE ------")
        joint_score = self._calculate_joint_score(trace, inputs)
        logger.debug("====" * 20)

        # --------
        logger.debug("----- JOINT LOG LIKELIHOOD RATIO ------")
        joint_log_likelihood_ratio = self._calculate_joint_log_likelihood_ratio(
            trace, inputs_num, inputs_den
        )
        logger.debug("====" * 20)


        return x, joint_score, joint_log_likelihood_ratio, joint_log_prob


    # ----------------------------------------------------------------------------
    def _replayed_trace(self, original_trace, inputs):
        if inputs is None:
            return original_trace

        return poutine.trace(poutine.replay(self.forward, original_trace)).get_trace(
            inputs
        )

    # -------------------------------
    @staticmethod
    def _calculate_x(trace):
        """
        Access the output values (this is why we use the key "_RETURN")
        param trace
        return
        """
        node = trace.nodes["_RETURN"]
        x = node["value"]
        return x

    # -------------------------------
    def _calculate_dist_joint_log_prob(self, trace, type= None):
        """
        We multiply the prob of making a decision at each step ( or add the log of the prob: Sum_t log[prob(x,z^i_t|theta)])
        param trace:
        param type:
        return:
        """
        log_p = 0.0
        # for distribution, z, param in self._get_branchings(trace):
        for i,(distribution, z, param) in enumerate(self._get_branchings(trace)):
            # if i==0: continue
            # logger.debug(f"items = {distribution.__dict__.items()}")
            if distribution.__class__.__name__ == str(type):
                logger.debug(f"i={i}")
                logger.debug(f" distribution ={distribution}")
                logger.debug(f" z ={z}")

                log_p = log_p + distribution.log_prob(z)

        return log_p

    def _calculate_joint_log_prob(self, trace, exponential=True, uniform=False):
        joint_log_prob_exp=0.
        joint_log_prob_unif=0.
        if exponential:
            joint_log_prob_exp = self._calculate_dist_joint_log_prob(trace, type='Exponential')
        if uniform:
            joint_log_prob_unif = self._calculate_dist_joint_log_prob(trace, type='Uniform')

        logger.debug(f"joint_log_prob_exp ={joint_log_prob_exp}")
        logger.debug(f"join_tog_prob_unif = {joint_log_prob_unif}")

        return joint_log_prob_exp+joint_log_prob_unif

    # -------------------------------
    def _calculate_joint_score(self, trace, inputs):
        score = 0.0

        # For distribution, node values (probaility at each step) and parameters of the distribution:
        for dist, z, _ in self._get_branchings(trace):
            logger.debug(f"+=+=" * 5)
            logger.debug(f"dist = {dist}")
            z = z.detach()
            log_p = dist.log_prob(z)
            logger.debug(f"z.detach() = {z.detach()}")
            logger.debug(f"log_p = {log_p}")

            try:
                score = (
                    score
                    + grad(
                        log_p,
                        inputs,
                        grad_outputs=torch.ones_like(log_p.data),
                        only_inputs=True,
                        retain_graph=True,
                    )[0]
                )
            except RuntimeError:
                # This can happen when individual distributions do not depend on the input params
                pass
        logger.debug(f"score =  {score}")

        return score


    # -------------------------------
    def _calculate_joint_log_likelihood_ratio(self, trace, inputs_num, inputs_den):
        '''
        We get the trace for each input argument
        :param trace:
        :param inputs_num:
        :param inputs_den:
        :return:
        '''
        trace_num = self._replayed_trace(trace, inputs_num)
        trace_den = self._replayed_trace(trace, inputs_den)

        log_p_num = self._calculate_joint_log_prob(trace_num)
        log_p_den = self._calculate_joint_log_prob(trace_den)

        return log_p_num - log_p_den


    # -------------------------------
    def _get_branchings(self, trace):
        for key in trace.nodes:

            # Skip if start or end positions
            if key in ["_INPUT", "_RETURN"]:
                continue
            node = trace.nodes[key]
            # print(' Get branchings nodes = ', node)
            # print('+++'*5)

            dist = node["fn"]  # distribution
            z = node["value"]  # drawn value

            # We get the parameters of the distribution, e.g. A uniform, distribution between low=0 and high=1.
            params = []
            for param_name in self._get_param_names(dist):
                param = getattr(dist, param_name)

                if len(param.size()) == 1:
                    param = param.view(-1, 1)
                params.append(param)

            # print('pyro params=', params)
            # print(np.shape(params))
            if np.shape(params)[0] > 1:
                # params = torch.cat(params, dim=0)
                params = torch.tensor(params)

            # print("Params = ", params)

            # We yield (keep track of the last state) the distribution, node values (probaility at each step) and parameters of the distribution
            yield dist, z, params

    # -------------------------------
    @staticmethod
    def _get_param_names(distribution):
        param_names_unsorted = list(distribution.arg_constraints.keys())
        sig = inspect.signature(distribution.__init__)

        # print('sig.parameters = ', sig.parameters)
        param_names_sorted = []
        for param in sig.parameters:
            if param in param_names_unsorted:
                param_names_sorted.append(param)
                param_names_unsorted.remove(param)

        # print('param_names_sorted = ',param_names_sorted)
        # print('param_names_unsorted = ', param_names_unsorted)

        return param_names_sorted + param_names_unsorted
