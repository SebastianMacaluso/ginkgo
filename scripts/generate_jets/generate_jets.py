import torch
import argparse
import numpy as np
from showerSim import exp2DShowerTree
from showerSim.utils import get_logger



def generate_jets(args):
    logger.info(f"Starting generation of {args.num_samples} jets")


    simulator = exp2DShowerTree.Simulator(jet_p=args.jet_p,
                                          Mw=torch.tensor(80.),
                                          pt_cut=args.pt_cut,
                                          Delta_0=args.Delta_0,
                                          num_samples=args.num_samples)



    if args.augmented_data==0:
        jet_list = simulator(args.rate)
        logger.info(f"Starting generation of {args.num_samples} jets")
        simulator.save(jet_list, args.outdir, args.filename)
        logger.info(f"Done!")

    else:

        jet_list, joint_score, joint_log_ratio, joint_log_prob = simulator.augmented_data(args.rate,
                                                                                        None,
                                                                                        args.rate_2,
                                                                                        exponential=True,
                                                                                        uniform=False)


        logger.info(f"---"*10)
        logger.debug(f"jet_list = {jet_list}")
        logger.info(f"joint_score = {joint_score}")
        logger.info(f"joint_log_likelihood_ratio= {joint_log_ratio}")
        logger.info(f"joint_log_prob= {joint_log_prob}")
        logger.info(f"---"*10)




        def jet_log_likelihood():
            '''
            Calulate the log likelihood for the Exponential distribution
            :return:
            '''

            Lambda = jet_list[0]['Lambda']

            if Lambda.requires_grad:
                Lambda = Lambda.detach().numpy()

            log_likelihood = 0
            for entry in jet_list[0]['draws'][1::]:
                if entry.requires_grad:
                    entry = entry.detach().numpy()

                log_likelihood += np.log(Lambda * np.exp(-Lambda * entry))

            return log_likelihood


        jet_log_likelihood_cross_check=jet_log_likelihood()
        logger.info(f" jet_log_likelihood cross-check = {jet_log_likelihood_cross_check}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a jet clustering algorithm using REINFORCE."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity"
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--filename", type=str, required=True, help="Name of output file"
    )
    parser.add_argument(
        "--num_samples", type=int, required=True, help="Number of jet samples"
    )
    parser.add_argument(
        "--Delta_0", type=float, default=60., help="Delta_0 for clustering"
    )
    parser.add_argument(
        "--jet_p", nargs='+', type=float, default=torch.tensor([800., 600.]), help="initial jet momentum"
    )
    parser.add_argument(
        "--pt_cut", type=float, default=0.04, help="IR cutoff"
    )
    parser.add_argument(
        "--rate", type=float, default=torch.tensor(4.), help="Emission rate"
    )

    parser.add_argument(
        "--rate_2", type=float, default=torch.tensor(10.), help="Emission rate"
    )

    parser.add_argument(
        "--augmented_data", type=int, required=True, help="If 0, do not get the augmented data"
    )


    logger = get_logger()

    args = parser.parse_args()

    generate_jets(args)








