import torch
import argparse
from showerSim import exp2DShowerTree
from showerSim.utils import get_logger


def generate_jets(args):
    logger.info(f"Starting generation of {args.num_samples} jets")
    kt_scale = torch.tensor([[args.kt_scale]])
    simulator = exp2DShowerTree.Simulator(jet_p=args.jet_p, rate=args.rate, Mw=80., pt_cut=args.pt_cut)
    jet_list = simulator(kt_scale, num_samples=args.num_samples)
    logger.info(f"Starting generation of {args.num_samples} jets")
    simulator.save(jet_list, args.outdir, args.filename)
    logger.info(f"Done!")


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
        "--kt_scale", type=float, default=60., help="kt-scale for clustering"
    )
    parser.add_argument(
        "--jet_p", nargs='+', type=float, default=[800., 600.], help="initial jet momentum"
    )
    parser.add_argument(
        "--pt_cut", type=float, default=0.04, help="IR cutoff"
    )
    parser.add_argument(
        "--rate", type=float, default=4., help="Emission rate"
    )

    logger = get_logger()

    args = parser.parse_args()

    generate_jets(args)








