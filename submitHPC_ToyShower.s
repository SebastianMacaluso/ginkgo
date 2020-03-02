#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=ClusteringAlgorithms
#SBATCH --mail-type=END
#SBATCH --mail-user=sm4511@nyu.edu
#SBATCH --output=logs/slurm_%j.out

module purge

## executable
##SRCDIR=$HOME/ReclusterTreeAlgorithms/scripts

RUNDIR=$SCRATCH/ToyJetsShower/runs/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR

##cd $SLURM_SUBMIT_DIR
##cp my_input_params.inp $RUNDIR

##cd $RUNDIR
##module load fftw/intel/3.3.5

cd $HOME/ToyJetsShower/


## W jets
##python run2DShower.py --Nsamples=100 --id=${SLURM_ARRAY_TASK_ID} --jetType=Wjets

##QCD jets
python run2DShower.py --Nsamples=100 --id=${SLURM_ARRAY_TASK_ID} --jetType=QCDjets

## to submit(for 2 jobs): sbatch --array 0-2 submitHPC_ToyShower.s






