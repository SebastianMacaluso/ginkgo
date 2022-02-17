#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-10:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=ginkgo
#SBATCH --mail-type=END
#SBATCH --mail-user=sm4511@nyu.edu
#SBATCH --output=logs/slurm_%j.out

module purge

## executable
##SRCDIR=$HOME/ReclusterTreeAlgorithms/scripts
#
##LOGSDIR=$SCRATCH/ginkgo/logs/.*
mkdir -p $SCRATCH/ginkgo/logs

RUNDIR=$SCRATCH/ginkgo/runs/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR


#outdir=/scratch/sm4511/ginkgo/data/invMassGinkgo
outdir=/scratch/sm4511/ginkgo/data/MCMC

##cd $SLURM_SUBMIT_DIR
##cp my_input_params.inp $RUNDIR

##cd $RUNDIR
##module load fftw/intel/3.3.5

#cd $SCRATCH
#pwd
######  Ginkgo 2D  ############
## W jets
##python run2DShower.py --Nsamples=100 --id=${SLURM_ARRAY_TASK_ID} --jetType=Wjets

##QCD jets
#python run2DShower.py --Nsamples=100 --id=${SLURM_ARRAY_TASK_ID} --jetType=QCDjets

##TELLIS
#python run2DShower.py --Nsamples=100 --id=${SLURM_ARRAY_TASK_ID} --jetType=TrellisMw300

#python run2DShower.py --Nsamples=100 --id=${SLURM_ARRAY_TASK_ID} --jetType=TrellisMw01

#python run2DShower.py --Nsamples=100 --id=${SLURM_ARRAY_TASK_ID} --jetType=TrellisMw01B

#To test
#python run2DShower.py --Nsamples=3 --id=${SLURM_ARRAY_TASK_ID} --jetType=QCDjets



#singularity exec --overlay pytorch1.7.0-cuda11.0.ext3 /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash
#source /ext3/env.sh

#cd $SCRATCH/ginkgo/src/ginkgo

#singularity exec --overlay pytorch1.7.0-cuda11.0.ext3 /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash

#singularity exec --nv \
#	    --overlay /scratch/sm4511/pytorch1.7.0-cuda11.0.ext3:ro \
#	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
#	    bash -c "source /ext3/env.sh; python $SCRATCH/ginkgo/src/ginkgo/run_invMassGinkgo.py --jetType=QCD --Nsamples=136000 --id=${SLURM_ARRAY_TASK_ID} --minLeaves=8 --maxLeaves=9 --maxNTry=20000 --out_dir=${outdir}"


singularity exec --nv \
	    --overlay /scratch/sm4511/pytorch1.7.0-cuda11.0.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    bash -c "source /ext3/env.sh; python $SCRATCH/ginkgo/src/ginkgo/run_invMassGinkgo_variableJet4vec.py --jetType=QCD --Nsamples=100000 --id=${SLURM_ARRAY_TASK_ID} --minLeaves=8 --maxLeaves=9 --maxNTry=2000000000 --out_dir=${outdir}"



######  Ginkgo invariant mass  ############
#python run_invMassGinkgo.py --Nsamples=100 --id=${SLURM_ARRAY_TASK_ID} --jetType=QCD
#python run_invMassGinkgo.py --jetType=QCD --Nsamples=10 --id=${SLURM_ARRAY_TASK_ID} --minLeaves=6 --maxLeaves=7 --maxNTry=20000

  ## to submit(for 3 jobs): sbatch --array 0-2 submitHPC_ginkgo.s





