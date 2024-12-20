#!/bin/bash 

#SBATCH --job-name=deepcam-torch
#SBATCH --nodes=8  
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --account=z19
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --output=slurm-clean-test-%j.out

JOB_OUTPUT_PATH=./results/${SLURM_JOB_ID}
mkdir -p ${JOB_OUTPUT_PATH}/logs

source ${HOME/home/work}/pyenvs/mlperf-pt/bin/activate

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cat $0 

time srun --distribution=block:block --hint=nomultithread --ntasks=64 --ntasks-per-node=8 python train.py \
        --wireup_method "mpi" \
        --run_tag test \
        --data_dir_prefix /work/z19/shared/mlperf-hpc/deepcam/mini/ \
        --output_dir ${JOB_OUTPUT_PATH} \
        --local_batch_size 1 \
        --max_epochs 5

        # Remember to use a different seed when running on CPUs to test performance
