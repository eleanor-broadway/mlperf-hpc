#!/bin/bash

#SBATCH --job-name=deepcam
#SBATCH --gpus=4
#SBATCH --time=00:40:00
#SBATCH --account=z19
#SBATCH --partition=gpu
#SBATCH --qos=gpu-exc
#SBATCH --exclusive 
#SBATCH --nodes=1
#SBATCH --output=slurm-clean-test-%j.out

JOB_OUTPUT_PATH=./results/${SLURM_JOB_ID}
mkdir -p ${JOB_OUTPUT_PATH}/logs

source ${HOME/home/work}/pyenvs/mlperf-pt-gpu/bin/activate
cat $0 

export OMP_NUM_THREADS=1 
export HOME=${HOME/home/work}

rocm-smi --showenergycounter 

time srun --ntasks=4 --tasks-per-node=4 --hint=nomultithread python train.py \
    --wireup_method "nccl-slurm" \
    --run_tag test \
    --output_dir ${JOB_OUTPUT_PATH} \
    --data_dir_prefix /work/z19/shared/mlperf-hpc/deepcam/mini \
    --local_batch_size 8 \
    --max_epochs 5

rocm-smi --showenergycounter 

# --checkpoint /work/z19/z19/eleanorb/ai4nz/inference/mini-tests/hpc/deepcam/src/deepCam/results/8296700/model_step_90.cpt \
