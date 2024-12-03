#!/bin/bash
# use gpus
#SBATCH --gres=gpu:1
nvidia-smi

/scratch/gi372/evaluation-pipeline-2024/eval_jblimp_both.sh $MODEL
