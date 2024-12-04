#!/bin/bash
# use gpus
#SBATCH --gres=gpu:1
nvidia-smi

# Check if an argument is provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 {opt-125m|opt-baby-medium} {10m|100m} {1|2|3}"
    exit 1
fi

# Store the arguments
arg1=$1 # model name
arg2=$2 # dataset size
arg3=$3 # seed

# echo "# MODEL: $arg1"
# echo "# DATASET_SIZE: $arg2"
# echo "# SEED: $arg3"

SEED=$3
SIZE=$2
MODEL_BASE_NAME=$1

REPO_ROOT=/scratch/gi372/japanese_curriculum
DATA_NAME=reazonspeech

if [ "$MODEL_BASE_NAME" = "opt-125m" ]; then
    TOKENIZER=$REPO_ROOT/tokenizer/opt-$SIZE-reazonspeech-concat-bpe-50265
    MODEL_CONFIG=$REPO_ROOT/config/opt-125m
elif [ "$MODEL_BASE_NAME" = "opt-baby-medium" ]; then
    TOKENIZER=$REPO_ROOT/tokenizer/opt-$SIZE-reazonspeech-concat-bpe-8192
    MODEL_CONFIG=$REPO_ROOT/config/opt-baby-medium-8192
else
    echo "Invalid tokenizer"
    exit 1
fi

MODEL_CONFIG_BASE=$(basename "$MODEL_CONFIG")
TOKENIZER_BASE=$(basename "$TOKENIZER")
MODEL_DIR=$REPO_ROOT/experiment/"$MODEL_BASE_NAME"-$SIZE-s$SEED

# raw
for PREPROCESS in "R" "S" "K" "B"; do
    CURRICULUM=$PREPROCESS$PREPROCESS$PREPROCESS$PREPROCESS
    MODEL=$MODEL_DIR/model--$MODEL_CONFIG_BASE.tokenizer--$TOKENIZER_BASE.size--10m.data--$DATA_NAME.curriculum--$CURRICULUM.s$SEED
    for i in $(seq 0 39); do
    /scratch/gi372/evaluation-pipeline-2024/eval_jblimp_both.sh $MODEL $i
    done
done

# rondom
for PREPROCESS in "S" "K" "B"; do
    CURRICULUM=RANDOM_$PREPROCESS
    MODEL=$MODEL_DIR/model--$MODEL_CONFIG_BASE.tokenizer--$TOKENIZER_BASE.size--10m.data--$DATA_NAME.curriculum--$CURRICULUM.s$SEED
    for i in $(seq 0 39); do
    /scratch/gi372/evaluation-pipeline-2024/eval_jblimp_both.sh $MODEL $i
    done
done

# early
for PREPROCESS in "S" "K" "B"; do
    CURRICULUM="$PREPROCESS"RRR
    MODEL=$MODEL_DIR/model--$MODEL_CONFIG_BASE.tokenizer--$TOKENIZER_BASE.size--10m.data--$DATA_NAME.curriculum--$CURRICULUM.s$SEED
    for i in $(seq 10 39); do
    /scratch/gi372/evaluation-pipeline-2024/eval_jblimp_both.sh $MODEL $i
    done
done

# mid-early
for PREPROCESS in "S" "K" "B"; do
    CURRICULUM=R"$PREPROCESS"RR
    MODEL=$MODEL_DIR/model--$MODEL_CONFIG_BASE.tokenizer--$TOKENIZER_BASE.size--10m.data--$DATA_NAME.curriculum--$CURRICULUM.s$SEED
    for i in $(seq 10 39); do
    /scratch/gi372/evaluation-pipeline-2024/eval_jblimp_both.sh $MODEL $i
    done
done

# mid-late
for PREPROCESS in "S" "K" "B"; do
    CURRICULUM=RR"$PREPROCESS"R
    MODEL=$MODEL_DIR/model--$MODEL_CONFIG_BASE.tokenizer--$TOKENIZER_BASE.size--10m.data--$DATA_NAME.curriculum--$CURRICULUM.s$SEED
    for i in $(seq 20 39); do
    /scratch/gi372/evaluation-pipeline-2024/eval_jblimp_both.sh $MODEL $i
    done
done

# late
for PREPROCESS in "S" "K" "B"; do
    CURRICULUM=RRR"$PREPROCESS"
    MODEL=$MODEL_DIR/model--$MODEL_CONFIG_BASE.tokenizer--$TOKENIZER_BASE.size--10m.data--$DATA_NAME.curriculum--$CURRICULUM.s$SEED
    for i in $(seq 30 39); do
    /scratch/gi372/evaluation-pipeline-2024/eval_jblimp_both.sh $MODEL $i
    done
done
#/scratch/gi372/evaluation-pipeline-2024/eval_jblimp_both.sh $MODEL
