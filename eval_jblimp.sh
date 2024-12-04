#!/bin/bash

MODEL_PATH=$1
CHECK_POINT=$2
MODEL_BASE=$(basename "$MODEL_PATH")
MODEL_BASENAME=$MODEL_BASE.epoch--$CHECK_POINT
MODEL_CHECKPOINT=$MODEL_PATH/epoch_$CHECK_POINT

python -m lm_eval --model hf \
    --model_args pretrained=$MODEL_CHECKPOINT,backend="causal" \
    --tasks jblimp \
    --device cuda:0 \
    --batch_size 1 \
    --log_samples \
    --output_path results/jblimp/${MODEL_BASENAME}/jblimp_results.json

# Use `--model hf-mlm` and `--model_args pretrained=$MODEL_PATH,backend="mlm"` if using a custom masked LM.
# Add `--trust_remote_code` if you need to load custom config/model files.
