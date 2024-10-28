#!/usr/bin/env bash

# Set the path to the config file
CONFIG_FILE="/stable-diffusion-webui/config.json"

# 1. Check if SD_CHECKPOINTS_LIMIT is set and update config.json if it is
if [ ! -z "$SD_CHECKPOINTS_LIMIT" ]; then
  jq --argjson limit "$SD_CHECKPOINTS_LIMIT" '.sd_checkpoints_limit = $limit' "$CONFIG_FILE" > tmp.json && mv tmp.json "$CONFIG_FILE"
  echo "Updated sd_checkpoints_limit to $SD_CHECKPOINTS_LIMIT"
fi

# 2. Check if SD_MODEL_CHECKPOINT is set and update config.json if it is
if [ ! -z "$SD_MODEL_CHECKPOINT" ]; then
  jq --arg checkpoint "$SD_MODEL_CHECKPOINT" '.sd_model_checkpoint = $checkpoint' "$CONFIG_FILE" > tmp.json && mv tmp.json "$CONFIG_FILE"
  echo "Updated sd_model_checkpoint to $SD_MODEL_CHECKPOINT"
fi

echo "Starting WebUI API"
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"
export PYTHONUNBUFFERED=true
export HF_HOME="/"
python /stable-diffusion-webui/webui.py \
  --xformers \
  --no-half-vae \
  --skip-python-version-check \
  --skip-torch-cuda-test \
  --skip-install \
  --lowram \
  --opt-sdp-attention \
  --disable-safe-unpickle \
  --port 3000 \
  --api \
  --nowebui \
  --skip-version-check \
  --no-hashing \
  --no-download-sd-model > /logs/webui.log 2>&1 &

echo "Starting The Job Queue Worker"
/usr/local/bin/salad-http-job-queue-worker &

echo "Starting The Handler"
python -u /salad_handler.py
