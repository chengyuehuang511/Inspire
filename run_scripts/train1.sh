#!/bin/bash

#SBATCH --partition="kira-lab,overcap"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:8"
#SBATCH --qos="long"
#SBATCH --mem-per-gpu=45G

export HOME="/coc/testnvme/chuang475"
export CONDA_BASE_PATH="${HOME}/miniconda3"
export CONDA_ENV_NAME="inspire"
export PYTHON_BIN="${CONDA_BASE_PATH}/envs/${CONDA_ENV_NAME}/bin/python"
export PRISMATIC_DATA_ROOT="/coc/testnvme/chuang475/datasets"
export HUGGINGFACE_HUB_CACHE="/coc/testnvme/chuang475/huggingface_cache"
name=baseline

cd /coc/testnvme/chuang475/projects/Inspire

srun -u ${PYTHON_BIN} -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  vla_scripts/train.py \
  --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90" \
  --data_root_dir "data/modified_libero_rlds" \
  --run_root_dir runs/$name \
  --wandb_project "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90" \
  --wandb_entity "chuang475-georgia-institute-of-technology" \
  --pretrained_checkpoint "runs/baseline/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90+n1+b16+x7/checkpoints/step-107500-epoch-23-loss=0.1219.pt" \
  --resume_step 107500 \
  --resume_epoch 23 \