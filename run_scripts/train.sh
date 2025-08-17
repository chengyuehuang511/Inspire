#!/bin/bash

#SBATCH --partition="kira-lab"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="l40s:1"
#SBATCH --qos="short"
#SBATCH --mem-per-gpu=45G

export HOME="/coc/testnvme/chuang475"
export CONDA_BASE_PATH="${HOME}/miniconda3"
export CONDA_ENV_NAME="openvla"
export PYTHON_BIN="${CONDA_BASE_PATH}/envs/${CONDA_ENV_NAME}/bin/python"
export PRISMATIC_DATA_ROOT="/coc/testnvme/chuang475/datasets"

cd /coc/testnvme/chuang475/projects/openvla-mini

# srun -u ${PYTHON_BIN} -m torch.distributed.run \
#   --standalone \
#   --nnodes=1 \
#   --nproc_per_node=1 \
#   vla-scripts/train.py \
#   --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90" \
#   --data_root_dir "/coc/testnvme/chuang475/projects/LIBERO/libero/libero/benchmark/" \
#   --run_root_dir "log/" \
#   --wandb_project "minivla" \
#   --wandb_entity "chuang475-georgia-institute-of-technology"

srun -u ${PYTHON_BIN} experiments/robot/libero/regenerate_libero_dataset.py \
    --libero_task_suite libero_90 \
    --libero_raw_data_dir /coc/testnvme/chuang475/projects/openvla-mini/data/modified_libero_rlds/libero_90_raw \
    --libero_target_dir /coc/testnvme/chuang475/projects/openvla-mini/data/modified_libero_rlds/libero_90_no_noops_new \