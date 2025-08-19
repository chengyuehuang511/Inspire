#!/bin/bash

#SBATCH --partition="kira-lab,overcap"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:8"
#SBATCH --qos="short"
#SBATCH --mem-per-gpu=45G
#SBATCH -x xaea-12

export HOME="/coc/testnvme/chuang475"
export CONDA_BASE_PATH="${HOME}/miniconda3"
export CONDA_ENV_NAME="inspire" # openvla rlds_env
export PYTHON_BIN="${CONDA_BASE_PATH}/envs/${CONDA_ENV_NAME}/bin/python"
export PRISMATIC_DATA_ROOT="/coc/testnvme/chuang475/datasets"
export HUGGINGFACE_HUB_CACHE="/coc/testnvme/chuang475/huggingface_cache"
export prismatic_data_root="data/prismatic"

cd /coc/testnvme/chuang475/projects/Inspire

num_gpus=8
num_processes=32
task_suite_names=(
    # "libero_90"
    "libero_goal"
    "libero_object"
    "libero_spatial"
    "libero_10"
)

# name=minivla-libero-90
# name=minivla-libero90-prismatic
name="baseline/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90+vq+n1+b16+x7"
# name="baseline/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90+n1+b16+x7"
# name="baseline/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90+vq+spd_3+n1+b16+x7"
steps="50000"
# steps="122500"

for task_suite_name in "${task_suite_names[@]}"; do
    srun -u ${PYTHON_BIN} -m vla_scripts.parallel_libero_evaluator \
        --num-trials-per-task 10 \
        --num-gpus $num_gpus \
        --num-processes $num_processes \
        --task-suite-name $task_suite_name \
        --pretrained-checkpoint runs/$name \
        --save-root results/$name \
        --with-vqa false \
        --steps $steps \
        --center-crop false
done
