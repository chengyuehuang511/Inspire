export cuda_visible_devices=0,1,2,3,4,5,6,7
export prismatic_data_root="data/prismatic"

num_gpus=8
num_processes=32
task_suite_names=(
    "libero_90"
    "libero_goal"
    "libero_object"
    "libero_spatial"
    "libero_10"
)

name=minivla-inspire-libero-90
steps="50000"

for task_suite_name in "${task_suite_names[@]}"; do
    python vla_scripts/parallel_libero_evaluator.py \
        --num-trails-per-task 10 \
        --num-gpus $num_gpus \
        --num-processes $num_processes \
        --task-suite-name $task_suite_name \
        --pretrained-checkpoint runs/$name \
        --save-root results/$name \
        --with-vqa true \
        --steps $steps 
done
