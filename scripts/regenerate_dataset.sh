(
name=libero_90
python scripts/parallel_libero_dataset_regenerator.py \
    --libero-task-suite $name \
    --libero-raw-data-dir data/$name \
    --libero-target-dir data/${name}_no_noops 
) &

(
name=libero_spatial
python scripts/parallel_libero_dataset_regenerator.py \
    --libero-task-suite $name \
    --libero-raw-data-dir data/$name \
    --libero-target-dir data/${name}_no_noops 
) &

(
name=libero_object
python scripts/parallel_libero_dataset_regenerator.py \
    --libero-task-suite $name \
    --libero-raw-data-dir data/$name \
    --libero-target-dir data/${name}_no_noops 
) &

(
name=libero_goal
python scripts/parallel_libero_dataset_regenerator.py \
    --libero-task-suite $name \
    --libero-raw-data-dir data/$name \
    --libero-target-dir data/${name}_no_noops 
) &

(
name=libero_10
python scripts/parallel_libero_dataset_regenerator.py \
    --libero-task-suite $name \
    --libero-raw-data-dir data/$name \
    --libero-target-dir data/${name}_no_noops 
) &

wait
