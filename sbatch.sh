#!/bin/bash

name="train1"

job_name="${name}_$(date +%Y%m%d_%H%M%S)"
output_dir="output/${job_name}"

mkdir -p "$output_dir"
sbatch --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" run_scripts/${name}.sh