num_gpus=8
name=inspire
data_root=path/to/data_root

torchrun --standalone --nnodes 1 --nproc-per-node $num_gpus vla_scripts/train_vqa.py \
  --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90" \
  --data_root_dir $data_root \
  --run_root_dir runs/$name