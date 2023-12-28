python train.py \
--batch_size 1 \
--num_workers 16 \
--num_steps 6400 \
--ckpt_path ./checkpoints \
--dataset_root /nvme-ssd/lzl/datasets \
--model_name cotracker \
--sequence_len 24 \
--traj_per_sample 384 \
--sliding_window_len 8 \
--updateformer_space_depth 6 \
--updateformer_time_depth 6 \
--save_freq 4 \
--save_every_n_epoch 4 \
--evaluate_every_n_epoch 4 \
--model_stride 16 \
--gpus 0
# --eval_datasets tapvid_davis_first badja \
# --dataset_root /hhd3/GroupProject2023Fall/datasets \
# --crop_size 512 640 \
# --sample_vis_1st_frame \