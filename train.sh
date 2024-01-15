python train.py \
--batch_size 1 \
--num_workers 16 \
--num_steps 114514 \
--ckpt_path ./checkpoints \
--dataset_root /nvme-ssd/lzl/datasets \
--eval_datasets tapvid_davis_first \
--model_name cotracker \
--num_virtual_tracks 64 \
--sequence_len 24 \
--traj_per_sample 384 \
--sliding_window_len 8 \
--save_freq 400 \
--save_every_n_epoch 1 \
--evaluate_every_n_epoch 1 \
--model_stride 4 \
--gpus 4 5 6 7 8
# --dataset_root /hhd3/GroupProject2023Fall/datasets \
# --crop_size 512 640 \
# --sample_vis_1st_frame \
