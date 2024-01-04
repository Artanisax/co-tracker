python train.py \
--batch_size 1 \
--num_workers 16 \
--num_steps 26000 \
--ckpt_path ./checkpoints \
--dataset_root /hhd3/GroupProject2023Fall/datasets \
--eval_datasets tapvid_davis_first \
--model_name cotracker \
--num_virtual_tracks 64 \
--sequence_len 24 \
--traj_per_sample 384 \
--sliding_window_len 8 \
--save_freq 400 \
--save_every_n_epoch 4 \
--evaluate_every_n_epoch 4 \
--model_stride 4 \
--gpus 5 6 7 8
# --dataset_root /nvme-ssd/lzl/datasets \
# --crop_size 512 640 \
# --sample_vis_1st_frame \
