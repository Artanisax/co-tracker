python train.py \
--batch_size 1 \
--num_workers 32 \
--num_steps 1024 \
--ckpt_path ./checkpoints \
--dataset_root /hhd3/GroupProject2023Fall/datasets/point_odyssey/train \
--model_name cotracker \
--save_freq 16 \
--sequence_len 8 \
--traj_per_sample 384 \
--crop_size 512 640 \
--sliding_window_len 8 \
--updateformer_space_depth 6 \
--updateformer_time_depth 6 \
--save_every_n_epoch 16 \
--evaluate_every_n_epoch 1145141919810 \
--model_stride 4 \
--gpus 1 2 8
# --sample_vis_1st_frame \
# --eval_datasets tapvid_davis_first badja \