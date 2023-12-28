python train.py \
--batch_size 1 \
--num_workers 16 \
--num_steps 6400 \
--ckpt_path ./checkpoints \
--dataset_root /hhd3/GroupProject2023Fall/datasets \
--model_name cotracker \
--sequence_len 24 \
--traj_per_sample 384 \
--crop_size 512 640 \
--sliding_window_len 8 \
--updateformer_space_depth 6 \
--updateformer_time_depth 6 \
--eval_datasets tapvid_davis_first badja \
--save_freq 16 \
--save_every_n_epoch 4 \
--evaluate_every_n_epoch 4 \
--model_stride 4 \
--gpus 1 2 3
# --sample_vis_1st_frame \