# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch

import imageio
import numpy as np

from cotracker.datasets.CoTrackerDataset import CoTrackerDataset
from cotracker.datasets.utils import CoTrackerData


class KubricMovifDataset(CoTrackerDataset):
    def __init__(
        self,
        data_root,
        crop_size=(384, 512),
        seq_len=24,
        traj_per_sample=768,
        sample_vis_1st_frame=False,
        use_augs=False,
    ):
        super(KubricMovifDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
            traj_per_sample=traj_per_sample,
            sample_vis_1st_frame=sample_vis_1st_frame,
            use_augs=use_augs,
        )

        self.pad_bounds = [0, 25]
        self.resize_lim = [0.75, 1.25]  # sample resizes from here
        self.resize_delta = 0.05
        self.max_crop_offset = 15
        self.seq_names = [
            fname
            for fname in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, fname))
        ]
        print("found %d unique videos in %s" % (len(self.seq_names), self.data_root))

    def getitem_helper(self, index):
        gotit = True
        seq_name = self.seq_names[index]

        npy_path = os.path.join(self.data_root, seq_name, seq_name + ".npy")
        rgb_path = os.path.join(self.data_root, seq_name, "frames")

        img_paths = sorted(os.listdir(rgb_path))
        rgbs = []
        for i, img_path in enumerate(img_paths):
            rgbs.append(imageio.v2.imread(os.path.join(rgb_path, img_path)))

        rgbs = np.stack(rgbs)
        annot_dict = np.load(npy_path, allow_pickle=True).item()
        traj_2d = annot_dict["coords"]
        visibility = annot_dict["visibility"]

        # random crop
        assert self.seq_len <= len(rgbs)
        if self.seq_len < len(rgbs):
            start_ind = np.random.choice(len(rgbs) - self.seq_len, 1)[0]

            rgbs = rgbs[start_ind : start_ind + self.seq_len]
            traj_2d = traj_2d[:, start_ind : start_ind + self.seq_len]
            visibility = visibility[:, start_ind : start_ind + self.seq_len]

        traj_2d = np.transpose(traj_2d, (1, 0, 2))
        visibility = np.transpose(np.logical_not(visibility), (1, 0))
        if self.use_augs:
            rgbs, traj_2d, visibility = self.add_photometric_augs(rgbs, traj_2d, visibility)
            rgbs, traj_2d = self.add_spatial_augs(rgbs, traj_2d, visibility)
        else:
            rgbs, traj_2d = self.crop(rgbs, traj_2d)

        visibility[traj_2d[:, :, 0] > self.crop_size[1] - 1] = False
        visibility[traj_2d[:, :, 0] < 0] = False
        visibility[traj_2d[:, :, 1] > self.crop_size[0] - 1] = False
        visibility[traj_2d[:, :, 1] < 0] = False

        visibility = torch.from_numpy(visibility)
        traj_2d = torch.from_numpy(traj_2d)

        visibile_pts_first_frame_inds = (visibility[0]).nonzero(as_tuple=False)[:, 0]

        if self.sample_vis_1st_frame:
            visibile_pts_inds = visibile_pts_first_frame_inds
        else:
            visibile_pts_mid_frame_inds = (visibility[self.seq_len // 2]).nonzero(as_tuple=False)[
                :, 0
            ]
            visibile_pts_inds = torch.cat(
                (visibile_pts_first_frame_inds, visibile_pts_mid_frame_inds), dim=0
            )
        point_inds = torch.randperm(len(visibile_pts_inds))[: self.traj_per_sample]
        if len(point_inds) < self.traj_per_sample:
            gotit = False

        visible_inds_sampled = visibile_pts_inds[point_inds]

        trajs = traj_2d[:, visible_inds_sampled].float()
        visibles = visibility[:, visible_inds_sampled]
        valids = torch.ones((self.seq_len, self.traj_per_sample))

        rgbs = torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).float()
        sample = CoTrackerData(
            video=rgbs,
            trajectory=trajs,
            visibility=visibles,
            valid=valids,
            seq_name=seq_name,
        )
        return sample, gotit

    def __len__(self):
        return len(self.seq_names)
