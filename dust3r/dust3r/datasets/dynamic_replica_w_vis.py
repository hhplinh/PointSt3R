import sys
sys.path.append('.')
import os
import json
import torch
import numpy as np
import os.path as osp
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch._C import dtype, set_flush_denormal
import dust3r.utils.po_utils.basic
import dust3r.utils.po_utils.improc
from dust3r.utils.po_utils.misc import farthest_point_sample_py
from dust3r.utils.po_utils.geom import apply_4x4_py, apply_pix_T_cam_py
import glob
import cv2
from torchvision.transforms import ColorJitter, GaussianBlur
from functools import partial
from tqdm import tqdm

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
from dust3r.utils.misc import get_stride_distribution

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')

class DynamicReplicaVISDUSt3R(BaseStereoViewDataset):
    def __init__(
            self,
            S,
            N,
            strides,
            clip_step,
            quick,
            verbose,
            dist_type,
            clip_step_last_skip,
            predict_occlusions,
            *args, split, ROOT, **kwargs
    ):
        self.ROOT = ROOT
        self.dataset_location = ROOT
        print('loading dynamic replica dataset...')
        super().__init__(*args, **kwargs)
        self.dataset_label = 'dynamic replica'
        self.split = split
        self.S = S # stride
        self.N = N # min num points
        self.verbose = verbose
        self.dset = split

        self.predict_occlusions = predict_occlusions

        self.rgb_paths = []
        self.depth_paths = []
        self.traj_paths = []
        self.annotation_paths = []
        self.full_idxs = []
        self.sample_stride = []
        if "all" in strides:
            self.strides = [i for i in range(1, int(strides.split("all_")[-1]) + 1)]
        else:
            self.strides = strides
        print(f"{self.strides=}")

        #pre_load_seqs_file = "/home/u5ad/rhodriguerrier.u5ad/mast3r_train_subsets/small_train_DR.json"
        #if os.path.isfile(pre_load_seqs_file):
        #    print(f"PO pre-load seqs file exists, loading from JSON...")
        #    with open(pre_load_seqs_file, "r") as f:
        #        pre_load_seqs = json.load(f)

        #    self.rgb_paths = []
        #    self.depth_paths = []
        #    self.annotation_paths = []
        #    for idx in range(len(pre_load_seqs["rgb_paths"])):
        #        seq_name = pre_load_seqs["rgb_paths"][idx][0].split("/")[-3]
        #        #seq_name = pre_load_seqs["annotation_paths"][idx].split("/")[-2]
        #        rgb_files = [rgb_path.split("/")[-1] for rgb_path in pre_load_seqs["rgb_paths"][idx]]
        #        depth_files = [depth_path.split("/")[-1] for depth_path in pre_load_seqs["depth_paths"][idx]]

        #        self.rgb_paths.append([f"{self.dataset_location}/{seq_name}/images/{rgb_file}" for rgb_file in rgb_files])
        #        self.depth_paths.append([f"{self.dataset_location}/{seq_name}/depths/{depth_file}" for depth_file in depth_files])
        #        self.annotation_paths.append(f"{self.dataset_location}/{seq_name}/{seq_name.split('_source')[0]}.npy")

        #    self.full_idxs = [np.array(ls) for ls in pre_load_seqs["full_idxs"]]
        #    print(f"{len(self.rgb_paths)=}, {len(self.depth_paths)=}, {len(set(self.annotation_paths))=}")
        #    print("DONE loading DynamicReplica train!")
        #    return

        # self.dataset_location = /root/dynamic_stereo/dynamic_replica_data/train
        #self.sequences = [seq for seq in glob.glob(f"{self.dataset_location}/*_left/")]
        self.sequences = [seq for seq in glob.glob(f"/home/u5ad/rhodriguerrier.u5ad/dynamic_stereo/dynamic_replica_mast3r_annots/train/*npy")]
        # e.g. /root/dynamic_stereo/dynamic_replica_mast3r_annots/train/06dcf6-3_obj.npy

        self.sequences = sorted(self.sequences)
        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s (self.dset=%s)' % (len(self.sequences), self.dataset_label, self.dset))
        
        ## load trajectories
        print('loading trajectories...')
 
        for seq in self.sequences:
            if self.verbose: 
                print('seq', seq)

            print(seq)
            seq_name = seq.split("/")[-1].split(".npy")[0] # e.g. 06dcf6-3_obj
            img_root = f"{self.dataset_location}/{seq_name}_source_left"
            annotations_path = seq

            for stride in self.strides:
                for ii in range(0, 300 - stride+1, clip_step):
                    if ii + stride >= 300:
                        continue
                    full_idx = ii + np.arange(self.S) * stride
                    self.rgb_paths.append([
                        f"{img_root}/images/{seq_name}_source_left-{str(idx).zfill(4)}.png"
                        for idx in full_idx
                    ])
                    self.depth_paths.append([
                        f"{img_root}/depths/{seq_name}_source_left_{str(idx).zfill(4)}.geometric.png"
                        for idx in full_idx
                    ])
                    self.annotation_paths.append(seq)
                    self.full_idxs.append(full_idx)
                    self.sample_stride.append(stride)
                if self.verbose:
                    sys.stdout.write('.')
                    sys.stdout.flush()
        
        self.stride_counts = {}
        self.stride_idxs = {}
        for stride in self.strides:
            self.stride_counts[stride] = 0
            self.stride_idxs[stride] = []
        for i, stride in enumerate(self.sample_stride):
            self.stride_counts[stride] += 1
            self.stride_idxs[stride].append(i)
        print('stride counts:', self.stride_counts)
        
        #if len(self.strides) > 1 and dist_type is not None:
        if len(self.strides) > 1 and dist_type is not None:
            num_clips_per_stride = self._resample_clips(self.strides, dist_type)
 
        print('collected %d clips of length %d in %s (self.dset=%s)' % (
            len(self.rgb_paths), self.S, self.dataset_location, self.dset))
         
        print(f"{len(self.sample_stride)=}")
        print(f"{len(set(self.annotation_paths))=}")
        print(f"{len(set(self.annotation_paths[:10000]))=}")
        
        print('### NEW ### collected %d clips of length %d in %s (self.dset=%s)' % (
            len(self.rgb_paths), self.S, self.dataset_location, self.dset))
        print(f"### {len(set(self.annotation_paths))=} ###")

        #print(self.rgb_paths[:10])
        #print(self.depth_paths[:10])
        #print(self.annotation_paths[:10])
        #print(self.full_idxs[:10])
        #output = {
        #  "rgb_paths": self.rgb_paths[:3334],
        #   "depth_paths": self.depth_paths[:3334],
        #    "annotation_paths": self.annotation_paths[:3334],
        #    "full_idxs": np.array(self.full_idxs[:3334]).tolist()
        #}
        #with open(f"/root/small_train_DR.json", "w") as f:
        #    json.dump(output, f)
        #sys.exit(0)


    def _resample_clips(self, strides, dist_type):

        # Get distribution of strides, and sample based on that
        dist = get_stride_distribution(strides, dist_type=dist_type)
        dist = dist / np.max(dist)
        max_num_clips = self.stride_counts[strides[np.argmax(dist)]]
        num_clips_each_stride = [min(self.stride_counts[stride], int(dist[i]*max_num_clips)) for i, stride in enumerate(strides)]
        print('resampled_num_clips_each_stride:', num_clips_each_stride)
        resampled_idxs = []
        for i, stride in enumerate(strides):
            resampled_idxs += np.random.choice(self.stride_idxs[stride], num_clips_each_stride[i], replace=False).tolist()
        
        self.rgb_paths = [self.rgb_paths[i] for i in resampled_idxs]
        self.depth_paths = [self.depth_paths[i] for i in resampled_idxs]
        self.annotation_paths = [self.annotation_paths[i] for i in resampled_idxs]
        self.full_idxs = [self.full_idxs[i] for i in resampled_idxs]
        self.sample_stride = [self.sample_stride[i] for i in resampled_idxs]
        return num_clips_each_stride

    def __len__(self):
        return len(self.rgb_paths)
    
    def _get_views(self, index, resolution, rng):
        rgb_paths = self.rgb_paths[index]
        depth_paths = self.depth_paths[index]
        full_idx = self.full_idxs[index]
        print(full_idx)

        annotations_path = self.annotation_paths[index]
        annotations = np.load(annotations_path, allow_pickle=True).item()
        cams_to_pixels = annotations['intrinsics'][full_idx].astype(np.float32)
        cams_to_world = annotations['cam_to_worlds'][full_idx].astype(np.float32)
        trajs_2d = annotations["trajs_2d"][full_idx].astype(np.float32)
        trajs_3d = annotations["trajs_3d"][full_idx].astype(np.float32)
        visibs = annotations["visibs"][full_idx]

        if self.predict_occlusions:
            #print("### DR ###")
            #print(f"{trajs_2d.shape=}, {visibs.shape=}")
            #print(f"{trajs_2d[0, visibs[1], :2].shape=}")
            #print(f"{visibs[1, visibs[0]].shape=}")
            trajs_2d_n_occ = [
                [trajs_2d[0, visibs[0], :2], visibs[1, visibs[0]]],
                [trajs_2d[1, visibs[1], :2], visibs[0, visibs[1]]]
            ]
 
        vis_in_both = (visibs[0] * visibs[1])
        trajs_2d = trajs_2d[:, vis_in_both]
        trajs_3d = trajs_3d[:, vis_in_both]

        num_trajs = trajs_2d.shape[1]
        dynamic_points = ~(np.sum(trajs_3d[0, :] == trajs_3d[1, :], axis=1) == 3)
        views = []
        for i in range(2):
            impath = rgb_paths[i]
            depthpath = depth_paths[i]

            # load camera params
            extrinsics = cams_to_world[i]
            intrinsics = cams_to_pixels[i]

            # load image and depth
            rgb_image = imread_cv2(impath)
            orig_y, orig_x, _ = rgb_image.shape

            with Image.open(depthpath) as depth_pil:
                depthmap = (
                    np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                    .astype(np.float32)
                    .reshape((depth_pil.size[1], depth_pil.size[0]))
                )
 
            rgb_image, depthmap, intrinsics, crop_bbox, size_before_crop = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath
            )

            trajs_2d_resize = trajs_2d[i]
            new_x, new_y = rgb_image.size
            x_mul, y_mul = (size_before_crop[0] / orig_x), (size_before_crop[1] / orig_y)
            trajs_2d_resize[:, 0] *= x_mul
            if crop_bbox[0] != 0:
                trajs_2d_resize[:, 0] -= crop_bbox[0]
            trajs_2d_resize[:, 1] *= y_mul
            if crop_bbox[1] != 0:
                trajs_2d_resize[:, 1] -= crop_bbox[1]
            trajs_2d_resize = trajs_2d_resize.astype(np.int32)

            temp_view_matches = np.zeros((new_y, new_x)).astype(np.int32)
            temp_dynamic_points = np.zeros((new_y, new_x)).astype(np.int32)
            counter = 0
            for temp_idx in range(trajs_2d_resize.shape[0]):
                x, y = trajs_2d_resize[temp_idx, 0], trajs_2d_resize[temp_idx, 1]
                if (0 <= x < new_x) and (0 <= y < new_y):
                    temp_view_matches[y, x] = temp_idx
                    temp_dynamic_points[y, x] = 1.0 if dynamic_points[temp_idx] else 0.0
 
            view_dict = dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=extrinsics,
                camera_intrinsics=intrinsics,
                dataset=self.dataset_label,
                label=rgb_paths[i].split("/")[-3],
                instance=rgb_paths[i].split("-")[-1].split(".png")[0],
                view_matches=temp_view_matches,
                dynamic_matches=temp_dynamic_points
            )

            ####
            if self.predict_occlusions:
                vis_in_other = np.zeros((new_y, new_x)).astype(np.float32)
                vis_in_other_valids = np.zeros((new_y, new_x)).astype(np.float32)
                vis_trajs, other_view_vis = trajs_2d_n_occ[i][0], trajs_2d_n_occ[i][1]
                vis_trajs[:, 0] *= x_mul
                if crop_bbox[0] != 0:
                    vis_trajs[:, 0] -= crop_bbox[0]
                vis_trajs[:, 1] *= y_mul
                if crop_bbox[1] != 0:
                    vis_trajs[:, 1] -= crop_bbox[1]
                vis_trajs = vis_trajs.astype(np.int32)
                total_counter, vis_counter, occ_counter = 0, 0, 0
                for temp_idx in range(vis_trajs.shape[0]):
                    x, y = vis_trajs[temp_idx, 0], vis_trajs[temp_idx, 1]
                    if (0 <= x < new_x) and (0 <= y < new_y):
                        total_counter += 1
                        vis_counter += 1 if other_view_vis[temp_idx] else 0
                        occ_counter += 0 if other_view_vis[temp_idx] else 1
                        vis_in_other[y, x] = 1 if other_view_vis[temp_idx] else 0
                        vis_in_other_valids[y, x] = 1
                view_dict["vis_in_other"] = vis_in_other
                view_dict["vis_in_other_valids"] = vis_in_other_valids
                #print(f"{vis_in_other.shape=}, {vis_in_other_valids.shape=}")
                #print(f"{np.sum(vis_in_other)=}, {np.sum(vis_in_other_valids)=}")
            #print(f"DR -> {view_dict.keys()=}")
            ####
 
            views.append(view_dict)
        return views
        

if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import gradio as gr
    import random

    dataset_location = 'data/point_odyssey'  # Change this to the correct path
    dset = 'train'
    use_augs = False
    S = 2
    N = 1
    strides = [1,2,3,4,5,6,7,8,9]
    clip_step = 2
    quick = False  # Set to True for quick testing

    def visualize_scene(idx):
        views = dataset[idx]
        assert len(views) == 2
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.25)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                        focal=views[view_idx]['camera_intrinsics'][0, 0],
                        color=(255, 0, 0),
                        image=colors,
                        cam_size=cam_size)
        os.makedirs('./tmp/po', exist_ok=True)
        path = f"./tmp/po/po_scene_{idx}.glb"
        return viz.save_glb(path)

    dataset = PointOdysseyDUSt3R(
        dataset_location=dataset_location,
        dset=dset,
        use_augs=use_augs,
        S=S,
        N=N,
        strides=strides,
        clip_step=clip_step,
        quick=quick,
        verbose=False,
        resolution=224, 
        aug_crop=16,
        dist_type='linear_9_1',
        aug_focal=1.5,
        z_far=80)
# around 514k samples

    idxs = np.arange(0, len(dataset)-1, (len(dataset)-1)//10)
    # idx = random.randint(0, len(dataset)-1)
    # idx = 0
    for idx in idxs:
        print(f"Visualizing scene {idx}...")
        visualize_scene(idx)
