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
from scipy.spatial.transform import Rotation as R
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

class CotrackerKubricVISDUSt3R(BaseStereoViewDataset):
    #def __init__(self,
    #             dataset_location='data/pointodyssey',
    #             dset='train',
    #             use_augs=False,
    #             S=2,
    #             N=16,
    #             strides=[1,2,3,4,5,6,7,8,9],
    #             clip_step=2,
    #             quick=False,
    #             verbose=True,
    #             dist_type=None,
    #             clip_step_last_skip = 0,
    #             load_masks=False,
    #             *args, 
    #             **kwargs
    #             ):
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
            remove_low_points,
            predict_occlusions,
            *args, split, ROOT, **kwargs
            #*args,
            #ROOT,
            #**kwargs
    ):
        self.ROOT = ROOT
        self.dataset_location = ROOT
        print('loading CoTracker3_Kubric dataset...')
        super().__init__(*args, **kwargs)
        self.dataset_label = 'cotracker_kubric'
        self.split = split
        self.S = S # stride
        self.N = N # min num points
        self.verbose = verbose

        self.remove_low_points = remove_low_points
        self.predict_occlusions = predict_occlusions

        self.dset = split

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

        self.sequences = []
        counter = 0
        for seq in glob.glob(f"{self.dataset_location}/*/"):
            counter += 1
            if counter == 1000:
                break
            self.sequences.append(seq)

        self.sequences = sorted(self.sequences)
        #if self.verbose:
        #    print(self.sequences)
        print('found %d unique videos in %s (self.dset=%s)' % (len(self.sequences), self.dataset_location, self.dset))
        
        ## load trajectories
        print('loading trajectories...')

        if quick:
           self.sequences = self.sequences[1:2]

        ###
        if self.remove_low_points:
            with open("/root/mast3r/kubric_train_high_point_pairs_MUCH_LARGER_STRIDES.json", "r") as f:
                high_point_samples = json.load(f)
        ###
        
        for seq in self.sequences:
            if self.verbose: 
                print('seq', seq)

            # 0000.npy  0000_trajs_2d.npy  0000_visibility.npy  0000_with_rank.npz  depths  frames
            seq_name = seq.split('/')[-2]
            rgb_path = os.path.join(seq, 'frames')
            annotations_path = os.path.join(seq, f"{seq_name}.npy")

            ###
            if self.remove_low_points:
                if seq_name not in high_point_samples:
                    print(f"Skipping {seq_name}...")
                    continue
                print(f"Reading samples from json for {seq_name}")
                if os.path.isfile(annotations_path):
                    for temp_sample in high_point_samples[seq_name]:
                        if high_point_samples[seq_name][temp_sample] < 1000:
                            continue
                        temp_start = int(temp_sample.split("_")[0])
                        temp_end = int(temp_sample.split("_")[1])
                        stride = abs(temp_end - temp_start)
                        full_idx = np.array([temp_start, temp_end])
                        self.rgb_paths.append([os.path.join(seq, 'frames', f'{str(idx).zfill(3)}.png' % idx) for idx in full_idx])
                        self.depth_paths.append([os.path.join(seq, 'depths', f'{str(idx).zfill(3)}.npy' % idx) for idx in full_idx])
                        self.annotation_paths.append(os.path.join(seq, f'{seq_name}.npy'))
                        self.full_idxs.append(full_idx)
                        self.sample_stride.append(stride)
                    if self.verbose:
                        sys.stdout.write('.')
                        sys.stdout.flush()
                continue
            ###

            if os.path.isfile(annotations_path):
                for stride in self.strides:
                    #for ii in range(0,len(os.listdir(rgb_path))-self.S*max(stride,clip_step_last_skip)+1, clip_step):
                    for ii in range(0, len(os.listdir(rgb_path)) - stride+1, clip_step):
                        if ii + stride >= 120:
                            continue
                        full_idx = ii + np.arange(self.S)*stride
                        self.rgb_paths.append([os.path.join(seq, 'frames', f'{str(idx).zfill(3)}.png') for idx in full_idx])
                        self.depth_paths.append([os.path.join(seq, 'depths', f'{str(idx).zfill(3)}.npy') for idx in full_idx])
                        self.annotation_paths.append(os.path.join(seq, f'{seq_name}.npy'))
                        self.full_idxs.append(full_idx)
                        self.sample_stride.append(stride)
                    if self.verbose:
                        sys.stdout.write('.')
                        sys.stdout.flush()
            elif self.verbose:
                print('rejecting seq for missing info or anno')
        
        self.stride_counts = {}
        self.stride_idxs = {}
        for stride in self.strides:
            self.stride_counts[stride] = 0
            self.stride_idxs[stride] = []
        for i, stride in enumerate(self.sample_stride):
            self.stride_counts[stride] += 1
            self.stride_idxs[stride].append(i)
        print('stride counts:', self.stride_counts)
        
        print(f"{self.rgb_paths[:2]=}")
        print(f"{self.depth_paths[:2]=}")
        if len(self.strides) > 1 and dist_type is not None and not self.remove_low_points:
            num_clips_per_stride = self._resample_clips(self.strides, dist_type)
        print(f"{self.rgb_paths[:2]=}")
        print(f"{self.depth_paths[:2]=}")

        # If reading from high points we need to shuffle in order to get diversity
        print(self.annotation_paths[:10])
        if self.remove_low_points:
            shuffle_idxs = np.random.permutation(len(self.annotation_paths))
            self.rgb_paths = [self.rgb_paths[i] for i in shuffle_idxs]
            self.depth_paths = [self.depth_paths[i] for i in shuffle_idxs]
            self.annotation_paths = [self.annotation_paths[i] for i in shuffle_idxs]
            self.full_idxs = [self.full_idxs[i] for i in shuffle_idxs]
            self.sample_stride = [self.sample_stride[i] for i in shuffle_idxs]
        print(self.annotation_paths[:10])

        print('collected %d clips of length %d in %s (self.dset=%s)' % (
            len(self.rgb_paths), self.S, self.dataset_location, self.dset))
        print(f"### {len(set(self.annotation_paths))=} ###")
        print(f"{len(set(self.annotation_paths))=}")
        print(f"{len(set(self.annotation_paths[:10000]))=}")
 

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

    def get_trajs_3d(self, intrinsics, extrinsics, coords, z_vals):
        img_coords = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)
        camera_coords = img_coords @ np.linalg.inv(np.transpose(intrinsics))
        camera_coords *= z_vals[:,None]
        camera_coords = np.concatenate([camera_coords, np.ones((camera_coords.shape[0], 1))], axis=1)
        return (camera_coords @ np.transpose(extrinsics))[:, :3] # [N, 3]
    
    def _get_views(self, index, resolution, rng):
        rgb_paths = self.rgb_paths[index]
        depth_paths = self.depth_paths[index]
        full_idx = self.full_idxs[index]

        annots = np.load(self.annotation_paths[index], allow_pickle=True).item()
        trajs_2d = annots["coords"][:, full_idx, :] # [N, views, 2]
        # For some reason CoTracker3 Kubric visibility is actually occluded
        visibs = ~(annots["visibility"][:, full_idx]) # [N, views]
        in_frame = np.logical_and(
            0.0 <= trajs_2d[:, :, 0], trajs_2d[:, :, 0] < 512.0
        ) * np.logical_and(
            0.0 <= trajs_2d[:, :, 1], trajs_2d[:, :, 1] < 512.0
        )

        if self.predict_occlusions:
            vis_n_valids = visibs * in_frame
            #print("### CoTracker Kubric ###")
            #print(f"{vis_n_valids.shape=}")
            #print(f"{trajs_2d.shape=}, {visibs.shape=}")
            #print(f"{trajs_2d[vis_n_valids[:,0], 0].shape=}")
            #print(f"{visibs[vis_n_valids[:,0], 1].shape=}")
            trajs_2d_n_occ = [
                [trajs_2d[vis_n_valids[:,0], 0], visibs[vis_n_valids[:,0], 1]],
                [trajs_2d[vis_n_valids[:,1], 1], visibs[vis_n_valids[:,1], 0]]
            ]

        vis_in_both = visibs[:,0] * visibs[:,1] * in_frame[:,0] * in_frame[:,1]
        trajs_2d = trajs_2d[vis_in_both, :, :] # [vis_N, views, 2]

        focal_length = annots["camera"]["focal_length"]
        sensor_width = annots["camera"]["sensor_width"]
        f_x = focal_length / sensor_width
        f_y = focal_length / sensor_width
        p_x = 0.5
        p_y = 0.5
        intrinsics = np.array([
            [f_x*512, 0., -p_x*512],
            [0., -f_y*512, -p_y*512],
            [0., 0., -1.]
        ]).astype(np.float32)
        camera_poses = []
        trajs_3d_ls = []
        for i, frame_idx in enumerate(full_idx):
            z_vals = np.load(depth_paths[i])[trajs_2d[:, i, 1].astype(int), trajs_2d[:, i, 0].astype(int)]
            position = annots["camera"]["positions"][frame_idx]
            quat = annots["camera"]["quaternions"][frame_idx]
            rotation_matrix = R.from_quat(np.concatenate([quat[1:], quat[0:1]], axis=0)).as_matrix()
            extrinsics = np.concatenate([rotation_matrix, position[:, None]], axis=1)
            camera_poses.append(np.concatenate([extrinsics, np.array([0.0, 0.0, 0.0, 1.0])[None, :]]).astype(np.float32))
            trajs_3d_ls.append(self.get_trajs_3d(intrinsics, extrinsics, trajs_2d[:, i, :], z_vals)[:, None, :])
        trajs_3d = np.concatenate(trajs_3d_ls, axis=1)

        num_trajs = trajs_2d.shape[0]
        dynamic_points = ~(np.sum(trajs_3d[:, 0] == trajs_3d[:, 1], axis=1) == 3)

        views = []
        trajs_2d_keep_ls = []
        for i in range(2):           
            impath = rgb_paths[i]
            depthpath = depth_paths[i]
            camera_pose = camera_poses[i]

            # load image and depth
            rgb_image = imread_cv2(impath)
            orig_y, orig_x, _ = rgb_image.shape
            depthmap = np.load(depthpath).astype(np.float32)

            rgb_image, depthmap, intrinsics, crop_bbox, size_before_crop = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath
            )

            trajs_2d_resize = trajs_2d[:, i, :].copy()
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
                    counter += 1
                    temp_view_matches[y, x] = temp_idx
                    temp_dynamic_points[y, x] = 1.0 if dynamic_points[temp_idx] else 0.0
 
            view_dict = dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset=self.dataset_label,
                label=rgb_paths[i].split('/')[-3],
                instance=osp.split(rgb_paths[i])[1],
                view_matches=temp_view_matches,
                dynamic_matches=temp_dynamic_points
            )

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
            #print(f"Kubric -> {view_dict.keys()=}")
            views.append(view_dict)
        ###
        #img1 = Image.open(rgb_paths[0])
        #draw1 = ImageDraw.Draw(img1)
        #img2 = Image.open(rgb_paths[1])
        #draw2 = ImageDraw.Draw(img2)
        ## 100 valid and invalid points to check
        #counter = 0
        #for i in range(trajs_2d.shape[0]):
        #    x1, y1 = trajs_2d[i, 0, 0], trajs_2d[i, 0, 1]
        #    x2, y2 = trajs_2d[i, 1, 0], trajs_2d[i, 1, 1]
        #    if y1 > 128 or x1 < 256:
        #        continue
        #    if counter < 200:
        #        counter += 1
        #        draw1.ellipse([(x1-2, y1-2), (x1+2, y1+2)], fill=(255,0,0))
        #        draw2.ellipse([(x2-2, y2-2), (x2+2, y2+2)], fill=(255,0,0))
        #    else:
        #        break
        #img1.save(f"kubric_points_view1_corres_BEFORE.jpg")
        #img2.save(f"kubric_points_view2_corres_BEFORE.jpg")
        #sys.exit(0)
        ###
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
