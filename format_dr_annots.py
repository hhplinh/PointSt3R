import gzip
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm


# Script formats DynamicReplica data to fit the dataloader
parser = argparse.ArgumentParser('PointSt3R DynamicReplica Formatter', add_help=False)
parser.add_argument("--dr_root", default="~/dynamic_stereo/dynamic_replica_data/train", help="Location of DR train data download")
parser.add_argument("--save_root", default="~/dynamic_stereo/dynamic_replica_mast3r_annots/train", help="Location to save re-formatted DR data")
args = parser.parse_args()
root = args.dr_root
save_root = args.save_root

with gzip.open(f"{root}/frame_annotations_train.jgz", "rt", encoding="utf8") as f:
    frame_annots_list = json.load(f)

seq_counter = 0
for i in range(0, len(frame_annots_list), 300):
    # Skip right side cameras
    if frame_annots_list[i]["camera_name"] != "left":
        print(f"Skipping right camera -> {frame_annots_list[i]['camera_name']}")
        continue

    sample_dicts = frame_annots_list[i:i+300]
    seq_name = sample_dicts[0]["sequence_name"]
    seq_counter += 1
    print(seq_name, seq_counter)

    cam_to_worlds_ls = []
    world_to_cams_ls = []
    intrinsics_ls = []
    trajs_2d_ls = []
    trajs_3d_ls = []
    visibs_ls = []

    for frame_dict in tqdm(sample_dicts):
        # Construct intrinsics matrix - convert from ndc to normal
        # See https://pytorch3d.org/docs/cameras for details
        H, W = frame_dict["image"]["size"]
        s = min(W, H)
        fx_ndc = frame_dict["viewpoint"]["focal_length"][0]
        fy_ndc = frame_dict["viewpoint"]["focal_length"][1]
        px_ndc = frame_dict["viewpoint"]["principal_point"][0]
        py_ndc = frame_dict["viewpoint"]["principal_point"][1]

        fx = (fx_ndc * s) / 2
        fy = (fy_ndc * s) / 2
        cx = W / 2 - px_ndc * s / 2
        cy = H / 2 - py_ndc * s / 2
        intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        intrinsics_ls.append(intrinsic[None])

        # Construct extrinsics matrix
        # pytorch3d R and T are designed for world to camera 
        # this is the opposite of usual, so we require 
        # R_cw = R^T and T_cw = -R^T * T
        cam_to_world = np.eye(4)
        R_wc = np.array(frame_dict["viewpoint"]["R"]).T
        t = np.array(frame_dict["viewpoint"]["T"]).reshape(3, 1)
        t_wc = -R_wc @ t
        cam_to_world[:3, :3] = R_wc
        cam_to_world[:3, 3] = t_wc.reshape(3)
        cam_to_worlds_ls.append(cam_to_world[None])

        # Get trajectories and visibilities from .pth file
        trajs = torch.load(f"{root}/{frame_dict['trajectories']['path']}")
        trajs_2d_ls.append(trajs["traj_2d"][None])
        trajs_3d_ls.append(trajs["traj_3d_world"][None])
        visibs_ls.append(trajs["verts_inds_vis"][None])

    new_annots = {
        "cam_to_worlds": np.concatenate(cam_to_worlds_ls, axis=0),
        "intrinsics": np.concatenate(intrinsics_ls, axis=0),
        "trajs_2d": np.concatenate(trajs_2d_ls, axis=0),
        "trajs_3d": np.concatenate(trajs_3d_ls, axis=0),
        "visibs": np.concatenate(visibs_ls, axis=0)
    }
    np.save(f"{save_root}/{seq_name}.npy", new_annots)
