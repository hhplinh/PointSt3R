import glob
import time
import sys
import os
import argparse
import pickle

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images, crop_img

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tvf
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2


ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
ToTensor = tvf.ToTensor()

mini_val = [
    "basketball_5.npz",
    "softball_25.npz",
    "boxes_22.npz",
    "boxes_19.npz",
    "juggle_8.npz",
    "boxes_12.npz",
    "boxes_6.npz",
    "basketball_29.npz",
    "tennis_28.npz",
    "tennis_22.npz",
    "basketball_9.npz",
    "basketball_24.npz",
    "football_3.npz",
    "tennis_17.npz",
    "softball_21.npz",
    "tennis_23.npz",
    "juggle_5.npz",
    "football_1.npz",
    "tennis_5.npz",
    "basketball_6.npz",
    "basketball_14.npz",
    "football_21.npz",
    "football_19.npz",
    "basketball_4.npz",
    "basketball_3.npz",
    "softball_2.npz",
    "boxes_11.npz",
    "juggle_4.npz",
    "softball_23.npz",
    "juggle_7.npz",
    "football_16.npz",
    "boxes_29.npz",
    "boxes_7.npz",
    "juggle_9.npz",
    "boxes_17.npz",
    "juggle_22.npz",
    "football_29.npz",
    "football_22.npz",
    "boxes_28.npz",
    "tennis_2.npz",
    "softball_9.npz",
    "basketball_13.npz",
    "tennis_4.npz",
    "football_7.npz",
    "softball_19.npz",
    "basketball_20.npz",
    "tennis_26.npz",
    "softball_14.npz",
    "boxes_5.npz",
    "boxes_27.npz",
]


def get_mast3r_ft(query_img, target_img, model, device):
    output = inference([(query_img, target_img)], model, device, batch_size=1, verbose=False)

    view1, pred1 = output["view1"], output["pred1"]
    view2, pred2 = output["view2"], output["pred2"]

    return pred1["desc"].detach().cpu(), pred2["desc"].detach().cpu(), \
        pred1["pts3d"].detach().cpu(), pred2["pts3d_in_other_view"].detach().cpu()


def decode_img_bytes(img_bytes):
    video = []
    for frame_bytes in img_bytes:
        arr = np.frombuffer(frame_bytes, np.uint8)
        image_bgr = cv2.imdecode(arr, flags=cv2.IMREAD_UNCHANGED)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        video.append(image_rgb)
    return np.stack(video, axis=0)


def load_img_for_pointst3r(rgb_frame, y_res, idx, instance_name):
    img = Image.fromarray(rgb_frame.astype(np.uint8))
    img = img.resize((512, y_res), Image.Resampling.LANCZOS)
    img_dict = dict(
        img=ImgNorm(img)[None],
        true_shape=np.int32([img.size[::-1]]),
        idx=idx,
        instance=instance_name,
        mask=~(ToTensor(img)[None].sum(1) <= 0.01)
    )
    img_dict['dynamic_mask'] = torch.zeros_like(img_dict['mask'])
    return img_dict, np.array(img).shape


def get_pstudio_results(
    split=0,
    dataset_location="~/tapvid3d_datasets",
    save_folder="pstudio_minival_results_PointSt3R_95",
    y_res=288
):
    os.makedirs(f"./{save_folder}/pstudio", exist_ok=True)
    done_files = sorted([
        path.split("/")[-1]
        for path in glob.glob(f"./{save_folder}/pstudio/*npz")
    ])
    for npz_counter, npz_file in enumerate(mini_val[(split*10):((split*10) + 10)]):
        if npz_file in done_files:
            print(f"Skipping {npz_file}, already computed...")
            continue

        npz_path = f"{dataset_location}/pstudio/{npz_file}"
        annots = np.load(npz_path)
        rgbs = decode_img_bytes(annots["images_jpeg_bytes"])
        visibs = annots["visibility"]
        tracks_xyz = annots["tracks_XYZ"]
        queries_xyt = annots["queries_xyt"]
        seq_name = npz_path.split("/")[-1].split(".npz")[0]

        pred_trajs = np.zeros((rgbs.shape[0], queries_xyt.shape[0], 3))
        pred_vals = np.zeros((rgbs.shape[0], queries_xyt.shape[0]))
        for point_idx in tqdm(range(queries_xyt.shape[0])):
            query_frame = int(queries_xyt[point_idx, 2])

            # Query image
            query_img_dict, _ = load_img_for_pointst3r(
                rgb_frame=rgbs[query_frame],
                y_res=y_res,
                idx=0,
                instance_name="query_img"
            )

            query_point = queries_xyt[point_idx, :2]
            H, W, _ = rgbs[0].shape
            H_, W_ = y_res, 512
            sy = H_/H
            sx = W_/W
            query_point[0] *= sx
            query_point[1] *= sy
            if not visibs[query_frame, point_idx]:
                continue

            target_frames = [i for i in range(rgbs.shape[0]) if i != query_frame]
            for frame_idx in target_frames:
                if visibs[frame_idx, point_idx]:
                    pred_vals[frame_idx, point_idx] = 1.0
                else:
                    continue

                # Target Image
                target_img_dict, _ = load_img_for_pointst3r(
                    rgb_frame=rgbs[frame_idx],
                    y_res=y_res,
                    idx=1,
                    instance_name="target_img"
                )
 
                # Get mast3r features
                query_ft, target_ft, query_pts3d, target_pts3d = get_mast3r_ft(
                    query_img_dict,
                    target_img_dict,
                    model,
                    device
                )
                _, ft_H, ft_W, D = query_ft.shape

                # Find target point
                query_3d = query_pts3d[:, int(query_point[1]), int(query_point[0]), :]
                if frame_idx == query_frame + 1:
                    pred_trajs[query_frame, point_idx, :] = query_3d
                query_point_ft = query_ft[0, int(query_point[1]), int(query_point[0]), :]

                target_ft_sim = F.cosine_similarity(
                    query_point_ft.unsqueeze(0).repeat(ft_H * ft_W, 1),
                    target_ft[0, :, :, :].reshape(ft_H * ft_W, D)
                ).reshape(ft_H, ft_W)

                max_cos_pos = (target_ft_sim == torch.max(target_ft_sim)).nonzero()[0]
                target_coords_pred = torch.tensor([
                    max_cos_pos[1], max_cos_pos[0]
                ]).float().to(device)
                target_3d = target_pts3d[:, max_cos_pos[0], max_cos_pos[1], :]
                pred_trajs[frame_idx, point_idx, :] = target_3d

        np.savez(
            f"./{save_folder}/pstudio/{seq_name}.npz",
            tracks_XYZ=pred_trajs,
            visibility=np.ones(pred_vals.shape[:2])
        )

device = 'cuda'
schedule = 'cosine'
lr = 0.01
niter = 300


parser = argparse.ArgumentParser('PointSt3R Pstudio Minival Evaluation', add_help=False)
parser.add_argument("--checkpoint", default="checkpoints/PointSt3R_95.pth")
parser.add_argument("--save_folder", default="./pstudio_minival_results_PointSt3R_95")
parser.add_argument("--dataset_location", default="~/tapvid3d_datasets")
parser.add_argument("--split", default=0, type=int)
parser.add_argument("--input_yres", default=288, type=int)
args = parser.parse_args()

model = AsymmetricMASt3R.from_pretrained(checkpoint_path).to(device)
model.eval()
get_pstudio_results(save_folder=args.save_folder, dataset_location=args.dataset_location, split=args.split, y_res=args.input_yres)
