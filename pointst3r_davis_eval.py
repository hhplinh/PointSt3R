import glob
import sys
import os
import argparse
import time
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

from pips2_utils.samp import bilinear_sample2d


ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
ToTensor = tvf.ToTensor()
TAG_FLOAT = 202021.25


class TapVidDavis(Dataset):
    def __init__(self, dataset_location="/lus/lfs1aip2/home/u5ad/rhodriguerrier.u5ad"):
        input_path = f"{dataset_location}/tapvid_davis.pkl"
        with open(input_path, "rb") as f:
            data = pickle.load(f)
            self.seq_names = list(data.keys())
            if isinstance(data, dict):
                data = list(data.values())
        self.data = data
        print(f"Found {len(self.data)} video in {input_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dat = self.data[index]
        rgbs = dat['video'] # list of H,W,C uint8 images
        trajs = dat['points'] # N,S,2 array
        valids = 1-dat['occluded'] # N,S array
        # note the annotations are only valid when not occluded

        trajs = trajs.transpose(1,0,2) # S,N,2
        valids = valids.transpose(1,0) # S,N

        ###
        #vis_ok = valids[0] > 0
        #trajs = trajs[:,vis_ok]
        #valids = valids[:,vis_ok]
        ###

        # 1.0,1.0 should lie at the bottom-right corner pixel
        H, W, C = rgbs[0].shape
        trajs[:,:,0] *= W-1
        trajs[:,:,1] *= H-1

        rgbs = torch.from_numpy(np.stack(rgbs,0)).permute(0,3,1,2) # S,C,H,W
        trajs = torch.from_numpy(trajs) # S,N,2
        valids = torch.from_numpy(valids) # S,N

        sample = {
            'seq_name': self.seq_names[index],
            'rgbs': rgbs,
            'trajs': trajs,
            'valids': valids,
            'visibs': valids,
        }
        return sample


def get_mast3r_ft(query_img, target_img, model, device):
    output = inference([(query_img, target_img)], model, device, batch_size=1, verbose=False)

    view1, pred1 = output["view1"], output["pred1"]
    view2, pred2 = output["view2"], output["pred2"]

    pred1_vis_mask = None if "vis_in_other" not in pred1 else F.sigmoid(pred1["vis_in_other"][0]).detach().cpu()
    pred2_vis_mask = None if "vis_in_other" not in pred2 else F.sigmoid(pred2["vis_in_other"][0]).detach().cpu()

    return pred1["desc"].detach().cpu(), pred2["desc"].detach().cpu(), pred1_vis_mask, pred2_vis_mask


def load_img_for_pointst3r(rgb_frame, y_res, idx, instance_name):
    img = Image.fromarray(rgb_frame)
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


def predict_point_trajectory(query_point, query_ft, target_ft):
    # Find the features that correspond to the query
    _, ft_H, ft_W, D = query_ft.shape
    query_point_ft = bilinear_sample2d(
        query_ft.permute(0,3,1,2),
        torch.tensor([[query_point[0]]]),
        torch.tensor([[query_point[1]]])
    )[0, :, 0]

    # Calculate cosine similarity of query with all target pixels
    target_ft_sim = F.cosine_similarity(
        query_point_ft.unsqueeze(0).repeat(ft_H * ft_W, 1),
        target_ft[0, :, :, :].reshape(ft_H * ft_W, D)
    ).reshape(ft_H, ft_W)

    # Define target trajectory as maximum cosine similarity
    max_cos_pos = (target_ft_sim == torch.max(target_ft_sim)).nonzero()[0]
    target_coords_pred = torch.tensor([max_cos_pos[1], max_cos_pos[0]]).float().to(device)

    return target_coords_pred


def eval_on_davis_points(y_res=384, w_vis=False, davis_root="/lus/lfs1aip2/home/u5ad/rhodriguerrier.u5ad"):
    thresholds = [1, 2, 4, 8, 16]

    dataset = TapVidDavis(dataset_location=davis_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    d_avg_ls = []
    vis_ls, vis_pos_ls, vis_neg_ls = [], [], []
    for sample_idx, sample in tqdm(enumerate(dataloader)):
        seq_name = sample["seq_name"][0]

        rgbs = sample["rgbs"][0]
        trajs_g = sample["trajs"][0].cuda().float()
        valids = sample["valids"][0].cuda().float()
        visibs_g = sample["visibs"][0].cuda().float()

        # Query Frame
        query_img_dict, query_img_shape = load_img_for_pointst3r(
            rgbs[0].permute(1,2,0).numpy().astype(np.uint8),
            y_res=y_res, idx=0, instance_name="query_img"
        )

        _, H, W = rgbs[0].shape
        H_, W_, _ = query_img_shape
        sy = H_/H
        sx = W_/W
        trajs_g[:,:,0] *= sx
        trajs_g[:,:,1] *= sy

        query_points = trajs_g[0]
        trajs_e = torch.zeros_like(trajs_g)
        visibs_e = torch.zeros_like(visibs_g)

        first_appearances = valids.argmax(dim=0)
        print(first_appearances)
        for frame_idx in tqdm(range(1, rgbs.shape[0])):
            # Target Frame
            target_img_dict, target_img_shape = load_img_for_pointst3r(
                rgbs[frame_idx].permute(1,2,0).numpy().astype(np.uint8),
                y_res=y_res, idx=1, instance_name="target_img"
            )

            query_ft, target_ft, query_vis_mask, _ = get_mast3r_ft(query_img_dict, target_img_dict, model, device)

            for point_idx in range(trajs_g.shape[1]):
                if w_vis:
                    visibs_e[frame_idx, point_idx] = 1 if query_vis_mask[int(query_points[point_idx, 1]), int(query_points[point_idx, 0])] > 0.5 else 0
                if first_appearances[point_idx] != 0:
                    continue
                if valids[frame_idx, point_idx] != 1:
                    continue

                trajs_e[frame_idx, point_idx, :] = predict_point_trajectory(query_points[point_idx, :], query_ft, target_ft)
 
        # Then, do new for remaining points
        for point_idx in tqdm(range(trajs_g.shape[1])):
            if first_appearances[point_idx] == 0:
                continue

            query_idx = first_appearances[point_idx]

            # Query image
            query_img_dict, query_img_shape = load_img_for_pointst3r(
                rgbs[query_idx].permute(1,2,0).numpy().astype(np.uint8),
                y_res=y_res, idx=0, instance_name="query_img"
            )

            query_point = trajs_g[query_idx, point_idx]
            for frame_idx in range(query_idx+1, rgbs.shape[0]):
                if valids[frame_idx, point_idx] != 1:
                    continue

                # Target image
                target_img_dict, target_img_shape = load_img_for_pointst3r(
                    rgbs[frame_idx].permute(1,2,0).numpy().astype(np.uint8),
                    y_res=y_res, idx=1, instance_name="target_img"
                )
       
                query_ft, target_ft, query_vis_mask, _ = get_mast3r_ft(query_img_dict, target_img_dict, model, device)

                if w_vis:
                    visibs_e[frame_idx, point_idx] = 1 if query_vis_mask[int(query_point[1]), int(query_point[0])] > 0.5 else 0

                if valids[frame_idx, point_idx] != 1:
                    continue

                trajs_e[frame_idx, point_idx, :] = predict_point_trajectory(query_point, query_ft, target_ft)
 
        sx = (256.0 / 512.0)
        sy = (256.0 / float(y_res))

        trajs_g[:, :, 0] *= sx
        trajs_g[:, :, 1] *= sy
        trajs_e[:, :, 0] *= sx
        trajs_e[:, :, 1] *= sy

        temp_d_sum = 0.0
        for thr in thresholds:
            points_correct, points_total = 0, 0
            # Need to loop through points as each has a different starting index
            for point_idx in range(trajs_g.shape[1]):
                first_app = first_appearances[point_idx]
                dists = torch.norm(
                    (trajs_g[first_app + 1:, point_idx] - trajs_e[first_app + 1:, point_idx]),
                    dim=-1
                )[torch.where(visibs_g[first_app + 1:, point_idx] == 1)]
                points_correct += torch.sum(dists < thr)
                points_total += torch.sum(visibs_g[first_app + 1:, point_idx])

            temp_d_sum += (points_correct / points_total) * 100.0
        
        sample_d_avg = temp_d_sum / len(thresholds)
        print(sample_idx, sample_d_avg)
        d_avg_ls.append(sample_d_avg)
        print(f"RUNNING d_avg = {sum(d_avg_ls) / len(d_avg_ls)}%")

        # Visibility Accuracy
        if w_vis:
            # All
            vis_matches = (visibs_g == visibs_e)
            vis_ls.append((torch.sum(vis_matches) / torch.numel(visibs_g)) * 100.0)

            # Positive
            vis_pos_matches = (visibs_g[torch.where(visibs_g == 1)] == visibs_e[torch.where(visibs_g == 1)])
            vis_pos_ls.append((torch.sum(vis_pos_matches) / torch.sum(visibs_g)) * 100.0)

            # Negative
            occ_g = (1 - visibs_g)
            vis_neg_matches = (visibs_g[torch.where(occ_g == 1)] == visibs_e[torch.where(occ_g == 1)])
            if torch.sum(occ_g) != 0:
                vis_neg_ls.append((torch.sum(vis_neg_matches) / torch.sum(occ_g)) * 100.0)

            print(f"RUNNING vis_acc={sum(vis_ls)/len(vis_ls)}%, vis_acc_pos={sum(vis_pos_ls)/len(vis_pos_ls)}%, vis_acc_neg={sum(vis_neg_ls)/len(vis_neg_ls) if len(vis_neg_ls) != 0 else None}%")
 
    metrics = {"d_avg": (sum(d_avg_ls) / len(d_avg_ls))}
    if w_vis:
        metrics["vis_acc"] = sum(vis_ls) / len(vis_ls)
        metrics["vis_acc_pos"] = sum(vis_pos_ls) / len(vis_pos_ls)
        metrics["vis_acc_neg"] = sum(vis_neg_ls) / len(vis_neg_ls)
    return metrics


device = 'cuda'
schedule = 'cosine'
lr = 0.01
niter = 300

parser = argparse.ArgumentParser('PointSt3R DAVIS Evaluation', add_help=False)
parser.add_argument("--checkpoint", default="checkpoints/PointSt3R_95.pth", type=str)
parser.add_argument("--input_yres", default=384, type=int)
parser.add_argument("--davis_root", default="/lus/lfs1aip2/home/u5ad/rhodriguerrier.u5ad", type=str)
args = parser.parse_args()
checkpoint_path = args.checkpoint
davis_root = args.davis_root
y_res = args.input_yres

model = AsymmetricMASt3R.from_pretrained(checkpoint_path).to(device)
model.eval() 
metrics = eval_on_davis_points(y_res=y_res, w_vis=True if "vis" in checkpoint_path else False, davis_root=davis_root)
print(metrics)
