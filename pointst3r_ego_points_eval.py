import glob
import sys
import os
import argparse
import json

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def get_mast3r_ft(query_img, target_img, model, device):
    output = inference([(query_img, target_img)], model, device, batch_size=1, verbose=False)

    view1, pred1 = output["view1"], output["pred1"]
    view2, pred2 = output["view2"], output["pred2"]

    return pred1["desc"].detach().cpu(), pred2["desc"].detach().cpu()


def get_annotated_frames(valids):
    all_valid_frames = []
    for point_idx in range(valids.shape[1]):
        valid_frames = np.where(valids[:, point_idx] == 1.0)[0]
        valid_frames = valid_frames[valid_frames != 0].tolist()
        all_valid_frames.extend(valid_frames)
    return sorted(list(set(all_valid_frames)))


def load_img_for_pointst3r(rgb_path, y_res, idx, instance_name):
    img = Image.open(rgb_path)
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


def eval_on_ego_points(annot_paths, ego_points_root, y_res=384):
    all_dists, all_dists_dyn, all_dists_static = [], [], []
    total_points, total_points_dyn, total_points_static = 0, 0, 0
    thresholds = [1, 2, 4, 8, 16]
    d_metrics = {thr: 0 for thr in thresholds}
    d_metrics_dyn = {thr: 0 for thr in thresholds}
    d_metrics_static = {thr: 0 for thr in thresholds}

    seqs_used = 0
    seqs_used_dyn, seqs_used_static = 0, 0
    avg_d_avg, avg_d_avg_dyn, avg_d_avg_static = 0, 0, 0
    for annot_path in tqdm(annot_paths):
        sample_d_metrics = {thr: 0 for thr in thresholds}
        sample_d_metrics_dyn = {thr: 0 for thr in thresholds}
        sample_d_metrics_static = {thr: 0 for thr in thresholds}
        sample_points, sample_points_dyn, sample_points_static = 0, 0, 0

        seq_name = annot_path.split("/")[-2]
        annots = np.load(annot_path)
        valids = annots["valids"]
        trajs_gt = annots["trajs_2d"]
        if "dynamic_obj_tracks" in annots:
            dynamic_label_available = True
            dynamic_obj_tracks = annots["dynamic_obj_tracks"]
        else:
            dynamic_label_available = False
            dynamic_obj_tracks = None
        seqs_used += 1
        
        #dynamic_obj_tracks = annots["dynamic_obj_tracks"]
        orig_H, orig_W, _ = np.array(Image.open(f"{ego_points_root}/{seq_name}/rgbs/frame_0000000001.jpg")).shape

        target_frames = get_annotated_frames(valids)

        # Get features for query compared with each target
        query_frame_idx = 0
        mast3r_ft = {}

        # Query Image
        query_img_dict, query_img_shape = load_img_for_pointst3r(
            rgb_path=f"{ego_points_root}/{seq_name}/rgbs/frame_{str(query_frame_idx+1).zfill(10)}.jpg",
            y_res=y_res,
            idx=0,
            instance_name="query_img"
        )
        for target_frame_idx in target_frames:
            # Target Image
            target_img_dict, target_img_shape = load_img_for_pointst3r(
                rgb_path=f"{ego_points_root}/{seq_name}/rgbs/frame_{str(target_frame_idx+1).zfill(10)}.jpg",
                y_res=y_res,
                idx=1,
                instance_name="target_img"
            )
 
            query_ft, target_ft = get_mast3r_ft(query_img_dict, target_img_dict, model, device)

            temp_query_coords_resize = trajs_gt[query_frame_idx].copy()
            _, new_H, new_W, D = query_ft.shape
            s_x = new_W / orig_W
            s_y = new_H / orig_H
            temp_query_coords_resize[:, 0] *= s_x
            temp_query_coords_resize[:, 1] *= s_y
            
            temp_query_visibs = annots["visibs"][query_frame_idx]
            temp_target_visibs = annots["visibs"][target_frame_idx]
            temp_query_vis_valids = annots["vis_valids"][query_frame_idx]
            temp_target_vis_valids = annots["vis_valids"][target_frame_idx]
            
            mast3r_ft[target_frame_idx] = {"query_ft": query_ft, "target_ft": target_ft}

        for point_idx in range(valids.shape[1]):
            if dynamic_label_available:
                is_dynamic = True if dynamic_obj_tracks[point_idx] == 1 else False
            else:
                is_dynamic = None
            target_frame_idxs = sorted([
                val for val in list(np.where(valids[:, point_idx] == 1.0)[0])
            ])[1:]

            query_coord = trajs_gt[query_frame_idx, point_idx, :]
            query_frame_path = f"{ego_points_root}/{seq_name}/rgbs/frame_{str(1).zfill(10)}.jpg"
            for img_idx, target_frame_idx in enumerate(target_frame_idxs):
                target_frame_path = f"{ego_points_root}/{seq_name}/rgbs/frame_{str(target_frame_idx+1).zfill(10)}.jpg"
                target_coord = trajs_gt[target_frame_idx, point_idx, :]

                desc1_ft = mast3r_ft[target_frame_idx]["query_ft"]
                desc2_ft = mast3r_ft[target_frame_idx]["target_ft"]

                _, new_H, new_W, D = desc1_ft.shape
                s_x = new_W / orig_W
                s_y = new_H / orig_H

                # Interpolate coordinates
                query_coords_resize = query_coord.copy()
                query_coords_resize[0] *= s_x
                query_coords_resize[1] *= s_y
                target_coords_resize = target_coord.copy()
                target_coords_resize[0] *= s_x
                target_coords_resize[1] *= s_y

                # Find the features that correspond to the query
                query_ft = bilinear_sample2d(desc1_ft.permute(0,3,1,2), torch.tensor([[query_coords_resize[0]]]), torch.tensor([[query_coords_resize[1]]]))[0, :, 0]

                all_target_gt = desc2_ft[0, :, :, :].reshape(new_H*new_W, D)
                target_ft_sim = F.cosine_similarity(query_ft.unsqueeze(0).repeat(new_H*new_W, 1), all_target_gt).reshape(new_H, new_W)
 
                max_cos_pos = (target_ft_sim == torch.max(target_ft_sim)).nonzero()[0]
                target_coords_gt = torch.from_numpy(target_coords_resize)
                target_coords_gt[0] *= (256.0/512.0)
                target_coords_gt[1] *= (256.0/float(y_res))
                target_coords_pred = torch.tensor([max_cos_pos[1], max_cos_pos[0]]).float()
                target_coords_pred[0] *= (256.0/512.0)
                target_coords_pred[1] *= (256.0/float(y_res))

                dist = torch.norm(
                    target_coords_gt - target_coords_pred,
                    dim=-1
                )
                total_points += 1
                sample_points += 1

                for thr in thresholds:
                    if dist < thr:
                        d_metrics[thr] += 1
                        sample_d_metrics[thr] += 1

                all_dists.append(dist.item())

                if not dynamic_label_available:
                    continue

                if is_dynamic:
                    total_points_dyn += 1
                    sample_points_dyn += 1
                    all_dists_dyn.append(dist.item())
                    for thr in thresholds:
                        if dist < thr:
                            d_metrics_dyn[thr] += 1
                            sample_d_metrics_dyn[thr] += 1
                else:
                    total_points_static += 1
                    sample_points_static += 1
                    all_dists_static.append(dist.item())
                    for thr in thresholds:
                        if dist < thr:
                            d_metrics_static[thr] += 1
                            sample_d_metrics_static[thr] += 1
 
        sample_d_avg_all = (sum([(val/sample_points)*100.0 for _, val in sample_d_metrics.items()]) / len(sample_d_metrics)) 
        avg_d_avg += sample_d_avg_all

        sample_d_avg_dyn = None
        if sample_points_dyn != 0:
            sample_d_avg_dyn = (sum([(val/sample_points_dyn)*100.0 for _, val in sample_d_metrics_dyn.items()]) / len(sample_d_metrics_dyn))
            seqs_used_dyn += 1
            avg_d_avg_dyn += sample_d_avg_dyn

        sample_d_avg_static = None
        if sample_points_static != 0:
            sample_d_avg_static = (sum([(val/sample_points_static)*100.0 for _, val in sample_d_metrics_static.items()]) / len(sample_d_metrics_static))
            seqs_used_static += 1
            avg_d_avg_static += sample_d_avg_static

        print(f"ALL -> {sum([(val/total_points)*100.0 for _, val in d_metrics.items()]) / len(d_metrics)}, {sample_d_avg_all}")
        print(f"DYN -> {sum([(val/total_points_dyn)*100.0 for _, val in d_metrics_dyn.items()]) / len(d_metrics_dyn)}, {sample_d_avg_dyn}")
        print(f"STA -> {sum([(val/total_points_static)*100.0 for _, val in d_metrics_static.items()]) / len(d_metrics_static)}, {sample_d_avg_static}")
        print(f"{seqs_used=}, {seqs_used_dyn=}, {seqs_used_static=}")
        print(f"AVG OVER VIDEOS {seqs_used} -> all={avg_d_avg/seqs_used}, dyn={avg_d_avg_dyn/seqs_used_dyn}, static={avg_d_avg_static/seqs_used_static}")
        ###

    mte = sum(all_dists)/len(all_dists)
    mte_dyn = sum(all_dists_dyn)/len(all_dists_dyn)
    mte_static = sum(all_dists_static)/len(all_dists_static)

    d_avg = sum([(val/total_points)*100.0 for _, val in d_metrics.items()]) / len(d_metrics)
    d_avg_dyn = sum([(val/total_points_dyn)*100.0 for _, val in d_metrics_dyn.items()]) / len(d_metrics_dyn)
    d_avg_static = sum([(val/total_points_static)*100.0 for _, val in d_metrics_static.items()]) / len(d_metrics_static)

    return {
        "mte": mte, "mte_static": mte_static, "mte_dyn": mte_dyn,
        "d_avg": (avg_d_avg/seqs_used), "d_avg_static": (avg_d_avg_static/seqs_used_static), "d_avg_dyn": (avg_d_avg_dyn/seqs_used_dyn)
    }



device = 'cuda'
schedule = 'cosine'
lr = 0.01
niter = 300


parser = argparse.ArgumentParser('MASt3R EgoPoints Evaluations', add_help=False)
parser.add_argument("--eval_folder", default="/home/u5ad/rhodriguerrier.u5ad/ego_points", type=str)
parser.add_argument("--checkpoint", default="checkpoints/PointSt3R_95.pth", type=str)
parser.add_argument("--input_yres", default=384, type=int)
args = parser.parse_args()

annot_paths = sorted(glob.glob(f"{args.eval_folder}/*/annot.npz"))

model = AsymmetricMASt3R.from_pretrained(args.checkpoint).to(device)
model.eval()
metrics = eval_on_ego_points(annot_paths, args.eval_folder, y_res=args.input_yres)
print(metrics)
