import glob
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

from pips2_utils.samp import bilinear_sample2d

np.random.seed(0)
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
ToTensor = tvf.ToTensor()


class POTest(Dataset):
    def __init__(self, dataset_location, annots_location, max_points=200, max_frames=100):
        self.dataset_location = dataset_location
        self.annots_location = annots_location
        self.seq_paths = [path for path in glob.glob(f"{dataset_location}/*") if ".mp4" not in path and ".py" not in path]
        self.max_points = max_points
        self.max_frames = max_frames

    def __len__(self):
        return len(self.seq_paths)

    def __getitem__(self, index):
        seq_path = self.seq_paths[index]
        seq_name = seq_path.split("/")[-1]
        rgb_paths = np.array(sorted(
            glob.glob(f"{seq_path}/rgbs/*jpg")
        ))
        annotations = np.load(
            f"{self.annots_location}/{seq_name}/anno.npy",
            allow_pickle=True
        ).item()
        
        trajs = annotations["trajs_2d"]
        valids = annotations["valids"]
        visibs = annotations["visibs"]
        static_points = annotations["static_points"]

        # select no more than self.max_frames for efficiency
        frame_idx_sample = np.linspace(
            0, len(rgb_paths)-1, self.max_frames
        ).astype(np.int32)
        rgb_paths = rgb_paths[frame_idx_sample]
        trajs = trajs[frame_idx_sample]
        visibs = visibs[frame_idx_sample]
        valids = valids[frame_idx_sample]

        # get rid of infs and nans
        valids_xy = np.ones_like(trajs)
        inf_idx = np.where(np.isinf(trajs))
        trajs[inf_idx] = 0
        valids_xy[inf_idx] = 0
        nan_idx = np.where(np.isnan(trajs))
        trajs[nan_idx] = 0
        valids_xy[nan_idx] = 0
        inv_idx = np.where(np.sum(valids_xy, axis=2) < 2)
        visibs[inv_idx] = 0
        valids[inv_idx] = 0

        # update visibility annotations
        H, W = 540, 960
        for si in range(trajs.shape[0]):
            # avoid 1px edge
            oob_inds = np.logical_or(
                np.logical_or(trajs[si,:,0] < 1, trajs[si,:,0] > W-2),
                np.logical_or(trajs[si,:,1] < 1, trajs[si,:,1] > H-2))
            visibs[si,oob_inds] = 0
            # exclude oob from eval
            valids[si,oob_inds] = 0

        # ensure that the point is good at frame0
        vis_and_val = valids * visibs
        vis0 = vis_and_val[0] > 0
        trajs = trajs[:,vis0]
        visibs = visibs[:,vis0]
        valids = valids[:,vis0]
        static_points = static_points[vis0]

        # ensure that the point is good in at least K frames total
        vis_and_val = valids * visibs
        val_ok = np.sum(vis_and_val, axis=0) >= 8
        trajs = trajs[:,val_ok]
        visibs = visibs[:,val_ok]
        valids = valids[:,val_ok]
        static_points = static_points[val_ok]

        sample = {
            "seq_name": seq_path.split("/")[-1],
            "rgb_paths": rgb_paths.tolist(),
            "trajs": trajs,
            "valids": valids,
            "visibs": visibs,
            "static_points": static_points
        }
        return sample


def get_mast3r_ft(query_img, target_img, model, device):
    output = inference([(query_img, target_img)], model, device, batch_size=1, verbose=False)

    view1, pred1 = output["view1"], output["pred1"]
    view2, pred2 = output["view2"], output["pred2"]

    return pred1["desc"].detach().cpu(), pred2["desc"].detach().cpu()


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


def eval_on_po_test_points(
    dataset_location="/home/u5ad/rhodriguerrier.u5ad/pointodyssey_v2/test",
    annots_location="~/pointodyssey_v2/static_dynamic_test",
    y_res=384
):
    thresholds = [1, 2, 4, 8, 16]
    dataset = POTest(dataset_location=dataset_location)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    d_avg_ls, d_avg_sta_ls, d_avg_dyn_ls = [], [], []
    for sample_idx, sample in tqdm(enumerate(dataloader)):
        seq_name = sample["seq_name"]
        rgb_paths = sample["rgb_paths"]
        trajs_g = sample["trajs"][0].cuda().float()
        valids = sample["valids"][0].cuda().float()
        visibs_g = sample["visibs"][0].cuda().float()
        static_points = sample["static_points"][0].cuda().float()

        d_metric = {thr: 0 for thr in thresholds}
        d_metric_dyn = {thr: 0 for thr in thresholds}
        d_metric_static = {thr: 0 for thr in thresholds}
        sample_points = 0
        sample_points_dyn = 0
        sample_points_static = 0

        # Query image
        init_rgb_img = Image.open(rgb_paths[0][0])
        query_img_dict, query_img_shape = load_img_for_pointst3r(
            rgb_path=rgb_paths[0][0],
            y_res=y_res,
            idx=0,
            instance_name="query_img"
        )

        W, H = init_rgb_img.size
        H_, W_, _ = query_img_shape
        sy = H_/H
        sx = W_/W
        trajs_g[:,:,0] *= sx
        trajs_g[:,:,1] *= sy

        query_points = trajs_g[0]
        query_points = query_points[torch.where(valids[0] == 1)]
        for frame_idx in tqdm(range(1, trajs_g.shape[0])):
            # Target image
            target_img_dict, _ = load_img_for_pointst3r(
                rgb_path=rgb_paths[frame_idx][0],
                y_res=y_res,
                idx=1,
                instance_name="target_img"
            )

            query_ft, target_ft = get_mast3r_ft(query_img_dict, target_img_dict, model, device)
            _, ft_H, ft_W, D = query_ft.shape

            for point_idx in range(query_points.shape[0]):
                if (valids[frame_idx, point_idx] * visibs_g[frame_idx, point_idx]) != 1:
                    continue

                query_point_ft = bilinear_sample2d(
                    query_ft.permute(0,3,1,2),
                    torch.tensor([[query_points[point_idx, 0]]]),
                    torch.tensor([[query_points[point_idx, 1]]])
                )[0, :, 0]

                target_ft_sim = F.cosine_similarity(
                    query_point_ft.unsqueeze(0).repeat(ft_H * ft_W, 1),
                    target_ft[0, :, :, :].reshape(ft_H * ft_W, D)
                ).reshape(ft_H, ft_W)

                max_cos_pos = (target_ft_sim == torch.max(target_ft_sim)).nonzero()[0]
                target_coords_gt = trajs_g[frame_idx, point_idx, :]
                target_coords_gt[0] *= (256.0/512.0)
                target_coords_gt[1] *= (256.0/float(y_res))
                target_coords_pred = torch.tensor([max_cos_pos[1], max_cos_pos[0]]).float().to(device)
                target_coords_pred[0] *= (256.0/512.0)
                target_coords_pred[1] *= (256.0/float(y_res))

                dist = torch.norm(
                    target_coords_gt - target_coords_pred,
                    dim=-1
                )
                sample_points += 1

                for thr in thresholds:
                    if dist < thr:
                        d_metric[thr] += 1

                # Point is static
                if static_points[point_idx] == 1:
                    sample_points_static += 1
                    for thr in thresholds:
                        if dist < thr:
                            d_metric_static[thr] += 1
                else: # Point is dynamic
                    sample_points_dyn += 1
                    for thr in thresholds:
                        if dist < thr:
                            d_metric_dyn[thr] += 1

        d_avg_ls.append(sum(
            [(val/sample_points)*100.0 for _, val in d_metric.items()]
        ) / len(d_metric))

        if sample_points_static != 0:
            d_avg_sta_ls.append(sum(
                [(val/sample_points_static)*100.0 for _, val in d_metric_static.items()]
            ) / len(d_metric_static))

        if sample_points_dyn != 0:
            d_avg_dyn_ls.append(sum(
                [(val/sample_points_dyn)*100.0 for _, val in d_metric_dyn.items()]
            ) / len(d_metric_dyn))

        print(f"RUNNING d_avg={sum(d_avg_ls)/len(d_avg_ls)}%, d_avg_sta={sum(d_avg_sta_ls)/len(d_avg_sta_ls) if len(d_avg_sta_ls) != 0 else None}%, d_avg_dyn={sum(d_avg_dyn_ls)/len(d_avg_dyn_ls) if len(d_avg_dyn_ls) != 0 else None}%")

    return {
        "d_avg": sum(d_avg_ls)/len(d_avg_ls),
        "d_avg_sta": sum(d_avg_sta_ls)/len(d_avg_sta_ls),
        "d_avg_dyn": sum(d_avg_dyn_ls)/len(d_avg_dyn_ls)
    }


device = 'cuda'
schedule = 'cosine'
lr = 0.01
niter = 300


parser = argparse.ArgumentParser('MASt3R EgoPoints Evaluations', add_help=False)
parser.add_argument("--checkpoint", default="checkpoints/PointSt3R_95.pth")
parser.add_argument("--input_yres", default=384, type=int)
parser.add_argument("--dataset_location", default="~/pointodyssey_v2/test", help="Location of full PointOdyssey dataset")
parser.add_argument("--annots_location", default="~/pointodyssey_v2/static_dynamic_test", help="Location of static/dynamic split annotation files")
args = parser.parse_args()

model = AsymmetricMASt3R.from_pretrained(args.checkpoint).to(device)
model.eval()
metrics = eval_on_po_test_points(
    dataset_location=args.dataset_location,
    annots_location=args.annots_location,
    y_res=args.input_yres
)

print(metrics)
