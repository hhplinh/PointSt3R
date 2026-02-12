import os
import glob
import argparse
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as tvf
from mast3r.model import AsymmetricMASt3R
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ImgNorm = tvf.Compose([
    tvf.ToTensor(),
    tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
ToTensor = tvf.ToTensor()

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

def get_mast3r_ft(query_img, target_img, model, device):
    output = inference([(query_img, target_img)], model, device, batch_size=1, verbose=False)
    view1, pred1 = output["view1"], output["pred1"]
    view2, pred2 = output["view2"], output["pred2"]
    return pred1["desc"].detach().cpu(), pred2["desc"].detach().cpu()

def main():
    parser = argparse.ArgumentParser("PointSt3R Inference on Custom Frames")
    parser.add_argument("--frames_dir", default="data/frames", required=True, help="Directory with input frames (jpg/png)")
    parser.add_argument("--checkpoint", default="checkpoints/PointSt3R_95.pth", help="Path to model checkpoint")
    parser.add_argument("--input_yres", default=288, type=int, help="Image height after resize")
    parser.add_argument("--num_points", default=10, type=int, help="Number of points to track")
    parser.add_argument("--output_dir", default="output_tracks", help="Directory to save visualizations")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, '*.jpg')) + glob.glob(os.path.join(args.frames_dir, '*.png')))
    if len(frame_paths) < 2:
        logging.error("Need at least 2 frames for tracking.")
        return

    # Load model
    model = AsymmetricMASt3R.from_pretrained(args.checkpoint).to(device)
    logging.info(f"Loaded model from {args.checkpoint}")
    model.eval()
    logging.info(f"Tracking {args.num_points} points across {len(frame_paths)} frames.")

    # Load first frame and select points randomly
    first_img = Image.open(frame_paths[0])
    w, h = first_img.size
    np.random.seed(42)
    points = np.stack([
        np.random.uniform(0, w, args.num_points),
        np.random.uniform(0, h, args.num_points)
    ], axis=-1)
    logging.info(f"Initial points: {points}")

    # Prepare first image for model
    query_img_dict, query_img_shape = load_img_for_pointst3r(frame_paths[0], args.input_yres, 0, "query_img")
    H_, W_, _ = query_img_shape
    sy = H_ / h
    sx = W_ / w
    points_scaled = points.copy()
    points_scaled[:, 0] *= sx
    points_scaled[:, 1] *= sy
    logging.info(f"Scaled points for model input: {points_scaled}")

    # Track points across frames
    tracks = [points.copy()]
    prev_img_dict = query_img_dict
    prev_points = points_scaled
    for idx, frame_path in enumerate(frame_paths[1:], 1):
        target_img_dict, _ = load_img_for_pointst3r(frame_path, args.input_yres, idx, f"frame_{idx}")
        query_ft, target_ft = get_mast3r_ft(prev_img_dict, target_img_dict, model, device)
        _, ft_H, ft_W, D = query_ft.shape
        new_points = []
        for pt in prev_points:
            # Feature matching
            from pips2_utils.samp import bilinear_sample2d
            import torch.nn.functional as F
            pt_tensor = torch.tensor([[pt[0]]]), torch.tensor([[pt[1]]])
            query_point_ft = bilinear_sample2d(query_ft.permute(0,3,1,2), pt_tensor[0], pt_tensor[1])[0, :, 0]
            target_ft_sim = F.cosine_similarity(
                query_point_ft.unsqueeze(0).repeat(ft_H * ft_W, 1),
                target_ft[0, :, :, :].reshape(ft_H * ft_W, D)
            ).reshape(ft_H, ft_W)
            max_cos_pos = (target_ft_sim == torch.max(target_ft_sim)).nonzero()[0]
            pred = [max_cos_pos[1].item() * (w/512.0), max_cos_pos[0].item() * (h/args.input_yres)]
            new_points.append(pred)
        logging.info(f"New points for frame {idx}: {new_points}")
        tracks.append(np.array(new_points))
        prev_img_dict = target_img_dict
        prev_points = np.array(new_points) * [sx, sy]
        logging.info(f"Updated previous points for next iteration: {prev_points}")
    
    # Visualization
    for idx, (frame_path, pts) in enumerate(zip(frame_paths, tracks)):
        img = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for pt in pts:
            x, y = pt
            r = 4
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(255,0,0))
        img.save(os.path.join(args.output_dir, f"tracked_{idx:04d}.jpg"))
    logging.info(f"Tracking results saved to {args.output_dir}")

if __name__ == "__main__":
    main()

# python pointst3r_infer_on_frames.py --frames_dir data/frames --checkpoint checkpoints/PointSt3R_95.pth --input_yres 288 --num_points 10 --output_dir output_tracks