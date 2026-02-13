import os
import glob
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as tvf
from mast3r.model import AsymmetricMASt3R
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
import logging
import gradio as gr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
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
    return pred1["desc"].detach(), pred2["desc"].detach()

def gradio_pointst3r(frames_dir, checkpoint, input_yres, output_dir, points):
    os.makedirs(output_dir, exist_ok=True)
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')) + glob.glob(os.path.join(frames_dir, '*.png')))
    if len(frame_paths) < 2:
        return "Need at least 2 frames for tracking.", None
    model = AsymmetricMASt3R.from_pretrained(checkpoint).to(device)
    model.eval()
    first_img = Image.open(frame_paths[0])
    w, h = first_img.size
    points_np = np.array(points)
    query_img_dict, query_img_shape = load_img_for_pointst3r(frame_paths[0], input_yres, 0, "query_img")
    H_, W_, _ = query_img_shape
    sy = H_ / h
    sx = W_ / w
    points_scaled = points_np.copy()
    points_scaled = points_scaled.astype(np.float64)
    points_scaled[:, 0] *= sx
    points_scaled[:, 1] *= sy
    tracks = [points_np.copy()]
    prev_img_dict = query_img_dict
    prev_points = points_scaled
    for idx, frame_path in enumerate(frame_paths[1:], 1):
        target_img_dict, _ = load_img_for_pointst3r(frame_path, input_yres, idx, f"frame_{idx}")
        query_ft, target_ft = get_mast3r_ft(prev_img_dict, target_img_dict, model, device)
        _, ft_H, ft_W, D = query_ft.shape
        new_points = []
        for pt in prev_points:
            from pips2_utils.samp import bilinear_sample2d
            import torch.nn.functional as F
            pt_np = np.array(pt).flatten()
            pt_tensor = torch.tensor([[pt_np[0]]], device=device), torch.tensor([[pt_np[1]]], device=device)
            # print('query_ft shape:', query_ft.shape)
            # print('query_ft permuted shape:', query_ft.permute(0,3,1,2).shape)
            # print('pt_tensor[0] shape:', pt_tensor[0].shape)
            # print('pt_tensor[1] shape:', pt_tensor[1].shape)
            query_ft = query_ft.to(device)
            target_ft = target_ft.to(device)
            query_point_ft = bilinear_sample2d(query_ft.permute(0,3,1,2), pt_tensor[0], pt_tensor[1])[0, :, 0]
            target_ft_sim = F.cosine_similarity(
                query_point_ft.unsqueeze(0).repeat(ft_H * ft_W, 1),
                target_ft[0, :, :, :].reshape(ft_H * ft_W, D)
            ).reshape(ft_H, ft_W)
            max_cos_pos = (target_ft_sim == torch.max(target_ft_sim)).nonzero()[0]
            pred = [max_cos_pos[1].item() * (w/512.0), max_cos_pos[0].item() * (h/input_yres)]
            new_points.append(pred)
        tracks.append(np.array(new_points))
        prev_img_dict = target_img_dict
        prev_points = np.array(new_points) * [sx, sy]
    # Visualization
    vis_images = []
    for idx, (frame_path, pts) in enumerate(zip(frame_paths, tracks)):
        img = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for pt in pts:
            pt_flat = np.array(pt).flatten()
            x, y = pt_flat[0], pt_flat[1]
            r = 4
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(255,0,0))
        img.save(os.path.join(output_dir, f"tracked_{idx:04d}.jpg"))
        vis_images.append(img)
    return f"Tracking results saved to {output_dir}", vis_images

def select_points_interface(frames_dir, checkpoint, input_yres, output_dir):
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')) + glob.glob(os.path.join(frames_dir, '*.png')))
    if not frame_paths:
        return None
    first_img = Image.open(frame_paths[0])
    return first_img

with gr.Blocks() as demo:
    gr.Markdown("## PointSt3R Interactive Tracking Demo")
    frames_dir = gr.Textbox(value="data/frames", label="Frames Directory")
    checkpoint = gr.Textbox(value="checkpoints/PointSt3R_95.pth", label="Checkpoint Path")
    input_yres = gr.Number(value=288, label="Image Height (resize)")
    output_dir = gr.Textbox(value="output_tracks", label="Output Directory")
    btn_load = gr.Button("Load First Frame")
    img = gr.Image(label="Click to select points")
    points_state = gr.State([])

    def select_handler(img_data, evt: gr.SelectData):
        points_state.value.append(evt.index)
        # Draw visual cue for selected points
        img = Image.fromarray(img_data) if isinstance(img_data, np.ndarray) else img_data
        draw = ImageDraw.Draw(img)
        for pt in points_state.value:
            x, y = pt
            r = 6
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(0,255,0), outline=(0,255,0))
        return np.array(img)

    img.select(select_handler, img, img)
    points_state = gr.State([])
    btn_track = gr.Button("Track Selected Points")
    gallery = gr.Gallery(label="Tracking Visualization")
    result = gr.Textbox(label="Status")

    def load_img(frames_dir, checkpoint, input_yres, output_dir):
        return select_points_interface(frames_dir, checkpoint, input_yres, output_dir)

    btn_load.click(load_img, [frames_dir, checkpoint, input_yres, output_dir], img)

    def track_points(frames_dir, checkpoint, input_yres, output_dir, points):
        status, vis_images = gradio_pointst3r(frames_dir, checkpoint, input_yres, output_dir, points)
        return status, vis_images

    btn_track.click(track_points, [frames_dir, checkpoint, input_yres, output_dir, img], [result, gallery])

if __name__ == "__main__":
    demo.launch()
