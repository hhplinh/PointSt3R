# PointSt3R: Point Tracking Through 3D Grounded Correspondence
This is the official implementation of PointSt3R, a variant of MASt3R fine-tuned to better handle dynamic point tracking.

## Environment
Our work was conducted on ARM achitecture, so the environment was created in the following way:

```
conda create -n pointst3r python=3.10
conda activate pointst3r
conda install pytorch-gpu -c conda-forge
conda install --no-deps conda-forge::torchvision
conda install imageio
conda install mediapy

pip install pillow-heif
pip install pyrender
pip install kapture
pip install numpy-quaternion
pip install boto3
pip install tensorflow
pip install wandb
pip install tensorboard
pip install prettytable
pip install scikit-image
pip install scikit-learn
pip install pypng
```

If you are not using ARM, it may be easier to follow the [`original MASt3R repo`](https://github.com/naver/mast3r).

## Evaluation
You can download the PointSt3R models with and without visibility, trained with 95% dynamic correspondences per batch from the following [`google drive link`](https://www.placeholder.com/PointSt3R).

Links for downloading the evaluatation datasets used in the paper are as follows:
- [`TAP-Vid-DAVIS, TAP-Vid-RGB-Stacking & RoboTAP`](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid)
- [`EgoPoints`](https://www.dropbox.com/scl/fo/tfvctluqu3cr17jr6q0td/AA6h6GlV-x6QeuupmeLejzA?rlkey=r0q12vbi6wour6qsteklivb6p&e=1&st=1e4b4dnn&dl=0)
- [`PointOdyssey Static/Dynamic Split`](https://www.placeholder.com/PO_static_dynamic)

### TAP-Vid
The evaluations for DAVIS, RoboTAP and RGB-Stacking can all be run with the following script:
```
python3 pointst3r_tapnet_eval.py --checkpoint=checkpoints/PointSt3R_95.pth --input_yres=384 --dataset_location=/your/path/that/contains/data --dataset_name=[davis, robo or rgb] --split=[0, 1, 2, 3, 4 or None]
```
Note that a split number can be defined when evaluating RoboTAP.

### EgoPoints
```
python3 pointst3r_ego_points_eval.py --checkpoint=checkpoints/PointSt3R_95.pth --input_yres=384 --eval_folder=/your/path/that/contains/ego_points
```

### PO Static/Dynamic Split
```
```

## Training
Download the baseline checkpoint [`MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`](https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth).

You will need to download the following datasets:
| Dataset | Download Link | Post-Processing |
| :---: | :---: | :---: |
| PointOdyssey | [`MonSt3R`](https://github.com/Junyi42/monst3r/blob/main/data/download_pointodyssey.sh) | N/A |
| Kubric | [`CoTracker3 Kubric dataset`](https://huggingface.co/datasets/facebook/CoTracker3_Kubric) | N/A|
| DynamicReplica | [`DynamicStereo`](https://github.com/facebookresearch/dynamic_stereo) | [`format_dr_annots.py`](./format_dr_annots.py) |

To train PointSt3R without visibility, run the following:
```bash
torchrun --nproc_per_node=4 --master_port=29350 train.py \
    --train_dataset "3_334 @ PointOdysseyDUSt3R(2, 16, [10,30,50,70,90,110,130,150,170], 2, False, True, 'linear_1_2', 0, False, False, False, split='train', ROOT='/your/path/to/pointodyssey_v2', aug_crop='auto', aug_monocular=0.005, aug_rot90='diff', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], n_corres=8192, nneg=0.5, use_soft_negs=False, dyn_ratio=0.95, transform=ColorJitter) + 3_333 @ CotrackerKubricDUSt3R(2, 16, [10,20,30,40,50,60,70,80,90], 2, False, True, 'linear_1_2', 0, False, split='train', ROOT='/your/path/to/CoTracker3_Kubric', aug_crop='auto', aug_monocular=0.005, aug_rot90='diff', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], n_corres=8192, nneg=0.5, use_soft_negs=False, dyn_ratio=0.95, transform=ColorJitter) + 3_333 @ DynamicReplicaDUSt3R(2, 16, [10,30,50,70,90,110,130,150,170], 2, False, True, 'linear_1_2', 0, split='train', ROOT='/your/path/to/dynamic_stereo/dynamic_replica_data/train', aug_crop='auto', aug_monocular=0.005, aug_rot90='diff', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], n_corres=8192, nneg=0.5, use_soft_negs=False, dyn_ratio=0.95, transform=ColorJitter)" \
    --test_dataset "1000 @ PointOdysseyDUSt3R(2, 16, [1,2,3,4,5,6,7,8,9], 2, False, True, None, 0, False, False, False, split='test', ROOT='/your/path/to/pointodyssey_v2', resolution=[(512, 384)], n_corres=1024, use_soft_negs=False, seed=777)" \
    --model "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf), freeze='encoder')" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2) + 0.075*ConfMatchingLoss(MatchingLoss(InfoNCE(mode='proper', temperature=0.05), negatives_padding=0, blocksize=8192), alpha=10.0, confmode='mean')" \
    --test_criterion "Regr3D(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) + -1.*MatchingLoss(APLoss(nq='torch', fp=torch.float16), negatives_padding=12288)" \
    --pretrained "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
    --lr 0.00005 --min_lr 1e-06 --warmup_epochs 3 --epochs 50 --batch_size 4 --accum_iter 1 \
    --save_freq 1 --keep_freq 1 --eval_freq 1 --print_freq=10 --disable_cudnn_benchmark \
    --output_dir "results/PointSt3R_95.pth"
```

To train PointSt3R with visibility, run the following:
```bash
torchrun --nproc_per_node=4 --master_port=29350 train.py \
    --train_dataset "3_334 @ PointOdysseyDUSt3R(2, 16, [10,30,50,70,90,110,130,150,170], 2, False, True, 'linear_1_2', 0, False, False, True, split='train', ROOT='/your/path/to/pointodyssey_v2', aug_crop='auto', aug_monocular=0.005, aug_rot90='diff', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], n_corres=8192, nneg=0.5, use_soft_negs=False, dyn_ratio=0.95, transform=ColorJitter) + 3_333 @ CotrackerKubricVISDUSt3R(2, 16, [10,20,30,40,50,60,70,80,90], 2, False, True, 'linear_1_2', 0, False, True, split='train', ROOT='/your/path/to/CoTracker3_Kubric', aug_crop='auto', aug_monocular=0.005, aug_rot90='diff', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], n_corres=8192, nneg=0.5, use_soft_negs=False, dyn_ratio=0.95, transform=ColorJitter) + 3_333 @ DynamicReplicaVISDUSt3R(2, 16, [10,30,50,70,90,110,130,150,170], 2, False, True, 'linear_1_2', 0, True, split='train', ROOT='/your/path/to/dynamic_stereo/dynamic_replica_data/train', aug_crop='auto', aug_monocular=0.005, aug_rot90='diff', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], n_corres=8192, nneg=0.5, use_soft_negs=False, dyn_ratio=0.95, transform=ColorJitter)" --test_dataset "1000 @ PointOdysseyDUSt3R(2, 16, [1,2,3,4,5,6,7,8,9], 2, False, True, None, 0, False, False, True, split='test', ROOT='/your/path/to/pointodyssey_v2', resolution=[(512, 384)], n_corres=1024, use_soft_negs=False, seed=777)" \
    --model "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24+vis', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf), freeze='encoder')" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2) + 0.075*ConfMatchingLoss(MatchingLoss(InfoNCE(mode='proper', temperature=0.05), negatives_padding=0, blocksize=8192), alpha=10.0, confmode='mean') + BalancedVisHeadLossV2(MaskCE(reduction='mean'))" \
    --test_criterion "Regr3D(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) + -1.*MatchingLoss(APLoss(nq='torch', fp=torch.float16), negatives_padding=12288) + BalancedVisHeadLossV2(MaskCE(reduction='mean'))" \
    --pretrained "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
    --lr 0.00005 --min_lr 1e-06 --warmup_epochs 3 --epochs 50 --batch_size 4 --accum_iter 1 \
    --save_freq 1 --keep_freq 1 --eval_freq 1 --print_freq=10 --disable_cudnn_benchmark \
    --output_dir "results/PointSt3R_95_w_vis.pth"
```
