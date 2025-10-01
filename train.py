#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training executable for MASt3R
# --------------------------------------------------------
from mast3r.model import AsymmetricMASt3R
from mast3r.losses import ConfMatchingLoss, MatchingLoss, APLoss, Regr3D, InfoNCE, Regr3D_ScaleShiftInv, DynamicMaskLoss, MaskCE, BalancedVisHeadLossV2
from mast3r.datasets import ARKitScenes, BlendedMVS, Co3d, MegaDepth, ScanNetpp, StaticThings3D, Waymo, WildRGBD, PointOdysseyDUSt3R, CotrackerKubricDUSt3R, DynamicReplicaDUSt3R, CotrackerKubricVISDUSt3R, DynamicReplicaVISDUSt3R

import mast3r.utils.path_to_dust3r  # noqa
# add mast3r classes to dust3r imports
import dust3r.training
dust3r.training.AsymmetricMASt3R = AsymmetricMASt3R
dust3r.training.Regr3D = Regr3D
dust3r.training.Regr3D_ScaleShiftInv = Regr3D_ScaleShiftInv
dust3r.training.MatchingLoss = MatchingLoss
dust3r.training.ConfMatchingLoss = ConfMatchingLoss
dust3r.training.InfoNCE = InfoNCE
dust3r.training.APLoss = APLoss
dust3r.training.DynamicMaskLoss = DynamicMaskLoss
dust3r.training.MaskCE = MaskCE
dust3r.training.BalancedVisHeadLossV2 = BalancedVisHeadLossV2

import dust3r.datasets
dust3r.datasets.ARKitScenes = ARKitScenes
dust3r.datasets.BlendedMVS = BlendedMVS
dust3r.datasets.Co3d = Co3d
dust3r.datasets.MegaDepth = MegaDepth
dust3r.datasets.ScanNetpp = ScanNetpp
dust3r.datasets.StaticThings3D = StaticThings3D
dust3r.datasets.Waymo = Waymo
dust3r.datasets.WildRGBD = WildRGBD
### PointSt3R ###
dust3r.datasets.PointOdysseyDUSt3R = PointOdysseyDUSt3R
dust3r.datasets.CotrackerKubricDUSt3R = CotrackerKubricDUSt3R
dust3r.datasets.DynamicReplicaDUSt3R = DynamicReplicaDUSt3R
dust3r.datasets.CotrackerKubricVISDUSt3R = CotrackerKubricVISDUSt3R
dust3r.datasets.DynamicReplicaVISDUSt3R = DynamicReplicaVISDUSt3R

from dust3r.training import get_args_parser as dust3r_get_args_parser  # noqa
from dust3r.training import train  # noqa


def get_args_parser():
    parser = dust3r_get_args_parser()
    # change defaults
    parser.prog = 'MASt3R training'
    parser.set_defaults(model="AsymmetricMASt3R(patch_embed_cls='ManyAR_PatchEmbed')")
    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    train(args)
