# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUST3R default transforms
# --------------------------------------------------------
import torchvision.transforms as tvf
from dust3r.utils.image import ImgNorm

# define the standard image transforms
ColorJitter = tvf.Compose([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), ImgNorm])
ColorJitterNBlur = tvf.Compose([tvf.ColorJitter(0.5,0.5,0.5,0.1), tvf.GaussianBlur(11, sigma=(0.1, 0.2)), ImgNorm])
