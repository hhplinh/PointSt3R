# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# cropping/match extraction
# --------------------------------------------------------
import numpy as np
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.device import to_numpy
from dust3r.utils.geometry import inv, geotrf


def reciprocal_1d(corres_1_to_2, corres_2_to_1, ret_recip=False):
    is_reciprocal1 = (corres_2_to_1[corres_1_to_2] == np.arange(len(corres_1_to_2)))
    pos1 = is_reciprocal1.nonzero()[0]
    pos2 = corres_1_to_2[pos1]
    if ret_recip:
        return is_reciprocal1, pos1, pos2
    return pos1, pos2


def extract_point_correspondences(view1, view2, target_n_corres, rng=np.random, ret_xy=True, nneg=0, only_dynamic_points=False, only_static_points=False, dyn_ratio=None):
    view1, view2 = to_numpy((view1, view2))

    coords1 = np.column_stack(np.nonzero(view1["view_matches"]))
    values1 = view1["view_matches"][coords1[:, 0], coords1[:, 1]]

    coords2 = np.column_stack(np.nonzero(view2["view_matches"]))
    values2 = view2["view_matches"][coords2[:, 0], coords2[:, 1]]

    # Find common values and their indices in both views
    common_values, idx1, idx2 = np.intersect1d(values1, values2, return_indices=True)

    # Get the corresponding coordinates
    new_pos1_before = coords1[idx1][:, [1, 0]]
    new_pos2_before = coords2[idx2][:, [1, 0]]
    dynamic_labels = view1['dynamic_matches'][new_pos1_before[:, 1], new_pos1_before[:, 0]].astype(bool)

    new_pos1 = ravel_xy(new_pos1_before, view1['pts3d'].shape[:2])
    new_pos2 = ravel_xy(new_pos2_before, view2['pts3d'].shape[:2])

    shape1, corres1_to_2 = reproject_view(view1['pts3d'], view2)
    shape2, corres2_to_1 = reproject_view(view2['pts3d'], view1)
    is_reciprocal1, pos1, pos2 = reciprocal_1d(corres1_to_2, corres2_to_1, ret_recip=True)
    is_reciprocal2 = (corres1_to_2[corres2_to_1] == np.arange(len(corres2_to_1)))

    if target_n_corres is None:
        if ret_xy:
            new_pos1 = unravel_xy(new_pos1, shape1)
            new_pos2 = unravel_xy(new_pos2, shape2)
        return new_pos1, new_pos2

    available_negatives = min((~is_reciprocal1).sum(), (~is_reciprocal2).sum())
    target_n_positives = int(target_n_corres * (1 - nneg))
    n_positives = min(len(new_pos1), target_n_positives)
    n_negatives = min(target_n_corres - n_positives, available_negatives)

    # If not enough negatives, we will have to create negatives
    extra_negatives = 0
    if n_negatives + n_positives != target_n_corres:
        extra_negatives = target_n_corres - (n_negatives + n_positives)

    assert n_positives <= len(new_pos1)
    assert n_positives <= len(new_pos2)

    assert n_negatives <= (~is_reciprocal1).sum()
    assert n_negatives <= (~is_reciprocal2).sum()
    assert n_positives + n_negatives + extra_negatives == target_n_corres

    valid = np.ones(n_positives, dtype=bool)
    if n_positives < len(new_pos1):
        perm = rng.permutation(len(new_pos1))[:n_positives]
        new_pos1 = new_pos1[perm]
        new_pos2 = new_pos2[perm]
        dynamic_labels = dynamic_labels[perm]

    if n_negatives > 0:
        # add false correspondences if not enough
        def norm(p): return p / p.sum()
        new_pos1 = np.r_[new_pos1, rng.choice(shape1[0] * shape1[1], size=n_negatives, replace=False, p=norm(~is_reciprocal1))]
        new_pos2 = np.r_[new_pos2, rng.choice(shape2[0] * shape2[1], size=n_negatives, replace=False, p=norm(~is_reciprocal2))]
        valid = np.r_[valid, np.zeros(n_negatives, dtype=bool)]
        dynamic_labels = np.r_[dynamic_labels, np.zeros(n_negatives, dtype=bool)]

    if extra_negatives > 0:
        max_idx = len(corres1_to_2)
        extra_negs_pos1, extra_negs_pos2 = [], []
        extra_negs_found = 0
        while extra_negs_found < extra_negatives:
            extra_neg_1, extra_neg_2 = np.random.randint(0, max_idx, size=1)[0], np.random.randint(0, max_idx, size=1)[0]
            temp_pos1_idx = np.where(new_pos1 == extra_neg_1)
            if np.sum(new_pos2[temp_pos1_idx] == extra_neg_2) == 0:
                extra_negs_found += 1
                extra_negs_pos1.append(extra_neg_1)
                extra_negs_pos2.append(extra_neg_2)
        new_pos1 = np.r_[new_pos1, np.array(extra_negs_pos1)]
        new_pos2 = np.r_[new_pos2, np.array(extra_negs_pos2)]
        valid = np.r_[valid, np.zeros(extra_negatives, dtype=bool)]
        dynamic_labels = np.r_[dynamic_labels, np.zeros(extra_negatives, dtype=bool)]

    # convert (x+W*y) back to 2d (x,y) coordinates
    if ret_xy:
        new_pos1 = unravel_xy(new_pos1, shape1)
        new_pos2 = unravel_xy(new_pos2, shape2)

 
    ### Control percentage of dynamic correspondences per batch
    if dyn_ratio is not None:
        # Get the number of dynamic and static points in the block
        num_dyn = int(np.sum(dynamic_labels * valid))
        num_sta = int(np.sum(~dynamic_labels * valid))

        # Calculate how many we need to drop from the block for dynamic and static
        reduce_sta, reduce_dyn = calculate_contributions(
            num_sta, num_dyn, (1-dyn_ratio), dyn_ratio
        )

        # Find where the dynamic and static correspondences are in the list
        dyn_idx = np.where((dynamic_labels * valid) == 1)[0]
        sta_idx = np.where((~dynamic_labels * valid) == 1)[0]

        # Assume all are invalid before re-filling the valid ones from the reduced block
        new_static_labels = np.zeros_like(dynamic_labels)
        sta_rand_perm = np.random.permutation(len(sta_idx))
        sta_rand_keep = sta_rand_perm[:(num_sta - reduce_sta)]
        sta_rand_drop = sta_rand_perm[(num_sta - reduce_sta):]
        new_static_labels[sta_idx[sta_rand_keep]] = 1

        new_dynamic_labels = np.zeros_like(dynamic_labels)
        dyn_rand_perm = np.random.permutation(len(dyn_idx))
        dyn_rand_keep = dyn_rand_perm[:(num_dyn - reduce_dyn)]
        dyn_rand_drop = dyn_rand_perm[(num_dyn - reduce_dyn):]
        new_dynamic_labels[dyn_idx[dyn_rand_keep]] = 1

        # Change the correspondence values for the dropped ones so they do not represent actual values
        if reduce_sta != 0:
            sta_new_inv1, sta_new_inv2 = np.random.rand(sta_rand_drop.shape[0], 2), np.random.rand(sta_rand_drop.shape[0], 2)

            sta_new_inv1[:, 0] *= view1['pts3d'].shape[1]
            sta_new_inv2[:, 0] *= view1['pts3d'].shape[1]

            sta_new_inv1[:, 1] *= view1['pts3d'].shape[0]
            sta_new_inv2[:, 1] *= view1['pts3d'].shape[0]

            sta_new_inv1, sta_new_inv2 = sta_new_inv1.astype(int), sta_new_inv2.astype(int)

            new_pos1[sta_idx[sta_rand_drop]] = sta_new_inv1
            new_pos2[sta_idx[sta_rand_drop]] = sta_new_inv2

        if reduce_dyn != 0:
            dyn_new_inv1, dyn_new_inv2 = np.random.rand(dyn_rand_drop.shape[0], 2), np.random.rand(dyn_rand_drop.shape[0], 2)

            dyn_new_inv1[:, 0] *= view1['pts3d'].shape[1]
            dyn_new_inv2[:, 0] *= view1['pts3d'].shape[1]

            dyn_new_inv1[:, 1] *= view1['pts3d'].shape[0]
            dyn_new_inv2[:, 1] *= view1['pts3d'].shape[0]

            dyn_new_inv1, dyn_new_inv2 = dyn_new_inv1.astype(int), dyn_new_inv2.astype(int)

            new_pos1[dyn_idx[dyn_rand_drop]] = dyn_new_inv1
            new_pos2[dyn_idx[dyn_rand_drop]] = dyn_new_inv2

        # Join static and dynamic labels into one mask
        new_joint_labels = (new_dynamic_labels + new_static_labels)

        return new_pos1, new_pos2, new_joint_labels, new_dynamic_labels

    if only_static_points:
        return new_pos1, new_pos2, (valid * (~dynamic_labels)), dynamic_labels
    if only_dynamic_points:
        return new_pos1, new_pos2, (valid*dynamic_labels), dynamic_labels
    return new_pos1, new_pos2, valid, dynamic_labels



def calculate_contributions(x, y, target_x_cont, target_y_cont):
    if target_x_cont == 0.0:
        return x, 0
    elif target_y_cont == 0.0:
        return 0, y
    elif x == 0:
        return 0, 0
    elif y == 0:
        return 0, 0

    max_T_x = x / target_x_cont
    max_T_y = y / target_y_cont
    T = min(max_T_x, max_T_y)

    new_x = target_x_cont * T
    new_y = target_y_cont * T

    reduce_x = int(x - new_x)
    reduce_y = int(y - new_y)

    return reduce_x, reduce_y



def extract_correspondences_from_pts3d(view1, view2, target_n_corres, rng=np.random, ret_xy=True, nneg=0):
    view1, view2 = to_numpy((view1, view2))

    po_valid_points = view1["vis_n_valids"] * view2["vis_n_valids"]
    _, temp_h, temp_w = view1['img'].shape
    new_pos1_before = view1['trajs_2d'][po_valid_points]
    new_pos1 = ravel_xy(new_pos1_before, view1['pts3d'].shape[:2])

    new_pos2_before = view2['trajs_2d'][po_valid_points]
    new_pos2 = ravel_xy(new_pos2_before, view2['pts3d'].shape[:2])

    ###
    dynamic_labels = (view1['dynamic_points'][po_valid_points] * view2['dynamic_points'][po_valid_points])
    ###

    # project pixels from image1 --> 3d points --> image2 pixels
    shape1, corres1_to_2 = reproject_view(view1['pts3d'], view2)
    shape2, corres2_to_1 = reproject_view(view2['pts3d'], view1)

    # compute reciprocal correspondences:
    # pos1 == valid pixels (correspondences) in image1
    is_reciprocal1, pos1, pos2 = reciprocal_1d(corres1_to_2, corres2_to_1, ret_recip=True)
    is_reciprocal2 = (corres1_to_2[corres2_to_1] == np.arange(len(corres2_to_1)))

    if target_n_corres is None:
        if ret_xy:
            #pos1 = unravel_xy(pos1, shape1)
            #pos2 = unravel_xy(pos2, shape2)
            new_pos1 = unravel_xy(new_pos1, shape1)
            new_pos2 = unravel_xy(new_pos2, shape2)
        #return pos1, pos2
        return new_pos1, new_pos2

    available_negatives = min((~is_reciprocal1).sum(), (~is_reciprocal2).sum())
    target_n_positives = int(target_n_corres * (1 - nneg))
    #print(f"{available_negatives=}, {target_n_positives=}, {target_n_corres=}")
    #n_positives = min(len(pos1), target_n_positives)
    n_positives = min(len(new_pos1), target_n_positives)
    n_negatives = min(target_n_corres - n_positives, available_negatives)

    #if n_negatives + n_positives != target_n_corres:
    #    # should be really rare => when there are not enough negatives
    #    # in that case, break nneg and add a few more positives ?
    #    n_positives = target_n_corres - n_negatives
    #    #assert n_positives <= len(pos1)
    #    assert n_positives <= len(new_pos1)

    # If not enough negatives, we will have to create negatives
    extra_negatives = 0
    if n_negatives + n_positives != target_n_corres:
        extra_negatives = target_n_corres - (n_negatives + n_positives)
    #print(f"{n_positives=}, {n_negatives=}, {extra_negatives=}")

    #assert n_positives <= len(pos1)
    #assert n_positives <= len(pos2)
    assert n_positives <= len(new_pos1)
    assert n_positives <= len(new_pos2)

    assert n_negatives <= (~is_reciprocal1).sum()
    assert n_negatives <= (~is_reciprocal2).sum()
    #assert n_positives + n_negatives == target_n_corres
    assert n_positives + n_negatives + extra_negatives == target_n_corres

    valid = np.ones(n_positives, dtype=bool)
    if n_positives < len(pos1):
        # random sub-sampling of valid correspondences
        #perm = rng.permutation(len(pos1))[:n_positives]
        #pos1 = pos1[perm]
        #pos2 = pos2[perm]
        perm = rng.permutation(len(new_pos1))[:n_positives]
        new_pos1 = new_pos1[perm]
        new_pos2 = new_pos2[perm]
        dynamic_labels = dynamic_labels[perm]
    #print(f"Positives -> {new_pos1.shape=}, {new_pos2.shape=}, {valid.shape=}")

    if n_negatives > 0:
        # add false correspondences if not enough
        def norm(p): return p / p.sum()
        #pos1 = np.r_[pos1, rng.choice(shape1[0] * shape1[1], size=n_negatives, replace=False, p=norm(~is_reciprocal1))]
        #pos2 = np.r_[pos2, rng.choice(shape2[0] * shape2[1], size=n_negatives, replace=False, p=norm(~is_reciprocal2))]
        new_pos1 = np.r_[new_pos1, rng.choice(shape1[0] * shape1[1], size=n_negatives, replace=False, p=norm(~is_reciprocal1))]
        new_pos2 = np.r_[new_pos2, rng.choice(shape2[0] * shape2[1], size=n_negatives, replace=False, p=norm(~is_reciprocal2))]
        valid = np.r_[valid, np.zeros(n_negatives, dtype=bool)]
        dynamic_labels = np.r_[dynamic_labels, np.zeros(n_negatives, dtype=bool)]
    #print(f"Negatives -> {new_pos1.shape=}, {new_pos2.shape=}, {valid.shape=}")

    if extra_negatives > 0:
        max_idx = len(corres1_to_2)
        extra_negs_pos1, extra_negs_pos2 = [], []
        extra_negs_found = 0
        while extra_negs_found < extra_negatives:
            extra_neg_1, extra_neg_2 = np.random.randint(0, max_idx, size=1)[0], np.random.randint(0, max_idx, size=1)[0]
            temp_pos1_idx = np.where(new_pos1 == extra_neg_1)
            if np.sum(new_pos2[temp_pos1_idx] == extra_neg_2) == 0:
                extra_negs_found += 1
                extra_negs_pos1.append(extra_neg_1)
                extra_negs_pos2.append(extra_neg_2)
        new_pos1 = np.r_[new_pos1, np.array(extra_negs_pos1)]
        new_pos2 = np.r_[new_pos2, np.array(extra_negs_pos2)]
        valid = np.r_[valid, np.zeros(extra_negatives, dtype=bool)]
        dynamic_labels = np.r_[dynamic_labels, np.zeros(extra_negatives, dtype=bool)]
    #print(f"Extra Negatives -> {new_pos1.shape=}, {new_pos2.shape=}, {valid.shape=}")

    # convert (x+W*y) back to 2d (x,y) coordinates
    if ret_xy:
        #pos1 = unravel_xy(pos1, shape1)
        #pos2 = unravel_xy(pos2, shape2)
        new_pos1 = unravel_xy(new_pos1, shape1)
        new_pos2 = unravel_xy(new_pos2, shape2)
    #return pos1, pos2, valid
    return new_pos1, new_pos2, valid, dynamic_labels


def reproject_view(pts3d, view2):
    shape = view2['pts3d'].shape[:2]
    return reproject(pts3d, view2['camera_intrinsics'], inv(view2['camera_pose']), shape)


def reproject(pts3d, K, world2cam, shape):
    H, W, THREE = pts3d.shape
    assert THREE == 3

    # reproject in camera2 space
    with np.errstate(divide='ignore', invalid='ignore'):
        pos = geotrf(K @ world2cam[:3], pts3d, norm=1, ncol=2)

    # quantize to pixel positions
    return (H, W), ravel_xy(pos, shape)


def ravel_xy(pos, shape):
    H, W = shape
    with np.errstate(invalid='ignore'):
        qx, qy = pos.reshape(-1, 2).round().astype(np.int32).T
    quantized_pos = qx.clip(min=0, max=W - 1, out=qx) + W * qy.clip(min=0, max=H - 1, out=qy)
    return quantized_pos


def unravel_xy(pos, shape):
    # convert (x+W*y) back to 2d (x,y) coordinates
    return np.unravel_index(pos, shape)[0].base[:, ::-1].copy()


def _rotation_origin_to_pt(target):
    """ Align the origin (0,0,1) with the target point (x,y,1) in projective space.
    Method: rotate z to put target on (x'+,0,1), then rotate on Y to get (0,0,1) and un-rotate z.
    """
    from scipy.spatial.transform import Rotation
    x, y = target
    rot_z = np.arctan2(y, x)
    rot_y = np.arctan(np.linalg.norm(target))
    R = Rotation.from_euler('ZYZ', [rot_z, rot_y, -rot_z]).as_matrix()
    return R


def _dotmv(Trf, pts, ncol=None, norm=False):
    assert Trf.ndim >= 2
    ncol = ncol or pts.shape[-1]

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    if Trf.ndim >= 3:
        n = Trf.ndim - 2
        assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
        Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

        if pts.ndim > Trf.ndim:
            # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
            pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
        elif pts.ndim == 2:
            # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
            pts = pts[:, None, :]

    if pts.shape[-1] + 1 == Trf.shape[-1]:
        Trf = Trf.swapaxes(-1, -2)  # transpose Trf
        pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]

    elif pts.shape[-1] == Trf.shape[-1]:
        Trf = Trf.swapaxes(-1, -2)  # transpose Trf
        pts = pts @ Trf
    else:
        pts = Trf @ pts.T
        if pts.ndim >= 2:
            pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def crop_to_homography(K, crop, target_size=None):
    """ Given an image and its intrinsics, 
        we want to replicate a rectangular crop with an homography, 
        so that the principal point of the new 'crop' is centered.
    """
    # build intrinsics for the crop
    crop = np.round(crop)
    crop_size = crop[2:] - crop[:2]
    K2 = K.copy()  # same focal
    K2[:2, 2] = crop_size / 2  # new principal point is perfectly centered

    # find which corner is the most far-away from current principal point
    # so that the final homography does not go over the image borders
    corners = crop.reshape(-1, 2)
    corner_idx = np.abs(corners - K[:2, 2]).argmax(0)
    corner = corners[corner_idx, [0, 1]]
    # align with the corresponding corner from the target view
    corner2 = np.c_[[0, 0], crop_size][[0, 1], corner_idx]

    old_pt = _dotmv(np.linalg.inv(K), corner, norm=1)
    new_pt = _dotmv(np.linalg.inv(K2), corner2, norm=1)
    R = _rotation_origin_to_pt(old_pt) @ np.linalg.inv(_rotation_origin_to_pt(new_pt))

    if target_size is not None:
        imsize = target_size
        target_size = np.asarray(target_size)
        scaling = min(target_size / crop_size)
        K2[:2] *= scaling
        K2[:2, 2] = target_size / 2
    else:
        imsize = tuple(np.int32(crop_size).tolist())

    return imsize, K2, R, K @ R @ np.linalg.inv(K2)


def gen_random_crops(imsize, n_crops, resolution, aug_crop, rng=np.random):
    """ Generate random crops of size=resolution, 
        for an input image upscaled to (imsize + randint(0 , aug_crop))
    """
    resolution_crop = np.array(resolution) * min(np.array(imsize) / resolution)

    # (virtually) upscale the input image
    # scaling = rng.uniform(1, 1+(aug_crop+1)/min(imsize))
    scaling = np.exp(rng.uniform(0, np.log(1 + aug_crop / min(imsize))))
    imsize2 = np.int32(np.array(imsize) * scaling)

    # generate some random crops
    topleft = rng.random((n_crops, 2)) * (imsize2 - resolution_crop)
    crops = np.c_[topleft, topleft + resolution_crop]
    # print(f"{scaling=}, {topleft=}")
    # reduce the resolution to come back to original size
    crops /= scaling
    return crops


def in2d_rect(corres, crops):
    # corres = (N,2)
    # crops = (M,4)
    # output = (N, M)
    is_sup = (corres[:, None] >= crops[None, :, 0:2])
    is_inf = (corres[:, None] < crops[None, :, 2:4])
    return (is_sup & is_inf).all(axis=-1)
