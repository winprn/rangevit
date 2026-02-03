'''
Adapted from R. Strudel et al.
https://github.com/rstrudel/segmenter

MIT License
Copyright (c) 2021 Robin Strudel
Copyright (c) INRIA
'''

import torch
import torch.nn.functional as F


def resize(im, smaller_size):
    h, w = im.shape[2:]
    if h < w:
        ratio = w / h
        h_res, w_res = smaller_size, ratio * smaller_size
    else:
        ratio = h / w
        h_res, w_res = ratio * smaller_size, smaller_size
    if min(h, w) < smaller_size:
        im_res = F.interpolate(im, (int(h_res), int(w_res)), mode='bilinear')
    else:
        im_res = im
    return im_res


def sliding_window(im, flip, window_size, window_stride):
    B, C, H, W = im.shape
    ws_h, ws_w = window_size
    
    windows = {'crop': [], 'anchors': []}
    h_anchors = torch.arange(0, H, window_stride[0])
    w_anchors = torch.arange(0, W, window_stride[1])
    h_anchors = [h.item() for h in h_anchors if h < H - ws_h] + [H - ws_h]
    w_anchors = [w.item() for w in w_anchors if w < W - ws_w] + [W - ws_w]
    for ha in h_anchors:
        for wa in w_anchors:
            window = im[:, :, ha : ha + ws_h, wa : wa + ws_w]
            windows['crop'].append(window)
            windows['anchors'].append((ha, wa))
    windows['flip'] = flip
    windows['shape'] = (H, W)
    return windows


def merge_windows(windows, window_size, ori_shape):
    ws_h, ws_w = window_size
    im_windows = windows['seg_maps']
    anchors = windows['anchors']
    C = im_windows[0].shape[0]
    H, W = windows['shape']
    flip = windows['flip']

    logit = torch.zeros((C, H, W), device=im_windows.device)
    count = torch.zeros((1, H, W), device=im_windows.device)
    for window, (ha, wa) in zip(im_windows, anchors):
        logit[:, ha : ha + ws_h, wa : wa + ws_w] += window
        count[:, ha : ha + ws_h, wa : wa + ws_w] += 1
    logit = logit / count
    logit = F.interpolate(
        logit.unsqueeze(0),
        ori_shape,
        mode='bilinear')[0]
    
    if flip:
        logit = torch.flip(logit, (2,))
    return logit


def inference(
    model,
    ims,
    ims_metas,
    ori_shape,
    window_size,
    window_stride,
    batch_size,
    use_kpconv=False):

    # window_size and window_stride have to be tuples or lists with two elements
    assert len(window_size) == len(window_stride) == 2

    wsize_h, wsize_w = window_size
    smaller_size = wsize_h if wsize_h < wsize_w else wsize_w

    seg_map = None
    for im, im_metas in zip(ims, ims_metas):
        im = resize(im, smaller_size)
        flip = im_metas['flip']
        windows = sliding_window(im, flip, window_size, window_stride)
        crops = torch.stack(windows.pop('crop'))[:, 0] # shape = [n_windows, in_channels, wsize_h, wsize_w]

        with torch.no_grad():
            if use_kpconv:
                seg_maps = model.forward_2d_features(crops) # shape = [n_windows, d_decoder, wsize_h, wsize_w]
            else:
                seg_maps = model.forward(crops) # shape = [n_windows, n_classes, wsize_h, wsize_w]
        windows['seg_maps'] = seg_maps
        im_seg_map = merge_windows(windows, window_size, ori_shape) # shape = [n_classes or d_decoder, ori_shape[0], ori_shape[1]]

        if seg_map is None:
            seg_map = im_seg_map
        else:
            seg_map += im_seg_map
    seg_map /= len(ims)
    return seg_map


def inference_fusion(
    model,
    ims,
    ims_metas,
    ori_shape,
    window_size,
    window_stride,
    batch_size):
    """
    Sliding window inference for fusion models.

    Processes image crops through the encoder and decoder without point fusion,
    merges the full-resolution features, and returns them for point processing.

    Args:
        model: RangeViTFusion model
        ims: List of input images (typically just one)
        ims_metas: List of image metadata dicts (flip info)
        ori_shape: Original image shape (H, W) to return features at
        window_size: (H, W) size of sliding window crops
        window_stride: (H, W) stride between windows
        batch_size: Batch size (kept for API compatibility)

    Returns:
        feat_map: (d_decoder, H, W) merged full-resolution pixel features
    """
    # window_size and window_stride have to be tuples or lists with two elements
    assert len(window_size) == len(window_stride) == 2

    wsize_h, wsize_w = window_size
    smaller_size = wsize_h if wsize_h < wsize_w else wsize_w

    feat_map = None
    for im, im_metas in zip(ims, ims_metas):
        im = resize(im, smaller_size)
        flip = im_metas['flip']
        windows = sliding_window(im, flip, window_size, window_stride)
        crops = torch.stack(windows.pop('crop'))[:, 0]  # shape = [n_windows, in_channels, wsize_h, wsize_w]

        with torch.no_grad():
            # Process each crop to get full-resolution features
            feat_maps = model.forward_2d_features(crops)  # shape = [n_windows, d_decoder, wsize_h, wsize_w]

        windows['seg_maps'] = feat_maps
        im_feat_map = merge_windows(windows, window_size, ori_shape)  # shape = [d_decoder, ori_shape[0], ori_shape[1]]

        if feat_map is None:
            feat_map = im_feat_map
        else:
            feat_map += im_feat_map

    feat_map /= len(ims)
    return feat_map


def inference_fusion_with_points(
    model,
    image,
    point_attrs,
    point_coords,
    point_labels,
    window_size,
    window_stride,
):
    """
    Sliding window inference for fusion models with per-window point fusion.

    This function properly matches the training behavior by processing each sliding
    window with its corresponding subset of points, enabling bidirectional fusion
    at intermediate layers (blocks 4, 8, 12).

    Args:
        model: RangeViTFusion model
        image: (1, C, H, W) full range image
        point_attrs: (N, 5) point attributes [x, y, z, intensity, range]
        point_coords: (N, 2) point coordinates [y, x] in full image space
        point_labels: (N,) per-point semantic labels
        window_size: (h, w) size of sliding window crops, e.g., (64, 768)
        window_stride: (h_s, w_s) stride between windows, e.g., (64, 256)

    Returns:
        dict containing:
            'point_logits': (N, n_classes) aggregated per-point predictions
            'losses': dict with loss terms
    """
    assert len(window_size) == len(window_stride) == 2
    assert image.shape[0] == 1, "Batch size must be 1 for validation"

    device = image.device
    B, C, H, W = image.shape
    ws_h, ws_w = window_size
    stride_h, stride_w = window_stride

    # Generate window positions
    h_anchors = torch.arange(0, H, stride_h)
    w_anchors = torch.arange(0, W, stride_w)
    h_anchors = [h.item() for h in h_anchors if h < H - ws_h] + [H - ws_h]
    w_anchors = [w.item() for w in w_anchors if w < W - ws_w] + [W - ws_w]

    # Dictionary to accumulate logits per point
    # point_logits_accum[point_idx] = list of logit tensors from different windows
    point_logits_accum = {}

    # Process each window
    for offset_y in h_anchors:
        for offset_x in w_anchors:
            # 1. Find points within this window
            x_in_window = (point_coords[:, 1] >= offset_x) & (point_coords[:, 1] < offset_x + ws_w)
            y_in_window = (point_coords[:, 0] >= offset_y) & (point_coords[:, 0] < offset_y + ws_h)
            point_mask = x_in_window & y_in_window

            # Skip empty windows
            if not point_mask.any():
                continue

            # Get indices of points in this window
            point_indices = torch.where(point_mask)[0]

            # 2. Adjust coordinates to window-relative
            window_coords = point_coords[point_mask].clone()
            window_coords[:, 0] -= offset_y  # adjust y
            window_coords[:, 1] -= offset_x  # adjust x

            # 3. Add batch index (always 0 for single sample)
            batch_indices = torch.zeros((window_coords.shape[0], 1), device=device, dtype=window_coords.dtype)
            window_coords_with_batch = torch.cat([batch_indices, window_coords], dim=1)  # (N_window, 3) [batch_idx, y, x]

            # 4. Crop image to window
            image_crop = image[:, :, offset_y:offset_y + ws_h, offset_x:offset_x + ws_w]

            # 5. Forward pass through full fusion model
            window_point_attrs = point_attrs[point_mask]
            window_point_labels = point_labels[point_mask]

            with torch.no_grad():
                model_outputs = model(
                    image_crop,
                    window_point_attrs,
                    window_coords_with_batch,
                    window_point_labels
                )
                window_point_logits = model_outputs['point_logits']  # (N_window, n_classes)

            # 6. Accumulate logits for each point
            for i, global_idx in enumerate(point_indices):
                global_idx_item = global_idx.item()
                if global_idx_item not in point_logits_accum:
                    point_logits_accum[global_idx_item] = []
                point_logits_accum[global_idx_item].append(window_point_logits[i])

    # 7. Aggregate predictions by averaging logits
    N = point_coords.shape[0]
    n_classes = model.n_cls
    final_logits = torch.zeros((N, n_classes), device=device)

    for point_idx, logits_list in point_logits_accum.items():
        # Average all logits for this point
        avg_logits = torch.stack(logits_list).mean(dim=0)
        final_logits[point_idx] = avg_logits

    # 8. Compute losses on aggregated predictions
    losses = model._compute_loss(
        point_logits=final_logits,
        labels=point_labels,
        aux_outputs=[],  # No auxiliary outputs during inference
        pixel_pseudo_labels=None,
    )

    return {
        'point_logits': final_logits,
        'losses': losses,
    }
