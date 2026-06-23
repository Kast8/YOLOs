"""Small NumPy demonstrations of the main idea introduced by each generation."""

from __future__ import annotations

import numpy as np


def _iou(box, boxes):
    box, boxes = np.asarray(box, float), np.asarray(boxes, float)
    lt, rb = np.maximum(box[:2], boxes[:, :2]), np.minimum(box[2:], boxes[:, 2:])
    inter = np.prod(np.maximum(rb - lt, 0), axis=1)
    a = np.prod(box[2:] - box[:2])
    b = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    return inter / np.maximum(a + b - inter, 1e-9)


def run_demo(version: int) -> dict:
    rng = np.random.RandomState(7)
    if version == 1:
        centers = np.array([[.12, .20], [.51, .47], [.88, .91]])
        cells = (centers * 7).astype(int)
        return {"grid_size": 7, "centers": centers.tolist(), "responsible_cells": cells.tolist()}
    if version == 2:
        wh = np.array([[.1, .2], [.2, .1], [.4, .5], [.12, .18]])
        anchors = np.array([[.1, .2], [.4, .5]])
        similarity = np.minimum(wh[:, None], anchors).prod(2) / np.maximum(
            np.maximum(wh[:, None], anchors).prod(2), 1e-9)
        return {"wh": wh.tolist(), "best_anchor": similarity.argmax(1).tolist()}
    if version == 3:
        return {"input": 416, "strides": [8, 16, 32], "grid_shapes": [[52, 52], [26, 26], [13, 13]]}
    if version == 4:
        tiles = np.arange(4 * 2 * 2).reshape(4, 2, 2)
        mosaic = np.block([[tiles[0], tiles[1]], [tiles[2], tiles[3]]])
        return {"four_tiles": tiles.tolist(), "mosaic": mosaic.tolist()}
    if version == 5:
        x = np.arange(1 * 4 * 4).reshape(1, 4, 4)
        focus = np.concatenate([x[:, ::2, ::2], x[:, 1::2, ::2], x[:, ::2, 1::2], x[:, 1::2, 1::2]])
        return {"input_shape": list(x.shape), "focus_shape": list(focus.shape), "focus": focus.tolist()}
    if version == 6:
        w3, w1 = rng.randn(3, 3), rng.randn()
        fused = w3.copy(); fused[1, 1] += w1
        return {"train_branches": ["3x3", "1x1"], "deploy_kernel": np.round(fused, 3).tolist()}
    if version == 7:
        x = rng.randn(2, 3)
        paths = [x, np.maximum(x, 0), x ** 2]
        return {"path_shapes": [list(p.shape) for p in paths], "concatenated_shape": list(np.concatenate(paths, 1).shape)}
    if version == 8:
        point = np.array([.5, .5]); ltrb = np.array([.1, .2, .3, .15])
        box = [point[0]-ltrb[0], point[1]-ltrb[1], point[0]+ltrb[2], point[1]+ltrb[3]]
        return {"anchor_point": point.tolist(), "predicted_ltrb": ltrb.tolist(), "decoded_box": box}
    if version == 9:
        main_grad = np.array([.01, .2, .0]); auxiliary = np.array([.15, .1, .08])
        return {"main_gradient": main_grad.tolist(), "pgi_augmented": (main_grad + auxiliary).tolist(), "note": "補助枝は推論時に除去"}
    if version == 10:
        boxes = np.array([[0,0,1,1],[.05,.05,1.05,1.05],[2,2,3,3]])
        scores = np.array([.9,.85,.7]); keep=[]
        for i in scores.argsort()[::-1]:
            if not keep or np.all(_iou(boxes[i], boxes[keep]) < .5): keep.append(int(i))
        return {"one_to_many_requires_nms_keep": keep, "one_to_one_training_target": [0, 2], "inference": "NMS不要"}
    if version == 11:
        q = rng.randn(4, 2); weights = np.exp(q @ q.T / np.sqrt(2)); weights /= weights.sum(1, keepdims=True)
        return {"tokens": 4, "attention_shape": list(weights.shape), "row_sums": np.round(weights.sum(1), 6).tolist()}
    if version == 12:
        x = rng.randn(8, 2); areas = x.reshape(2, 4, 2)
        costs = {"global_qk_pairs": 8*8, "area_qk_pairs": 2*4*4}
        return {"tokens": list(x.shape), "areas": list(areas.shape), **costs}
    if version == 13:
        incidence = np.array([[1,0],[1,1],[1,0],[0,1]])
        node = np.arange(8).reshape(4,2); edge = incidence.T @ node / incidence.sum(0)[:,None]
        propagated = incidence @ edge / incidence.sum(1)[:,None]
        return {"incidence": incidence.tolist(), "hyperedge_features": edge.tolist(), "propagated_nodes": propagated.tolist()}
    if version == 26:
        # End-to-end one-to-one predictions can be consumed directly; no NMS loop.
        predictions = np.array([[.91, 10, 12, 40, 50], [.76, 70, 20, 95, 60]])
        return {"raw_predictions": predictions.tolist(), "postprocess": "confidence threshold only", "nms": False}
    raise ValueError(f"unknown version: {version}")
