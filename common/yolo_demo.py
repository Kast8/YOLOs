"""YOLOv1 staged demonstration helpers for the notebook."""

from __future__ import annotations

__all__ = [
    "YOLOV1_CLASSES",
    "YOLOV1_OBJECTS",
    "build_yolov1_stages",
    "compute_yolov1_loss_demo",
    "display_yolov1_architecture",
    "display_yolov1_output_flow",
    "display_yolov1_responsible_cells",
    "display_yolov1_stages",
]

from pathlib import Path

import numpy as np


YOLOV1_CLASSES = ["person", "bicycle", "motorcycle", "car", "bus"]
YOLOV1_OBJECTS = [
    {"label": "person", "xyxy": [0.14, 0.18, 0.30, 0.78]},
    {"label": "motorcycle", "xyxy": [0.24, 0.36, 0.52, 0.82]},
    {"label": "bicycle", "xyxy": [0.57, 0.34, 0.78, 0.78]},
    {"label": "car", "xyxy": [0.64, 0.18, 0.92, 0.52]},
]


def _image_size(image_path):
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Install pillow to run the YOLOv1 image demo") from exc
    with Image.open(image_path) as image:
        return image.size


def _cell_assignment(box, grid_size):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    col = min(int(cx * grid_size), grid_size - 1)
    row = min(int(cy * grid_size), grid_size - 1)
    x_in_cell = cx * grid_size - col
    y_in_cell = cy * grid_size - row
    return cx, cy, row, col, x_in_cell, y_in_cell


def build_yolov1_stages(image_path="images/sample.jpg", grid_size=7, boxes_per_cell=2, classes=None, objects=None):
    """Build the three YOLOv1 concept stages shown in the notebook."""
    image_path = Path(image_path)
    classes = list(classes or YOLOV1_CLASSES)
    objects = list(objects or YOLOV1_OBJECTS)
    width, height = _image_size(image_path)

    output_depth = boxes_per_cell * 5 + len(classes)
    assignments = []
    target = np.zeros((grid_size, grid_size, output_depth), dtype=float)
    occupied = {}

    for obj in objects:
        x1, y1, x2, y2 = map(float, obj["xyxy"])
        cx, cy, row, col, x_cell, y_cell = _cell_assignment([x1, y1, x2, y2], grid_size)
        w = x2 - x1
        h = y2 - y1
        class_id = classes.index(obj["label"])
        conflict = (row, col) in occupied
        if not conflict:
            target[row, col, 0:5] = [x_cell, y_cell, w, h, 1.0]
            target[row, col, boxes_per_cell * 5 + class_id] = 1.0
            occupied[(row, col)] = obj["label"]
        assignments.append({
            "label": obj["label"],
            "xyxy": [x1, y1, x2, y2],
            "center": [cx, cy],
            "cell": [row, col],
            "cell_offset_xy": [x_cell, y_cell],
            "wh": [w, h],
            "conflict": conflict,
            "encoded": not conflict,
        })

    active_cells = []
    for row in range(grid_size):
        for col in range(grid_size):
            if target[row, col, 4] > 0:
                class_scores = target[row, col, boxes_per_cell * 5:]
                class_id = int(class_scores.argmax())
                x_cell, y_cell, w, h, conf = target[row, col, 0:5]
                cx = (col + x_cell) / grid_size
                cy = (row + y_cell) / grid_size
                active_cells.append({
                    "cell": [row, col],
                    "label": classes[class_id],
                    "vector": {
                        "x_in_cell": float(x_cell),
                        "y_in_cell": float(y_cell),
                        "w_image": float(w),
                        "h_image": float(h),
                        "confidence": float(conf),
                        "class_probs": {classes[i]: float(v) for i, v in enumerate(class_scores)},
                    },
                    "decoded_xyxy": [float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2)],
                })

    return {
        "stage1_whole_image": {
            "image_path": str(image_path),
            "image_size_px": [width, height],
            "grid_size": grid_size,
            "boxes_per_cell": boxes_per_cell,
            "classes": classes,
            "single_forward_output_shape": [grid_size, grid_size, output_depth],
            "meaning": "1回のforwardで全セルのbbox候補・confidence・class確率をまとめて出す",
        },
        "stage2_responsible_cells": assignments,
        "stage3_joint_regression": {
            "target_tensor_shape": list(target.shape),
            "active_cells": active_cells,
            "note": "各active cellは [x_cell, y_cell, w_image, h_image, confidence] とclass確率を同時に持つ",
        },
    }


def _to_pixels(box, width, height):
    x1, y1, x2, y2 = box
    return [x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height]


def display_yolov1_stages(stages):
    """Visualize the three YOLOv1 stages side by side."""
    try:
        from PIL import Image
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise ImportError("Install pillow and matplotlib to display the YOLOv1 staged demo") from exc

    image_path = stages["stage1_whole_image"]["image_path"]
    grid_size = stages["stage1_whole_image"]["grid_size"]
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]

    def base(ax, title):
        ax.imshow(image)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")

    base(axes[0], "1. Whole image -> one output tensor")
    axes[0].text(
        8, 20,
        f"output: {stages['stage1_whole_image']['single_forward_output_shape']}",
        color="white", fontsize=10, bbox={"facecolor": "black", "alpha": 0.65, "pad": 4},
    )
    for i, obj in enumerate(stages["stage2_responsible_cells"]):
        x, y, w, h = _to_pixels(obj["xyxy"], width, height)
        axes[0].add_patch(Rectangle((x, y), w, h, fill=False, lw=2, ec=colors[i % len(colors)]))
        axes[0].text(x, y - 4, obj["label"], color=colors[i % len(colors)], fontsize=9, weight="bold")

    base(axes[1], "2. Object center chooses responsible cell")
    for k in range(1, grid_size):
        axes[1].axvline(width * k / grid_size, color="white", lw=1, alpha=0.8)
        axes[1].axhline(height * k / grid_size, color="white", lw=1, alpha=0.8)
    for i, obj in enumerate(stages["stage2_responsible_cells"]):
        cx, cy = obj["center"]
        px, py = cx * width, cy * height
        row, col = obj["cell"]
        axes[1].scatter([px], [py], s=60, c=colors[i % len(colors)], edgecolors="white", zorder=3)
        axes[1].text(px + 4, py, f"{obj['label']} -> ({row},{col})", color="white", fontsize=8,
                     bbox={"facecolor": "black", "alpha": 0.55, "pad": 2})

    base(axes[2], "3. Cell vector decodes bbox + class together")
    for k in range(1, grid_size):
        axes[2].axvline(width * k / grid_size, color="white", lw=1, alpha=0.35)
        axes[2].axhline(height * k / grid_size, color="white", lw=1, alpha=0.35)
    for i, cell in enumerate(stages["stage3_joint_regression"]["active_cells"]):
        x, y, w, h = _to_pixels(cell["decoded_xyxy"], width, height)
        row, col = cell["cell"]
        axes[2].add_patch(Rectangle((x, y), w, h, fill=False, lw=2.5, ec=colors[i % len(colors)]))
        axes[2].text(x, y - 4, f"cell({row},{col}) {cell['label']}", color=colors[i % len(colors)], fontsize=9, weight="bold")

    fig.tight_layout()
    return fig



def display_yolov1_output_flow(stages):
    """Draw the YOLOv1 input-to-output tensor flow for the notebook."""
    try:
        from PIL import Image
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, FancyArrowPatch
    except ImportError as exc:
        raise ImportError("Install pillow and matplotlib to display the YOLOv1 flow diagram") from exc

    info = stages["stage1_whole_image"]
    image = Image.open(info["image_path"]).convert("RGB")
    grid_size = info["grid_size"]
    boxes_per_cell = info["boxes_per_cell"]
    classes = info["classes"]
    depth = info["single_forward_output_shape"][2]

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, title, body, face="#f8fafc", edge="#334155"):
        rect = Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=11, weight="bold", color="#111827")
        ax.text(x + 0.18, y + h - 0.78, body, ha="left", va="top", fontsize=9, color="#111827", linespacing=1.45)
        return rect

    def arrow(x1, y1, x2, y2, label):
        arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color="#475569")
        ax.add_patch(arr)
        ax.text((x1 + x2) / 2, y1 + 0.18, label, ha="center", va="bottom", fontsize=9, color="#334155")

    # Left: actual input image thumbnail.
    ax.imshow(image, extent=(0.35, 3.0, 1.6, 4.5), zorder=1)
    ax.add_patch(Rectangle((0.35, 1.6), 2.65, 2.9, fill=False, edgecolor="#111827", linewidth=1.8, zorder=2))
    ax.text(1.675, 4.95, "Input image", ha="center", va="top", fontsize=11, weight="bold")
    ax.text(1.675, 1.12, "No object crops\none pass over whole image", ha="center", va="top", fontsize=9)

    arrow(3.15, 3.05, 4.15, 3.05, "one forward")

    # Middle: grid assignment.
    box(
        4.35, 1.0, 3.0, 4.1,
        f"Grid: S x S = {grid_size} x {grid_size}",
        f"S={grid_size}: split width/height\n{grid_size * grid_size} cells total\n\nObject center chooses\nthe responsible cell",
        face="#eff6ff", edge="#2563eb",
    )
    gx0, gy0, gw, gh = 5.05, 1.45, 1.6, 1.6
    for i in range(grid_size + 1):
        ax.plot([gx0, gx0 + gw], [gy0 + gh * i / grid_size] * 2, color="#2563eb", linewidth=0.6)
        ax.plot([gx0 + gw * i / grid_size] * 2, [gy0, gy0 + gh], color="#2563eb", linewidth=0.6)
    for obj in stages["stage2_responsible_cells"]:
        cx, cy = obj["center"]
        ax.scatter(gx0 + cx * gw, gy0 + (1 - cy) * gh, s=36, color="#dc2626", edgecolor="white", linewidth=0.8, zorder=3)

    arrow(7.55, 3.05, 8.55, 3.05, "each cell stores")

    # Right middle: per-cell vector.
    per_cell = box(
        8.75, 1.0, 2.9, 4.1,
        f"One cell vector: B*5+C = {depth}",
        f"B={boxes_per_cell}: box candidates\n5: x,y,w,h,confidence\nC={len(classes)}: class scores\n\nIn this demo\n{boxes_per_cell}*5+{len(classes)}={depth} values",
        face="#f0fdf4", edge="#16a34a",
    )
    vx, vy = 9.15, 1.45
    segment_widths = [0.75, 0.75, 0.95]
    labels = ["box1\n5", "box2\n5", "class\n5"]
    colors = ["#bbf7d0", "#86efac", "#dcfce7"]
    cur = vx
    for sw, label, color in zip(segment_widths, labels, colors):
        ax.add_patch(Rectangle((cur, vy), sw, 0.55, facecolor=color, edgecolor="#166534", linewidth=1.1))
        ax.text(cur + sw / 2, vy + 0.275, label, ha="center", va="center", fontsize=8)
        cur += sw

    arrow(11.85, 3.05, 12.75, 3.05, "stack for all cells")

    # Right: output tensor.
    box(
        12.9, 1.0, 1.75, 4.1,
        "Output tensor",
        f"shape\n{grid_size} x {grid_size} x {depth}\n\nOne output contains\ndetection fields\nfor all cells",
        face="#fff7ed", edge="#ea580c",
    )
    # Draw tensor as stacked grids.
    tx, ty = 13.25, 1.6
    for offset in [0.28, 0.14, 0.0]:
        ax.add_patch(Rectangle((tx + offset, ty + offset), 0.72, 0.72, facecolor="none", edgecolor="#ea580c", linewidth=1.0))
    ax.text(tx + 0.55, ty + 1.15, f"{grid_size}x{grid_size}x{depth}", ha="center", fontsize=8, color="#9a3412")

    fig.tight_layout()
    return fig



def display_yolov1_responsible_cells(stages):
    """Draw how ground-truth box centers choose YOLOv1 responsible cells."""
    try:
        from PIL import Image
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, FancyArrowPatch
    except ImportError as exc:
        raise ImportError("Install pillow and matplotlib to display the YOLOv1 responsible-cell diagram") from exc

    image_path = stages["stage1_whole_image"]["image_path"]
    grid_size = stages["stage1_whole_image"]["grid_size"]
    assignments = stages["stage2_responsible_cells"]
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    colors = ["#dc2626", "#2563eb", "#16a34a", "#ea580c", "#9333ea"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def setup(ax, title):
        ax.imshow(image)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")

    def draw_grid(ax, alpha=0.85):
        for k in range(1, grid_size):
            ax.axvline(width * k / grid_size, color="white", lw=1, alpha=alpha)
            ax.axhline(height * k / grid_size, color="white", lw=1, alpha=alpha)

    def draw_arrow(ax, start, end, color):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.8, color=color))

    # 1. Ground-truth boxes and centers.
    setup(axes[0], "1. Ground-truth boxes")
    for i, obj in enumerate(assignments):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = obj["xyxy"]
        px, py = x1 * width, y1 * height
        bw, bh = (x2 - x1) * width, (y2 - y1) * height
        cx, cy = obj["center"][0] * width, obj["center"][1] * height
        axes[0].add_patch(Rectangle((px, py), bw, bh, fill=False, lw=2.4, ec=color))
        axes[0].scatter([cx], [cy], s=60, color=color, edgecolor="white", linewidth=1.0, zorder=4)
        axes[0].text(px, max(12, py - 5), obj["label"], color=color, fontsize=9, weight="bold")
    axes[0].text(
        8, height - 12,
        "Training uses annotated boxes.\nCenter = midpoint of each box.",
        color="white", fontsize=9, va="bottom",
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 4},
    )

    # 2. Centers fall into cells.
    setup(axes[1], "2. Center falls into one grid cell")
    draw_grid(axes[1])
    for i, obj in enumerate(assignments):
        color = colors[i % len(colors)]
        cx, cy = obj["center"][0] * width, obj["center"][1] * height
        row, col = obj["cell"]
        cell_x = col * width / grid_size
        cell_y = row * height / grid_size
        axes[1].add_patch(Rectangle((cell_x, cell_y), width / grid_size, height / grid_size,
                                    facecolor=color, alpha=0.25, edgecolor=color, lw=2.0))
        axes[1].scatter([cx], [cy], s=70, color=color, edgecolor="white", linewidth=1.0, zorder=4)
        axes[1].text(cx + 4, cy, f"cell({row},{col})", color="white", fontsize=8,
                     bbox={"facecolor": "black", "alpha": 0.55, "pad": 2})
    axes[1].text(
        8, height - 12,
        "The cell containing the center\nbecomes responsible for that object.",
        color="white", fontsize=9, va="bottom",
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 4},
    )

    # 3. Cell-local offset target.
    setup(axes[2], "3. Target stored as cell-local offset")
    draw_grid(axes[2], alpha=0.55)
    for i, obj in enumerate(assignments):
        color = colors[i % len(colors)]
        row, col = obj["cell"]
        x_cell, y_cell = obj["cell_offset_xy"]
        cell_x = col * width / grid_size
        cell_y = row * height / grid_size
        cell_w = width / grid_size
        cell_h = height / grid_size
        target_x = cell_x + x_cell * cell_w
        target_y = cell_y + y_cell * cell_h
        axes[2].add_patch(Rectangle((cell_x, cell_y), cell_w, cell_h,
                                    facecolor=color, alpha=0.18, edgecolor=color, lw=2.0))
        axes[2].scatter([cell_x], [cell_y], s=38, marker="s", color="white", edgecolor=color, linewidth=1.2, zorder=4)
        axes[2].scatter([target_x], [target_y], s=70, color=color, edgecolor="white", linewidth=1.0, zorder=5)
        draw_arrow(axes[2], (cell_x, cell_y), (target_x, target_y), color)
        axes[2].text(target_x + 4, target_y, f"x={x_cell:.2f}\ny={y_cell:.2f}", color="white", fontsize=8,
                     bbox={"facecolor": "black", "alpha": 0.55, "pad": 2})
    axes[2].text(
        8, height - 12,
        "The model learns x_cell,y_cell:\noffset from the cell's top-left corner.",
        color="white", fontsize=9, va="bottom",
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 4},
    )

    fig.tight_layout()
    return fig


def compute_yolov1_loss_demo(stages, lambda_coord=5.0, lambda_noobj=0.5):
    """Compute a small YOLOv1-style loss breakdown from hand-made predictions."""
    classes = stages["stage1_whole_image"]["classes"]
    active_cells = stages["stage3_joint_regression"]["active_cells"]
    rows = []
    totals = {"coord_xy": 0.0, "coord_wh": 0.0, "object_conf": 0.0, "class": 0.0}

    for i, cell in enumerate(active_cells):
        target = cell["vector"]
        label = cell["label"]
        # Small deterministic errors so learners can see which term reacts to what.
        pred = {
            "x_in_cell": min(max(target["x_in_cell"] + 0.08 - 0.03 * (i % 2), 0.0), 1.0),
            "y_in_cell": min(max(target["y_in_cell"] - 0.06 + 0.02 * (i % 2), 0.0), 1.0),
            "w_image": max(target["w_image"] * (1.0 + 0.12 - 0.04 * (i % 2)), 1e-6),
            "h_image": max(target["h_image"] * (1.0 - 0.10 + 0.05 * (i % 2)), 1e-6),
            "confidence": max(min(0.78 + 0.04 * (i % 2), 1.0), 0.0),
            "class_probs": {name: 0.05 for name in classes},
        }
        pred["class_probs"][label] = 0.78

        coord_xy = (pred["x_in_cell"] - target["x_in_cell"]) ** 2 + (pred["y_in_cell"] - target["y_in_cell"]) ** 2
        coord_wh = (pred["w_image"] ** 0.5 - target["w_image"] ** 0.5) ** 2 + (pred["h_image"] ** 0.5 - target["h_image"] ** 0.5) ** 2
        object_conf = (pred["confidence"] - 1.0) ** 2
        class_loss = sum((pred["class_probs"][name] - target["class_probs"][name]) ** 2 for name in classes)

        weighted = {
            "coord_xy": lambda_coord * coord_xy,
            "coord_wh": lambda_coord * coord_wh,
            "object_conf": object_conf,
            "class": class_loss,
        }
        for key, value in weighted.items():
            totals[key] += value

        rows.append({
            "cell": cell["cell"],
            "label": label,
            "target": target,
            "prediction": pred,
            "weighted_loss": weighted,
        })

    # One no-object cell to show the confidence penalty for false positives.
    noobj_conf_pred = 0.30
    noobj_loss = lambda_noobj * (noobj_conf_pred - 0.0) ** 2
    totals["noobject_conf"] = noobj_loss
    totals["total"] = sum(totals.values())

    return {
        "lambda_coord": lambda_coord,
        "lambda_noobj": lambda_noobj,
        "rows": rows,
        "noobject_example": {
            "target_confidence": 0.0,
            "predicted_confidence": noobj_conf_pred,
            "weighted_loss": noobj_loss,
        },
        "totals": totals,
    }


def display_yolov1_architecture(stages):
    """Draw a Fig.3-style YOLOv1 architecture diagram."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, FancyArrowPatch
    except ImportError as exc:
        raise ImportError("Install matplotlib to display the YOLOv1 architecture diagram") from exc

    info = stages["stage1_whole_image"]
    grid_size = info["grid_size"]
    depth = info["single_forward_output_shape"][2]
    boxes_per_cell = info["boxes_per_cell"]
    class_count = len(info["classes"])

    layers = [
        {
            "title": "Input",
            "body": "448 x 448 x 3\nwhole image",
            "color": "#e5e7eb",
            "height": 1.35,
        },
        {
            "title": "Conv stem",
            "body": "7x7 conv, 64, s=2\nmaxpool 2x2, s=2\n-> 112 x 112",
            "color": "#dbeafe",
            "height": 1.75,
        },
        {
            "title": "Conv block",
            "body": "3x3 conv, 192\nmaxpool\n-> 56 x 56",
            "color": "#bfdbfe",
            "height": 1.95,
        },
        {
            "title": "1x1/3x3 block",
            "body": "1x1 reduce\n3x3 conv\nchannels up to 512\nmaxpool -> 28 x 28",
            "color": "#bbf7d0",
            "height": 2.25,
        },
        {
            "title": "Repeated convs",
            "body": "4x {1x1 256, 3x3 512}\n1x1 512, 3x3 1024\nmaxpool -> 14 x 14",
            "color": "#86efac",
            "height": 2.75,
        },
        {
            "title": "Deep convs",
            "body": "2x {1x1 512, 3x3 1024}\n3x3 1024\n3x3 1024, s=2\n-> 7 x 7",
            "color": "#fde68a",
            "height": 3.05,
        },
        {
            "title": "Final convs",
            "body": "3x3 1024\n3x3 1024\nfeature map: 7 x 7",
            "color": "#fcd34d",
            "height": 2.65,
        },
        {
            "title": "FC layers",
            "body": "2 fully connected layers\nFC 4096 + Dropout\nFinal FC output",
            "color": "#fed7aa",
            "height": 2.1,
        },
        {
            "title": "Detection tensor",
            "body": f"{grid_size} x {grid_size} x {depth}\nB={boxes_per_cell}, C={class_count}\nper cell: B*5+C",
            "color": "#fecaca",
            "height": 1.85,
        },
    ]

    fig, ax = plt.subplots(figsize=(18, 6.2))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    x = 0.35
    y_base = 1.55
    width = 1.65
    gap = 0.28

    for i, layer in enumerate(layers):
        h = layer["height"]
        y = y_base + (3.1 - h) / 2
        ax.add_patch(Rectangle((x, y), width, h, facecolor=layer["color"], edgecolor="#111827", linewidth=1.5))
        ax.text(x + width / 2, y + h - 0.18, layer["title"], ha="center", va="top", fontsize=9.5, weight="bold")
        ax.text(x + 0.08, y + h - 0.55, layer["body"], ha="left", va="top", fontsize=7.7, linespacing=1.25)
        if i < len(layers) - 1:
            ax.add_patch(FancyArrowPatch((x + width + 0.03, 3.1), (x + width + gap - 0.04, 3.1),
                                         arrowstyle="-|>", mutation_scale=12, linewidth=1.35, color="#475569"))
        x += width + gap

    ax.text(
        9.0, 5.78,
        "YOLOv1 architecture: 24 conv layers + 2 FC layers, one global CNN over the whole image",
        ha="center", va="top", fontsize=12, weight="bold", color="#111827",
    )
    ax.text(
        9.0, 0.7,
        "Training strategy: classification pretraining on ImageNet, then detection fine-tuning with bbox/conf/class loss.",
        ha="center", va="center", fontsize=10, color="#111827",
        bbox={"facecolor": "#f3f4f6", "edgecolor": "#9ca3af", "pad": 5},
    )
    ax.text(
        9.0, 0.38,
        "Activation: Leaky ReLU after layers except the final output; regularization: Dropout in FC layers.",
        ha="center", va="center", fontsize=9, color="#374151",
    )

    # Small per-cell output legend.
    lx, ly = 12.3, 0.18
    segment_w = [0.65, 0.65, 0.9]
    labels = ["box1\n5", "box2\n5", "classes\nC"]
    colors = ["#fee2e2", "#fecaca", "#fca5a5"]
    cur = lx
    ax.text(lx - 0.15, ly + 0.34, "one cell vector:", ha="right", va="center", fontsize=8)
    for sw, label, color in zip(segment_w, labels, colors):
        ax.add_patch(Rectangle((cur, ly), sw, 0.62, facecolor=color, edgecolor="#991b1b", linewidth=1.0))
        ax.text(cur + sw / 2, ly + 0.31, label, ha="center", va="center", fontsize=7.5)
        cur += sw

    fig.tight_layout()
    return fig

