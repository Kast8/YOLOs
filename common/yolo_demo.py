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
    "display_yolo_architecture",
    "display_yolo_technology_demo",
    "display_yolo_element_demo",
]

from pathlib import Path

import numpy as np


def _display_figure(fig):
    try:
        from IPython.display import display
        display(fig)
    except Exception:
        try:
            fig.show()
        except Exception:
            pass
    finally:
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass
    return None


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
    return _display_figure(fig)



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
    return _display_figure(fig)



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
    return _display_figure(fig)


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
    return _display_figure(fig)




def _version_title(version):
    if version == 26:
        return "YOLO26"
    if version == 11:
        return "YOLO11"
    return f"YOLOv{version}"


def _arch_spec(version):
    specs = {
        2: {
            "blocks": ["Input\n416 x 416", "Darknet-19\n13x13 deep feat", "Passthrough\n26x26 reorg + concat", "Anchor head\ncell x anchor", "NMS\nfinal boxes"],
            "notes": ["YOLOv1の直接bbox回帰から、anchorごとのoffset予測へ移行", "分類データと検出データを階層ラベルで共同学習するYOLO9000も提案"],
        },
        3: {
            "blocks": ["Input\n416 x 416", "Darknet-53\nresidual backbone", "FPN-like\nfeature fusion", "3 scale heads\n52/26/13 grids", "NMS\nper class boxes"],
            "notes": ["深い特徴だけでなく、浅い高解像度特徴も検出に使う", "softmaxではなく独立ロジスティック分類でmulti-label性を扱いやすくした"],
        },
        4: {
            "blocks": ["Input\nMosaic/SAT", "CSPDarknet53\nMish", "SPP + PAN\nfeature aggregation", "YOLO heads\nanchor based", "NMS\nfinal boxes"],
            "notes": ["Backbone, neck, head, training tricksを体系的に組み合わせた", "BoFは推論コストを増やさず学習を改善し、BoSは少しコストを払い精度を上げる"],
        },
        5: {
            "blocks": ["Input\nPyTorch pipeline", "Focus/CSP\nbackbone", "PAN-FPN\nneck", "Anchor head\nAutoAnchor", "Export/API\nONNX/TensorRT"],
            "notes": ["論文より実装体系としての影響が大きい", "学習、評価、推論、exportを同じCLI/APIで扱う実務上の流れを普及させた"],
        },
        6: {
            "blocks": ["Input", "Rep backbone\ntrain-time branches", "Efficient neck", "Decoupled head\ncls/reg separated", "Deploy model\nfused convs"],
            "notes": ["学習時は複数枝で最適化しやすく、推論時は単一畳み込みへ融合", "分類と回帰のheadを分け、各タスクに適した特徴を使いやすくした"],
        },
        7: {
            "blocks": ["Input", "E-ELAN\nmulti-path expand", "Concat/merge\ngradient diversity", "Scaled heads", "Planned re-param\ndeploy"],
            "notes": ["特徴変換の経路を増やし、結合して表現力を上げる", "モデルサイズを変えるときも計算量と精度の関係を崩しにくい設計を重視"],
        },
        8: {
            "blocks": ["Input", "C2f backbone", "PAN-FPN neck", "Anchor-free\ndecoupled head", "Task API\ndetect/seg/pose"],
            "notes": ["anchorの幅高さ候補ではなく、特徴点からbbox境界までの距離を予測", "検出以外のタスクも同じ実装体系に統合"],
        },
        9: {
            "blocks": ["Input", "GELAN backbone", "Main branch\ninference path", "PGI auxiliary\ntraining only", "YOLO head"],
            "notes": ["推論経路を重くせず、学習時だけ補助的な勾配情報を追加", "深い/軽量なモデルで情報が失われる問題を学習設計から扱う"],
        },
        10: {
            "blocks": ["Input", "Efficient backbone", "Dual heads\none-to-many + one-to-one", "Consistent assignment", "NMS-free\nend-to-end output"],
            "notes": ["学習時は豊富な割当で信号を増やし、推論時は重複の少ないone-to-one出力を使う", "後処理NMSの遅延と実装差を設計対象にした"],
        },
        11: {
            "blocks": ["Input", "C3k2 backbone", "C2PSA attention", "Multi-task heads", "Unified API\ndetect/seg/pose/OBB"],
            "notes": ["CNNの局所特徴に加えて、軽量attentionで位置間の関係を使う", "Ultralytics系列として複数タスクを同じ操作体系で扱う"],
        },
        12: {
            "blocks": ["Input", "R-ELAN backbone", "Area Attention\nregional tokens", "YOLO head", "Real-time output"],
            "notes": ["global attentionの計算量を避け、領域内attentionで文脈と速度を両立", "attentionをリアルタイム検出の制約に合わせて再設計"],
        },
        13: {
            "blocks": ["Input", "Backbone/neck", "HyperACE\nhypergraph correlation", "FullPAD + DS assignment", "Detection output"],
            "notes": ["2点間ではなく複数特徴をまとめるhyperedgeで高次相関を見る", "遮蔽や密集など、離れた特徴のまとまりが重要なケースを意識"],
        },
        26: {
            "blocks": ["Input", "Efficient backbone", "NMS-free head", "No DFL\nlean regression", "Deployment\nCPU/edge/export"],
            "notes": ["番号順のYOLOv14ではなくUltralyticsの製品系列として扱う", "後処理を単純化し、配備時の遅延と実装複雑性を下げる方向"],
        },
    }
    return specs[version]


def _arch_notes_en(version):
    notes = {
        2: ["Anchor offsets replace direct box regression.", "High-resolution classification pretraining matters for detection fine-tuning."],
        3: ["Three detection scales handle small, medium, and large objects.", "Upsampled semantic features are fused with higher-resolution features."],
        4: ["Backbone, neck, loss, activation, and augmentation are optimized together.", "Training-time tricks become central to detector quality."],
        5: ["PyTorch workflow makes training, inference, and export practical.", "CSP + PAN-FPN + AutoAnchor form a deployable baseline."],
        6: ["Train-time branches are fused into deploy-time convolutions.", "Classification and regression heads are separated."],
        7: ["E-ELAN preserves feature diversity through multi-path aggregation.", "Scaling and re-parameterization are planned with the architecture."],
        8: ["Anchor-free decoding predicts distances from feature points.", "One API covers detection and adjacent vision tasks."],
        9: ["PGI adds training-only gradient paths.", "Auxiliary branches are removed for inference."],
        10: ["Dual assignments keep rich training signals and direct inference outputs.", "NMS latency becomes part of the architecture problem."],
        11: ["Lightweight attention augments CNN feature extraction.", "The model family is optimized for multiple task heads."],
        12: ["Area Attention limits token interactions to reduce cost.", "Attention is adapted to real-time detection constraints."],
        13: ["Hyperedges aggregate higher-order feature relations.", "Assignment and padding details support dense detection quality."],
        26: ["Deployment cost is treated as a first-class design target.", "NMS-free output and lean regression simplify inference."],
    }
    return notes[version]


def display_yolo_architecture(version):
    """Draw a compact architecture diagram for a YOLO generation after v1."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, Rectangle
    except ImportError as exc:
        raise ImportError("Install matplotlib to display YOLO architecture diagrams") from exc

    spec = _arch_spec(version)
    blocks = spec["blocks"]
    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    colors = ["#dbeafe", "#dcfce7", "#fef3c7", "#fee2e2", "#f3e8ff"]
    edges = ["#2563eb", "#16a34a", "#d97706", "#dc2626", "#9333ea"]
    w = 2.25
    gap = 0.45
    x0 = 0.45
    y = 2.05
    h = 1.45
    for i, label in enumerate(blocks):
        x = x0 + i * (w + gap)
        ax.add_patch(Rectangle((x, y), w, h, facecolor=colors[i % len(colors)], edgecolor=edges[i % len(edges)], linewidth=1.8))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10, weight="bold", linespacing=1.25)
        if i < len(blocks) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.05, y + h / 2), (x + w + gap - 0.05, y + h / 2), arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color="#475569"))
    ax.text(7.5, 4.45, f"{_version_title(version)} architecture overview", ha="center", va="top", fontsize=13, weight="bold")
    for i, note in enumerate(_arch_notes_en(version)):
        ax.text(0.55, 1.2 - i * 0.38, f"- {note}", ha="left", va="top", fontsize=9.5, color="#111827")
    fig.tight_layout()
    return _display_figure(fig)



def _draw_yolov2_overview_demo(ax):
    ax.set_title("YOLOv2: direct boxes -> cell x anchor predictions", fontsize=11, weight="bold")
    ax.axis("off")

    def block(x, y, w, h, label, face, edge):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.6))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.5, weight="bold", linespacing=1.25)

    # v1 side
    block(0.35, 2.55, 1.7, 0.7, "YOLOv1\ncell", "#dbeafe", "#2563eb")
    block(2.55, 2.55, 2.2, 0.7, "direct box\n(x, y, w, h)", "#dbeafe", "#2563eb")
    ax.annotate("", xy=(2.55, 2.9), xytext=(2.05, 2.9), arrowprops={"arrowstyle":"-|>", "color":"#475569"})

    # v2 side
    block(0.35, 1.2, 1.7, 0.7, "YOLOv2\ncell", "#dcfce7", "#16a34a")
    block(2.55, 1.2, 1.7, 0.7, "anchor 1\nprior shape", "#fef3c7", "#d97706")
    block(4.65, 1.2, 1.7, 0.7, "anchor 2\nprior shape", "#fef3c7", "#d97706")
    block(6.75, 1.2, 2.45, 0.7, "predict offsets\ntx, ty, tw, th", "#fee2e2", "#dc2626")
    block(9.75, 1.2, 2.55, 0.7, "box + objectness\n+ class scores", "#f3e8ff", "#9333ea")
    for x1, x2 in [(2.05,2.55),(4.25,4.65),(6.35,6.75),(9.2,9.75)]:
        ax.annotate("", xy=(x2, 1.55), xytext=(x1, 1.55), arrowprops={"arrowstyle":"-|>", "color":"#475569"})

    # Supporting changes
    block(2.7, 0.15, 2.35, 0.62, "k-means anchors\nfrom training bbox (w,h)", "#fff7ed", "#ea580c")
    block(5.55, 0.15, 2.35, 0.62, "passthrough\nmid feature concat", "#e0f2fe", "#0284c7")
    block(8.4, 0.15, 2.35, 0.62, "WordTree\njoint class learning", "#f0fdf4", "#16a34a")

    ax.text(6.35, 3.55, "main change: each grid cell predicts offsets from multiple learned anchor shapes", ha="center", fontsize=9.5)
    ax.text(6.35, 0.95, "other v2 improvements support anchors, fine features, and many-class learning", ha="center", fontsize=9)
    ax.set_xlim(0, 12.8)
    ax.set_ylim(-0.05, 3.85)

def _draw_anchor_demo(ax):
    ax.set_title("Anchor boxes: object shape -> best prior", fontsize=11, weight="bold")
    ax.set_xlim(0, 0.55)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("box width")
    ax.set_ylabel("box height")
    ax.grid(True, alpha=0.25)
    wh = np.array([[.1, .2], [.2, .1], [.4, .5], [.12, .18]])
    anchors = np.array([[.1, .2], [.4, .5]])
    colors = ["#2563eb", "#dc2626"]
    for i, a in enumerate(anchors):
        ax.scatter([a[0]], [a[1]], s=180, marker="s", color=colors[i], label=f"anchor {i}")
    for w, h in wh:
        best = np.argmin(np.abs(anchors[:, 0] * anchors[:, 1] - w * h))
        ax.scatter([w], [h], s=70, color=colors[best], edgecolor="white", linewidth=1.2)
        ax.plot([w, anchors[best, 0]], [h, anchors[best, 1]], color=colors[best], alpha=0.5)
    ax.legend(loc="upper left")


def _draw_multiscale_demo(ax):
    ax.set_title("Multi-scale prediction", fontsize=11, weight="bold")
    ax.axis("off")
    sizes = [(52, "small objects"), (26, "medium objects"), (13, "large objects")]
    x = 0.4
    for grid, label in sizes:
        scale = grid / 52
        side = 1.8 * scale + 0.65
        ax.add_patch(Rectangle((x, 1.2), side, side, fill=False, edgecolor="#2563eb", linewidth=1.8))
        steps = min(grid, 13)
        for k in range(1, steps):
            ax.plot([x, x + side], [1.2 + side * k / steps] * 2, color="#93c5fd", linewidth=0.4)
            ax.plot([x + side * k / steps] * 2, [1.2, 1.2 + side], color="#93c5fd", linewidth=0.4)
        ax.text(x + side / 2, 0.85, f"{grid} x {grid}\n{label}", ha="center", va="top", fontsize=9)
        x += 3.25
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 4.2)



def _draw_yolov3_fusion_demo(ax):
    ax.set_title("YOLOv3 FPN-like fusion: upsample + concat", fontsize=11, weight="bold")
    ax.axis("off")

    def block(x, y, w, h, label, face, edge):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.6))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.5, weight="bold", linespacing=1.25)

    block(0.4, 2.6, 2.1, 0.7, "deep feature\n13 x 13", "#dbeafe", "#2563eb")
    block(3.0, 2.6, 1.65, 0.7, "detect\nlarge", "#fef3c7", "#d97706")
    block(0.4, 1.45, 2.1, 0.7, "upsample x2\n26 x 26", "#dcfce7", "#16a34a")
    block(3.0, 1.45, 2.1, 0.7, "concat with\nmid feature", "#f3e8ff", "#9333ea")
    block(5.65, 1.45, 1.65, 0.7, "detect\nmedium", "#fef3c7", "#d97706")
    block(0.4, 0.3, 2.1, 0.7, "upsample x2\n52 x 52", "#dcfce7", "#16a34a")
    block(3.0, 0.3, 2.1, 0.7, "concat with\nshallow feature", "#f3e8ff", "#9333ea")
    block(5.65, 0.3, 1.65, 0.7, "detect\nsmall", "#fef3c7", "#d97706")

    arrows = [
        ((2.5, 2.95), (3.0, 2.95)),
        ((1.45, 2.6), (1.45, 2.15)),
        ((2.5, 1.8), (3.0, 1.8)),
        ((5.1, 1.8), (5.65, 1.8)),
        ((1.45, 1.45), (1.45, 1.0)),
        ((2.5, 0.65), (3.0, 0.65)),
        ((5.1, 0.65), (5.65, 0.65)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle":"-|>", "color":"#475569", "linewidth":1.4})
    ax.text(4.05, 2.35, "semantic deep features are reused at finer grids", ha="center", fontsize=9)
    ax.set_xlim(0, 7.8)
    ax.set_ylim(0.0, 3.55)


def _draw_yolov3_overview_demo(ax):
    ax.set_title("YOLOv3: one scale -> three detection scales", fontsize=11, weight="bold")
    ax.axis("off")

    def block(x, y, w, h, label, face, edge):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.6))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.5, weight="bold", linespacing=1.2)

    block(0.4, 2.55, 1.7, 0.65, "YOLOv2\n13 x 13", "#dbeafe", "#2563eb")
    block(2.8, 2.55, 2.3, 0.65, "single detection\nfeature map", "#dbeafe", "#2563eb")
    ax.annotate("", xy=(2.8, 2.875), xytext=(2.1, 2.875), arrowprops={"arrowstyle":"-|>", "color":"#475569"})

    block(0.4, 1.25, 1.8, 0.65, "YOLOv3\nDarknet-53", "#dcfce7", "#16a34a")
    block(2.85, 1.95, 1.8, 0.58, "13 x 13\nlarge objects", "#fef3c7", "#d97706")
    block(2.85, 1.15, 1.8, 0.58, "26 x 26\nmedium objects", "#fef3c7", "#d97706")
    block(2.85, 0.35, 1.8, 0.58, "52 x 52\nsmall objects", "#fef3c7", "#d97706")
    block(5.4, 1.15, 2.4, 0.7, "per scale:\n3 anchors x (box + obj + class)", "#f3e8ff", "#9333ea")
    for y in [2.24, 1.44, 0.64]:
        ax.annotate("", xy=(2.85, y), xytext=(2.2, 1.575), arrowprops={"arrowstyle":"-|>", "color":"#475569"})
        ax.annotate("", xy=(5.4, 1.5), xytext=(4.65, y), arrowprops={"arrowstyle":"-|>", "color":"#475569"})
    ax.text(4.2, 3.35, "main change: predictions are made at multiple feature-map resolutions", ha="center", fontsize=9.5)
    ax.set_xlim(0, 8.2)
    ax.set_ylim(0.05, 3.65)

def _draw_mosaic_demo(ax):
    ax.set_title("Mosaic augmentation", fontsize=11, weight="bold")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.axis("off")
    colors = ["#dbeafe", "#dcfce7", "#fef3c7", "#fee2e2"]
    labels = ["image A", "image B", "image C", "image D"]
    positions = [(0, 1), (1, 1), (0, 0), (1, 0)]
    for (x, y), c, label in zip(positions, colors, labels):
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=c, edgecolor="#334155", linewidth=1.5))
        ax.text(x + 0.5, y + 0.5, label, ha="center", va="center", fontsize=10, weight="bold")
    ax.text(1, -0.12, "one training input mixes contexts and scales", ha="center", va="top", fontsize=9)


def _draw_yolov4_overview_demo(ax):
    ax.set_title("YOLOv4: YOLOv3 detector + stronger backbone, neck, loss, and training", fontsize=11, weight="bold")
    ax.axis("off")

    def block(x, y, w, h, label, face, edge, fontsize=7.6):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.35))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, weight="bold", linespacing=1.12)

    def arrow(x1, y1, x2, y2, color="#475569"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.25})

    blue = ("#dbeafe", "#2563eb")
    green = ("#dcfce7", "#16a34a")
    yellow = ("#fef3c7", "#d97706")
    red = ("#fee2e2", "#dc2626")
    purple = ("#f3e8ff", "#9333ea")
    gray = ("#f8fafc", "#64748b")

    ax.text(0.45, 3.42, "YOLOv3 baseline", fontsize=9.2, weight="bold", color="#334155")
    block(0.45, 2.58, 1.55, 0.56, "Darknet-53\nbackbone", *blue)
    block(2.35, 2.58, 1.55, 0.56, "FPN-like\n3-scale fusion", *green)
    block(4.25, 2.58, 1.55, 0.56, "anchor heads\n13/26/52", *yellow)
    block(6.15, 2.58, 1.55, 0.56, "IoU / coord\nloss + NMS", *gray)
    for x1, x2 in [(2.0, 2.35), (3.9, 4.25), (5.8, 6.15)]:
        arrow(x1, 2.86, x2, 2.86)

    ax.text(0.45, 1.94, "YOLOv4 changes from v3", fontsize=9.2, weight="bold", color="#334155")
    block(0.45, 1.08, 1.55, 0.64, "CSPDarknet53\nless duplicate\ngradient flow", *blue, fontsize=7.0)
    block(2.35, 1.08, 1.55, 0.64, "SPP + PAN\nwider context\nbidirectional neck", *green, fontsize=7.0)
    block(4.25, 1.08, 1.55, 0.64, "same YOLO\nanchor heads\n3 scales", *yellow, fontsize=7.0)
    block(6.15, 1.08, 1.55, 0.64, "CIoU loss\nMosaic / BoF\nBoS", *red, fontsize=7.0)
    for x1, x2 in [(2.0, 2.35), (3.9, 4.25), (5.8, 6.15)]:
        arrow(x1, 1.4, x2, 1.4)

    for x in [1.225, 3.125, 5.025, 6.925]:
        arrow(x, 2.58, x, 1.72, color="#9333ea")
    ax.text(1.22, 0.55, "backbone\nreplaced", ha="center", va="center", fontsize=7.2, color="#1e40af")
    ax.text(3.12, 0.55, "neck\nstrengthened", ha="center", va="center", fontsize=7.2, color="#166534")
    ax.text(5.02, 0.55, "head mostly\nkept", ha="center", va="center", fontsize=7.2, color="#92400e")
    ax.text(6.92, 0.55, "training/loss\nupgraded", ha="center", va="center", fontsize=7.2, color="#991b1b")

    ax.text(4.05, 3.82, "The main v4 story is not a new output format; it is a stronger full detection recipe.",
            ha="center", va="center", fontsize=8.3, color="#334155")
    ax.set_xlim(0.1, 8.05)
    ax.set_ylim(0.15, 4.05)


def _draw_focus_demo(ax):
    ax.set_title("Focus: spatial pixels -> channels", fontsize=11, weight="bold")
    ax.axis("off")
    vals = np.arange(16).reshape(4, 4)
    for r in range(4):
        for c in range(4):
            fc = "#dbeafe" if (r + c) % 2 == 0 else "#dcfce7"
            ax.add_patch(Rectangle((c, 4 - r), 0.9, 0.9, facecolor=fc, edgecolor="#334155"))
            ax.text(c + 0.45, 4.45 - r, str(vals[r, c]), ha="center", va="center", fontsize=8)
    ax.text(1.8, 5.2, "4 x 4 x C", ha="center", fontsize=9)
    ax.annotate("slice + concat", xy=(5.2, 3.0), xytext=(4.25, 3.0), arrowprops={"arrowstyle": "-|>"}, ha="center")
    names = ["even/even", "odd/even", "even/odd", "odd/odd"]
    for i, name in enumerate(names):
        ax.add_patch(Rectangle((6.0, 4.4 - i * 0.65), 1.8, 0.48, facecolor="#fef3c7", edgecolor="#d97706"))
        ax.text(6.9, 4.64 - i * 0.65, name, ha="center", va="center", fontsize=8)
    ax.text(6.9, 1.35, "2 x 2 x 4C", ha="center", fontsize=9)
    ax.set_xlim(-0.2, 8.2)
    ax.set_ylim(0.8, 5.5)


def _draw_reparam_demo(ax):
    ax.set_title("Re-parameterization: train-time branches -> one deploy 3x3 Conv", fontsize=11, weight="bold")
    ax.axis("off")

    def block(x, y, w, h, label, face, edge, fontsize=7.6):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.35))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, weight="bold", linespacing=1.1)

    def arrow(x1, y1, x2, y2, color="#475569"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.25})

    def kernel_grid(x, y, values, title, edge="#334155", fill="#f8fafc"):
        cell = 0.18
        ax.text(x + 0.27, y + 0.66, title, ha="center", va="bottom", fontsize=6.8, weight="bold", color=edge)
        for r in range(3):
            for c in range(3):
                v = values[r][c]
                fc = fill if v else "#ffffff"
                ax.add_patch(Rectangle((x + c * cell, y + (2 - r) * cell), cell, cell,
                                       facecolor=fc, edgecolor=edge, linewidth=0.8))
                if v:
                    ax.text(x + c * cell + cell / 2, y + (2 - r) * cell + cell / 2, v,
                            ha="center", va="center", fontsize=5.8, weight="bold", color=edge)

    blue = ("#dbeafe", "#2563eb")
    yellow = ("#fef3c7", "#d97706")
    green = ("#dcfce7", "#16a34a")
    purple = ("#f3e8ff", "#9333ea")

    # Training graph branches.
    block(0.35, 2.85, 1.35, 0.52, "3x3 Conv\n+ BN", *blue)
    block(0.35, 1.95, 1.35, 0.52, "1x1 Conv\n+ BN", *yellow)
    block(0.35, 1.05, 1.35, 0.52, "Identity\n+ BN", *purple)
    ax.text(1.02, 3.62, "train-time branches", ha="center", fontsize=8.2, color="#334155", weight="bold")

    # Equivalent kernels after folding BN.
    block(2.35, 2.86, 1.0, 0.48, "fold BN", "#f8fafc", "#64748b", fontsize=7.0)
    block(2.35, 1.96, 1.0, 0.48, "fold BN", "#f8fafc", "#64748b", fontsize=7.0)
    block(2.35, 1.06, 1.0, 0.48, "fold BN", "#f8fafc", "#64748b", fontsize=7.0)
    for y in [3.11, 2.21, 1.31]:
        arrow(1.7, y, 2.35, y)

    kernel_grid(3.85, 2.78, [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]], "3x3 kernel", "#2563eb", "#dbeafe")
    kernel_grid(3.85, 1.88, [["", "", ""], ["", "w", ""], ["", "", ""]], "1x1 padded", "#d97706", "#fef3c7")
    kernel_grid(3.85, 0.98, [["", "", ""], ["", "1", ""], ["", "", ""]], "identity padded", "#9333ea", "#f3e8ff")
    for y in [3.11, 2.21, 1.31]:
        arrow(3.35, y, 3.85, y)

    ax.text(4.9, 2.22, "+", ha="center", va="center", fontsize=18, weight="bold", color="#334155")
    arrow(4.75, 2.22, 5.55, 2.22)

    kernel_grid(5.65, 1.88, [["a", "b", "c"], ["d", "e+w+1", "f"], ["g", "h", "i"]], "summed 3x3", "#16a34a", "#dcfce7")
    block(6.65, 1.95, 1.55, 0.56, "single 3x3 Conv\nfor inference", *green, fontsize=7.4)
    arrow(6.22, 2.22, 6.65, 2.22, "#16a34a")

    ax.text(5.05, 3.53, "1x1 and identity are embedded at the center of a 3x3 kernel",
            ha="center", va="center", fontsize=7.7, color="#334155")
    ax.text(6.9, 1.35, "same output shape\nfewer runtime ops", ha="center", va="center", fontsize=7.2, color="#166534")
    ax.set_xlim(0, 8.45)
    ax.set_ylim(0.65, 3.85)



def _draw_yolov6_overview_demo(ax):
    from matplotlib.patches import Rectangle

    ax.set_title("YOLOv6: Rep blocks + TAL + deployment-oriented design", fontsize=11, weight="bold")
    ax.axis("off")

    def box(x, y, w, h, text, face, edge, fontsize=7.6):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.45))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, weight="bold", linespacing=1.12)

    def arrow(x1, y1, x2, y2, color="#475569", label=None, yoff=0.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "-|>", "linewidth": 1.35, "color": color})
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + yoff, label,
                    ha="center", va="center", fontsize=6.9, color=color, weight="bold")

    # Context inherited from YOLOX, kept visually separate so it is not presented as v6's new part.
    box(0.25, 2.9, 1.6, 0.55, "YOLOX context\nanchor-free / decoupled", "#f8fafc", "#94a3b8", 6.9)
    box(0.25, 2.05, 1.6, 0.55, "SimOTA\ndynamic assign", "#f8fafc", "#94a3b8", 6.9)
    ax.text(1.05, 3.65, "already covered", ha="center", va="center", fontsize=7.4, color="#64748b", weight="bold")

    # Main v6 additions.
    box(2.45, 3.05, 1.75, 0.62, "Rep block\ntrain: 3x3 + 1x1 + id", "#dbeafe", "#2563eb")
    box(2.45, 2.05, 1.75, 0.62, "TAL assigner\nt = s^alpha u^beta", "#dcfce7", "#16a34a")
    box(2.45, 1.05, 1.75, 0.62, "Latency-aware\nefficient design", "#fef3c7", "#d97706")
    ax.text(3.33, 3.9, "YOLOv6 focus", ha="center", va="center", fontsize=8.3, color="#111827", weight="bold")

    # Deployment side.
    box(5.05, 3.05, 1.75, 0.62, "Deploy conv\nfused 3x3", "#bfdbfe", "#1d4ed8")
    box(5.05, 2.05, 1.75, 0.62, "Positive samples\ncls and IoU aligned", "#bbf7d0", "#15803d", 7.1)
    box(5.05, 1.05, 1.75, 0.62, "Runtime path\nsimple ops", "#fde68a", "#b45309")

    arrow(1.85, 3.18, 2.45, 3.36, "#94a3b8")
    arrow(1.85, 2.33, 2.45, 2.36, "#94a3b8")
    arrow(4.2, 3.36, 5.05, 3.36, "#2563eb", "fuse at deploy", 0.22)
    arrow(4.2, 2.36, 5.05, 2.36, "#16a34a", "select high alignment", 0.22)
    arrow(4.2, 1.36, 5.05, 1.36, "#d97706", "optimize actual latency", 0.22)

    # Small equation card for the TAL part.
    box(0.55, 0.68, 6.0, 0.42,
        "TAL: high class score s and high IoU u must coincide; SimOTA details are not repeated here",
        "#ffffff", "#cbd5e1", 7.1)
    ax.text(5.92, 2.84, "classification score\nshould reflect box quality", ha="center", va="center",
            fontsize=6.9, color="#166534")
    ax.text(5.92, 3.84, "training graph !=\ninference graph", ha="center", va="center",
            fontsize=6.9, color="#1e40af")

    ax.set_xlim(0, 7.2)
    ax.set_ylim(0.45, 4.15)

def _draw_elan_demo(ax):
    ax.set_title("ELAN/GELAN-style multi-path aggregation", fontsize=11, weight="bold")
    ax.axis("off")
    ax.add_patch(Rectangle((0.3, 2.0), 1.5, 0.65, facecolor="#dbeafe", edgecolor="#2563eb", linewidth=1.5))
    ax.text(1.05, 2.33, "input", ha="center", va="center", weight="bold")
    ys = [3.1, 2.1, 1.1]
    names = ["short path", "conv path", "deep path"]
    for y, name in zip(ys, names):
        ax.add_patch(Rectangle((3.0, y), 2.0, 0.55, facecolor="#dcfce7", edgecolor="#16a34a", linewidth=1.5))
        ax.text(4.0, y + 0.275, name, ha="center", va="center", fontsize=9)
        ax.annotate("", xy=(3.0, y + 0.275), xytext=(1.8, 2.33), arrowprops={"arrowstyle": "-|>"})
        ax.annotate("", xy=(6.2, 2.33), xytext=(5.0, y + 0.275), arrowprops={"arrowstyle": "-|>"})
    ax.add_patch(Rectangle((6.2, 2.0), 1.7, 0.65, facecolor="#fef3c7", edgecolor="#d97706", linewidth=1.5))
    ax.text(7.05, 2.33, "concat", ha="center", va="center", weight="bold")
    ax.set_xlim(0, 8.4)
    ax.set_ylim(0.7, 3.9)


def _draw_anchor_free_demo(ax):
    ax.set_title("Anchor-free box decoding", fontsize=11, weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    p = np.array([.5, .5])
    ltrb = np.array([.1, .2, .3, .15])
    box = [p[0] - ltrb[0], p[1] - ltrb[1], p[0] + ltrb[2], p[1] + ltrb[3]]
    ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor="#dc2626", linewidth=2.2))
    ax.scatter([p[0]], [p[1]], s=80, color="#2563eb", edgecolor="white", zorder=3)
    labels = [(box[0], p[1], "l"), (p[0], box[1], "t"), (box[2], p[1], "r"), (p[0], box[3], "b")]
    for x, y, label in labels:
        ax.plot([p[0], x], [p[1], y], color="#2563eb", linewidth=1.6)
        ax.text((p[0] + x) / 2, (p[1] + y) / 2, label, fontsize=10, weight="bold")


def _draw_pgi_demo(ax):
    ax.set_title("PGI: auxiliary gradient during training", fontsize=11, weight="bold")
    ax.axis("off")
    items = [("main loss", 0.2, 2.2), ("auxiliary loss", 0.2, 1.1), ("backbone update", 5.1, 1.65)]
    for label, x, y in items:
        ax.add_patch(Rectangle((x, y), 2.0, 0.65, facecolor="#dbeafe" if x < 5 else "#dcfce7", edgecolor="#2563eb", linewidth=1.5))
        ax.text(x + 1.0, y + 0.325, label, ha="center", va="center", weight="bold", fontsize=9)
    ax.annotate("gradient", xy=(5.1, 1.98), xytext=(2.2, 2.52), arrowprops={"arrowstyle": "-|>"})
    ax.annotate("programmable gradient", xy=(5.1, 1.98), xytext=(2.2, 1.42), arrowprops={"arrowstyle": "-|>", "color": "#dc2626"}, color="#dc2626")
    ax.text(3.8, 0.7, "auxiliary path is removed for inference", ha="center", fontsize=9)
    ax.set_xlim(0, 7.5)
    ax.set_ylim(0.5, 3.2)


def _draw_nms_free_demo(ax):
    ax.set_title("Dual assignment -> NMS-free inference", fontsize=11, weight="bold")
    ax.axis("off")
    left = [("one-to-many\ntraining", 0.4, 2.4), ("many positives\nrich supervision", 3.2, 2.4)]
    right = [("one-to-one\ntraining", 0.4, 1.1), ("unique predictions\nused at inference", 3.2, 1.1)]
    for items, color in [(left, "#dbeafe"), (right, "#dcfce7")]:
        for label, x, y in items:
            ax.add_patch(Rectangle((x, y), 2.2, 0.7, facecolor=color, edgecolor="#334155", linewidth=1.5))
            ax.text(x + 1.1, y + 0.35, label, ha="center", va="center", fontsize=9, weight="bold")
        ax.annotate("", xy=(3.2, items[1][2] + 0.35), xytext=(2.6, items[0][2] + 0.35), arrowprops={"arrowstyle": "-|>"})
    ax.text(3.9, 0.55, "inference consumes one-to-one outputs directly", ha="center", fontsize=9)
    ax.set_xlim(0, 6.0)
    ax.set_ylim(0.3, 3.4)


def _draw_attention_demo(ax, area=False):
    ax.set_title("Area Attention" if area else "Position attention", fontsize=11, weight="bold")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)
    ax.axis("off")
    pts = [(0.8, 2.2), (2.0, 2.35), (3.2, 2.0), (1.2, 0.9), (2.5, 0.75), (3.4, 1.0)]
    for i, (x, y) in enumerate(pts):
        ax.scatter([x], [y], s=120, color="#dbeafe", edgecolor="#2563eb", linewidth=1.5, zorder=3)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=9, weight="bold")
    if area:
        ax.add_patch(Rectangle((0.35, 1.65), 3.25, 0.95, fill=False, edgecolor="#16a34a", linewidth=2))
        ax.add_patch(Rectangle((0.35, 0.35), 3.25, 0.95, fill=False, edgecolor="#dc2626", linewidth=2))
        pairs = [(0, 1), (1, 2), (3, 4), (4, 5)]
    else:
        pairs = [(0, 1), (0, 4), (2, 3), (1, 5), (3, 5)]
    for a, b in pairs:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color="#64748b", alpha=0.65)
    ax.text(2.0, 0.08, "restricted regions reduce QK pairs" if area else "features can refer to other positions", ha="center", fontsize=9)


def _draw_hypergraph_demo(ax):
    ax.set_title("Hypergraph feature correlation", fontsize=11, weight="bold")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    nodes = [(0.8, 2.4), (1.5, 1.7), (0.8, 1.0), (3.6, 2.0), (4.1, 1.1)]
    for i, (x, y) in enumerate(nodes):
        ax.scatter([x], [y], s=120, color="#dbeafe", edgecolor="#2563eb", linewidth=1.5, zorder=3)
        ax.text(x, y, f"n{i}", ha="center", va="center", fontsize=8, weight="bold")
    ax.add_patch(Rectangle((0.45, 0.75), 1.45, 1.9, fill=False, edgecolor="#16a34a", linewidth=2.2))
    ax.add_patch(Rectangle((3.25, 0.85), 1.15, 1.45, fill=False, edgecolor="#dc2626", linewidth=2.2))
    ax.text(1.18, 2.82, "hyperedge A", ha="center", fontsize=9, color="#166534")
    ax.text(3.82, 2.5, "hyperedge B", ha="center", fontsize=9, color="#991b1b")
    ax.annotate("aggregate and return", xy=(2.6, 1.65), xytext=(2.1, 1.65), arrowprops={"arrowstyle": "<->"}, ha="center")


def _draw_deployment_demo(ax):
    ax.set_title("Deployment-oriented output", fontsize=11, weight="bold")
    ax.axis("off")
    labels = ["raw predictions", "confidence threshold", "final detections"]
    xs = [0.4, 3.0, 5.9]
    for x, label in zip(xs, labels):
        ax.add_patch(Rectangle((x, 1.5), 2.0, 0.75, facecolor="#dbeafe", edgecolor="#2563eb", linewidth=1.5))
        ax.text(x + 1.0, 1.875, label, ha="center", va="center", fontsize=9, weight="bold")
    ax.annotate("no NMS loop", xy=(3.0, 1.875), xytext=(2.4, 1.875), arrowprops={"arrowstyle": "-|>"}, ha="center")
    ax.annotate("direct consume", xy=(5.9, 1.875), xytext=(5.0, 1.875), arrowprops={"arrowstyle": "-|>"}, ha="center")
    ax.set_xlim(0, 8.2)
    ax.set_ylim(1.0, 2.8)


def display_yolo_technology_demo(version):
    """Draw the main technical difference introduced by a YOLO generation."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle as _Rectangle
    except ImportError as exc:
        raise ImportError("Install matplotlib to display YOLO technology demos") from exc

    globals()["Rectangle"] = _Rectangle
    fig, ax = plt.subplots(figsize=(9, 4.8))
    drawers = {
        2: _draw_yolov2_overview_demo,
        3: _draw_yolov3_overview_demo,
        4: _draw_yolov4_overview_demo,
        5: _draw_focus_demo,
        6: _draw_yolov6_overview_demo,
        7: _draw_elan_demo,
        8: _draw_anchor_free_demo,
        9: _draw_pgi_demo,
        10: _draw_nms_free_demo,
        11: lambda axis: _draw_attention_demo(axis, area=False),
        12: lambda axis: _draw_attention_demo(axis, area=True),
        13: _draw_hypergraph_demo,
        26: _draw_deployment_demo,
    }
    if version not in drawers:
        raise ValueError(f"No technology demo for version: {version}")
    drawers[version](ax)
    fig.tight_layout()
    return _display_figure(fig)




def _draw_iou_distance_demo(ax):
    ax.set_title("IoU distance for anchor k-means", fontsize=11, weight="bold")
    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(0.45, -0.45)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("normalized width")
    ax.set_ylabel("normalized height")

    # Boxes are placed at the same center because anchor clustering compares shape,
    # not object position in the image.
    box = (-0.10, -0.25, 0.20, 0.50)
    centroid = (-0.22, -0.14, 0.44, 0.28)
    inter = (-0.10, -0.14, 0.20, 0.28)

    for x, y, w, h, color, label, alpha in [
        (*centroid, "#2563eb", "cluster centroid", 0.18),
        (*box, "#dc2626", "training box", 0.18),
        (*inter, "#16a34a", "overlap", 0.45),
    ]:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor=color, linewidth=2.0, alpha=alpha, label=label))
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2.2))

    ax.scatter([0], [0], s=45, color="#111827", zorder=4)
    ax.text(0.02, 0.03, "same center", fontsize=8.5, color="#111827")
    ax.text(
        -0.42,
        0.36,
        "shape distance\nd(box, centroid) = 1 - IoU",
        ha="left",
        va="top",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9, "pad": 4},
    )
    ax.legend(loc="upper right", fontsize=8)

def _draw_bn_demo(ax):
    ax.set_title("Batch normalization", fontsize=11, weight="bold")
    ax.axis("off")
    xs = [0.6, 3.2, 5.8]
    labels = ["conv output\nx", "normalize\n(x - mean) / std", "scale + shift\ngamma * xhat + beta"]
    for x, label in zip(xs, labels):
        ax.add_patch(Rectangle((x, 1.6), 2.0, 0.9, facecolor="#dbeafe", edgecolor="#2563eb", linewidth=1.5))
        ax.text(x + 1.0, 2.05, label, ha="center", va="center", fontsize=9, weight="bold")
    for x1, x2 in [(2.6, 3.2), (5.2, 5.8)]:
        ax.annotate("", xy=(x2, 2.05), xytext=(x1, 2.05), arrowprops={"arrowstyle": "-|>"})
    ax.set_xlim(0, 8.2)
    ax.set_ylim(1.0, 3.0)


def _draw_passthrough_demo(ax):
    ax.set_title("YOLOv2 passthrough: mid-level feature -> concat before head", fontsize=11, weight="bold")
    ax.axis("off")

    def block(x, y, w, h, label, face, edge):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.6))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.5, weight="bold", linespacing=1.25)

    block(0.25, 2.55, 1.35, 0.65, "input\n416 x 416", "#dbeafe", "#2563eb")
    block(2.05, 2.55, 1.75, 0.65, "Darknet-19\nearly/mid convs", "#dcfce7", "#16a34a")
    block(4.35, 2.55, 1.65, 0.65, "mid feature\n26 x 26 x C", "#fef3c7", "#d97706")
    block(6.65, 2.55, 1.35, 0.65, "reorg\n2 x 2 -> ch", "#fee2e2", "#dc2626")
    block(8.65, 2.55, 1.85, 0.65, "shallow path\n13 x 13 x 4C", "#fee2e2", "#dc2626")

    block(4.35, 1.15, 1.95, 0.65, "later/deeper convs", "#dcfce7", "#16a34a")
    block(6.9, 1.15, 1.85, 0.65, "deep feature\n13 x 13 x D", "#dbeafe", "#2563eb")
    block(9.05, 1.15, 1.35, 0.65, "concat", "#f3e8ff", "#9333ea")
    block(11.0, 1.15, 2.05, 0.65, "head input\n13 x 13 x (D + 4C)", "#f3e8ff", "#9333ea")
    block(13.55, 1.15, 1.35, 0.65, "anchor\nhead", "#e0f2fe", "#0284c7")

    arrows = [
        ((1.6, 2.875), (2.05, 2.875)),
        ((3.8, 2.875), (4.35, 2.875)),
        ((6.0, 2.875), (6.65, 2.875)),
        ((8.0, 2.875), (8.65, 2.875)),
        ((5.2, 2.55), (5.05, 1.8)),
        ((6.3, 1.475), (6.9, 1.475)),
        ((8.75, 1.475), (9.05, 1.475)),
        ((10.5, 1.475), (11.0, 1.475)),
        ((13.05, 1.475), (13.55, 1.475)),
        ((9.6, 2.55), (9.65, 1.8)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": "#475569", "linewidth": 1.4})

    ax.text(9.1, 0.58, "channels increase only at the concat point: D -> D + 4C", ha="center", fontsize=9.5, color="#111827")
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0.35, 3.55)


def _draw_hierarchy_demo(ax):
    ax.set_title("Hierarchical classification", fontsize=11, weight="bold")
    ax.axis("off")
    nodes = [("entity", 3.5, 3.0), ("animal", 2.0, 2.1), ("vehicle", 5.0, 2.1), ("dog", 1.3, 1.2), ("cat", 2.7, 1.2), ("car", 4.4, 1.2), ("bus", 5.6, 1.2)]
    for name, x, y in nodes:
        ax.add_patch(Rectangle((x - 0.45, y - 0.2), 0.9, 0.4, facecolor="#fef3c7", edgecolor="#d97706", linewidth=1.2))
        ax.text(x, y, name, ha="center", va="center", fontsize=8, weight="bold")
    for a, b in [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)]:
        ax.plot([nodes[a][1], nodes[b][1]], [nodes[a][2]-0.2, nodes[b][2]+0.2], color="#64748b")
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0.7, 3.4)


def _draw_residual_demo(ax):
    ax.set_title("Residual block", fontsize=11, weight="bold")
    ax.axis("off")
    ax.add_patch(Rectangle((0.6, 1.8), 1.3, 0.6, facecolor="#dbeafe", edgecolor="#2563eb", linewidth=1.5))
    ax.text(1.25, 2.1, "x", ha="center", va="center", weight="bold")
    ax.add_patch(Rectangle((3.0, 1.8), 2.0, 0.6, facecolor="#dcfce7", edgecolor="#16a34a", linewidth=1.5))
    ax.text(4.0, 2.1, "F(x)", ha="center", va="center", weight="bold")
    ax.add_patch(Rectangle((6.0, 1.8), 1.4, 0.6, facecolor="#fef3c7", edgecolor="#d97706", linewidth=1.5))
    ax.text(6.7, 2.1, "x + F(x)", ha="center", va="center", weight="bold")
    ax.annotate("", xy=(3.0, 2.1), xytext=(1.9, 2.1), arrowprops={"arrowstyle": "-|>"})
    ax.annotate("", xy=(6.0, 2.1), xytext=(5.0, 2.1), arrowprops={"arrowstyle": "-|>"})
    ax.annotate("skip", xy=(6.0, 2.32), xytext=(1.9, 2.65), arrowprops={"arrowstyle": "-|>"}, ha="center")
    ax.set_xlim(0, 8)
    ax.set_ylim(1.4, 3.0)


def _draw_logistic_demo(ax):
    ax.set_title("Independent logistic class prediction", fontsize=11, weight="bold")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.05, 1.05)
    x = np.linspace(-5, 5, 200)
    y = 1 / (1 + np.exp(-x))
    ax.plot(x, y, color="#2563eb", linewidth=2)
    ax.axhline(0.5, color="#64748b", linewidth=1, linestyle="--")
    ax.set_xlabel("logit")
    ax.set_ylabel("probability")
    ax.grid(True, alpha=0.25)


def _draw_spp_pan_demo(ax):
    ax.set_title("YOLOv4 neck: SPP, FPN top-down, PAN bottom-up", fontsize=11, weight="bold")
    ax.axis("off")

    def block(x, y, w, h, label, face, edge, fontsize=7.2):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.35))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, weight="bold", linespacing=1.12)

    def arrow(x1, y1, x2, y2, color="#475569", label=None, label_xy=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.35,
                                "shrinkA": 1, "shrinkB": 1})
        if label:
            lx, ly = label_xy if label_xy is not None else ((x1 + x2) / 2, (y1 + y2) / 2)
            ax.text(lx, ly, label, ha="center", va="center", fontsize=6.6, color=color,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.4})

    blue = ("#dbeafe", "#2563eb")
    yellow = ("#fef3c7", "#d97706")
    green = ("#dcfce7", "#16a34a")
    red = ("#fee2e2", "#dc2626")
    gray = ("#f8fafc", "#64748b")

    # Row centers: high resolution at top, low resolution at bottom.
    y3, y4, y5 = 3.0, 2.0, 1.0
    h = 0.5
    x_c, x_p, x_n, x_h = 0.35, 2.55, 4.85, 7.05
    w = 1.45

    ax.text(x_c + w / 2, 3.55, "backbone", ha="center", fontsize=7.5, color="#1e40af", weight="bold")
    ax.text(x_p + w / 2, 3.55, "SPP + FPN", ha="center", fontsize=7.5, color="#166534", weight="bold")
    ax.text(x_n + w / 2, 3.55, "PAN neck", ha="center", fontsize=7.5, color="#991b1b", weight="bold")
    ax.text(x_h + w / 2, 3.55, "heads", ha="center", fontsize=7.5, color="#475569", weight="bold")

    block(x_c, y3 - h / 2, w, h, "C3\n52x52", *blue)
    block(x_c, y4 - h / 2, w, h, "C4\n26x26", *blue)
    block(x_c, y5 - h / 2, w, h, "C5\n13x13", *blue)

    block(x_p, y5 - h / 2, w, h, "SPP + P5\n13x13", *yellow)
    block(x_p, y4 - h / 2, w, h, "P4\nconcat(C4, up P5)\n26x26", *green, fontsize=6.5)
    block(x_p, y3 - h / 2, w, h, "P3\nconcat(C3, up P4)\n52x52", *green, fontsize=6.5)

    block(x_n, y3 - h / 2, w, h, "N3\nfrom P3\n52x52", *red, fontsize=6.6)
    block(x_n, y4 - h / 2, w, h, "N4\nconcat(P4, down N3)\n26x26", *red, fontsize=6.2)
    block(x_n, y5 - h / 2, w, h, "N5\nconcat(P5, down N4)\n13x13", *red, fontsize=6.2)

    block(x_h, y3 - h / 2, w, h, "small\nhead", *gray)
    block(x_h, y4 - h / 2, w, h, "medium\nhead", *gray)
    block(x_h, y5 - h / 2, w, h, "large\nhead", *gray)

    # Backbone to SPP/FPN lateral connections.
    arrow(x_c + w, y5, x_p, y5, blue[1])
    arrow(x_c + w, y4, x_p, y4, blue[1])
    arrow(x_c + w, y3, x_p, y3, blue[1])

    # FPN top-down path: low resolution to high resolution by upsampling.
    arrow(x_p + w / 2, y5 + h / 2, x_p + w / 2, y4 - h / 2, green[1], "up x2", (x_p + w / 2 + 0.34, 1.5))
    arrow(x_p + w / 2, y4 + h / 2, x_p + w / 2, y3 - h / 2, green[1], "up x2", (x_p + w / 2 + 0.34, 2.5))

    # PAN bottom-up path: high resolution to low resolution by downsampling.
    arrow(x_p + w, y3, x_n, y3, red[1])
    arrow(x_n + w / 2, y3 - h / 2, x_n + w / 2, y4 + h / 2, red[1], "down x2", (x_n + w / 2 + 0.42, 2.5))
    arrow(x_p + w, y4, x_n, y4, red[1])
    arrow(x_n + w / 2, y4 - h / 2, x_n + w / 2, y5 + h / 2, red[1], "down x2", (x_n + w / 2 + 0.42, 1.5))
    arrow(x_p + w, y5, x_n, y5, red[1])

    # Heads consume PAN features only; no arrows return from heads.
    arrow(x_n + w, y3, x_h, y3, gray[1])
    arrow(x_n + w, y4, x_h, y4, gray[1])
    arrow(x_n + w, y5, x_h, y5, gray[1])

    ax.text(2.0, 0.35, "SPP is applied on deepest C5 and keeps 13x13 H,W",
            ha="center", va="center", fontsize=7.0, color="#92400e")
    ax.text(3.7, 3.28, "green = FPN top-down", ha="center", va="center", fontsize=7.0, color="#166534")
    ax.text(5.85, 3.28, "red = PAN bottom-up", ha="center", va="center", fontsize=7.0, color="#991b1b")
    ax.set_xlim(0.0, 8.75)
    ax.set_ylim(0.15, 3.75)


def _draw_activation_demo(ax, kind="mish"):
    if kind == "mish":
        ax.set_title("Activation comparison: ReLU vs GELU vs Mish", fontsize=11, weight="bold")
    else:
        ax.set_title(kind.capitalize() + " activation", fontsize=11, weight="bold")

    x = np.linspace(-5, 5, 500)
    relu = np.maximum(x, 0)
    # Tanh approximation of GELU used in many implementations.
    gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    mish = x * np.tanh(np.log1p(np.exp(x)))

    if kind == "mish":
        ax.plot(x, relu, color="#64748b", linewidth=1.8, linestyle="--", label="ReLU: max(0, x)")
        ax.plot(x, gelu, color="#d97706", linewidth=2.0, label="GELU: x Phi(x)")
        ax.plot(x, mish, color="#2563eb", linewidth=2.4, label="Mish: x tanh(softplus(x))")
        ax.text(-3.65, -0.55, "negative side\nkept smoothly", ha="center", va="center", fontsize=8, color="#1d4ed8")
        ax.text(2.65, 2.25, "all behave\nroughly like x", ha="center", va="center", fontsize=8, color="#334155")
        ax.legend(loc="upper left", fontsize=8, frameon=True)
    elif kind == "relu":
        ax.plot(x, relu, color="#2563eb", linewidth=2.2)
    else:
        ax.plot(x, mish, color="#2563eb", linewidth=2.2)

    ax.axhline(0, color="#94a3b8", linewidth=1)
    ax.axvline(0, color="#94a3b8", linewidth=1)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1.0, 5.0)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")


def _draw_ciou_demo(ax):
    ax.set_title("CIoU components", fontsize=11, weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.add_patch(Rectangle((0.18, 0.25), 0.42, 0.35, fill=False, edgecolor="#2563eb", linewidth=2.2))
    ax.add_patch(Rectangle((0.38, 0.38), 0.42, 0.32, fill=False, edgecolor="#dc2626", linewidth=2.2))
    ax.scatter([0.39, 0.59], [0.425, 0.54], color=["#2563eb", "#dc2626"], zorder=3)
    ax.plot([0.39, 0.59], [0.425, 0.54], color="#64748b", linestyle="--")
    ax.text(0.5, 0.18, "IoU + center distance + aspect ratio", ha="center", fontsize=9)


def _draw_csp_demo(ax):
    ax.set_title("CSP block: split channels, shortcut one part, transform the other", fontsize=11, weight="bold")
    ax.axis("off")

    def block(x, y, w, h, label, face, edge, fontsize=8.2):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.5))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, weight="bold", linespacing=1.15)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "-|>", "color": "#475569", "linewidth": 1.35})

    block(0.35, 1.85, 1.3, 0.7, "input\nfeature C", "#dbeafe", "#2563eb")
    block(2.25, 2.65, 1.35, 0.62, "split A\nC/2", "#ecfeff", "#0891b2")
    block(2.25, 0.95, 1.35, 0.62, "split B\nC/2", "#ecfeff", "#0891b2")

    block(4.0, 2.65, 1.45, 0.62, "1x1 Conv\nshortcut", "#dcfce7", "#16a34a")
    block(3.85, 1.45, 1.35, 0.62, "1x1 Conv", "#fef3c7", "#d97706")
    block(5.55, 1.25, 1.7, 1.02, "residual /\nconv blocks\nx N", "#fee2e2", "#dc2626")
    block(7.6, 1.45, 1.35, 0.62, "1x1 Conv", "#fef3c7", "#d97706")

    block(9.7, 1.85, 1.45, 0.72, "concat\nchannels", "#f3e8ff", "#9333ea")
    block(11.75, 1.85, 1.45, 0.72, "1x1 Conv\nfuse", "#dbeafe", "#2563eb")

    arrow(1.65, 2.2, 2.25, 2.96)
    arrow(1.65, 2.2, 2.25, 1.26)
    arrow(3.6, 2.96, 4.0, 2.96)
    arrow(3.6, 1.26, 3.85, 1.76)
    arrow(5.2, 1.76, 5.55, 1.76)
    arrow(7.25, 1.76, 7.6, 1.76)
    arrow(5.45, 2.96, 9.7, 2.28)
    arrow(8.95, 1.76, 9.7, 2.12)
    arrow(11.15, 2.21, 11.75, 2.21)

    ax.text(6.55, 3.45, "A: light path preserves features and gradients",
            ha="center", va="center", fontsize=8.4, color="#166534")
    ax.text(6.45, 0.58, "B: only part of the channels goes through heavy blocks",
            ha="center", va="center", fontsize=8.4, color="#991b1b")
    ax.text(10.45, 1.42, "concat increases channels;\nfuse conv mixes them",
            ha="center", va="top", fontsize=8.1, color="#581c87")
    ax.set_xlim(0, 13.55)
    ax.set_ylim(0.25, 3.75)


def _draw_decoupled_head_demo(ax):
    ax.set_title("Decoupled detection head", fontsize=11, weight="bold")
    ax.axis("off")
    ax.add_patch(Rectangle((0.5, 2.0), 1.8, 0.7, facecolor="#dbeafe", edgecolor="#2563eb", linewidth=1.5))
    ax.text(1.4,2.35,"shared feature",ha="center",va="center",weight="bold",fontsize=9)
    for label, y in [("class branch",2.8),("box branch",1.25)]:
        ax.add_patch(Rectangle((4.0, y), 2.0, 0.65, facecolor="#fef3c7", edgecolor="#d97706", linewidth=1.5))
        ax.text(5.0,y+0.325,label,ha="center",va="center",weight="bold",fontsize=9)
        ax.annotate("", xy=(4.0,y+0.325), xytext=(2.3,2.35), arrowprops={"arrowstyle":"-|>"})
    ax.set_xlim(0,7)
    ax.set_ylim(1.0,3.8)


def _draw_anchor_point_grid_demo(ax):
    ax.set_title("Grid point, anchor box, and anchor point", fontsize=11, weight="bold")
    ax.axis("off")

    def draw_grid(x0, y0, size, n=4):
        step = size / n
        for i in range(n + 1):
            ax.plot([x0, x0 + size], [y0 + i * step, y0 + i * step], color="#cbd5e1", linewidth=1)
            ax.plot([x0 + i * step, x0 + i * step], [y0, y0 + size], color="#cbd5e1", linewidth=1)
        pts = []
        for r in range(n):
            for c in range(n):
                pts.append((x0 + (c + 0.5) * step, y0 + (r + 0.5) * step))
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=18, color="#64748b", zorder=3)
        return step

    def rect_center(cx, cy, w, h, edge, linestyle="-", lw=1.8):
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False,
                               edgecolor=edge, linewidth=lw, linestyle=linestyle))

    # Left: anchor-based detector. Anchor boxes are width/height priors placed at grid points.
    ax.text(1.75, 3.62, "anchor-based YOLOv2-v7 style", ha="center", fontsize=8.3, weight="bold", color="#334155")
    step = draw_grid(0.45, 0.75, 2.6, n=4)
    gp = (0.45 + 2.5 * step, 0.75 + 1.5 * step)
    ax.scatter([gp[0]], [gp[1]], s=80, color="#2563eb", edgecolor="white", zorder=5)
    rect_center(gp[0], gp[1], 0.58, 0.36, "#d97706")
    rect_center(gp[0], gp[1], 0.34, 0.70, "#d97706", linestyle="--")
    ax.text(gp[0] + 0.18, gp[1] - 0.25, "grid point\n(cell center)", fontsize=7.0, color="#1d4ed8")
    ax.text(1.75, 0.38, "anchor boxes = prior widths/heights\nplaced on each grid point", ha="center", fontsize=7.2, color="#92400e")

    # Right: anchor-free / anchor-point detector. The point is a reference, not a pre-sized box.
    ax.text(6.0, 3.62, "anchor-free / anchor-point style", ha="center", fontsize=8.3, weight="bold", color="#334155")
    draw_grid(4.7, 0.75, 2.6, n=4)
    ap = (4.7 + 1.5 * step, 0.75 + 1.5 * step)
    ax.scatter([ap[0]], [ap[1]], s=80, color="#16a34a", edgecolor="white", zorder=5)
    box = (ap[0] - 0.52, ap[1] - 0.33, 1.18, 0.78)
    ax.add_patch(Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor="#dc2626", linewidth=2.0))
    # l,t,r,b distances from anchor point to box sides.
    ax.annotate("", xy=(box[0], ap[1]), xytext=(ap[0], ap[1]), arrowprops={"arrowstyle":"<->", "color":"#16a34a", "linewidth":1.2})
    ax.annotate("", xy=(ap[0], box[1]), xytext=(ap[0], ap[1]), arrowprops={"arrowstyle":"<->", "color":"#16a34a", "linewidth":1.2})
    ax.annotate("", xy=(box[0] + box[2], ap[1]), xytext=(ap[0], ap[1]), arrowprops={"arrowstyle":"<->", "color":"#16a34a", "linewidth":1.2})
    ax.annotate("", xy=(ap[0], box[1] + box[3]), xytext=(ap[0], ap[1]), arrowprops={"arrowstyle":"<->", "color":"#16a34a", "linewidth":1.2})
    ax.text(ap[0] - 0.1, ap[1] - 0.18, "anchor point\nreference only", fontsize=7.0, color="#166534", ha="right")
    ax.text(6.0, 0.38, "predict distances l,t,r,b\nfrom the point to box sides", ha="center", fontsize=7.2, color="#166534")

    ax.text(3.82, 2.15, "same grid locations,\ndifferent meaning", ha="center", va="center", fontsize=7.4, color="#334155")
    ax.annotate("", xy=(4.45, 2.05), xytext=(3.25, 2.05), arrowprops={"arrowstyle":"-|>", "color":"#64748b", "linewidth":1.2})
    ax.set_xlim(0.1, 7.75)
    ax.set_ylim(0.15, 3.9)


def _draw_assignment_demo(ax):
    ax.set_title("Label assignment: candidate point vs predicted box", fontsize=11, weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.18)

    gt_xy = (0.42, 0.28)
    gt_w, gt_h = 0.32, 0.34
    gt_center = (gt_xy[0] + gt_w / 2, gt_xy[1] + gt_h / 2)
    ax.add_patch(Rectangle(gt_xy, gt_w, gt_h, fill=False, edgecolor="#dc2626", linewidth=2.4))
    ax.scatter([gt_center[0]], [gt_center[1]], s=95, marker="*", color="#dc2626", edgecolor="white", zorder=5)
    ax.text(gt_center[0] + 0.025, gt_center[1] - 0.035, "GT center", fontsize=7.5, color="#dc2626", weight="bold")

    # Candidate A is close to the object center and predicts a good box.
    cand_a = np.array([0.56, 0.45])
    pred_a = (0.40, 0.29, 0.34, 0.32)
    pred_a_center = (pred_a[0] + pred_a[2] / 2, pred_a[1] + pred_a[3] / 2)
    ax.add_patch(Rectangle(pred_a[:2], pred_a[2], pred_a[3], fill=False, edgecolor="#2563eb", linewidth=2.0))
    ax.scatter([cand_a[0]], [cand_a[1]], s=85, color="#2563eb", edgecolor="white", zorder=4)
    ax.scatter([pred_a_center[0]], [pred_a_center[1]], s=70, marker="x", color="#2563eb", linewidth=2.2, zorder=4)
    ax.annotate("", xy=pred_a_center, xytext=cand_a, arrowprops={"arrowstyle":"->", "color":"#2563eb", "linewidth":1.25})
    ax.text(cand_a[0] - 0.17, cand_a[1] + 0.085, "candidate point A\n(anchor/grid point)", fontsize=7.3, color="#1d4ed8")
    ax.text(pred_a_center[0] + 0.03, pred_a_center[1] + 0.05, "pred box center\nnot necessarily A", fontsize=7.1, color="#1d4ed8")

    # Candidate B is far from the object center but can still regress a partly overlapping box.
    cand_b = np.array([0.18, 0.72])
    pred_b = (0.33, 0.35, 0.34, 0.30)
    pred_b_center = (pred_b[0] + pred_b[2] / 2, pred_b[1] + pred_b[3] / 2)
    ax.add_patch(Rectangle(pred_b[:2], pred_b[2], pred_b[3], fill=False, edgecolor="#f97316", linewidth=1.8, linestyle="--"))
    ax.scatter([cand_b[0]], [cand_b[1]], s=85, color="#f97316", edgecolor="white", zorder=4)
    ax.scatter([pred_b_center[0]], [pred_b_center[1]], s=65, marker="x", color="#f97316", linewidth=2.0, zorder=4)
    ax.annotate("", xy=pred_b_center, xytext=cand_b, arrowprops={"arrowstyle":"->", "color":"#f97316", "linewidth":1.2})
    ax.text(cand_b[0] - 0.11, cand_b[1] + 0.08, "candidate B\nfar from center", fontsize=7.2, color="#c2410c")

    # Center region used as a simple visual proxy for center prior/penalty.
    center_region = Rectangle((gt_center[0] - 0.13, gt_center[1] - 0.13), 0.26, 0.26,
                              fill=False, edgecolor="#16a34a", linewidth=1.5, linestyle=":")
    ax.add_patch(center_region)
    ax.text(gt_center[0], gt_center[1] + 0.18, "center region\npenalty=0 inside", ha="center", fontsize=7.1, color="#166534")

    ax.text(0.50, 0.08, "L_box judges predicted box quality; center penalty judges whether the candidate point should own this object",
            ha="center", fontsize=7.4, color="#334155")


def _draw_scaling_demo(ax):
    ax.set_title("Model scaling", fontsize=11, weight="bold")
    ax.set_xlabel("width multiplier")
    ax.set_ylabel("relative compute")
    widths = np.array([0.5, 0.75, 1.0, 1.25])
    compute = widths ** 2 * np.array([0.7, 0.9, 1.0, 1.15])
    ax.plot(widths, compute, marker="o", color="#2563eb", linewidth=2)
    ax.grid(True, alpha=0.25)


def _draw_c2f_demo(ax):
    ax.set_title("C2f feature reuse", fontsize=11, weight="bold")
    ax.axis("off")
    xs = [0.4, 2.4, 4.4, 6.4]
    labels = ["split", "bottleneck 1", "bottleneck 2", "concat"]
    for x,label in zip(xs,labels):
        ax.add_patch(Rectangle((x,1.8),1.5,0.7,facecolor="#dcfce7",edgecolor="#16a34a",linewidth=1.5))
        ax.text(x+.75,2.15,label,ha="center",va="center",fontsize=8.5,weight="bold")
    for x1,x2 in [(1.9,2.4),(3.9,4.4),(5.9,6.4)]:
        ax.annotate("", xy=(x2,2.15), xytext=(x1,2.15), arrowprops={"arrowstyle":"-|>"})
    ax.text(4.0,1.25,"intermediate features are reused before concat",ha="center",fontsize=9)
    ax.set_xlim(0,8.4)
    ax.set_ylim(1.0,3.0)


def _draw_dfl_demo(ax):
    ax.set_title("Distribution Focal Loss intuition", fontsize=11, weight="bold")
    bins = np.arange(8)
    probs = np.exp(-0.5 * (bins - 3.4) ** 2)
    probs = probs / probs.sum()
    ax.bar(bins, probs, color="#93c5fd", edgecolor="#2563eb")
    ax.axvline((bins * probs).sum(), color="#dc2626", linewidth=2, label="expected distance")
    ax.set_xlabel("distance bin")
    ax.set_ylabel("probability")
    ax.legend()


def _draw_pipeline_demo(ax):
    ax.set_title("Training / inference / export pipeline", fontsize=11, weight="bold")
    ax.axis("off")
    labels = ["dataset", "train", "validate", "predict", "export"]
    for i,label in enumerate(labels):
        x = 0.35 + i*1.55
        ax.add_patch(Rectangle((x,1.7),1.2,0.7,facecolor="#dbeafe",edgecolor="#2563eb",linewidth=1.4))
        ax.text(x+.6,2.05,label,ha="center",va="center",fontsize=8.5,weight="bold")
        if i < len(labels)-1:
            ax.annotate("", xy=(x+1.55,2.05), xytext=(x+1.2,2.05), arrowprops={"arrowstyle":"-|>"})
    ax.set_xlim(0,8.2)
    ax.set_ylim(1.2,3.0)


def _draw_efficiency_demo(ax):
    ax.set_title("Efficiency trade-off", fontsize=11, weight="bold")
    names = ["model", "postprocess", "export", "runtime"]
    vals = [0.45, 0.25, 0.15, 0.15]
    ax.bar(names, vals, color=["#2563eb", "#16a34a", "#d97706", "#dc2626"])
    ax.set_ylabel("latency / complexity share")
    ax.set_ylim(0, 0.6)
    ax.grid(True, axis="y", alpha=0.25)


def _draw_optimizer_demo(ax):
    ax.set_title("Optimizer step", fontsize=11, weight="bold")
    ax.axis("off")
    labels = ["gradient", "momentum / update rule", "new weights"]
    xs = [0.7, 3.0, 6.0]
    for x,label in zip(xs,labels):
        ax.add_patch(Rectangle((x,1.7),1.8,0.75,facecolor="#fef3c7",edgecolor="#d97706",linewidth=1.5))
        ax.text(x+.9,2.075,label,ha="center",va="center",fontsize=8.5,weight="bold")
    ax.annotate("", xy=(3.0,2.075), xytext=(2.5,2.075), arrowprops={"arrowstyle":"-|>"})
    ax.annotate("", xy=(6.0,2.075), xytext=(4.8,2.075), arrowprops={"arrowstyle":"-|>"})
    ax.set_xlim(0,8.5)
    ax.set_ylim(1.2,3.0)


_ELEMENT_DRAWERS = {
    "anchor": _draw_anchor_demo,
    "kmeans": _draw_iou_distance_demo,
    "bn": _draw_bn_demo,
    "passthrough": _draw_passthrough_demo,
    "hierarchy": _draw_hierarchy_demo,
    "residual": _draw_residual_demo,
    "multiscale": _draw_multiscale_demo,
    "fusion": _draw_spp_pan_demo,
    "yolov3_fusion": _draw_yolov3_fusion_demo,
    "logistic": _draw_logistic_demo,
    "csp": _draw_csp_demo,
    "spp_pan": _draw_spp_pan_demo,
    "mish": lambda ax: _draw_activation_demo(ax, "mish"),
    "ciou": _draw_ciou_demo,
    "mosaic": _draw_mosaic_demo,
    "pipeline": _draw_pipeline_demo,
    "focus": _draw_focus_demo,
    "reparam": _draw_reparam_demo,
    "decoupled": _draw_decoupled_head_demo,
    "anchor_point_grid": _draw_anchor_point_grid_demo,
    "assignment": _draw_assignment_demo,
    "efficiency": _draw_efficiency_demo,
    "elan": _draw_elan_demo,
    "scaling": _draw_scaling_demo,
    "anchor_free": _draw_anchor_free_demo,
    "c2f": _draw_c2f_demo,
    "dfl": _draw_dfl_demo,
    "pgi": _draw_pgi_demo,
    "nms_free": _draw_nms_free_demo,
    "attention": lambda ax: _draw_attention_demo(ax, area=False),
    "area_attention": lambda ax: _draw_attention_demo(ax, area=True),
    "hypergraph": _draw_hypergraph_demo,
    "deployment": _draw_deployment_demo,
    "optimizer": _draw_optimizer_demo,
}


def display_yolo_element_demo(version, element):
    """Draw a compact demo for one named technical element."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle as _Rectangle
    except ImportError as exc:
        raise ImportError("Install matplotlib to display YOLO element demos") from exc

    globals()["Rectangle"] = _Rectangle
    if element not in _ELEMENT_DRAWERS:
        raise ValueError(f"No element demo for {version=}, {element=}")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    _ELEMENT_DRAWERS[element](ax)
    fig.tight_layout()
    return _display_figure(fig)
