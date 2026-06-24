"""Small demos for related techniques used in the notebook."""

from __future__ import annotations

__all__ = [
    "box_iou_xyxy",
    "darknet_file_roles",
    "display_darknet_format_demo",
    "display_leaky_relu_demo",
    "display_nms_demo",
    "leaky_relu",
    "leaky_relu_demo_values",
    "nms_demo_boxes",
    "relu",
    "run_nms_demo",
]

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


def relu(x):
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0)


def leaky_relu(x, alpha=0.1):
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x, alpha * x)


def leaky_relu_demo_values(alpha=0.1):
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    return {
        "alpha": alpha,
        "x": x.tolist(),
        "relu(x)": relu(x).tolist(),
        "leaky_relu(x)": leaky_relu(x, alpha).tolist(),
    }


def display_leaky_relu_demo(alpha=0.1):
    """Plot ReLU and Leaky ReLU curves and their gradients."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install matplotlib to display the Leaky ReLU demo") from exc

    x = np.linspace(-4, 4, 401)
    y_relu = relu(x)
    y_leaky = leaky_relu(x, alpha)
    grad_relu = np.where(x > 0, 1.0, 0.0)
    grad_leaky = np.where(x > 0, 1.0, alpha)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(x, y_relu, label="ReLU", linewidth=2)
    axes[0].plot(x, y_leaky, label=f"Leaky ReLU (alpha={alpha})", linewidth=2)
    axes[0].axhline(0, color="#9ca3af", linewidth=1)
    axes[0].axvline(0, color="#9ca3af", linewidth=1)
    axes[0].set_title("Activation")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, grad_relu, label="ReLU gradient", linewidth=2)
    axes[1].plot(x, grad_leaky, label="Leaky ReLU gradient", linewidth=2)
    axes[1].axvline(0, color="#9ca3af", linewidth=1)
    axes[1].set_title("Gradient")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("df/dx")
    axes[1].set_ylim(-0.05, 1.1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    return _display_figure(fig)


def darknet_file_roles():
    """Return a tiny summary of Darknet-format files used by old YOLO versions."""
    return {
        "cfg": "Network structure and inference settings: layers, filters, anchors, classes, input size.",
        "weights": "Binary learned parameters: convolution kernels, biases, batch-normalization statistics.",
        "data/names": "Optional class-name metadata used by Darknet-style tooling.",
    }


def display_darknet_format_demo():
    """Draw how Darknet cfg/weights files are loaded for inference."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, FancyArrowPatch
    except ImportError as exc:
        raise ImportError("Install matplotlib to display the Darknet format diagram") from exc

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    def block(x, y, w, h, title, body, fc, ec):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.8))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=11, weight="bold")
        ax.text(x + 0.16, y + h - 0.68, body, ha="left", va="top", fontsize=9, linespacing=1.35)

    def arrow(x1, y1, x2, y2, label):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color="#475569"))
        ax.text((x1 + x2) / 2, y1 + 0.16, label, ha="center", va="bottom", fontsize=8.5, color="#334155")

    block(0.4, 2.35, 2.2, 1.45, ".cfg", "text file\nnetwork layers\nanchors/classes\ninput size", "#dbeafe", "#2563eb")
    block(0.4, 0.65, 2.2, 1.45, ".weights", "binary file\nlearned params\nkernels/biases\nBN statistics", "#dcfce7", "#16a34a")
    arrow(2.75, 3.05, 4.0, 2.55, "read")
    arrow(2.75, 1.35, 4.0, 2.05, "read")

    block(4.15, 1.35, 2.45, 1.85, "Darknet loader", "build model graph\nthen fill layers\nwith learned weights", "#fef3c7", "#d97706")
    arrow(6.78, 2.27, 8.0, 2.27, "run image")

    block(8.15, 1.35, 2.25, 1.85, "Detector", "forward pass\nraw boxes\nclass scores", "#fee2e2", "#dc2626")
    arrow(10.58, 2.27, 11.5, 2.27, "NMS")

    block(11.65, 1.35, 1.1, 1.85, "Final\ndetections", "boxes\nlabels\nscores", "#f3e8ff", "#9333ea")

    ax.text(6.5, 4.22, "Darknet format = architecture file (.cfg) + learned weights file (.weights)", ha="center", va="top", fontsize=12, weight="bold")
    fig.tight_layout()
    return _display_figure(fig)


def nms_demo_boxes():
    """Return small overlapping boxes for an NMS demonstration."""
    return [
        {"id": "A", "xyxy": [0.12, 0.18, 0.58, 0.72], "score": 0.92, "label": "person"},
        {"id": "B", "xyxy": [0.18, 0.22, 0.62, 0.75], "score": 0.84, "label": "person"},
        {"id": "C", "xyxy": [0.55, 0.20, 0.88, 0.62], "score": 0.76, "label": "person"},
        {"id": "D", "xyxy": [0.58, 0.24, 0.90, 0.65], "score": 0.63, "label": "person"},
    ]


def box_iou_xyxy(box, boxes):
    box = np.asarray(box, dtype=float)
    boxes = np.asarray(boxes, dtype=float)
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    wh = np.maximum(rb - lt, 0.0)
    inter = wh[:, 0] * wh[:, 1]
    area_box = np.prod(box[2:] - box[:2])
    area_boxes = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    return inter / np.maximum(area_box + area_boxes - inter, 1e-12)


def run_nms_demo(boxes=None, iou_threshold=0.5):
    """Run class-agnostic NMS on a tiny box set and expose every decision."""
    boxes = list(boxes or nms_demo_boxes())
    order = sorted(range(len(boxes)), key=lambda i: boxes[i]["score"], reverse=True)
    kept = []
    suppressed = []
    steps = []

    while order:
        current = order.pop(0)
        current_box = boxes[current]
        kept.append(current_box["id"])
        remaining = []
        comparisons = []
        for idx in order:
            other = boxes[idx]
            iou = float(box_iou_xyxy(current_box["xyxy"], np.array([other["xyxy"]]))[0])
            suppress = iou >= iou_threshold
            comparisons.append({"with": other["id"], "iou": iou, "suppressed": suppress})
            if suppress:
                suppressed.append(other["id"])
            else:
                remaining.append(idx)
        steps.append({"keep": current_box["id"], "comparisons": comparisons})
        order = remaining

    return {"iou_threshold": iou_threshold, "boxes": boxes, "kept": kept, "suppressed": suppressed, "steps": steps}


def display_nms_demo(nms_result):
    """Visualize candidate boxes before and after NMS."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise ImportError("Install matplotlib to display the NMS diagram") from exc

    boxes = nms_result["boxes"]
    kept = set(nms_result["kept"])
    colors = {"kept": "#16a34a", "suppressed": "#dc2626", "candidate": "#2563eb"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, title in zip(axes, ["Before NMS: overlapping candidates", "After NMS: keep high-score boxes"]):
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        axes[0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2.2, edgecolor=colors["candidate"]))
        axes[0].text(x1, y1 - 0.015, f"{b['id']} {b['score']:.2f}", color=colors["candidate"], fontsize=9, weight="bold")

    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        is_kept = b["id"] in kept
        color = colors["kept"] if is_kept else colors["suppressed"]
        alpha = 1.0 if is_kept else 0.35
        axes[1].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2.4, edgecolor=color, alpha=alpha))
        label = "keep" if is_kept else "suppress"
        axes[1].text(x1, y1 - 0.015, f"{b['id']} {label}", color=color, fontsize=9, weight="bold", alpha=alpha)

    fig.tight_layout()
    return _display_figure(fig)
