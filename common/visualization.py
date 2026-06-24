"""Notebook visualization helpers for inference results."""

from __future__ import annotations

__all__ = [
    "display_image_grid",
    "latest_detect_dir",
    "prediction_images_for",
]

from pathlib import Path


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


def _load_image(path: Path):
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Install pillow to display image grids") from exc
    return Image.open(path).convert("RGB")


def display_image_grid(image_paths, titles=None, columns: int = 3, figsize=(14, 8)):
    """Display images in a compact matplotlib grid."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install matplotlib to display image grids") from exc

    paths = [Path(p) for p in image_paths]
    if not paths:
        raise ValueError("image_paths is empty")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing image(s): " + ", ".join(missing))

    titles = list(titles) if titles is not None else [p.name for p in paths]
    columns = max(1, min(columns, len(paths)))
    rows = (len(paths) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")
    for ax, path, title in zip(axes.ravel(), paths, titles):
        ax.imshow(_load_image(path))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.tight_layout()
    return _display_figure(fig)


def latest_detect_dir(runs_dir="runs/detect"):
    """Return the newest Ultralytics detect output directory."""
    runs_path = Path(runs_dir)
    candidates = [p for p in runs_path.glob("predict*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No predict* directories found under {runs_path}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def prediction_images_for(source_images, runs_dir="runs/detect", detect_dir=None):
    """Find saved prediction images matching source image basenames."""
    detect_path = Path(detect_dir) if detect_dir is not None else latest_detect_dir(runs_dir)
    outputs = []
    for src in source_images:
        src_path = Path(src)
        candidates = [
            detect_path / src_path.name,
            detect_path / f"{src_path.stem}.jpg",
            detect_path / f"{src_path.stem}.png",
            detect_path / f"{src_path.stem}.jpeg",
        ]
        match = next((p for p in candidates if p.exists()), None)
        if match is None:
            raise FileNotFoundError(f"No saved prediction image for {src_path.name} in {detect_path}")
        outputs.append(match)
    return outputs
