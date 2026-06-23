"""Legacy placeholder for the former notebook generator.

The tutorial notebook is now maintained directly because it contains detailed,
hand-written per-element explanations. Use common.runner.main from the notebook.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_notebook():
    notebook = ROOT / "yolo_history_tutorial.ipynb"
    if not notebook.exists():
        raise FileNotFoundError(notebook)
    return notebook


if __name__ == "__main__":
    print(write_notebook())
