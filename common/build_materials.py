"""Regenerate thin version entry points and the course notebook."""

from __future__ import annotations

import json
from pathlib import Path

from common.versions import VERSIONS

ROOT = Path(__file__).resolve().parents[1]


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def write_entry_points():
    template = '''"""Run the {title} lesson, demo, or inference adapter."""\nfrom common.runner import main\n\nif __name__ == "__main__":\n    main({version})\n'''
    for version, info in VERSIONS.items():
        (ROOT / f"yolov{version}.py").write_text(template.format(title=info["title"], version=version), encoding="utf-8")


def write_notebook():
    cells = [
        markdown("""# YOLOの進化を動かして理解する

## 全体概要

YOLOは **画像全体を一度に見て物体の位置と種類を予測する one-stage detector** として始まりました。ただし、v1から現在までを管理する単一組織はありません。v5以降の番号は異なるチームが提案した系列を含むため、番号が大きいだけで常に同じ設計の直接後継、または全用途で優秀、という意味ではありません。

本教材は番号付きの主要リリースを v1→v13 の順に比較し、最後に Ultralytics の現行製品系列 YOLO26 を扱います。**v14〜v25という公開版があるわけではなく、YOLO26は年に合わせた命名**です。YOLOX、YOLO-NAS、RT-DETRなどの重要な派生は、番号順という今回の範囲から除外します。

### 学び方

1. 各節の「追加技術」と「使えるようになった場面」を読む
2. NumPyだけの小規模デモで中心概念を確認する
3. 対応する公式実装・重みで実画像推論を行う

小規模デモは本物のネットワーク精度を再現するものではなく、差分となる演算・データ表現を分離して観察する教材です。精度や速度の数値比較には、同じデータ、解像度、ハードウェア、測定条件が必要です。"""),
        code("""from pathlib import Path
import sys, json, subprocess

ROOT = Path.cwd()
if not (ROOT / 'common').exists():
    raise RuntimeError('このノートブックをリポジトリ直下から実行してください')
sys.path.insert(0, str(ROOT))

from common.versions import VERSIONS
from common.demos import run_demo
print('収録:', ', '.join(VERSIONS[v]['title'] for v in VERSIONS))"""),
        markdown("""## 推論環境

技術デモのみ: `python -m pip install -r requirements-demo.txt`

画像推論も行う: `python -m pip install -r requirements.txt`

- v1〜v4: Darknet形式の `.cfg` と `.weights` を用意し、OpenCV DNNで実行
- v5/v8〜v12/YOLO26: `ultralytics` が読める対応 `.pt` を指定
- v6/v7/v13: 公式リポジトリごとに環境が異なるので、その推論コマンドを `--command` 経由で実行

推論だけならデータセットは不要で、手元のJPEG/PNGを1枚使えます。学習用データは末尾を参照してください。"""),
    ]
    for version, info in VERSIONS.items():
        lineage = "## 補足: " if version == 26 else "## "
        links = f"[公式/著者実装]({info['repo']})"
        if info.get("paper"):
            links += f" / [論文]({info['paper']})"
        if info.get("docs"):
            links += f" / [公式ドキュメント]({info['docs']})"
        cells.append(markdown(f"""{lineage}{info['title']} ({info['year']})

**新しく加わった技術:** {info['innovation']}。

**その結果使いやすくなった場面:** {info['use_case']}。

**注意:** この節のデモは `{info['demo']}` の要点だけを可視化します。モデル全体の再実装ではありません。

出典: {links}"""))
        cells.append(code(f"""# {info['title']} の技術差分デモ
print(json.dumps(run_demo({version}), ensure_ascii=False, indent=2))"""))
        if info["backend"] == "darknet":
            infer = f"python yolov{version}.py --mode infer --image data/sample.jpg --config path/to/yolov{version}.cfg --weights path/to/yolov{version}.weights"
        elif info["backend"] == "ultralytics":
            example = {5:"yolov5nu.pt",8:"yolov8n.pt",9:"yolov9c.pt",10:"yolov10n.pt",11:"yolo11n.pt",12:"yolo12n.pt",26:"yolo26n.pt"}.get(version, "model.pt")
            infer = f"python yolov{version}.py --mode infer --image data/sample.jpg --weights {example} --save"
        else:
            infer = f"python yolov{version}.py --mode infer --image data/sample.jpg --weights path/to/model --command 'python path/to/official/infer.py --weights {{weights}} --source {{image}}'"
        cells.append(markdown(f"""### 実際の推論体験

まず別ターミナルで次を実行します（重み名・公式CLIの引数は、リンク先で取得した版に合わせてください）。

```bash
{infer}
```

`--mode info` は世代情報、`--mode demo` は上と同じ小規模デモを端末に表示します。外部実装用の `--command` はシェルを介さず引数列として実行されます。信頼できるコマンドだけを指定してください。"""))
    cells += [
        markdown("""## 同じ画像で結果を比較する

以下では選んだ世代のスクリプトを実行します。`VERSION`, `IMAGE`, `WEIGHTS` を変更してください。v1〜v4では `CONFIG` も必要です。実行前に、上の各節にあるバックエンド条件を確認してください。"""),
        code("""VERSION = 8
IMAGE = 'data/sample.jpg'
WEIGHTS = 'yolov8n.pt'
CONFIG = None

cmd = [sys.executable, f'yolov{VERSION}.py', '--mode', 'infer', '--image', IMAGE, '--weights', WEIGHTS, '--save']
if CONFIG:
    cmd += ['--config', CONFIG]
print(' '.join(cmd))
# ダウンロードと推論を実行するときに次行のコメントを外す
# subprocess.run(cmd, check=True)"""),
        markdown("""## データセット

**推論体験だけならダウンロード不要**です。`data/sample.jpg` に任意の写真を置いてください。

独自学習まで進む場合の小規模候補:

- [COCO128](https://docs.ultralytics.com/datasets/detect/coco128/): COCO train2017の先頭128枚。パイプライン確認向けで、性能評価には小さすぎます。
- [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/): v1/v2時代との比較に適した20クラス。
- [COCO](https://cocodataset.org/#download): 標準的な80クラス。容量・学習時間が大きい本評価向け。

データ利用条件と画像の権利を確認してください。世代比較では同じtrain/validation分割、入力解像度、augmentation、評価指標（通常 COCO mAP）を固定します。"""),
        markdown("""## まとめ

- v1〜v3: 統一回帰 → anchor → multi-scale と、検出器の基本形を確立
- v4〜v7: 学習手法、特徴融合、再パラメータ化で速度精度比を改善
- v8〜v10: anchor-free、勾配設計、NMS-free end-to-endへ移行
- v11〜v13: attentionと高次特徴相関をリアルタイム制約へ導入
- YOLO26: 番号連番ではない製品系列として、配備とend-to-end効率を重視

「最新版」を選ぶ前に、ライセンス、対象タスク、対応export形式、実測レイテンシ、保守性を同じ重要度で確認してください。"""),
    ]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.9"}}, "nbformat": 4, "nbformat_minor": 5}
    (ROOT / "yolo_history_tutorial.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_entry_points()
    write_notebook()
