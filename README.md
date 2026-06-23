# YOLO version-by-version tutorial

解説本体は `yolo_history_tutorial.ipynb` にあります。ノートブックでは `common.runner.main` を直接呼び出して、各世代の情報表示、技術デモ、実画像推論を行います。

```python
from common.runner import main as run_yolo

run_yolo(1, ['--mode', 'info'])
run_yolo(10, ['--mode', 'demo'])
run_yolo(8, ['--mode', 'infer', '--image', 'images/sample.jpg', '--weights', 'yolov8n.pt', '--save'])
```

v5以降は単一チームの直系ではありません。番号付き主要研究版はv13まで、Ultralytics製品系列の現行版はYOLO26です。v14〜v25は存在しません。

## データ

推論だけならデータセットは不要です。任意のJPEG/PNGを `images/sample.jpg` として置けます。学習には [COCO128](https://docs.ultralytics.com/datasets/detect/coco128/)（動作確認）、[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)（古い世代との比較）、[COCO](https://cocodataset.org/#download)（本評価）が候補です。

重みと設定ファイルは各節に記載した公式/著者実装から取得してください。各プロジェクトのライセンスは異なります。
