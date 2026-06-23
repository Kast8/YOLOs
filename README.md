# YOLO version-by-version tutorial

`yolov1.py`〜`yolov13.py` と `yolov26.py` は、それぞれの世代について情報表示、技術デモ、実画像推論を行う入口です。解説本体は `yolo_history_tutorial.ipynb` にあります。

```bash
python yolov1.py --mode info
python yolov10.py --mode demo
python -m pip install -r requirements.txt
python yolov8.py --mode infer --image data/sample.jpg --weights yolov8n.pt --save
```

v5以降は単一チームの直系ではありません。番号付き主要研究版はv13まで、Ultralytics製品系列の現行版はYOLO26です。v14〜v25は存在しないためファイルも作っていません。

## データ

推論だけならデータセットは不要です。任意のJPEG/PNGを `data/sample.jpg` として置けます。学習には [COCO128](https://docs.ultralytics.com/datasets/detect/coco128/)（動作確認）、[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)（古い世代との比較）、[COCO](https://cocodataset.org/#download)（本評価）が候補です。

重みと設定ファイルは各節に記載した公式/著者実装から取得してください。各プロジェクトのライセンスは異なります。

## 再生成

```bash
python -m common.build_materials
```

