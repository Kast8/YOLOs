"""CLI shared by yolov1.py ... yolov13.py."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from common.demos import run_demo
from common.versions import VERSIONS


def _darknet_infer(args):
    if not args.config or not args.weights:
        raise SystemExit("Darknet inference needs --config CFG and --weights WEIGHTS")
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Install opencv-python for the Darknet backend") from exc
    net = cv2.dnn.readNetFromDarknet(args.config, args.weights)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(args.imgsz, args.imgsz), scale=1 / 255, swapRB=True)
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")
    ids, scores, boxes = model.detect(image, args.conf, args.iou)
    result = [{"class_id": int(i), "score": float(s), "xywh": list(map(int, b))}
              for i, s, b in zip(ids, scores, boxes)]
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _ultralytics_infer(args):
    if not args.weights:
        raise SystemExit("Inference needs --weights (for example yolov8n.pt)")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install ultralytics for this backend") from exc
    model = YOLO(args.weights)
    results = model.predict(args.image, imgsz=args.imgsz, conf=args.conf, iou=args.iou, save=args.save)
    print(json.dumps([json.loads(r.to_json()) for r in results], ensure_ascii=False, indent=2))


def _external_infer(args):
    if not args.command:
        raise SystemExit("This generation uses its official repository. Pass its inference command after --command, using {image}/{weights} placeholders.")
    values = {"image": args.image or "", "weights": args.weights or "", "imgsz": args.imgsz,
              "conf": args.conf, "iou": args.iou}
    command = [part.format(**values) for part in shlex.split(args.command)]
    subprocess.run(command, check=True)


def main(version: int):
    info = VERSIONS[version]
    parser = argparse.ArgumentParser(description=f"{info['title']}: {info['innovation']}")
    parser.add_argument("--mode", choices=["info", "demo", "infer"], default="info")
    parser.add_argument("--image")
    parser.add_argument("--weights")
    parser.add_argument("--config", help="Darknet .cfg (v1-v4)")
    parser.add_argument("--backend", choices=["auto", "darknet", "ultralytics", "external"], default="auto")
    parser.add_argument("--command", help="Official repo command template for external backend")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--iou", type=float, default=.45)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    if args.mode == "info":
        print(json.dumps(info, ensure_ascii=False, indent=2))
    elif args.mode == "demo":
        print(json.dumps(run_demo(version), ensure_ascii=False, indent=2))
    else:
        if not args.image:
            parser.error("--mode infer requires --image")
        backend = info["backend"] if args.backend == "auto" else args.backend
        {"darknet": _darknet_infer, "ultralytics": _ultralytics_infer, "external": _external_infer}[backend](args)

