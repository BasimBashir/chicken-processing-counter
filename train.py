"""
YOLO Training Script for Slaughtered Chicken Counting
3-class detection: empty_shackles, single_legged, slaughtered_chicken

Dataset setup:
    1. Unzip "Birds Counting.v1i.yolo26.zip" into the project root.
       The zip should produce a folder structure like:
           dataset/
               train/images/  train/labels/
               valid/images/  valid/labels/
               test/images/   test/labels/   (optional)
               data.yaml

    2. Run:  python train.py

After training, best weights are saved to:
    runs/chicken_counter/weights/best.pt

Copy to project root to deploy:
    copy runs\\chicken_counter\\weights\\best.pt best.pt   (Windows)
    cp runs/chicken_counter/weights/best.pt best.pt        (Linux/macOS)
"""

import os
from ultralytics import YOLO


def main():
    # Absolute path so Ultralytics resolves the relative train/val paths inside
    # data.yaml correctly regardless of working directory
    data_path = os.path.abspath(os.path.join("dataset", "data.yaml"))

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"data.yaml not found at '{data_path}'.\n"
            "Unzip 'Birds Counting.v1i.yolo26.zip' into the project root first.\n"
            "Expected structure:\n"
            "  dataset/\n"
            "    data.yaml\n"
            "    train/images/  train/labels/\n"
            "    valid/images/  valid/labels/"
        )

    # Load YOLO26s pretrained on COCO (transfer learning)
    model = YOLO("yolo26m.pt")

    model.train(
        data=data_path,
        epochs=50,
        imgsz=1280,
        multi_scale=0.3,
        batch=-1,
        patience=10,
        device=0,               # GPU 0; change to "cpu" if no GPU

        # Optimizer
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,             # conveyor moves horizontally — no vertical flip
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,

        # Scheduler / precision
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        cache="disk",
        workers=8,
        seed=42,
        verbose=True,
        plots=True,

        # Output
        project="runs",
        name="chicken_counter",
        exist_ok=True,
    )

    metrics = model.val()
    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    best_path = "runs/chicken_counter/weights/best.pt"
    print(f"\nBest model saved to: {best_path}")
    print("Copy best.pt to the project root to use with the counting app:")
    print("  copy runs\\chicken_counter\\weights\\best.pt best.pt")


if __name__ == "__main__":
    main()
