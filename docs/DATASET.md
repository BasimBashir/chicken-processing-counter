# Dataset & Training Guide — Chicken Counter

The model is the limiter for **recall** (missed birds → undercount). The single
biggest lever is **training on data that matches the deployed stream**, covering
the hard conditions. Hyperparameters are secondary.

## 0. The golden rule: match the deployment domain
Models trained on sharp/high-res frames **degrade on a lower-res, compressed
stream** (domain gap). The API now processes every stream at a **fixed 1280×720**,
so:
- Capture training frames **from that exact pipeline output** (the 1280×720 the
  API produces), not from 4K originals.
- Include the **failure conditions** the live system misses: belt **stopped**,
  **slow**, **fast / motion-blurred**, **dense / occluded**, and the lighting of
  those shifts. The current misses are concentrated here.

## 1. Roboflow — Preprocessing
| Setting | Value | Why |
|---|---|---|
| Auto-Orient | **On** | strip EXIF rotation |
| Resize | **Stretch to 1280×1280** | match `imgsz=1280` and the API's fixed resize; do NOT downsize below 1280 |
| Tiling | **Off** | birds are ~100px, not tiny (<32px); tiling only helps tiny objects |
| Grayscale / auto-contrast | **Off** | keep natural color |

## 2. Roboflow — Augmentation (light & realistic: 1–2 per image, ~3× dataset)
Keep ✅:
- **Horizontal flip** (birds ~symmetric)
- **Brightness ±15–25%**, **Exposure ±10%** (plant lighting drifts)
- **Blur (light) + Motion blur** ← important: fast-belt motion blur is a likely
  cause of missed birds
- **Noise** (small) — emulates stream compression artifacts

Use sparingly / avoid:
- ⚠️ **Rotation ±3–5°** max, or skip (camera is fixed)
- ❌ **Vertical flip, 90° rotation, shear, heavy crop, strong hue shift**
- ❌ **Do NOT add Roboflow Mosaic** — YOLO already does mosaic at train time
  (doubling it distorts the distribution)

## 3. Training (`train.py`)
Already good: `yolo26m`, `imgsz=1280`, `multi_scale=0.3`, `mosaic=1.0`,
`close_mosaic=10`, `flipud=0.0`. Applied tweaks: `degrees=3.0` (fixed camera),
`hsv_v=0.5`, `erasing=0.4` (occlusion robustness).
- Keep **mosaic on** — it helps dense/small-object recall (Ultralytics guidance).
- Consider **`yolo26l`** for more capacity on the dense, occluded chain (3090 can
  handle it).
- With the expanded dataset, raise **`epochs` to ~100**.
- Optimise for **recall**: after training, read the PR curve and set the inference
  confidence at the point that **maximises recall** at acceptable precision —
  don't just keep 0.42.

## 4. Validation discipline (so the numbers mean something)
- Compare to BAADER over **matched windows**. ⚠️ The BAADER↔camera 1:52 offset is
  only valid at **constant belt speed** — during **stops** the travel time
  changes, so BAADER's window and the camera's window contain **different birds**.
  Validate model recall on **stop-free windows**, or align by belt position, not a
  fixed time offset.
- Counting accuracy and detection accuracy are **separate**: confirm by eye that
  missed counts are *visible un-boxed birds* (detection → retrain) vs birds simply
  not in the window (alignment → not a model problem).

## Sources
- Ultralytics — [Data augmentation guide](https://docs.ultralytics.com/guides/yolo-data-augmentation), [Model evaluation insights](https://docs.ultralytics.com/guides/model-evaluation-insights)
- Roboflow — [Preprocessing](https://docs.roboflow.com/datasets/dataset-versions/image-preprocessing), [Augmentation](https://docs.roboflow.com/datasets/dataset-versions/image-augmentation)
