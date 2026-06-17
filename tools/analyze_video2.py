"""
Deeper analysis of mix.mp4:
1. Stop-event duration histogram — distinguish chicken gaps from true belt stops
2. Chicken centroid speed via background subtraction + contour tracking
3. Recommended parameter values
"""
import cv2
import numpy as np
import sys
from collections import Counter

VIDEO = r"C:\Users\mianx\Downloads\mix.mp4"
PROC_W, PROC_H = 1280, 720
SMALL_W, SMALL_H = 160, 90
STOP_THRESH = 0.4

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    sys.exit(f"Cannot open {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
print(f"fps={fps}")

prev_small = None
motion_vals = []

# Background subtractor to isolate moving foreground (chickens)
bg_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)

chicken_speeds = []       # x-displacement per frame in proc-space
frame_idx = 0
prev_centroids = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    frame = cv2.resize(frame, (PROC_W, PROC_H))
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (SMALL_W, SMALL_H))

    # Motion diff
    if prev_small is not None:
        motion_vals.append(float(cv2.absdiff(small, prev_small).mean()))
    prev_small = small

    # Foreground mask → chicken centroids
    fg = bg_sub.apply(frame)
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.dilate(fg, kernel, iterations=2)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 800:  # skip small noise
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroids.append((cx, cy, area))

    # Match centroids frame-to-frame by proximity → get x-displacement
    if prev_centroids and centroids:
        for (px, py, _) in prev_centroids:
            best_dist = 9999
            best_dx = None
            for (cx, cy, _) in centroids:
                dist = abs(cx - px) + abs(cy - py)
                if dist < best_dist and dist < 80:  # max 80px between frames
                    best_dist = dist
                    best_dx = cx - px
            if best_dx is not None and best_dx > 0:  # belt moves left→right
                chicken_speeds.append(best_dx)

    prev_centroids = centroids

cap.release()

motion_arr = np.array(motion_vals)
speed_arr  = np.array(chicken_speeds)

# ── Stop-event duration analysis ──────────────────────────────────────────────
stopped = motion_arr < STOP_THRESH

# Run-length encode to get duration of each stopped/running segment
runs = []
current_val = stopped[0]
current_len = 1
for v in stopped[1:]:
    if v == current_val:
        current_len += 1
    else:
        runs.append((bool(current_val), current_len))
        current_val = v
        current_len = 1
runs.append((bool(current_val), current_len))

stop_runs   = [r[1] for r in runs if r[0]]   # frames where belt looks stopped
moving_runs = [r[1] for r in runs if not r[0]]

print(f"\n=== STOPPED SEGMENTS (frames below stop_thresh={STOP_THRESH}) ===")
print(f"  Total stop segments: {len(stop_runs)}")
if stop_runs:
    sr = np.array(stop_runs)
    print(f"  Duration distribution (frames):")
    for bucket, lo, hi in [("1-4 frames  (< {:.0f}ms)".format(4/fps*1000),   1,  4),
                            ("5-15 frames (< {:.0f}ms)".format(15/fps*1000),  5, 15),
                            ("16-30 frames(< {:.0f}ms)".format(30/fps*1000), 16, 30),
                            ("31-90 frames(< {:.1f}s)".format(90/fps),       31, 90),
                            ("> 90 frames (> {:.1f}s)".format(90/fps),       91, 99999)]:
        count = int(((sr >= lo) & (sr <= hi)).sum())
        pct   = 100 * count / len(sr)
        print(f"    {bucket}: {count}  ({pct:.1f}%)")
    print(f"  min={sr.min()}  p50={np.median(sr):.0f}  p90={np.percentile(sr,90):.0f}  p99={np.percentile(sr,99):.0f}  max={sr.max()}")
    print(f"  In seconds: p50={np.median(sr)/fps:.2f}s  p90={np.percentile(sr,90)/fps:.2f}s  max={sr.max()/fps:.1f}s")

print(f"\n=== MOVING SEGMENTS ===")
if moving_runs:
    mr = np.array(moving_runs)
    print(f"  Total: {len(mr)}  p50={np.median(mr):.0f}  p90={np.percentile(mr,90):.0f} frames")
    print(f"  In seconds: p50={np.median(mr)/fps:.2f}s")

# ── Chicken/bird speed from contour tracking ───────────────────────────────────
print(f"\n=== CHICKEN CENTROID X-SPEED (proc-space px/frame) — {len(speed_arr)} matched pairs ===")
if len(speed_arr) >= 10:
    print(f"  min:    {speed_arr.min():.1f}")
    print(f"  p10:    {np.percentile(speed_arr, 10):.1f}")
    print(f"  p25:    {np.percentile(speed_arr, 25):.1f}")
    print(f"  median: {np.median(speed_arr):.1f}")
    print(f"  p75:    {np.percentile(speed_arr, 75):.1f}")
    print(f"  p90:    {np.percentile(speed_arr, 90):.1f}")
    print(f"  p99:    {np.percentile(speed_arr, 99):.1f}")
    print(f"  max:    {speed_arr.max():.1f}")
else:
    print("  Not enough matched pairs — contour tracking may need tuning")

# ── Final recommendations ──────────────────────────────────────────────────────
print("\n=== CALIBRATED RECOMMENDATIONS ===")

# Belt stop: threshold for "true stop" = runs lasting > 1 second
TRUE_STOP_FRAMES = int(fps)  # 1 second
true_stops = [r for r in stop_runs if r >= TRUE_STOP_FRAMES]
chicken_gaps = [r for r in stop_runs if r < TRUE_STOP_FRAMES]
print(f"  Chicken inter-bird gaps (< 1s): {len(chicken_gaps)} events, "
      f"max={max(chicken_gaps) if chicken_gaps else 0} frames "
      f"({max(chicken_gaps)/fps*1000:.0f}ms)" if chicken_gaps else "  No short gaps found")
print(f"  True belt stops (>= 1s):        {len(true_stops)} events, "
      f"min={min(true_stops) if true_stops else 0} frames "
      f"({min(true_stops)/fps:.1f}s min)" if true_stops else "  No long stops found")

# Recommended _stop_run threshold: just above max inter-bird gap
if chicken_gaps:
    max_gap = max(chicken_gaps)
    rec_stop_run = max_gap + int(fps * 0.5)  # gap + 0.5s buffer
    print(f"\n  Recommended _stop_run threshold: {rec_stop_run} frames "
          f"({rec_stop_run/fps:.1f}s) — safely above max gap of {max_gap} frames ({max_gap/fps:.2f}s)")
else:
    print(f"\n  No inter-bird gaps found; keep _stop_run=4")

# Belt speed
if len(speed_arr) >= 10:
    med_speed = np.median(speed_arr)
    p90_speed = np.percentile(speed_arr, 90)
    p10_speed = np.percentile(speed_arr, 10)
    print(f"\n  Belt speed (median): {med_speed:.1f} px/frame  "
          f"(= {med_speed*fps:.0f} px/s at {fps}fps)")
    print(f"  Belt speed range:   {p10_speed:.1f} – {p90_speed:.1f} px/frame (p10–p90)")

    # zone_half: must catch a bird at p90 speed in at least 1 frame
    # bird box width at 1280 ≈ 80-120px, so catch window = bird_w + 2*zone_half
    # For 1-frame guarantee: zone_half >= p90_speed / 2
    min_zone = int(np.ceil(p90_speed / 2))
    rec_zone = max(15, min_zone)
    zone_factor = rec_zone / med_speed if med_speed > 0 else 0.8
    print(f"\n  Recommended zone_half:       {rec_zone} px  "
          f"(was 15 — guarantees catch at p90 speed in 1 frame)")
    print(f"  Recommended zone_speed_factor: {zone_factor:.2f}")

    # stop_resume_thresh: above running noise floor
    running_motion = motion_arr[motion_arr >= STOP_THRESH]
    if len(running_motion):
        resume_thresh = np.percentile(running_motion, 10)
        print(f"\n  stop_motion_thresh (keep):   {STOP_THRESH}")
        print(f"  stop_resume_thresh (new):    {resume_thresh:.2f}  "
              f"(p10 of running frames — safely above noise)")
