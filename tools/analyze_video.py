"""
Analyze mix.mp4 to derive calibration parameters:
  - Frame-to-frame motion stats (for stop_motion_thresh / stop_resume_thresh)
  - Belt speed in px/frame (for zone_half / zone_speed_factor)
  - Belt stop/start events
"""
import cv2
import numpy as np
import sys

VIDEO = r"C:\Users\mianx\Downloads\mix.mp4"
PROC_W, PROC_H = 1280, 720
SMALL_W, SMALL_H = 160, 90

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    sys.exit(f"Cannot open {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total / fps
print(f"Video: {total} frames, {fps:.1f} fps, {duration:.1f}s")

prev_gray = None
prev_small = None

motion_vals = []       # mean abs intensity diff on 160x90 (for stop thresh)
flow_speeds = []       # sparse OF speed in proc-space px/frame (for belt speed)
frame_idx = 0

# Sparse OF feature params
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(maxCorners=80, qualityLevel=0.01, minDistance=7, blockSize=7)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # Resize to processing resolution
    frame = cv2.resize(frame, (PROC_W, PROC_H))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Small downscale for motion diff (matches VideoProcessor logic)
    small = cv2.resize(gray, (SMALL_W, SMALL_H))

    if prev_small is not None:
        motion = float(cv2.absdiff(small, prev_small).mean())
        motion_vals.append(motion)

    # Sparse optical flow every 5 frames to get real displacement
    if prev_gray is not None and frame_idx % 5 == 0:
        pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if pts is not None and len(pts) >= 5:
            pts_next, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, pts, None, **lk_params)
            good_prev = pts[status.ravel() == 1]
            good_next = pts_next[status.ravel() == 1]
            if len(good_prev) >= 5:
                dx = good_next[:, 0, 0] - good_prev[:, 0, 0]
                # Use median x-displacement (belt moves horizontally)
                speed = float(np.median(np.abs(dx)))
                flow_speeds.append(speed)

    prev_gray = gray
    prev_small = small

cap.release()

motion_arr = np.array(motion_vals, dtype=float)
flow_arr   = np.array(flow_speeds, dtype=float)

print(f"\n=== MOTION DIFF (160x90 intensity diff) — {len(motion_arr)} samples ===")
print(f"  min:    {motion_arr.min():.3f}")
print(f"  p5:     {np.percentile(motion_arr, 5):.3f}")
print(f"  p10:    {np.percentile(motion_arr, 10):.3f}")
print(f"  p25:    {np.percentile(motion_arr, 25):.3f}")
print(f"  median: {np.median(motion_arr):.3f}")
print(f"  p75:    {np.percentile(motion_arr, 75):.3f}")
print(f"  p90:    {np.percentile(motion_arr, 90):.3f}")
print(f"  max:    {motion_arr.max():.3f}")

# Detect stop/start events using candidate threshold
STOP_THRESH = 0.4
stopped_frames = motion_arr < STOP_THRESH
transitions = np.diff(stopped_frames.astype(int))
stop_starts  = np.where(transitions == 1)[0]   # belt just stopped
stop_ends    = np.where(transitions == -1)[0]  # belt just resumed

print(f"\n  At stop_thresh={STOP_THRESH}: {stopped_frames.sum()} stopped frames "
      f"({100*stopped_frames.mean():.1f}%)")
print(f"  Stop events: {len(stop_starts)}, Resume events: {len(stop_ends)}")

# Show motion values during the first 30 frames after each resume (slow-start ramp)
print(f"\n=== SLOW-START RAMP (motion values 0-30 frames after each resume) ===")
for i, resume_f in enumerate(stop_ends[:5]):  # first 5 resume events
    ramp = motion_arr[resume_f : resume_f + 31]
    print(f"  Resume {i+1} (frame {resume_f}): {[round(v,2) for v in ramp[:15]]}...")

print(f"\n=== OPTICAL FLOW SPEED (proc-space px/frame, every 5 frames) — {len(flow_arr)} samples ===")
if len(flow_arr):
    # Filter to moving frames only (flow > 1 px/frame)
    moving = flow_arr[flow_arr > 1.0]
    print(f"  Moving samples (>1 px): {len(moving)}")
    if len(moving):
        print(f"  min:    {moving.min():.1f} px/frame")
        print(f"  p10:    {np.percentile(moving, 10):.1f} px/frame")
        print(f"  p25:    {np.percentile(moving, 25):.1f} px/frame")
        print(f"  median: {np.median(moving):.1f} px/frame")
        print(f"  p75:    {np.percentile(moving, 75):.1f} px/frame")
        print(f"  p90:    {np.percentile(moving, 90):.1f} px/frame")
        print(f"  max:    {moving.max():.1f} px/frame")

print(f"\n=== RECOMMENDATIONS ===")
if len(motion_arr):
    stopped_p = np.percentile(motion_arr[motion_arr < STOP_THRESH], 90) if stopped_frames.sum() > 0 else STOP_THRESH
    running_p10 = np.percentile(motion_arr[motion_arr >= STOP_THRESH], 10) if (~stopped_frames).sum() > 0 else STOP_THRESH*3
    print(f"  stop_motion_thresh:   {STOP_THRESH} (existing)")
    print(f"  90th pct of stopped frames motion:  {stopped_p:.3f}")
    print(f"  10th pct of running frames motion:  {running_p10:.3f}")
    ratio = running_p10 / STOP_THRESH if STOP_THRESH > 0 else 3.0
    print(f"  Suggested stop_resume_thresh:       {running_p10:.2f}  (={ratio:.1f}x stop_thresh)")

if len(flow_arr) and len(flow_arr[flow_arr > 1.0]):
    moving = flow_arr[flow_arr > 1.0]
    med_speed = np.median(moving)
    p90_speed = np.percentile(moving, 90)
    print(f"  Median belt speed:    {med_speed:.1f} px/frame")
    print(f"  P90 belt speed:       {p90_speed:.1f} px/frame")
    # zone_half should catch a bird at p90 speed in at least 1 frame
    # bird width ~80px at 1280 proc width, so effective zone = bird_width + 2*zone_half
    # minimum: zone_half >= p90_speed / 2 so zone covers 1 full frame of travel
    min_zone_half = int(np.ceil(p90_speed / 2))
    print(f"  Suggested zone_half:  >= {min_zone_half} px  (covers p90 speed in 1 frame)")
    zone_speed_factor = min_zone_half / med_speed if med_speed > 0 else 0.8
    print(f"  Suggested zone_speed_factor: {zone_speed_factor:.2f}  (zone_half = factor * median_speed)")
