"""People counting test: RF-DETR (detection) + Supervision (ByteTrack + LineZone).
Usage: python client_test_submission.py people-walking.mp4
Outputs: output_video.mp4, crossings_log.csv, run_info.txt (next to the script).
Direction: line (W,540)->(0,540); walking DOWN (towards camera) = IN, UP = OUT.
"""
import argparse, csv, time, os
import cv2, numpy as np, torch, supervision as sv
from rfdetr import RFDETRLarge

ap = argparse.ArgumentParser()
ap.add_argument("video", nargs="?", default="فيديوالعميل/people-walking (1).mp4")
ap.add_argument("--threshold", type=float, default=0.35)
args = ap.parse_args()

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise FileNotFoundError(args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
W, H = int(cap.get(3)), int(cap.get(4))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

device = f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GB)" \
         if torch.cuda.is_available() else "CPU"
print("Device:", device)
model = RFDETRLarge()   # downloads/loads official COCO weights

# Reversed start/end so that walking DOWN counts as IN
line = sv.LineZone(start=sv.Point(W, 540), end=sv.Point(0, 540),
                   triggering_anchors=(sv.Position.BOTTOM_CENTER,))
line_ann = sv.LineZoneAnnotator(thickness=3, text_thickness=2, text_scale=0.8)
box_ann, label_ann = sv.BoxAnnotator(thickness=2), sv.LabelAnnotator(text_scale=0.6)
trace_ann = sv.TraceAnnotator(trace_length=30)
tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30, frame_rate=int(fps))

out_video_path = os.path.join(os.path.dirname(args.video) or ".", "output_video.mp4")
out_csv_path = os.path.join(os.path.dirname(args.video) or ".", "crossings_log.csv")
out_info_path = os.path.join(os.path.dirname(args.video) or ".", "run_info.txt")

out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
f = open(out_csv_path, "w", newline="", encoding="utf-8")
wr = csv.writer(f)
wr.writerow(["Timestamp (sec)", "Formatted Time", "Frame Index", "Track ID", "Direction"])

def log(idx, tid, d):
    t = idx / fps
    wr.writerow([round(t, 2), f"{int(t//60):02d}:{t%60:05.2f}", idx, int(tid), d])

idx, t0 = 0, time.time()
with torch.inference_mode():
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        try:
            det = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=args.threshold, include_source_image=False)
        except Exception:
            det = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=args.threshold)

        det = det[det.class_id == 1] if len(det) else det          # COCO "person"
        tr = tracker.update_with_detections(det)
        in_m, out_m = line.trigger(detections=tr)
        for tid, i_, o_ in zip(tr.tracker_id, in_m, out_m):
            if i_: log(idx, tid, "IN")
            if o_: log(idx, tid, "OUT")
        img = frame.copy()
        if len(tr):
            img = trace_ann.annotate(img, tr)
            img = box_ann.annotate(img, tr)
            img = label_ann.annotate(img, tr, labels=[f"ID #{t}" for t in tr.tracker_id])
        out.write(line_ann.annotate(img, line))
        idx += 1
        if idx % 25 == 0:
            print(f"{idx}/{total} frames processed...", flush=True)

cap.release(); out.release(); f.close()

el = time.time() - t0
info = (f"Model: RFDETRLarge | Device: {device} | Frames: {idx} | Time: {el:.1f}s | "
        f"Speed: {idx/el:.2f} FPS | Final IN={line.in_count} OUT={line.out_count}")
open(out_info_path, "w").write(info + "\n")
print("\n================ SUMMARY ================")
print(info)
print("=========================================")
