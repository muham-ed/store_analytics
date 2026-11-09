import os
import time
import csv
import cv2
import torch
import numpy as np
import supervision as sv
import rfdetr

# Paths
input_video_path = os.path.abspath("فيديوالعميل/people-walking (1).mp4")
output_video_path = os.path.abspath("فيديوالعميل/output_video.mp4")
output_csv_path = os.path.abspath("فيديوالعميل/crossings_log.csv")

print(f"Loading video from: {input_video_path}")
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Video file not found: {input_video_path}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video Specs: {width}x{height} @ {fps} FPS, Total Frames: {total_frames}")

# Device check
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Execution Hardware Device: {device}")

# Load RF-DETR Model (Apache 2.0)
print("Loading RF-DETR Model (RFDETRLarge)...")
model = rfdetr.RFDETRLarge()
if hasattr(model, "to"):
    model.to(device)
if hasattr(model, "eval"):
    model.eval()

# Correct Direction Line Setup:
# Downwards (towards camera, top-to-bottom) = IN
# Upwards (away from camera, bottom-to-top) = OUT
# By setting start=(width, 540) and end=(0, 540), crossing top-to-bottom triggers crossed_in
start_point = sv.Point(x=width, y=540)
end_point = sv.Point(x=0, y=540)
line_zone = sv.LineZone(
    start=start_point,
    end=end_point,
    triggering_anchors=(sv.Position.BOTTOM_CENTER,)
)

# Annotators
line_annotator = sv.LineZoneAnnotator(thickness=3, text_thickness=2, text_scale=0.8)
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.6)
trace_annotator = sv.TraceAnnotator(trace_length=30)

# Tracker (Supervision ByteTrack)
tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30, frame_rate=int(fps))

# Output Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# CSV Logging Setup
csv_file = open(output_csv_path, mode='w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp (sec)", "Formatted Time", "Frame Index", "Track ID", "Direction"])

events_log = []
frame_idx = 0
start_proc_time = time.time()

print("Processing video frames...")
with torch.inference_mode():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_idx / fps

        # RF-DETR Detection on standard image frame
        try:
            detections = model.predict(frame, threshold=0.35, include_source_image=False)
        except Exception:
            detections = model.predict(frame, threshold=0.35)

        if not isinstance(detections, sv.Detections):
            detections = sv.Detections.empty()

        # Filter Person Class (class_id == 1 in RF-DETR or class_id == 0)
        if len(detections) > 0 and detections.class_id is not None:
            person_mask = (detections.class_id == 1) | (detections.class_id == 0)
            detections = detections[person_mask]

        # Supervision ByteTrack
        tracked = tracker.update_with_detections(detections)

        # Line Crossing Detection
        in_mask, out_mask = line_zone.trigger(detections=tracked)

        # Log IN events (Downwards)
        if tracked.tracker_id is not None and len(tracked) > 0:
            for tid, is_in in zip(tracked.tracker_id, in_mask):
                if is_in:
                    mins = int(timestamp_sec // 60)
                    secs = timestamp_sec % 60
                    time_str = f"{mins:02d}:{secs:05.2f}"
                    row = [round(timestamp_sec, 2), time_str, frame_idx, int(tid), "IN"]
                    csv_writer.writerow(row)
                    events_log.append(row)
                    print(f"[EVENT] Frame {frame_idx} ({time_str}) - ID #{tid} crossed IN (downwards)")

            for tid, is_out in zip(tracked.tracker_id, out_mask):
                if is_out:
                    mins = int(timestamp_sec // 60)
                    secs = timestamp_sec % 60
                    time_str = f"{mins:02d}:{secs:05.2f}"
                    row = [round(timestamp_sec, 2), time_str, frame_idx, int(tid), "OUT"]
                    csv_writer.writerow(row)
                    events_log.append(row)
                    print(f"[EVENT] Frame {frame_idx} ({time_str}) - ID #{tid} crossed OUT (upwards)")

        # Annotation
        labels = [f"ID #{tid}" for tid in tracked.tracker_id] if tracked.tracker_id is not None else []
        annotated = frame.copy()
        if len(tracked) > 0:
            annotated = trace_annotator.annotate(scene=annotated, detections=tracked)
            annotated = box_annotator.annotate(scene=annotated, detections=tracked)
            annotated = label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)

        try:
            annotated = line_annotator.annotate(frame=annotated, line_counter=line_zone)
        except Exception:
            annotated = line_annotator.annotate(scene=annotated, line_counter=line_zone)

        out.write(annotated)
        frame_idx += 1

cap.release()
out.release()
csv_file.close()

elapsed = time.time() - start_proc_time
proc_fps = total_frames / elapsed if elapsed > 0 else 0.0

print("\n================ SUMMARY REPORT ================")
print(f"Total Video Duration: {total_frames / fps:.2f} seconds ({total_frames} frames)")
print(f"Total Processing Time: {elapsed:.2f} seconds ({proc_fps:.2f} FPS)")
print(f"Final Count -> IN (Downwards): {line_zone.in_count} | OUT (Upwards): {line_zone.out_count}")
print(f"CSV Events saved to: {output_csv_path}")
print(f"Annotated Video saved to: {output_video_path}")
print("================================================")
