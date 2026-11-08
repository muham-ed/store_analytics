import cv2
import numpy as np
import supervision as sv
from typing import List, Tuple

class VideoVisualizer:
    """
    Helper class for drawing bounding boxes, track paths, lines, and polygon zones on frames
    for verification, visual debugging, or rendering output streams.
    """
    def __init__(self):
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)
        self.trace_annotator = sv.TraceAnnotator(trace_length=30)

    def draw_detections(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        labels = []
        if detections.tracker_id is not None:
            labels = [f"ID: {tid} ({conf:.2f})" for tid, conf in zip(detections.tracker_id, detections.confidence)]
        else:
            labels = [f"Person ({conf:.2f})" for conf in detections.confidence]

        annotated_frame = frame.copy()
        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        if detections.tracker_id is not None:
            annotated_frame = self.trace_annotator.annotate(scene=annotated_frame, detections=detections)

        return annotated_frame

    def draw_zone_polygons(self, frame: np.ndarray, polygons: List[np.ndarray], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        annotated_frame = frame.copy()
        for poly in polygons:
            cv2.polylines(annotated_frame, [poly], isClosed=True, color=color, thickness=2)
        return annotated_frame
