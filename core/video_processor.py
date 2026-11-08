import cv2
import time
import logging
import numpy as np
from typing import Dict, Any
from config.settings import settings
from database.db_manager import DatabaseManager
from core.detector import RFDetrDetector
from core.tracker import SupervisionTracker
from scenarios.registry import ScenarioRegistry

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Core Video Stream Processor:
    Loads video, executes RF-DETR object detection and Supervision ByteTrack,
    passes defensive frame copies (`frame.copy()`) and accurate timestamps to scenario plugins,
    and commits all results atomically in a single transaction.
    """
    def __init__(self, detector: RFDetrDetector, tracker: SupervisionTracker):
        self.detector = detector
        self.tracker = tracker

    def process_video_task(self, task_id: str, camera_id: str, video_path: str) -> Dict[str, Any]:
        logger.info(f"Starting analysis for Task [{task_id}], Camera [{camera_id}], File [{video_path}]")

        camera_config = DatabaseManager.get_camera_config(camera_id)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video specs: {width}x{height} @ {fps:.1f} FPS, Total Frames: {total_frames}")

        scenarios = ScenarioRegistry.get_all_scenarios()
        for scenario in scenarios:
            scenario.initialize(camera_config=camera_config, fps=fps, width=width, height=height)

        frame_idx = 0
        start_time = time.time()
        frame_skip = max(1, settings.FRAME_SKIP)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            # Calculate accurate video timestamp in seconds
            timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp_sec <= 0:
                timestamp_sec = frame_idx / fps

            # 1. RF-DETR Detection
            detections = self.detector.detect(frame)

            # 2. Supervision Tracking
            tracked_detections = self.tracker.update(detections)

            # 3. Pass defensive frame.copy() to all scenario plugins
            frame_copy = frame.copy()
            for scenario in scenarios:
                try:
                    scenario.process_frame(
                        frame=frame_copy,
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        detections=tracked_detections
                    )
                except Exception as sc_err:
                    logger.error(f"Error in scenario [{scenario.scenario_name}] at frame {frame_idx}: {sc_err}")

            if frame_idx % 300 == 0:
                elapsed = time.time() - start_time
                proc_fps = frame_idx / elapsed if elapsed > 0 else 0
                logger.info(f"Task [{task_id}] progress: {frame_idx}/{total_frames} frames ({proc_fps:.1f} FPS)")

        cap.release()

        # 4. Collect and save all scenario results in a single database transaction
        all_results = {}
        for scenario in scenarios:
            all_results[scenario.scenario_name] = scenario.get_final_results()

        DatabaseManager.save_all_scenario_results(
            task_id=task_id,
            camera_id=camera_id,
            results_dict=all_results
        )

        total_elapsed = time.time() - start_time
        logger.info(f"Task [{task_id}] completed successfully in {total_elapsed:.2f}s.")

        return all_results
