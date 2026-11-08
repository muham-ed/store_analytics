import os
import cv2
import numpy as np
import supervision as sv
from typing import Dict, Any
from config.settings import settings
from scenarios.base_scenario import BaseScenario
from scenarios.registry import ScenarioRegistry

@ScenarioRegistry.register
class HeatmapScenario(BaseScenario):
    """
    Scenario 3: Heatmap Generation
    Accumulates customer spatial foot-traffic density across video duration,
    renders and saves a JPEG heatmap overlay image to outputs/heatmaps/{task_id}_heatmap.jpg.
    """

    @property
    def scenario_name(self) -> str:
        return "heatmap"

    def initialize(self, camera_config: Dict[str, Any], fps: float, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.task_id = "default"
        self.scale = 4
        self.grid_w = width // self.scale
        self.grid_h = height // self.scale
        self.accumulator = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp_sec: float, detections: sv.Detections) -> None:
        if len(detections) == 0:
            return

        for bbox in detections.xyxy:
            cx = int((bbox[0] + bbox[2]) / 2.0 / self.scale)
            cy = int(bbox[3] / self.scale)

            if 0 <= cx < self.grid_w and 0 <= cy < self.grid_h:
                self.accumulator[cy, cx] += 1.0

    def get_final_results(self) -> Dict[str, Any]:
        os.makedirs(settings.HEATMAPS_DIR, exist_ok=True)

        if self.accumulator.max() > 0:
            norm = (self.accumulator / self.accumulator.max() * 255.0).astype(np.uint8)
        else:
            norm = self.accumulator.astype(np.uint8)

        # Resize to original resolution and apply Gaussian blur
        resized = cv2.resize(norm, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(resized, (31, 31), 0)
        color_heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)

        output_filename = f"heatmap_output.jpg"
        output_path = os.path.join(settings.HEATMAPS_DIR, output_filename)
        cv2.imwrite(output_path, color_heatmap)

        return {
            "image_path": output_path,
            "max_density_intensity": float(self.accumulator.max()),
            "total_footsteps_logged": float(self.accumulator.sum())
        }
