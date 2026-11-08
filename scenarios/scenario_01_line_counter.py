import numpy as np
import supervision as sv
from typing import Dict, Any
from scenarios.base_scenario import BaseScenario
from scenarios.registry import ScenarioRegistry

@ScenarioRegistry.register
class LineCounterScenario(BaseScenario):
    """
    Scenario 1: In/Out Line Counter
    Counts people crossing a virtual line using sv.LineZone with bottom-center anchor.
    """

    @property
    def scenario_name(self) -> str:
        return "line_counter"

    def initialize(self, camera_config: Dict[str, Any], fps: float, width: int, height: int) -> None:
        self.fps = fps
        line_cfg = camera_config.get("entry_line", {
            "start": [int(width * 0.1), int(height * 0.5)],
            "end": [int(width * 0.9), int(height * 0.5)]
        })
        start_pt = sv.Point(x=line_cfg["start"][0], y=line_cfg["start"][1])
        end_pt = sv.Point(x=line_cfg["end"][0], y=line_cfg["end"][1])

        self.line_zone = sv.LineZone(
            start=start_pt,
            end=end_pt,
            triggering_anchors=(sv.Position.BOTTOM_CENTER,)
        )

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp_sec: float, detections: sv.Detections) -> None:
        if len(detections) == 0:
            return

        self.line_zone.trigger(detections=detections)

    def get_final_results(self) -> Dict[str, Any]:
        return {
            "total_in": int(self.line_zone.in_count),
            "total_out": int(self.line_zone.out_count),
            "net_flow": int(self.line_zone.in_count - self.line_zone.out_count)
        }
