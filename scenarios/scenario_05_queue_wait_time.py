import numpy as np
import supervision as sv
from typing import Dict, Any
from scenarios.base_scenario import BaseScenario
from scenarios.registry import ScenarioRegistry

@ScenarioRegistry.register
class QueueWaitTimeScenario(BaseScenario):
    """
    Scenario 5: Queue Wait Time
    Measures duration spent in cashier queues. Aggregated numerical privacy-compliant output.
    """

    @property
    def scenario_name(self) -> str:
        return "queue_wait_time"

    def initialize(self, camera_config: Dict[str, Any], fps: float, width: int, height: int) -> None:
        self.fps = fps
        self.queue_zones = {}
        self.person_queue_records = {}

        configured_queues = camera_config.get("queue_zones", [
            {
                "name": "Cashier_Queue_1",
                "polygon": [[int(width*0.6), int(height*0.5)], [int(width*0.9), int(height*0.5)],
                            [int(width*0.9), int(height*0.9)], [int(width*0.6), int(height*0.9)]]
            }
        ])

        for q in configured_queues:
            poly_np = np.array(q["polygon"], dtype=np.int32)
            self.queue_zones[q["name"]] = sv.PolygonZone(
                polygon=poly_np,
                triggering_anchors=(sv.Position.BOTTOM_CENTER,)
            )
            self.person_queue_records[q["name"]] = {}

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp_sec: float, detections: sv.Detections) -> None:
        if detections.tracker_id is None or len(detections) == 0:
            return

        for q_name, zone in self.queue_zones.items():
            inside_mask = zone.trigger(detections=detections)
            inside_ids = detections.tracker_id[inside_mask]

            records = self.person_queue_records[q_name]

            for tid in inside_ids:
                if tid not in records:
                    records[tid] = {
                        "first_seen": timestamp_sec,
                        "last_seen": timestamp_sec
                    }
                else:
                    records[tid]["last_seen"] = timestamp_sec

    def get_final_results(self) -> Dict[str, Any]:
        results = {}
        for q_name, records in self.person_queue_records.items():
            durations = [
                round(rec["last_seen"] - rec["first_seen"], 2)
                for rec in records.values()
                if rec["last_seen"] > rec["first_seen"]
            ]

            avg_wait = round(float(np.mean(durations)), 2) if durations else 0.0
            max_wait = round(float(np.max(durations)), 2) if durations else 0.0
            min_wait = round(float(np.min(durations)), 2) if durations else 0.0

            results[q_name] = {
                "average_wait_time_seconds": avg_wait,
                "maximum_wait_time_seconds": max_wait,
                "minimum_wait_time_seconds": min_wait,
                "total_customers_served": len(durations)
            }
        return results
