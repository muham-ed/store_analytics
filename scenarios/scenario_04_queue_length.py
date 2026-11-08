import numpy as np
import supervision as sv
from typing import Dict, Any, List
from scenarios.base_scenario import BaseScenario
from scenarios.registry import ScenarioRegistry

@ScenarioRegistry.register
class QueueLengthScenario(BaseScenario):
    """
    Scenario 4: Cashier Queue Length (طول طابور الكاشير)
    Monitors person count inside cashier queue zones over time.
    """

    @property
    def scenario_name(self) -> str:
        return "queue_length"

    def initialize(self, camera_config: Dict[str, Any], fps: float, width: int, height: int) -> None:
        self.queue_zones = {}
        self.length_samples = {} # {queue_name: [count1, count2, ...]}

        configured_queues = camera_config.get("queue_zones", [
            {
                "name": "Cashier_Queue_1",
                "polygon": [[int(width*0.6), int(height*0.5)], [int(width*0.9), int(height*0.5)],
                            [int(width*0.9), int(height*0.9)], [int(width*0.6), int(height*0.9)]]
            }
        ])

        for q in configured_queues:
            poly_np = np.array(q["polygon"], dtype=np.int32)
            self.queue_zones[q["name"]] = sv.PolygonZone(polygon=poly_np)
            self.length_samples[q["name"]] = []

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp_sec: float, detections: sv.Detections) -> None:
        for q_name, zone in self.queue_zones.items():
            if len(detections) > 0:
                is_inside = zone.trigger(detections=detections)
                current_count = int(np.sum(is_inside))
            else:
                current_count = 0
            self.length_samples[q_name].append(current_count)

    def get_final_results(self) -> Dict[str, Any]:
        results = {}
        for q_name, samples in self.length_samples.items():
            if len(samples) == 0:
                continue
            avg_len = round(float(np.mean(samples)), 2)
            max_len = int(np.max(samples))
            p90_len = round(float(np.percentile(samples, 90)), 2)

            results[q_name] = {
                "average_queue_length": avg_len,
                "maximum_queue_length": max_len,
                "90th_percentile_queue_length": p90_len,
                "samples_count": len(samples)
            }
        return results
