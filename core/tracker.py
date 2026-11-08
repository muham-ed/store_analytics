import supervision as sv
import logging

logger = logging.getLogger(__name__)

class SupervisionTracker:
    """
    Wraps Supervision ByteTrack tracker for multi-object person tracking across video frames.
    Preserves persistent track_id per detected person.
    """
    def __init__(self, frame_rate: int = 25, track_activation_threshold: float = 0.25, lost_track_buffer: int = 30):
        logger.info(f"Initializing Supervision ByteTrack tracker with frame_rate={frame_rate}")

        if hasattr(sv, "ByteTrack"):
            self.tracker = sv.ByteTrack(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
                frame_rate=frame_rate
            )
        elif hasattr(sv, "Tracker"):
            self.tracker = sv.Tracker(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
                frame_rate=frame_rate
            )
        else:
            try:
                from supervision.tracker.byte_track.core import ByteTrack
                self.tracker = ByteTrack(
                    track_activation_threshold=track_activation_threshold,
                    lost_track_buffer=lost_track_buffer,
                    frame_rate=frame_rate
                )
            except Exception:
                self.tracker = sv.ByteTrack()

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates tracking state with new detections and attaches tracker_id.
        """
        if len(detections) == 0:
            return self.tracker.update_with_detections(sv.Detections.empty())

        tracked_detections = self.tracker.update_with_detections(detections)
        return tracked_detections
