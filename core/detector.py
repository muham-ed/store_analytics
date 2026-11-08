import cv2
import numpy as np
import torch
import supervision as sv
import logging
import time
from config.settings import settings

logger = logging.getLogger(__name__)

class RFDetrDetector:
    """
    Person Detector using RF-DETR (Official Apache 2.0 rfdetr library).
    Applies night vision CLAHE enhancement and executes GPU inference via predictor.
    """
    def __init__(
        self,
        model_variant: str = settings.RFDETR_MODEL_VARIANT,
        confidence_threshold: float = settings.RFDETR_CONFIDENCE_THRESHOLD,
        device: str = settings.DEVICE
    ):
        self.confidence_threshold = confidence_threshold
        self.device = device if torch.cuda.is_available() else "cpu"
        self.person_class_id = settings.PERSON_CLASS_ID

        logger.info(f"Initializing RF-DETR ({model_variant}) on device: {self.device}")

        self.model = self._load_model_with_retry(model_variant)

    def _load_model_with_retry(self, variant: str):
        import rfdetr

        models_to_try = [variant, "RFDETRBase", "RFDETRLarge"]
        last_exception = None

        for model_name in models_to_try:
            if not hasattr(rfdetr, model_name):
                continue

            model_cls = getattr(rfdetr, model_name)
            for attempt in range(1, 4):
                try:
                    logger.info(f"Attempting to load {model_name} (Attempt {attempt}/3)...")
                    model = model_cls()
                    if hasattr(model, "to"):
                        model.to(self.device)
                    if hasattr(model, "eval"):
                        model.eval()
                    logger.info(f"Successfully loaded {model_name}!")
                    return model
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt} for {model_name} failed: {e}. Retrying in 2s...")
                    time.sleep(2)

        err_msg = f"Fatal: Failed to load official rfdetr model after retries: {last_exception}"
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    def _enhance_night_vision(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Applies CLAHE on Lightness (L) channel in LAB color space to enhance night/low-light vision.
        """
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def detect(self, frame_bgr: np.ndarray) -> sv.Detections:
        """
        Enhances low-light video frame, runs RF-DETR inference using torch.inference_mode(),
        and formats output into Supervision `sv.Detections`.
        """
        enhanced_frame = self._enhance_night_vision(frame_bgr)

        with torch.inference_mode():
            try:
                detections = self.model.predict(
                    enhanced_frame,
                    threshold=self.confidence_threshold,
                    include_source_image=False
                )
            except Exception:
                try:
                    detections = self.model.predict(enhanced_frame, threshold=self.confidence_threshold)
                except Exception:
                    detections = self.model(enhanced_frame)

        if not isinstance(detections, sv.Detections):
            detections = sv.Detections.from_ultralytics(detections) if hasattr(sv.Detections, "from_ultralytics") else sv.Detections.empty()

        if len(detections) > 0 and detections.class_id is not None:
            person_mask = (detections.class_id == self.person_class_id) & (detections.confidence >= self.confidence_threshold)
            detections = detections[person_mask]

        return detections
