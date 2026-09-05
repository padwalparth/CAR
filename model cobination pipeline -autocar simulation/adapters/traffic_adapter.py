"""
Traffic Model Adapter.

Converts raw YOLO detection outputs into standardized List[Detection] objects.
Decouples YOLOv8 tensor formats and Ultralytics abstractions from downstream
perception fusion and simulation.
"""

from typing import Any, Dict, List, Optional
from perception.schemas import Detection, BoundingBox


class TrafficAdapter:
    """
    Adapter converting raw YOLO detection dictionaries/objects into List[Detection].
    """
    def __init__(self, min_confidence: float = 0.35):
        self.min_confidence = min_confidence

    def convert(self, raw_output: Any) -> List[Detection]:
        """
        Convert raw YOLO output to List[Detection].
        Accepts:
          - Dict with 'detections': List of detection dicts (class_id, class_name, confidence, bbox)
          - Ultralytics Results object
          - Direct list of detection dictionaries
        """
        if raw_output is None:
            return []

        raw_list: List[Dict[str, Any]] = []

        if isinstance(raw_output, dict):
            raw_list = raw_output.get("detections", [])
        elif isinstance(raw_output, list):
            raw_list = raw_output
        elif hasattr(raw_output, "boxes"):
            # Ultralytics Results object directly passed
            boxes = raw_output.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                name = raw_output.names.get(cls_id, f"class_{cls_id}") if hasattr(raw_output, "names") else f"class_{cls_id}"
                raw_list.append({
                    "class_id": cls_id,
                    "class_name": name,
                    "confidence": conf,
                    "bbox": (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                })

        detections: List[Detection] = []
        for idx, item in enumerate(raw_list):
            conf = float(item.get("confidence", 0.0))
            if conf < self.min_confidence:
                continue

            raw_box = item.get("bbox")
            if raw_box is None or len(raw_box) < 4:
                continue

            xmin, ymin, xmax, ymax = raw_box[:4]
            # Clean coordinate ordering
            b_xmin = min(float(xmin), float(xmax))
            b_xmax = max(float(xmin), float(xmax))
            b_ymin = min(float(ymin), float(ymax))
            b_ymax = max(float(ymin), float(ymax))

            bbox = BoundingBox(
                xmin=b_xmin,
                ymin=b_ymin,
                xmax=b_xmax,
                ymax=b_ymax,
                is_normalized=False,
            )

            det = Detection(
                object_id=idx + 1,
                class_name=str(item.get("class_name", "unknown")),
                class_id=int(item.get("class_id", -1)),
                confidence=float(conf),
                bbox=bbox,
                track_id=None,  # Tracking happens in perception/tracker.py
                estimated_distance=None,  # 2D bbox alone does not provide physical distance
                road_association=0.0,  # Associated in fusion layer
                lane_id=None,
            )
            detections.append(det)

        return detections
