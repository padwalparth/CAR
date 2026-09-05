"""
Lightweight Multi-Object Tracker for Traffic Objects.

Assigns persistent track_id values to Detection objects across consecutive frames
using spatial IoU and centroid distance matching.
"""

from typing import Dict, List, Tuple
from perception.schemas import Detection, BoundingBox
from perception.coordinate_utils import bbox_iou


class TrackedTarget:
    """Internal representation of an actively tracked object."""
    def __init__(self, track_id: int, detection: Detection):
        self.track_id = track_id
        self.last_detection = detection
        self.class_name = detection.class_name
        self.disappeared_count = 0

    @property
    def bbox(self) -> BoundingBox:
        return self.last_detection.bbox

    def update(self, detection: Detection):
        self.last_detection = detection
        self.disappeared_count = 0


class SimpleTracker:
    """
    IoU & Centroid based multi-object tracker.
    
    Provides persistent IDs for traffic objects between frames so that
    simulators can track individual vehicles and pedestrians continuously.
    """
    def __init__(
        self,
        min_iou_threshold: float = 0.25,
        max_disappeared: int = 5,
        max_centroid_distance_norm: float = 0.15,
    ):
        self.min_iou_threshold = min_iou_threshold
        self.max_disappeared = max_disappeared
        self.max_centroid_distance_norm = max_centroid_distance_norm
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedTarget] = {}

    def reset(self):
        """Reset tracker state."""
        self.next_track_id = 1
        self.tracks.clear()

    def update(
        self,
        detections: List[Detection],
        img_width: int,
        img_height: int,
    ) -> List[Detection]:
        """
        Match incoming detections against existing tracks and assign persistent track_ids.
        Returns the detections list with track_id populated.
        """
        if not detections:
            # Mark all current tracks as disappeared
            to_remove = []
            for tid, target in self.tracks.items():
                target.disappeared_count += 1
                if target.disappeared_count > self.max_disappeared:
                    to_remove.append(tid)
            for tid in to_remove:
                del self.tracks[tid]
            return []

        # Ensure all incoming detections are normalized for distance calculations
        norm_det_boxes = [
            d.bbox.to_normalized(img_width, img_height) for d in detections
        ]

        if not self.tracks:
            # Initialize tracks for all incoming detections
            for i, det in enumerate(detections):
                tid = self.next_track_id
                self.next_track_id += 1
                det.track_id = tid
                self.tracks[tid] = TrackedTarget(tid, det)
            return detections

        # Active track IDs and their normalized boxes
        active_tids = list(self.tracks.keys())
        active_boxes = [
            self.tracks[tid].bbox.to_normalized(img_width, img_height)
            for tid in active_tids
        ]

        # 1. IoU Matching Matrix
        matched_tracks = set()
        matched_dets = set()

        iou_pairs: List[Tuple[float, int, int]] = []
        for d_idx, d_box in enumerate(norm_det_boxes):
            for t_idx, t_box in enumerate(active_boxes):
                # Only match same class family (or vehicle types)
                tid = active_tids[t_idx]
                if self.tracks[tid].class_name == detections[d_idx].class_name:
                    iou = bbox_iou(d_box, t_box)
                    if iou >= self.min_iou_threshold:
                        iou_pairs.append((iou, d_idx, t_idx))

        # Sort pairs by highest IoU descending (greedy matching)
        iou_pairs.sort(key=lambda p: p[0], reverse=True)
        for iou, d_idx, t_idx in iou_pairs:
            tid = active_tids[t_idx]
            if d_idx not in matched_dets and tid not in matched_tracks:
                matched_dets.add(d_idx)
                matched_tracks.add(tid)
                detections[d_idx].track_id = tid
                self.tracks[tid].update(detections[d_idx])

        # 2. Centroid distance matching for remaining unmatched detections
        dist_pairs: List[Tuple[float, int, int]] = []
        for d_idx in range(len(detections)):
            if d_idx in matched_dets:
                continue
            dcx, dcy = norm_det_boxes[d_idx].center
            for t_idx, tid in enumerate(active_tids):
                if tid in matched_tracks:
                    continue
                if self.tracks[tid].class_name == detections[d_idx].class_name:
                    tcx, tcy = active_boxes[t_idx].center
                    dist = ((dcx - tcx) ** 2 + (dcy - tcy) ** 2) ** 0.5
                    if dist <= self.max_centroid_distance_norm:
                        dist_pairs.append((dist, d_idx, t_idx))

        dist_pairs.sort(key=lambda p: p[0])  # Shortest distance first
        for dist, d_idx, t_idx in dist_pairs:
            tid = active_tids[t_idx]
            if d_idx not in matched_dets and tid not in matched_tracks:
                matched_dets.add(d_idx)
                matched_tracks.add(tid)
                detections[d_idx].track_id = tid
                self.tracks[tid].update(detections[d_idx])

        # 3. Create new tracks for remaining unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                tid = self.next_track_id
                self.next_track_id += 1
                det.track_id = tid
                self.tracks[tid] = TrackedTarget(tid, det)

        # 4. Handle unmatched tracks (disappeared)
        to_remove = []
        for tid in active_tids:
            if tid not in matched_tracks:
                self.tracks[tid].disappeared_count += 1
                if self.tracks[tid].disappeared_count > self.max_disappeared:
                    to_remove.append(tid)

        for tid in to_remove:
            del self.tracks[tid]

        return detections
