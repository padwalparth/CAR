#!/usr/bin/env python3
"""
RoadVision AI - Production ML Inference REST API Server.

Serves the real computer-vision perception and fusion pipeline:
1. YOLOv8.3 IDD Object Detector
2. U-Net Road Segmentation (ONNX)
3. Res2Net-50d Pothole Detector
4. Perception Fusion Engine

Endpoints:
- GET  /api/models/health
- POST /api/inference/upload
- POST /api/inference/<session_id>/start
- GET  /api/inference/<session_id>
- GET  /api/inference/history
- POST /api/inference/<session_id>/cancel
- GET  /uploads/<filename>
- GET  /output/<path:filename>
"""

import os
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Ensure repo root and pipeline modules are on sys.path
PIPELINE_ROOT = os.path.abspath(os.path.dirname(__file__))
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

from config.loader import load_models_config, load_pipeline_config
from models.factory import create_road_model, create_traffic_model, create_pothole_model
from processing.frame_processor import FrameProcessor
from perception.schemas import ModelStatus

app = Flask(__name__)
# Enable CORS for all routes so frontend on port 3000 can access seamlessly
CORS(app, resources={r"/*": {"origins": "*"}})

# Directory setup
UPLOADS_DIR = os.path.join(PIPELINE_ROOT, "uploads")
OUTPUT_DIR = os.path.join(PIPELINE_ROOT, "output")
MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "mp4", "avi", "mov"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# In-memory session tracking and history
sessions_db: Dict[str, Dict[str, Any]] = {}
sessions_history: List[str] = []

# Global ML Pipeline instance and model health registry
processor: Optional[FrameProcessor] = None
model_health_registry: Dict[str, Dict[str, Any]] = {
    "yolo": {
        "id": "yolo",
        "name": "YOLO Object Detector",
        "description": "Detects vehicles, pedestrians, animals, tractors & traffic obstacles",
        "status": "Offline",
        "version": "YOLOv8.3 IDD",
        "latency_ms": 0.0,
        "endpoint": "/api/models/yolo",
        "device": "cpu",
        "error": None,
    },
    "unet": {
        "id": "unet",
        "name": "U-Net Road Segmentation",
        "description": "Segments drivable road boundaries & computes coverage percentage",
        "status": "Offline",
        "version": "U-Net ONNX (160x160)",
        "latency_ms": 0.0,
        "endpoint": "/api/models/unet",
        "device": "cpu",
        "error": None,
    },
    "pothole": {
        "id": "pothole",
        "name": "Pothole Detector",
        "description": "Identifies road surface hazards, deep cracks & pothole bounding boxes",
        "status": "Offline",
        "version": "Res2Net-50d",
        "latency_ms": 0.0,
        "endpoint": "/api/models/potholes",
        "device": "cpu",
        "error": None,
    },
    "fusion": {
        "id": "fusion",
        "name": "Fusion Engine",
        "description": "Fuses multi-model detections to generate safety scores & warnings",
        "status": "Offline",
        "version": "Rule/Spatial Fusion Engine v1.0",
        "latency_ms": 0.0,
        "endpoint": "/api/models/fusion",
        "device": "cpu",
        "error": None,
    },
}


def initialize_ml_models() -> None:
    """Initialize real computer-vision models once at startup."""
    global processor, model_health_registry
    print("=" * 60)
    print("[RoadVision AI] Initializing Real Machine Learning Models...")
    print("=" * 60)

    models_config_path = os.path.join(PIPELINE_ROOT, "config", "models.yaml")
    pipeline_config_path = os.path.join(PIPELINE_ROOT, "config", "pipeline.yaml")

    try:
        models_cfg = load_models_config(models_config_path)
    except Exception as e:
        print(f"[Error] Failed to load models config: {e}")
        models_cfg = {}

    try:
        pipeline_cfg = load_pipeline_config(pipeline_config_path)
    except Exception as e:
        print(f"[Warning] Failed to load pipeline config: {e}")
        pipeline_cfg = {}

    # Log resolved absolute paths for absolute audit verification
    for model_key in ["road_model", "traffic_model", "pothole_model"]:
        if model_key in models_cfg and "path" in models_cfg[model_key]:
            abs_p = os.path.abspath(os.path.join(PIPELINE_ROOT, models_cfg[model_key]["path"]))
            print(f"  [Config Path] {model_key:14s} -> {abs_p}")

    # 1. Road Segmentation (U-Net)
    road_model = None
    try:
        print("  -> Loading U-Net Road Segmentation Model...")
        t0 = time.perf_counter()
        road_model = create_road_model(models_cfg.get("road_model", {}))
        road_model.load()
        dt = (time.perf_counter() - t0) * 1000.0
        model_health_registry["unet"]["status"] = "Ready"
        model_health_registry["unet"]["latency_ms"] = round(dt, 2)
        model_health_registry["unet"]["error"] = None
        print(f"  [OK] U-Net loaded successfully in {dt:.1f}ms")
    except Exception as e:
        print(f"  [Error] Failed to load U-Net model: {e}")
        model_health_registry["unet"]["status"] = "Error"
        model_health_registry["unet"]["error"] = str(e)

    # 2. Traffic Detector (YOLOv8 IDD)
    traffic_model = None
    try:
        print("  -> Loading YOLOv8 IDD Traffic Model...")
        t0 = time.perf_counter()
        traffic_model = create_traffic_model(models_cfg.get("traffic_model", {}))
        traffic_model.load()
        dt = (time.perf_counter() - t0) * 1000.0
        model_health_registry["yolo"]["status"] = "Ready"
        model_health_registry["yolo"]["latency_ms"] = round(dt, 2)
        model_health_registry["yolo"]["error"] = None
        print(f"  [OK] YOLO loaded successfully in {dt:.1f}ms")
    except Exception as e:
        print(f"  [Error] Failed to load YOLO model: {e}")
        model_health_registry["yolo"]["status"] = "Error"
        model_health_registry["yolo"]["error"] = str(e)

    # 3. Pothole Detector (Res2Net)
    pothole_model = None
    try:
        print("  -> Loading Res2Net Pothole Model...")
        t0 = time.perf_counter()
        pothole_model = create_pothole_model(models_cfg.get("pothole_model", {}))
        pothole_model.load()
        dt = (time.perf_counter() - t0) * 1000.0
        model_health_registry["pothole"]["status"] = "Ready"
        model_health_registry["pothole"]["latency_ms"] = round(dt, 2)
        model_health_registry["pothole"]["error"] = None
        print(f"  [OK] Res2Net loaded successfully in {dt:.1f}ms")
    except Exception as e:
        print(f"  [Error] Failed to load Pothole model: {e}")
        model_health_registry["pothole"]["status"] = "Error"
        model_health_registry["pothole"]["error"] = str(e)

    # 4. Initialize Coordinator FrameProcessor
    try:
        processor = FrameProcessor(
            road_model=road_model,
            traffic_model=traffic_model,
            pothole_model=pothole_model,
            config=pipeline_cfg,
        )
        processor.initialize(warmup=False)
        model_health_registry["fusion"]["status"] = "Ready"
        model_health_registry["fusion"]["latency_ms"] = 1.5
        model_health_registry["fusion"]["error"] = None
        print("  [OK] FrameProcessor & Perception Fusion Initialized.")
    except Exception as e:
        print(f"  [Error] Failed to initialize FrameProcessor: {e}")
        model_health_registry["fusion"]["status"] = "Error"
        model_health_registry["fusion"]["error"] = str(e)

    print("=" * 60)
    print("[RoadVision AI] Startup Complete. Ready for Inference Requests.")
    print("=" * 60)


# ==============================================================================
# Health Endpoints
# ==============================================================================

@app.route("/api/models/health", methods=["GET"])
def get_models_health():
    """Return live status of all perception and fusion modules."""
    now_iso = datetime.utcnow().isoformat() + "Z"
    result = []
    for model_key in ["yolo", "unet", "pothole", "fusion"]:
        info = dict(model_health_registry[model_key])
        info["last_check_at"] = now_iso
        result.append(info)
    return jsonify(result)


# ==============================================================================
# Media Upload Endpoint
# ==============================================================================

@app.route("/api/inference/upload", methods=["POST"])
def upload_inference_media():
    """Accept and store user media file and initialize session tracking."""
    if "file" not in request.files:
        return jsonify({"message": "No file uploaded in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "message": f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    original_filename = secure_filename(file.filename) or "upload.jpg"
    session_id = f"sess_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    saved_filename = f"{session_id}_{original_filename}"
    saved_filepath = os.path.join(UPLOADS_DIR, saved_filename)
    file.save(saved_filepath)

    ext = original_filename.rsplit(".", 1)[-1].lower()
    media_type = "video" if ext in {"mp4", "avi", "mov"} else "image"

    session_obj = {
        "id": session_id,
        "media_id": f"med_{session_id}",
        "media_type": media_type,
        "media_name": original_filename,
        "saved_filename": saved_filename,
        "saved_filepath": saved_filepath,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "status": "queued",
        "models": {
            "yolo": "waiting",
            "unet": "waiting",
            "pothole": "waiting",
            "fusion": "waiting",
        },
        "result": None,
    }

    sessions_db[session_id] = session_obj
    sessions_history.insert(0, session_id)
    if len(sessions_history) > 50:
        sessions_history.pop()

    # Return session without internal file path
    client_session = {k: v for k, v in session_obj.items() if k not in {"saved_filepath", "saved_filename"}}
    return jsonify(client_session), 200


# ==============================================================================
# Inference Execution Endpoint
# ==============================================================================

@app.route("/api/inference/<session_id>/start", methods=["POST"])
def start_inference_analysis(session_id: str):
    """Execute real ML perception & fusion pipeline on uploaded media."""
    global processor
    session = sessions_db.get(session_id)
    if not session:
        return jsonify({"message": f"Session {session_id} not found"}), 404

    if processor is None or not processor.is_initialized:
        return jsonify({"message": "ML Pipeline is not initialized or still starting"}), 503

    saved_filepath = session.get("saved_filepath")
    if not saved_filepath or not os.path.exists(saved_filepath):
        return jsonify({"message": "Uploaded media file not found on disk"}), 404

    # Timestamp for total end-to-end server execution time calculation
    t_end_to_end_start = time.perf_counter()

    session["status"] = "processing"
    session["models"] = {
        "yolo": "processing",
        "unet": "processing",
        "pothole": "processing",
        "fusion": "waiting",
    }

    try:
        frame = cv2.imread(saved_filepath)
        if frame is None:
            raise ValueError(f"OpenCV could not decode image file: {session['media_name']}")

        img_h, img_w = frame.shape[:2]

        # Execute FrameProcessor
        world_state = processor.process_frame(frame, frame_id=1, source="image")

        # 1. Road Segmentation Mask rendering
        mask_url = None
        road_coverage = 0.0
        road_status = "completed" if world_state.perception_status.road == "ok" else "failed"

        if world_state.road and world_state.road.mask is not None and road_status == "completed":
            raw_mask = world_state.road.mask
            road_coverage = float(world_state.road.road_area_ratio)

            # Resize binary mask to original image dimensions
            mask_resized = cv2.resize(
                raw_mask.astype(np.uint8),
                (img_w, img_h),
                interpolation=cv2.INTER_NEAREST,
            )

            # Generate translucent RGBA overlay (Cyber Blue: [77, 124, 254])
            rgba_mask = np.zeros((img_h, img_w, 4), dtype=np.uint8)
            road_indices = mask_resized > 0
            rgba_mask[road_indices] = [77, 124, 254, 180]  # RGBA

            mask_filename = f"{session_id}_mask.png"
            mask_filepath = os.path.join(MASKS_DIR, mask_filename)
            cv2.imwrite(mask_filepath, cv2.cvtColor(rgba_mask, cv2.COLOR_RGBA2BGRA))
            mask_url = f"http://localhost:8000/output/masks/{mask_filename}"

        # 2. Traffic Object Detections (YOLO)
        yolo_detections = []
        traffic_status = "completed" if world_state.perception_status.traffic == "ok" else "failed"

        if traffic_status == "completed":
            all_traffic = (
                world_state.traffic.vehicles +
                world_state.traffic.pedestrians +
                world_state.traffic.other_objects
            )
            for idx, obj in enumerate(all_traffic):
                bbox = obj.bbox
                if bbox.is_normalized:
                    bbox = bbox.to_pixel(img_w, img_h)

                yolo_detections.append({
                    "id": f"obj_{idx + 1}",
                    "class_name": str(obj.class_name),
                    "confidence": round(float(obj.confidence), 3),
                    "bbox": {
                        "x1": round(float(bbox.xmin), 1),
                        "y1": round(float(bbox.ymin), 1),
                        "x2": round(float(bbox.xmax), 1),
                        "y2": round(float(bbox.ymax), 1),
                    },
                })

        # 3. Pothole Detections (with configurable spatial road-surface validation & U-Net failure handling)
        pothole_detections = []
        pothole_status = "completed" if world_state.perception_status.pothole == "ok" else "failed"
        
        # Load configurable road association threshold
        assoc_cfg = processor.config.get("road_association", {})
        min_overlap_threshold = float(assoc_cfg.get("min_overlap", 0.20))
        
        road_mask_available = bool(world_state.road and world_state.road.mask is not None and road_status == "completed")

        if pothole_status == "completed":
            for idx, pot in enumerate(world_state.potholes.potholes):
                # Spatial Validation: If U-Net succeeded, filter out non-road false positive boxes.
                # If U-Net failed, do NOT discard potholes; include all detections with status notice.
                if road_mask_available:
                    if pot.road_association < min_overlap_threshold:
                        continue

                p_bbox = pot.bbox
                if p_bbox.is_normalized:
                    p_bbox = p_bbox.to_pixel(img_w, img_h)

                pothole_detections.append({
                    "id": f"pot_{len(pothole_detections) + 1}",
                    "confidence": round(float(pot.confidence), 3),
                    "bbox": {
                        "x1": round(float(p_bbox.xmin), 1),
                        "y1": round(float(p_bbox.ymin), 1),
                        "x2": round(float(p_bbox.xmax), 1),
                        "y2": round(float(p_bbox.ymax), 1),
                    },
                    "severity": pot.estimated_severity.value if pot.estimated_severity else "Not available",
                })

        filter_info = {
            "status": "active" if road_mask_available else "unavailable",
            "threshold": min_overlap_threshold,
            "reason": None if road_mask_available else "U-Net road segmentation mask unavailable"
        }

        # 4. Perception Fusion & Safety Analytics
        is_pipeline_healthy = (traffic_status == "completed" and road_status == "completed" and pothole_status == "completed")
        fusion_status = "completed" if is_pipeline_healthy else "degraded"

        pothole_count = len(pothole_detections)
        obj_count = len(yolo_detections)

        # Risk score formula
        base_road_score = float(world_state.road_condition.score)  # 0 to 100
        hazard_from_road = (100.0 - base_road_score) * 0.5
        hazard_from_potholes = min(40.0, pothole_count * 15.0)
        hazard_from_traffic = min(20.0, obj_count * 3.0)

        risk_score = round(min(100.0, max(5.0, hazard_from_road + hazard_from_potholes + hazard_from_traffic)), 1)

        if risk_score >= 70.0 or pothole_count >= 2:
            risk_level = "critical"
        elif risk_score >= 35.0 or pothole_count == 1:
            risk_level = "caution"
        else:
            risk_level = "safe"

        warnings: List[str] = []
        if not is_pipeline_healthy:
            warnings.append("Perception pipeline degraded: one or more models experienced errors")
        if pothole_count > 0:
            warnings.append(f"{pothole_count} road surface pothole(s) detected in travel corridor")
        if world_state.traffic.congestion_level.value != "LOW":
            warnings.append(f"Traffic density elevated: {world_state.traffic.congestion_level.value}")
        if road_coverage < 0.20 and road_status == "completed":
            warnings.append("Low drivable road area detected — drive with caution")
        if not warnings:
            warnings.append("Road surface clear. No critical hazards identified.")

        # End-to-end wall-clock latency measurement
        total_end_to_end_ms = round((time.perf_counter() - t_end_to_end_start) * 1000.0, 1)

        unified_result = {
            "session_id": session_id,
            "input": {
                "media_url": f"http://localhost:8000/uploads/{session['saved_filename']}",
                "media_type": session["media_type"],
                "width": img_w,
                "height": img_h,
            },
            "yolo": {
                "detections": yolo_detections,
                "processing_time_ms": round(world_state.metadata.traffic_inference_ms, 1),
                "status": traffic_status,
            },
            "road_segmentation": {
                "mask_url": mask_url,
                "coverage_ratio": round(road_coverage, 4),
                "coverage_percent": round(road_coverage * 100.0, 2),
                "processing_time_ms": round(world_state.metadata.road_inference_ms, 1),
                "status": road_status,
            },
            "potholes": {
                "detections": pothole_detections,
                "road_association_filter": filter_info,
                "processing_time_ms": round(world_state.metadata.pothole_inference_ms, 1),
                "status": pothole_status,
            },
            "fusion": {
                "risk_level": risk_level,
                "risk_score": risk_score,
                "warnings": warnings,
                "processing_time_ms": round(world_state.metadata.fusion_ms, 1),
                "status": fusion_status,
            },
            "total_processing_time_ms": total_end_to_end_ms,
        }

        session["status"] = "completed"
        session["models"] = {
            "yolo": traffic_status,
            "unet": road_status,
            "pothole": pothole_status,
            "fusion": fusion_status,
        }
        session["result"] = unified_result

        client_session = {k: v for k, v in session.items() if k not in {"saved_filepath", "saved_filename"}}
        return jsonify(client_session), 200

    except Exception as e:
        session["status"] = "failed"
        session["error"] = {"code": "INFERENCE_FAILED", "message": str(e)}
        session["models"] = {
            "yolo": "failed",
            "unet": "failed",
            "pothole": "failed",
            "fusion": "failed",
        }
        client_session = {k: v for k, v in session.items() if k not in {"saved_filepath", "saved_filename"}}
        return jsonify(client_session), 500


# ==============================================================================
# Session Query & History Endpoints
# ==============================================================================

@app.route("/api/inference/<session_id>", methods=["GET"])
def get_inference_session(session_id: str):
    """Fetch current inference session status and results."""
    session = sessions_db.get(session_id)
    if not session:
        return jsonify({"message": f"Session {session_id} not found"}), 404

    client_session = {k: v for k, v in session.items() if k not in {"saved_filepath", "saved_filename"}}
    return jsonify(client_session), 200


@app.route("/api/inference/history", methods=["GET"])
def get_inference_history():
    """Return historical completed inference sessions."""
    history = []
    for sid in sessions_history:
        sess = sessions_db.get(sid)
        if sess:
            client_sess = {k: v for k, v in sess.items() if k not in {"saved_filepath", "saved_filename"}}
            history.append(client_sess)
    return jsonify(history), 200


@app.route("/api/inference/<session_id>/cancel", methods=["POST"])
def cancel_inference_session(session_id: str):
    """Cancel a queued or processing inference session."""
    session = sessions_db.get(session_id)
    if not session:
        return jsonify({"message": f"Session {session_id} not found"}), 404

    session["status"] = "cancelled"
    for k in session["models"]:
        if session["models"][k] in {"waiting", "processing"}:
            session["models"][k] = "cancelled"

    return "", 204


# ==============================================================================
# Analytics Endpoint — derived from real completed inference sessions
# ==============================================================================

@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    """
    Build analytics summary from all completed inference sessions in memory.
    Returns object distribution, pothole timeline, road coverage trend, and
    model latency metrics, all derived from real ML inference results.
    """
    completed = [
        sessions_db[sid]
        for sid in sessions_history
        if sessions_db.get(sid, {}).get("status") == "completed"
           and sessions_db[sid].get("result") is not None
    ]

    if not completed:
        return jsonify(None), 200

    total_sessions = len(completed)
    object_counts: Dict[str, int] = {}
    total_objects = 0
    total_potholes = 0
    coverage_sum = 0.0
    pothole_timeline = []
    coverage_trend = []
    latency_yolo: List[float] = []
    latency_unet: List[float] = []
    latency_pothole: List[float] = []

    for sess in completed:
        result = sess.get("result", {})
        ts_raw = sess.get("created_at", "")
        # Shorten to HH:MM or session suffix for chart axis label
        ts_label = ts_raw[11:16] if len(ts_raw) >= 16 else sess.get("id", "")[-6:]

        # YOLO detections → object distribution
        yolo = result.get("yolo", {})
        dets = yolo.get("detections", [])
        total_objects += len(dets)
        for d in dets:
            cls = d.get("class_name", "unknown")
            object_counts[cls] = object_counts.get(cls, 0) + 1

        # Pothole count
        potholes = result.get("potholes", {})
        ph_count = len(potholes.get("detections", []))
        total_potholes += ph_count
        pothole_timeline.append({"timestamp": ts_label, "count": ph_count})

        # Road coverage
        road = result.get("road_segmentation", {})
        cov = float(road.get("coverage_ratio", 0.0))
        coverage_sum += cov
        coverage_trend.append({
            "timestamp": ts_label,
            "coverage_percentage": round(cov * 100.0, 1),
        })

        # Latency
        y_ms = yolo.get("processing_time_ms")
        u_ms = road.get("processing_time_ms")
        p_ms = potholes.get("processing_time_ms")
        if y_ms is not None: latency_yolo.append(float(y_ms))
        if u_ms is not None: latency_unet.append(float(u_ms))
        if p_ms is not None: latency_pothole.append(float(p_ms))

    avg_coverage = coverage_sum / max(1, total_sessions)

    def avg(lst: List[float]) -> float:
        return round(sum(lst) / len(lst), 1) if lst else 0.0

    latency_metrics = [
        {"model": "YOLO Detector",      "latency_ms": avg(latency_yolo)},
        {"model": "U-Net Road Seg",     "latency_ms": avg(latency_unet)},
        {"model": "Res2Net Pothole",    "latency_ms": avg(latency_pothole)},
    ]

    object_distribution = [
        {"class_name": cls, "count": cnt}
        for cls, cnt in sorted(object_counts.items(), key=lambda x: -x[1])
    ]

    analytics = {
        "total_sessions": total_sessions,
        "total_objects_detected": total_objects,
        "total_potholes_detected": total_potholes,
        "average_road_coverage": round(avg_coverage, 4),
        "object_distribution": object_distribution,
        "pothole_timeline": pothole_timeline,
        "road_coverage_trend": coverage_trend,
        "latency_metrics": latency_metrics,
    }
    return jsonify(analytics), 200



# ==============================================================================
# Static Media Streaming Endpoints
# ==============================================================================

@app.route("/uploads/<path:filename>")
def serve_upload(filename: str):
    """Serve uploaded image or video files."""
    return send_from_directory(UPLOADS_DIR, filename)


@app.route("/output/<path:filename>")
def serve_output(filename: str):
    """Serve generated masks and visualization artifacts."""
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    # Preload models before starting HTTP server
    initialize_ml_models()
    port = int(os.environ.get("PORT", 8000))
    print(f"\n[RoadVision AI Server] Running on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
