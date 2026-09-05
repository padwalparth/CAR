"""
Pipeline Runner and Lifecycle Coordinator.

Coordinates the end-to-end execution of the road perception and simulation pipeline:
1. Loads and validates YAML configurations.
2. Initializes model runners (Mock or Real ML) via ModelFactory.
3. Configures ingestion FrameSource (Synthetic, Image, Video, Camera).
4. Coordinates FrameProcessor, OverlayRenderer, and SimulationAdapter.
5. Manages main processing loop, resource cleanup, and statistics aggregation.
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from config.loader import (
    load_models_config,
    load_pipeline_config,
    load_simulation_config,
    load_yaml_file,
    validate_models_config,
    validate_pipeline_config,
)
from models.factory import (
    create_road_model,
    create_traffic_model,
    create_pothole_model,
)
from perception.schemas import WorldState
from processing.frame_processor import FrameProcessor
from processing.frame_source import (
    CameraSource,
    FrameSource,
    ImageSource,
    SyntheticSource,
    VideoSource,
)
from processing.statistics import RuntimeStatistics
from simulation.mock_simulator import MockSimulator
from simulation.simulation_adapter import SimulationAdapter
from visualization.overlay import OverlayRenderer

logger = logging.getLogger("pipeline_runner")


def setup_logger(verbose: bool = False) -> logging.Logger:
    """Configure console logging level and formatting."""
    level = logging.DEBUG if verbose else logging.INFO
    log = logging.getLogger("pipeline")
    log.setLevel(level)
    
    # Avoid duplicate handlers on re-entry
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        log.addHandler(handler)
    else:
        log.handlers[0].setLevel(level)
        
    return log


def build_frame_source(source_arg: str, max_frames: Optional[int] = None) -> FrameSource:
    """
    Construct appropriate FrameSource based on user specification.
    
    Conventions:
    - 'mock' -> SyntheticSource
    - 'camera:<index>' or numeric index -> CameraSource
    - Image extensions (.jpg, .png, etc.) -> ImageSource
    - Video extensions (.mp4, .avi, etc.) -> VideoSource
    """
    src_clean = source_arg.strip()
    src_lower = src_clean.lower()

    if src_lower == "mock":
        limit = max_frames if (max_frames is not None and max_frames > 0) else 10
        return SyntheticSource(max_frames=limit)

    # Explicit camera source: e.g. "camera:0" or numeric index if file doesn't exist
    if src_lower.startswith("camera:"):
        cam_idx_str = src_clean.split(":", 1)[1].strip()
        try:
            cam_idx = int(cam_idx_str)
        except ValueError:
            raise ValueError(f"Invalid camera index in '{source_arg}'. Expected format: 'camera:0'")
        return CameraSource(camera_index=cam_idx)

    if src_clean.isdigit() and not os.path.exists(src_clean):
        # Fallback numeric camera index
        return CameraSource(camera_index=int(src_clean))

    # File path handling
    if not os.path.exists(src_clean):
        raise FileNotFoundError(
            f"Input source file not found: '{src_clean}'.\n"
            f"Ensure the path is correct or specify '--source mock' for testing."
        )

    ext = os.path.splitext(src_lower)[1]
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}

    if ext in image_exts:
        return ImageSource(src_clean)
    elif ext in video_exts:
        return VideoSource(src_clean)
    else:
        # Default probe: try opening as ImageSource
        return ImageSource(src_clean)


def save_image_frame(frame_bgr: np.ndarray, file_path: str) -> None:
    """
    Save image frame using Pillow or OpenCV without requiring opencv-python in core mode.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        import cv2
        cv2.imwrite(file_path, frame_bgr)
    except ImportError:
        # Fallback to Pillow: convert BGR -> RGB
        rgb = frame_bgr[:, :, ::-1].copy()
        pil_img = Image.fromarray(rgb)
        pil_img.save(file_path)


def format_frame_result(world_state: WorldState) -> str:
    """
    Format single-frame perception summary string.
    """
    rc = world_state.road_condition
    tf = world_state.traffic
    ph = world_state.potholes
    meta = world_state.metadata

    road_cond_str = rc.category.value if hasattr(rc.category, "value") else str(rc.category)
    road_score_str = f"{rc.score:.1f}" if rc.score is not None else "N/A"

    cong_str = tf.congestion_level.value if hasattr(tf.congestion_level, "value") else str(tf.congestion_level)

    total_latency_str = f"{meta.processing_time_ms:.1f} ms" if meta.processing_time_ms > 0 else "N/A"
    fps_str = f"{meta.fps:.1f}" if meta.fps > 0 else "N/A"

    large_potholes = ph.severity_counts.get("LARGE", 0) if ph.severity_counts else 0

    return f"""
========================================
FRAME RESULT
========================================
Frame ID: {world_state.frame_id}
Resolution: {world_state.image_width}x{world_state.image_height}

Road:
  Condition: {road_cond_str}
  Score: {road_score_str}

Traffic:
  Objects: {tf.total_objects}
  Vehicles: {tf.vehicle_count}
  Pedestrians: {tf.pedestrian_count}
  Congestion: {cong_str}

Potholes:
  Count: {ph.count}
  Large: {large_potholes}

Processing:
  Total: {total_latency_str}
  FPS: {fps_str}
========================================
""".strip()


def run_pipeline(
    source: str = "mock",
    config_dir: str = "config",
    mock_models: bool = False,
    visualize: bool = False,
    simulate: bool = False,
    save_output: Optional[str] = None,
    save_json: Optional[str] = None,
    max_frames: Optional[int] = None,
    display: bool = False,
    verbose: bool = False,
) -> RuntimeStatistics:
    """
    Execute perception and simulation pipeline lifecycle.

    Returns:
      RuntimeStatistics accumulated across all processed frames.
    """
    log = setup_logger(verbose)
    stats = RuntimeStatistics()
    stats.start()

    log.info("Loading configuration...")

    # 1. Resolve configuration files
    if os.path.isdir(config_dir):
        models_cfg_path = os.path.join(config_dir, "models.yaml")
        pipeline_cfg_path = os.path.join(config_dir, "pipeline.yaml")
        sim_cfg_path = os.path.join(config_dir, "simulation.yaml")
    else:
        # If a single configuration file was passed
        models_cfg_path = config_dir
        pipeline_cfg_path = config_dir
        sim_cfg_path = config_dir

    try:
        models_cfg = load_models_config(models_cfg_path)
    except Exception as e:
        log.error(f"Failed to load models configuration: {e}")
        raise

    try:
        pipeline_cfg = load_pipeline_config(pipeline_cfg_path)
    except Exception as e:
        log.error(f"Failed to load pipeline configuration: {e}")
        raise

    sim_cfg = {}
    if os.path.exists(sim_cfg_path):
        try:
            sim_cfg = load_simulation_config(sim_cfg_path)
        except Exception as e:
            log.warning(f"Could not load simulation config, using defaults: {e}")

    # 2. Configure model backends
    if mock_models:
        log.info("Forcing MOCK perception model runners (--mock-models).")
        models_cfg["road_model"]["backend"] = "mock"
        models_cfg["traffic_model"]["backend"] = "mock"
        models_cfg["pothole_model"]["backend"] = "mock"
    else:
        log.info("Using configured model backends (Real Model Mode).")

    # 3. Instantiate model runners with actionable error handling
    try:
        road_model = create_road_model(models_cfg.get("road_model", {}))
        traffic_model = create_traffic_model(models_cfg.get("traffic_model", {}))
        pothole_model = create_pothole_model(models_cfg.get("pothole_model", {}))
    except (FileNotFoundError, ImportError, ValueError) as err:
        log.error(
            f"\nERROR: Model creation failed: {err}\n"
            f"Suggested action:\n"
            f"  - To run in lightweight deterministic test mode, use: --mock-models\n"
            f"  - To run real models, install required ML dependencies (see requirements-ml.txt) "
            f"and verify model checkpoints exist."
        )
        raise

    log.info(f"Road model:    {road_model.model_name} (backend: {road_model.backend})")
    log.info(f"Traffic model: {traffic_model.model_name} (backend: {traffic_model.backend})")
    log.info(f"Pothole model: {pothole_model.model_name} (backend: {pothole_model.backend})")

    # 4. Construct FrameProcessor
    processor = FrameProcessor(
        road_model=road_model,
        traffic_model=traffic_model,
        pothole_model=pothole_model,
        config=pipeline_cfg,
    )

    # 5. Construct FrameSource
    try:
        frame_source = build_frame_source(source, max_frames=max_frames)
    except Exception as err:
        log.error(f"Failed to initialize source '{source}': {err}")
        raise

    log.info(f"Frame source:  {frame_source.source_name}")

    # 6. Optional Visualization
    renderer = OverlayRenderer() if visualize else None
    if visualize:
        log.info("Visualization overlay: enabled")

    # 7. Optional Simulation
    sim_adapter = None
    if simulate:
        simulator = MockSimulator(sim_cfg)
        sim_adapter = SimulationAdapter(simulator=simulator, config=pipeline_cfg)
        log.info("Simulation adapter:    enabled (MockSimulator)")

    # 8. Main execution loop with guaranteed cleanup
    try:
        frame_source.open()
        processor.initialize()
        if sim_adapter:
            sim_adapter.initialize()

        log.info("Pipeline initialized. Beginning execution...")

        # Determine single-frame vs streaming
        is_single_image = isinstance(frame_source, ImageSource) and (max_frames is None or max_frames == 1)

        while frame_source.has_next():
            if max_frames is not None and stats.frames_processed >= max_frames:
                break

            t0 = time.time()
            success, raw_frame, meta = frame_source.read()
            if not success or raw_frame is None:
                break

            frame_id = meta.frame_id if meta else stats.frames_processed + 1

            # Perception processing
            world_state = processor.process_frame(
                frame=raw_frame,
                frame_id=frame_id,
                timestamp=meta.timestamp if meta else None,
                source=frame_source.source_name,
            )

            # Simulation update
            if sim_adapter:
                sim_adapter.update(world_state)
                sim_adapter.step()

            # Visualization
            rendered_frame = raw_frame
            if renderer:
                rendered_frame = renderer.render(raw_frame, world_state)

            latency_ms = (time.time() - t0) * 1000.0
            stats.record_frame(world_state, latency_ms)

            # Output saving
            if save_output:
                img_path = os.path.join(save_output, f"frame_{frame_id:06d}.jpg")
                save_image_frame(rendered_frame, img_path)

            if save_json:
                json_path = os.path.join(save_json, f"world_state_{frame_id:06d}.json")
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                with open(json_path, "w", encoding="utf-8") as jf:
                    jf.write(world_state.to_json())

            # Display handling (headless guard)
            if display:
                try:
                    import cv2
                    cv2.imshow("Road Perception Pipeline", rendered_frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                        log.info("Display interrupted by user (ESC).")
                        break
                except Exception as e:
                    log.warning(f"Display rendering failed (continuing headless): {e}")
                    display = False

            # Frame reporting
            if is_single_image:
                print(format_frame_result(world_state))
            else:
                road_cond = world_state.road_condition
                road_str = f"{road_cond.category.value} ({road_score_str(road_cond.score)})"
                log.info(
                    f"[FRAME {frame_id}] Road: {road_str} | "
                    f"Traffic: {world_state.traffic.total_objects} objects | "
                    f"Potholes: {world_state.potholes.count} | "
                    f"Latency: {latency_ms:.1f} ms | "
                    f"FPS: {stats.avg_fps:.1f}"
                )

    except Exception as e:
        log.error(f"Pipeline execution error: {e}")
        raise
    finally:
        stats.stop()
        if sim_adapter:
            try:
                sim_adapter.close()
            except Exception:
                pass
        try:
            processor.close()
        except Exception:
            pass
        try:
            frame_source.close()
        except Exception:
            pass

        if display:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass

    # Print summary
    print(stats.format_summary(sim_enabled=simulate, vis_enabled=visualize))

    # Save summary JSON if output directory is configured
    if save_output:
        summary_path = os.path.join(save_output, "run_summary.json")
        try:
            os.makedirs(save_output, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as sf:
                json.dump(stats.to_dict(sim_enabled=simulate, vis_enabled=visualize), sf, indent=2)
            log.info(f"Saved run summary to: {summary_path}")
        except Exception as e:
            log.warning(f"Could not save run summary: {e}")

    return stats


def road_score_str(score: Optional[float]) -> str:
    """Helper to safely format road condition score."""
    return f"{score:.1f}" if score is not None else "N/A"
