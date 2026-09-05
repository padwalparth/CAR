#!/usr/bin/env python3
"""
SIH 2026 Road-Scene Perception & Simulation Pipeline CLI.

Command-line entry point to run perception models, perception fusion,
telemetry visualization, and simulation adapters across mock, image,
video, and live camera sources.

Examples:
  # 1. Primary Smoke Test (Mock mode, zero ML dependencies)
  python scripts/run_pipeline.py --source mock --mock-models --simulate --max-frames 10 --no-display

  # 2. Single Image Inspection
  python scripts/run_pipeline.py --source sample.jpg --mock-models --visualize

  # 3. Streaming Video with Simulation
  python scripts/run_pipeline.py --source road_video.mp4 --mock-models --simulate --max-frames 100

  # 4. Camera Ingestion
  python scripts/run_pipeline.py --source camera:0 --mock-models --max-frames 50

  # 5. Full Real Model Mode (Requires installed PyTorch, ONNX, Ultralytics)
  python scripts/run_pipeline.py --source road_video.mp4 --visualize --simulate
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from processing.pipeline_runner import run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct argument parser for the pipeline CLI."""
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="SIH 2026 Intelligent Road Perception & Simulation Pipeline Runner.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--source",
        type=str,
        default="mock",
        help=(
            "Input frame source to process:\n"
            "  - 'mock'        : Synthetic generated road scene (zero dependencies)\n"
            "  - '<path>.jpg'  : Single image file\n"
            "  - '<path>.mp4'  : Streaming video file\n"
            "  - 'camera:0'    : Live camera stream by device index (e.g. camera:0)"
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config",
        help="Path to configuration directory containing models.yaml, pipeline.yaml, simulation.yaml (default: 'config').",
    )

    parser.add_argument(
        "--mock-models",
        action="store_true",
        default=False,
        help="Force lightweight mock perception runners (zero ML dependencies required, deterministic).",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Enable HUD and bounding-box overlay rendering over output frames.",
    )

    parser.add_argument(
        "--simulate",
        action="store_true",
        default=False,
        help="Enable SimulationAdapter to build SimulationScenario and advance simulator.",
    )

    parser.add_argument(
        "--save-output",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory to save rendered/processed image frames and run_summary.json.",
    )

    parser.add_argument(
        "--save-json",
        type=str,
        nargs="?",
        const="output",
        default=None,
        metavar="DIR",
        help="Directory to save per-frame WorldState JSON snapshots (default: 'output' if passed without path).",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Optional maximum number of frames to process (useful for testing or batch runs).",
    )

    parser.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="Enable live GUI display window if display server is available (default: disabled).",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        default=False,
        help="Explicit headless mode: suppress all GUI popups (default behavior).",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose DEBUG level logging.",
    )

    return parser


def main() -> int:
    """Main CLI execution routine."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Determine GUI display flag: --no-display explicitly takes precedence
    # Default is headless unless --display was passed and --no-display was not passed
    show_display = False
    if args.display and not args.no_display:
        show_display = True

    try:
        run_pipeline(
            source=args.source,
            config_dir=args.config,
            mock_models=args.mock_models,
            visualize=args.visualize,
            simulate=args.simulate,
            save_output=args.save_output,
            save_json=args.save_json,
            max_frames=args.max_frames,
            display=show_display,
            verbose=args.verbose,
        )
        return 0

    except FileNotFoundError as fnf:
        print(f"\n[ERROR] File Not Found: {fnf}", file=sys.stderr)
        return 1
    except ImportError as imp:
        print(f"\n[ERROR] Dependency Missing: {imp}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[INFO] Pipeline execution halted by user.", file=sys.stderr)
        return 0
    except Exception as exc:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"\n[ERROR] Pipeline failed: {exc}", file=sys.stderr)
            print("Run with --verbose for complete traceback.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
