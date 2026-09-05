"""
End-to-End API Test Script for RoadVision AI.
Tests: Upload, Inference, Verify Schema, Failure Modes, Health Check.
"""
import os
import sys
import json
import time
import requests

PIPELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_URL = "http://localhost:8000"

TEST_IMAGE = os.path.join(
    PIPELINE_ROOT,
    "Road-segmentation-UNET-model-main",
    "data", "data_temp_folder",
    "road_seg_kitti", "default", "image_2", "0.jpg"
)

def sep(title=""):
    print("=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)

def ok(msg): print(f"  [PASS] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")

def test_health():
    sep("TEST 1: Model Health Endpoint")
    r = requests.get(f"{BASE_URL}/api/models/health", timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    models = r.json()
    if isinstance(models, list):
        items = models
    else:
        items = models.get("models", models.get("value", []))

    for m in items:
        status = m.get("status", "")
        name = m.get("name", m.get("id", "?"))
        if status.lower() in ("ready", "online"):
            ok(f"{name}: {status}")
        else:
            fail(f"{name}: {status} (expected Ready)")

def test_upload_and_inference():
    sep("TEST 2: Upload + Inference End-to-End")
    assert os.path.exists(TEST_IMAGE), f"Test image not found: {TEST_IMAGE}"

    with open(TEST_IMAGE, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/api/inference/upload",
            files={"file": ("0.jpg", f, "image/jpeg")},
            timeout=30
        )
    assert r.status_code == 200, f"Upload failed: {r.status_code} — {r.text[:200]}"
    session = r.json()
    session_id = session.get("id", "")
    assert session_id, "No session ID returned from upload"
    ok(f"Upload OK — Session: {session_id}")

    # Start inference
    t_start = time.perf_counter()
    r2 = requests.post(
        f"{BASE_URL}/api/inference/{session_id}/start",
        json={},
        timeout=120
    )
    elapsed = round((time.perf_counter() - t_start) * 1000, 1)
    assert r2.status_code == 200, f"Inference failed: {r2.status_code} — {r2.text[:400]}"
    ok(f"Inference OK — Wall-clock time: {elapsed} ms")

    # Response is the session wrapper — unified_result is under session["result"]
    session_response = r2.json()

    # Verify session ID at top-level (session.id) and inside result (result.session_id)
    resp_id = session_response.get("id", "")
    assert resp_id == session_id, f"Session ID mismatch — got '{resp_id}', expected '{session_id}'"
    ok(f"Session ID consistent: {session_id}")

    result = session_response.get("result", {})
    assert result, "No result inside session response"

    # Also verify nested session_id inside unified result
    nested_id = result.get("session_id", "")
    assert nested_id == session_id, f"Nested session_id mismatch — got '{nested_id}', expected '{session_id}'"
    ok(f"Nested session_id in result consistent: {session_id}")

    return result, session_id

def test_schema(result):
    sep("TEST 3: API Schema Validation")

    # YOLO
    yolo = result.get("yolo", {})
    assert "status" in yolo, "Missing yolo.status"
    assert "detections" in yolo, "Missing yolo.detections"
    ok(f"YOLO: status={yolo['status']}, detections={len(yolo['detections'])}")

    # Road Segmentation
    road = result.get("road_segmentation", {})
    assert "status" in road, "Missing road_segmentation.status"
    assert "coverage_ratio" in road, "Missing coverage_ratio (old 'coverage' field must be removed)"
    assert "coverage_percent" in road, "Missing coverage_percent"
    assert "coverage" not in road, f"Ambiguous 'coverage' field still present — must be removed"
    ok(f"Road Seg: status={road['status']}, coverage_ratio={road['coverage_ratio']}, coverage_percent={road['coverage_percent']}%")

    # Potholes
    potholes = result.get("potholes", {})
    assert "status" in potholes, "Missing potholes.status"
    assert "detections" in potholes, "Missing potholes.detections"
    filter_info = potholes.get("road_association_filter", {})
    ok(f"Potholes: status={potholes['status']}, count={len(potholes['detections'])}, filter={filter_info.get('status','N/A')}")

    # Fusion
    fusion = result.get("fusion", {})
    assert "status" in fusion, "Missing fusion.status"
    assert "risk_level" in fusion, "Missing fusion.risk_level"
    assert "risk_score" in fusion, "Missing fusion.risk_score"
    ok(f"Fusion: status={fusion['status']}, risk_level={fusion['risk_level']}, risk_score={fusion['risk_score']}")

    # Timing
    total_ms = result.get("total_processing_time_ms")
    ok(f"Total Processing Time: {total_ms} ms")

def test_zero_detections(result):
    sep("TEST 4: Zero Detections Separation from Failures")
    yolo = result.get("yolo", {})
    potholes = result.get("potholes", {})

    if yolo.get("detections") == [] and yolo.get("status") == "completed":
        ok("YOLO: status=completed with 0 detections (distinct from error)")
    elif yolo.get("status") == "failed":
        fail("YOLO returned failed — different from expected completed+[]")
    else:
        ok(f"YOLO: {yolo.get('status')}, {len(yolo.get('detections',[]))} detections")

    if potholes.get("detections") == [] and potholes.get("status") == "completed":
        ok("Potholes: status=completed with 0 detections (distinct from error)")
    elif potholes.get("status") == "failed":
        fail("Potholes returned failed — different from expected completed+[]")
    else:
        ok(f"Potholes: {potholes.get('status')}, {len(potholes.get('detections',[]))} detections")

def test_mask_url(result, session_id):
    sep("TEST 5: Mask URL Accessibility")
    road = result.get("road_segmentation", {})
    mask_url = road.get("mask_url")
    if mask_url:
        r = requests.get(mask_url, timeout=10)
        if r.status_code == 200:
            ok(f"Mask URL accessible: {mask_url}")
        else:
            fail(f"Mask URL returned {r.status_code}: {mask_url}")
    else:
        print("  [INFO] No mask URL (coverage may be 0 or U-Net returned no road area)")

def test_coverage_fields(result):
    sep("TEST 6: Coverage Field Standardization")
    road = result.get("road_segmentation", {})
    ratio = road.get("coverage_ratio")
    pct = road.get("coverage_percent")
    has_old = "coverage" in road

    if has_old:
        fail("Old 'coverage' field still present — must be removed")
    else:
        ok("Old 'coverage' field removed")

    if ratio is not None and pct is not None:
        expected_pct = round(ratio * 100, 2)
        if abs(pct - expected_pct) < 0.1:
            ok(f"coverage_ratio={ratio}, coverage_percent={pct}% — consistent")
        else:
            fail(f"coverage_ratio={ratio} but coverage_percent={pct}% (expected ~{expected_pct})")
    else:
        fail(f"Missing coverage_ratio or coverage_percent: ratio={ratio}, pct={pct}")

def main():
    sep("ROADVISION AI — END-TO-END API VALIDATION SUITE")
    print(f"  Backend: {BASE_URL}")
    print(f"  Test Image: {os.path.basename(TEST_IMAGE)}")
    print()

    try:
        test_health()
        result, session_id = test_upload_and_inference()
        test_schema(result)
        test_zero_detections(result)
        test_mask_url(result, session_id)
        test_coverage_fields(result)

        sep("FULL RESULT DUMP")
        print(json.dumps(result, indent=2))

        sep("ALL TESTS COMPLETED")
    except AssertionError as e:
        print(f"\n  [ASSERTION ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  [EXCEPTION] {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
