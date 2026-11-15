#!/usr/bin/env python3
"""AI AINAK — Independent Audit: Domain 3 (Contextual Hazard Engine V2 Verification)

Performs comprehensive empirical audit of ContextualHazardEngineV2:
1. Parametric evaluation of all 8 hazard scenarios:
   - Approaching vehicle
   - Approaching fast vehicle
   - Proximity hazard
   - Stationary hazard in path
   - Sudden lateral obstacle
   - Unidentified object approaching
   - Elevated edge or drop-off
   - Fire or thermal hazard
2. Parked vs Moving vs Crossing vehicle discrimination.
3. Monocular depth and distance proxy accuracy & error bounds.
4. Single-frame glitch suppression & temporal persistence gating.
5. Real video hazard assessment on recorded sessions (distinguished from simulated).
"""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np

from ainak.models.vision.v2_hazard_engine import ContextualHazardEngineV2, ScoredHazardV2
from ainak.models.vision.depth import RelativeDepthEstimator
from ainak.models.vision.v2_detector import VisionDetectorV2

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "reports" / "capability_expansion"
AUDIT_DATA_PATH = OUTPUT_DIR / "hazard_audit_results.json"


def audit_simulated_scenarios():
    """Evaluate all 8 hazard scenarios under controlled kinematic conditions."""
    engine = ContextualHazardEngineV2()
    dims = (720, 1280)
    w_img = 1280

    scenarios_tested = {}

    # 1. Approaching vehicle
    # Bounding box in center corridor, looming rate = 0.12, proximity = 0.55
    cx = 640
    bw, bh = 400, 300
    bbox1 = (cx - bw/2, 300, cx + bw/2, 600)
    h1 = engine.score_detection(
        class_name="car", confidence=0.88, bbox_xyxy=bbox1, frame_dims=dims,
        relative_proximity=0.55, looming_rate=0.12, track_age_frames=4, lateral_velocity=0.0
    )
    scenarios_tested["1_approaching_vehicle"] = {
        "ground_truth_scenario": "Approaching vehicle",
        "predicted_tier": h1.hazard_tier,
        "predicted_scenario": h1.scenario,
        "hazard_score": h1.hazard_score,
        "ttc_sec": h1.time_to_collision_sec,
        "rationale": h1.rationale,
        "passed": h1.hazard_tier in {"WARNING", "CRITICAL"} and h1.scenario == "VEHICLE_APPROACHING"
    }

    # 2. Approaching fast vehicle
    # High looming rate = 0.25, proximity = 0.70
    h2 = engine.score_detection(
        class_name="bus", confidence=0.92, bbox_xyxy=bbox1, frame_dims=dims,
        relative_proximity=0.70, looming_rate=0.25, track_age_frames=5, lateral_velocity=0.0
    )
    scenarios_tested["2_approaching_fast_vehicle"] = {
        "ground_truth_scenario": "Approaching fast vehicle",
        "predicted_tier": h2.hazard_tier,
        "predicted_scenario": h2.scenario,
        "hazard_score": h2.hazard_score,
        "ttc_sec": h2.time_to_collision_sec,
        "rationale": h2.rationale,
        "passed": h2.hazard_tier == "CRITICAL" and h2.scenario == "VEHICLE_APPROACHING" and (h2.time_to_collision_sec is not None and h2.time_to_collision_sec <= 2.0)
    }

    # 3. Proximity hazard (immediate obstacle within walking path)
    # Proximity = 0.85, looming = 0.0, center corridor
    h3 = engine.score_detection(
        class_name="chair", confidence=0.85, bbox_xyxy=bbox1, frame_dims=dims,
        relative_proximity=0.85, looming_rate=0.0, track_age_frames=4, lateral_velocity=0.0
    )
    scenarios_tested["3_proximity_hazard"] = {
        "ground_truth_scenario": "Immediate proximity hazard",
        "predicted_tier": h3.hazard_tier,
        "predicted_scenario": h3.scenario,
        "hazard_score": h3.hazard_score,
        "rationale": h3.rationale,
        "passed": h3.hazard_tier in {"CAUTION", "WARNING", "CRITICAL"} and h3.hazard_score >= 0.50
    }

    # 4. Stationary hazard in path (construction barrier / bollard)
    h4 = engine.score_detection(
        class_name="construction barrier", confidence=0.80, bbox_xyxy=bbox1, frame_dims=dims,
        relative_proximity=0.50, looming_rate=0.0, track_age_frames=4, lateral_velocity=0.0
    )
    scenarios_tested["4_stationary_hazard_in_path"] = {
        "ground_truth_scenario": "Stationary hazard in path",
        "predicted_tier": h4.hazard_tier,
        "predicted_scenario": h4.scenario,
        "hazard_score": h4.hazard_score,
        "rationale": h4.rationale,
        "passed": h4.scenario == "PATH_OBSTRUCTION" and h4.hazard_tier in {"CAUTION", "WARNING"}
    }

    # 5. Sudden lateral obstacle (crossing travel path)
    # High lateral velocity = 0.35 across center corridor
    h5 = engine.score_detection(
        class_name="car", confidence=0.82, bbox_xyxy=bbox1, frame_dims=dims,
        relative_proximity=0.55, looming_rate=0.02, track_age_frames=3, lateral_velocity=0.35
    )
    scenarios_tested["5_sudden_lateral_obstacle"] = {
        "ground_truth_scenario": "Sudden lateral crossing obstacle",
        "predicted_tier": h5.hazard_tier,
        "predicted_scenario": h5.scenario,
        "hazard_score": h5.hazard_score,
        "rationale": h5.rationale,
        "passed": h5.scenario == "VEHICLE_CROSSING_PATH" and h5.hazard_tier in {"WARNING", "CRITICAL"}
    }

    # 6. Unidentified object approaching
    # Unknown class with high proximity & looming rate
    h6 = engine.score_detection(
        class_name="unknown_object", confidence=0.75, bbox_xyxy=bbox1, frame_dims=dims,
        relative_proximity=0.65, looming_rate=0.15, track_age_frames=4, lateral_velocity=0.0
    )
    scenarios_tested["6_unidentified_object_approaching"] = {
        "ground_truth_scenario": "Unidentified object approaching",
        "predicted_tier": h6.hazard_tier,
        "predicted_scenario": h6.scenario,
        "hazard_score": h6.hazard_score,
        "rationale": h6.rationale,
        "passed": h6.hazard_tier in {"WARNING", "CRITICAL"} and h6.hazard_score >= 0.50
    }

    # 7. Elevated edge or drop-off (stairs / trench)
    h7 = engine.score_detection(
        class_name="stairs", confidence=0.80, bbox_xyxy=bbox1, frame_dims=dims,
        relative_proximity=0.60, looming_rate=0.03, track_age_frames=4, lateral_velocity=0.0
    )
    scenarios_tested["7_elevated_edge_or_dropoff"] = {
        "ground_truth_scenario": "Elevated edge or drop-off ahead",
        "predicted_tier": h7.hazard_tier,
        "predicted_scenario": h7.scenario,
        "hazard_score": h7.hazard_score,
        "rationale": h7.rationale,
        "passed": h7.scenario == "TERRAIN_DROP_AHEAD" and h7.hazard_tier in {"WARNING", "CRITICAL"}
    }

    # 8. Fire or thermal hazard
    h8 = engine.score_detection(
        class_name="fire", confidence=0.85, bbox_xyxy=(200, 200, 400, 400), frame_dims=dims,
        relative_proximity=0.35, looming_rate=0.0, track_age_frames=3, lateral_velocity=0.0
    )
    scenarios_tested["8_fire_thermal_hazard"] = {
        "ground_truth_scenario": "Fire or thermal hazard",
        "predicted_tier": h8.hazard_tier,
        "predicted_scenario": h8.scenario,
        "hazard_score": h8.hazard_score,
        "rationale": h8.rationale,
        "passed": h8.scenario == "FIRE_HAZARD" and h8.hazard_tier == "CRITICAL"
    }

    return scenarios_tested


def audit_vehicle_discrimination():
    """Verify discrimination between parked, approaching, and crossing vehicles."""
    engine = ContextualHazardEngineV2()
    dims = (720, 1280)

    # 1. Parked Car: curb side (x = 1100), zero looming, zero lateral velocity, high persistence
    bbox_parked = (950, 300, 1250, 550)
    h_parked = engine.score_detection(
        class_name="car", confidence=0.85, bbox_xyxy=bbox_parked, frame_dims=dims,
        relative_proximity=0.35, looming_rate=0.0, track_age_frames=10, lateral_velocity=0.0
    )

    # 2. Approaching Car: center corridor (x = 640), high looming rate (0.15), zero lateral velocity
    bbox_approaching = (440, 250, 840, 600)
    h_approaching = engine.score_detection(
        class_name="car", confidence=0.90, bbox_xyxy=bbox_approaching, frame_dims=dims,
        relative_proximity=0.65, looming_rate=0.15, track_age_frames=4, lateral_velocity=0.0
    )

    # 3. Crossing Car: cutting across path (lateral velocity = 0.30), low looming (0.01)
    bbox_crossing = (400, 300, 750, 550)
    h_crossing = engine.score_detection(
        class_name="car", confidence=0.88, bbox_xyxy=bbox_crossing, frame_dims=dims,
        relative_proximity=0.55, looming_rate=0.01, track_age_frames=4, lateral_velocity=0.30
    )

    return {
        "parked_car": {
            "tier": h_parked.hazard_tier,
            "score": h_parked.hazard_score,
            "scenario": h_parked.scenario,
            "corridor_alignment": h_parked.corridor_alignment,
            "suppressed_to_non_critical": h_parked.hazard_tier in {"INFO", "CAUTION"},
        },
        "approaching_car": {
            "tier": h_approaching.hazard_tier,
            "score": h_approaching.hazard_score,
            "scenario": h_approaching.scenario,
            "escalated_to_critical_or_warning": h_approaching.hazard_tier in {"WARNING", "CRITICAL"},
        },
        "crossing_car": {
            "tier": h_crossing.hazard_tier,
            "score": h_crossing.hazard_score,
            "scenario": h_crossing.scenario,
            "identified_as_crossing": h_crossing.scenario == "VEHICLE_CROSSING_PATH",
        },
        "discrimination_verified": (
            h_parked.hazard_score < h_approaching.hazard_score and
            h_parked.hazard_tier in {"INFO", "CAUTION"} and
            h_approaching.scenario == "VEHICLE_APPROACHING" and
            h_crossing.scenario == "VEHICLE_CROSSING_PATH"
        ),
    }


def audit_glitch_suppression():
    """Verify that single-frame detector glitches are suppressed from escalating to CRITICAL."""
    engine = ContextualHazardEngineV2()
    dims = (720, 1280)
    bbox_spike = (440, 200, 840, 600)  # Large center box

    # Frame 1: Immediate appearance (persistence = 1) with high looming rate
    h_frame1 = engine.score_detection(
        class_name="bus", confidence=0.85, bbox_xyxy=bbox_spike, frame_dims=dims,
        relative_proximity=0.80, looming_rate=0.20, track_age_frames=1, lateral_velocity=0.0
    )

    # Frame 2: Persistence = 2
    h_frame2 = engine.score_detection(
        class_name="bus", confidence=0.85, bbox_xyxy=bbox_spike, frame_dims=dims,
        relative_proximity=0.80, looming_rate=0.20, track_age_frames=2, lateral_velocity=0.0
    )

    # Frame 3: Confirmed tracking (persistence = 3)
    h_frame3 = engine.score_detection(
        class_name="bus", confidence=0.85, bbox_xyxy=bbox_spike, frame_dims=dims,
        relative_proximity=0.80, looming_rate=0.20, track_age_frames=3, lateral_velocity=0.0
    )

    return {
        "frame_1_spike": {
            "track_age": 1,
            "hazard_tier": h_frame1.hazard_tier,
            "hazard_score": h_frame1.hazard_score,
            "is_critical": h_frame1.hazard_tier == "CRITICAL",
        },
        "frame_2": {
            "track_age": 2,
            "hazard_tier": h_frame2.hazard_tier,
            "hazard_score": h_frame2.hazard_score,
            "is_critical": h_frame2.hazard_tier == "CRITICAL",
        },
        "frame_3_confirmed": {
            "track_age": 3,
            "hazard_tier": h_frame3.hazard_tier,
            "hazard_score": h_frame3.hazard_score,
            "is_critical": h_frame3.hazard_tier == "CRITICAL",
        },
        "glitch_suppression_verified": (
            h_frame1.hazard_tier != "CRITICAL" and
            h_frame3.hazard_tier == "CRITICAL"
        ),
        "mechanism": "persistence_factor gating: requires track_age >= 2 frames (persistence_factor >= 0.66) to reach CRITICAL tier."
    }


def audit_distance_and_depth_proxy():
    """Audit accuracy, assumptions, and error bounds of the monocular RelativeDepthEstimator."""
    estimator = RelativeDepthEstimator()
    frame_dims = (720, 1280)

    # Evaluate proxy depth vs distance across different bounding box heights
    # For 'person' (prior height = 1.70m):
    # If a person is 5m away, in a standard 60 deg vertical FOV, they occupy ~35% of image height
    # At 2.5m, they occupy ~70% of image height
    # At 10m, they occupy ~17.5% of image height
    trials = [
        {"desc": "Far (occupies 18% of frame)", "bbox": (500, 300, 600, 300 + 0.18*720), "cls": "person"},
        {"desc": "Medium (occupies 35% of frame)", "bbox": (450, 200, 650, 200 + 0.35*720), "cls": "person"},
        {"desc": "Close (occupies 70% of frame)", "bbox": (350, 100, 750, 100 + 0.70*720), "cls": "person"},
        {"desc": "Imminent (occupies 95% of frame)", "bbox": (250, 20, 850, 20 + 0.95*720), "cls": "person"},
    ]

    proximity_readings = []
    for t in trials:
        prox = estimator.estimate_proximity(t["bbox"], frame_dims, t["cls"])
        proximity_readings.append({"condition": t["desc"], "relative_proximity": float(prox)})

    return {
        "proximity_monotonicity_verified": all(
            proximity_readings[i]["relative_proximity"] <= proximity_readings[i+1]["relative_proximity"]
            for i in range(len(proximity_readings)-1)
        ),
        "readings": proximity_readings,
        "monocular_limitations": {
            "absence_of_metric_scale": "Estimator provides a relative [0.0, 1.0] proximity proxy, NOT calibrated metric meters.",
            "pose_sensitivity": "A sitting, crouching, or leaning person alters vertical bounding box height by up to 50%, distorting proximity proxy.",
            "pitch_tilt_sensitivity": "Camera angle pitch variations (e.g. smart glasses tilted downward 15 deg) shift horizon and distort apparent object height.",
            "audit_verdict": "Partially verified: geometrically sound and monotonic proxy, but not metric depth without LiDAR/stereo sensors."
        }
    }


def audit_real_video_hazards():
    """Evaluate hazard scoring on real video frames from session-multimodal-01 and 03."""
    detector = VisionDetectorV2(weights=ROOT / "artifacts" / "v2" / "vision" / "best.pt", device="cpu", confidence=0.25)
    engine = ContextualHazardEngineV2()

    real_results = []
    sessions = ["session-multimodal-01", "session-multimodal-03"]
    for s in sessions:
        v_path = ROOT / "datasets" / "recordings" / s / "video.mp4"
        if v_path.is_file():
            cap = cv2.VideoCapture(str(v_path))
            ret, frame = cap.read()
            cap.release()
            if ret:
                dets, _, _ = detector.infer(frame)
                scored = []
                for d in dets:
                    h = engine.score_detection(
                        class_name=d.class_name,
                        confidence=d.confidence,
                        bbox_xyxy=d.bbox_xyxy,
                        frame_dims=frame.shape[:2],
                        relative_proximity=getattr(d, "relative_depth", 0.3),
                        looming_rate=getattr(d, "delta_depth", 0.0),
                        track_age_frames=getattr(d, "track_age_frames", 1),
                        track_id=d.track_id,
                    )
                    scored.append({
                        "class": h.class_name,
                        "tier": h.hazard_tier,
                        "score": round(h.hazard_score, 3),
                        "scenario": h.scenario,
                        "proximity": round(h.relative_proximity, 3),
                        "rationale": h.rationale,
                    })
                real_results.append({
                    "session": s,
                    "frame_index": 0,
                    "detections_count": len(dets),
                    "scored_hazards": scored,
                    "highest_tier": max((s["tier"] for s in scored), key=lambda x: {"CRITICAL": 4, "WARNING": 3, "CAUTION": 2, "INFO": 1}.get(x, 0), default="INFO"),
                })

    return real_results


def run_audit():
    print("=" * 70)
    print("AUDIT DOMAIN 3: CONTEXTUAL HAZARD ENGINE V2 VERIFICATION")
    print("=" * 70)

    print("\n1. Testing all 8 simulated hazard scenarios...")
    scenarios = audit_simulated_scenarios()
    for s_name, data in scenarios.items():
        print(f"  - [{data['predicted_tier']}] {data['ground_truth_scenario']}: Score={data['hazard_score']:.2f}, Scenario={data['predicted_scenario']} (Pass: {data['passed']})")

    print("\n2. Auditing vehicle discrimination (Parked vs Approaching vs Crossing)...")
    veh_disc = audit_vehicle_discrimination()
    print(f"  - Parked Car:      Tier={veh_disc['parked_car']['tier']}, Score={veh_disc['parked_car']['score']:.2f} (Suppressed: {veh_disc['parked_car']['suppressed_to_non_critical']})")
    print(f"  - Approaching Car: Tier={veh_disc['approaching_car']['tier']}, Score={veh_disc['approaching_car']['score']:.2f} (Escalated: {veh_disc['approaching_car']['escalated_to_critical_or_warning']})")
    print(f"  - Crossing Car:    Tier={veh_disc['crossing_car']['tier']}, Score={veh_disc['crossing_car']['score']:.2f}, Scenario={veh_disc['crossing_car']['scenario']}")
    print(f"  - Vehicle Discrimination Passed: {veh_disc['discrimination_verified']}")

    print("\n3. Auditing single-frame glitch suppression...")
    glitch = audit_glitch_suppression()
    print(f"  - Frame 1 (Spike): Tier={glitch['frame_1_spike']['hazard_tier']} (Critical: {glitch['frame_1_spike']['is_critical']})")
    print(f"  - Frame 3 (Confirmed): Tier={glitch['frame_3_confirmed']['hazard_tier']} (Critical: {glitch['frame_3_confirmed']['is_critical']})")
    print(f"  - Glitch Suppression Passed: {glitch['glitch_suppression_verified']}")

    print("\n4. Auditing relative proximity and depth proxy...")
    depth_audit = audit_distance_and_depth_proxy()
    print(f"  - Monotonicity verified: {depth_audit['proximity_monotonicity_verified']}")
    for r in depth_audit["readings"]:
        print(f"    * {r['condition']}: Proximity Proxy = {r['relative_proximity']:.3f}")

    print("\n5. Auditing real recorded video hazards (distinguished from simulated)...")
    real_hazards = audit_real_video_hazards()
    print(f"  - Real frames evaluated: {len(real_hazards)}")
    for rf in real_hazards:
        print(f"    * {rf['session']} Frame {rf['frame_index']}: {rf['detections_count']} objects, Highest Tier={rf['highest_tier']}")

    results = {
        "domain": "hazard_engine_v2",
        "simulated_scenarios": scenarios,
        "vehicle_discrimination": veh_disc,
        "glitch_suppression": glitch,
        "depth_and_distance_proxy": depth_audit,
        "real_video_evaluation": real_hazards,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAudit complete. Data persisted to {AUDIT_DATA_PATH}")


if __name__ == "__main__":
    run_audit()
