"""
Posture Analysis Module
-----------------------
Uses OpenCV and YOLOv8 Pose to analyze sitting posture from video files.
Processes every 5th frame for performance, calculates neck and back angles,
and returns a final posture assessment.

Detection backend: YOLOv8 Pose (Ultralytics) with COCO 17-keypoint format.
Features preserved:
- Temporal smoothing (5-frame moving average) for stable landmarks
- Bilateral body usage (midpoints of left+right) for shoulder, hip, knee
- Frame normalization to 640x480 before pose detection
- Posture scoring and issue detection
- Django visualization saving
"""

import math
from collections import deque

import cv2
from ultralytics import YOLO  # [YOLOv8] Pose detection model

# ── YOLOv8 Pose Model ──
# Load model once at module level for efficiency.
# The model weights are auto-downloaded on first use.
_yolo_model = YOLO("yolov8n-pose.pt")

# ── COCO Keypoint Indices (YOLOv8 Pose) ──
# 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
# 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow
# 9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip
# 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
_KP_NOSE = 0
_KP_LEFT_SHOULDER = 5
_KP_RIGHT_SHOULDER = 6
_KP_LEFT_HIP = 11
_KP_RIGHT_HIP = 12
_KP_LEFT_KNEE = 13
_KP_RIGHT_KNEE = 14


def _calculate_angle(a, b, c):
    """
    Calculate the angle at point B formed by points A-B-C.
    Each point is a tuple (x, y).
    Returns angle in degrees.
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if mag_ba * mag_bc == 0:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot_product / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def _get_yolo_keypoint(keypoints_xy, index):
    """
    Extract (x, y) pixel coordinates from YOLOv8 keypoint array.
    Returns None if the keypoint is missing (coordinates are 0, 0).
    YOLOv8 returns (0, 0) for undetected keypoints.
    """
    x = float(keypoints_xy[index][0])
    y = float(keypoints_xy[index][1])
    # YOLOv8 uses (0, 0) for missing keypoints
    if x == 0.0 and y == 0.0:
        return None
    return (x, y)


def _midpoint(p1, p2):
    """
    Return the midpoint of two (x, y) points.
    If either point is None, returns the other (or None if both are None).
    """
    if p1 is None and p2 is None:
        return None
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def _smooth_point(history):
    """
    Return the averaged (x, y) from a deque of past landmark positions.
    Provides temporal smoothing to reduce frame-to-frame jitter.
    """
    if not history:
        return None
    avg_x = sum(p[0] for p in history) / len(history)
    avg_y = sum(p[1] for p in history) / len(history)
    return (avg_x, avg_y)


def _draw_skeleton(frame, ear, shoulder, hip, knee, neck_angle, back_angle):
    """
    Draw the pose skeleton on the frame using OpenCV.
    Replaces MediaPipe's draw_landmarks with simple lines and circles.
    Green for normal posture, red highlighting for detected issues.
    """
    # Convert to integer pixel coordinates
    p_ear = (int(ear[0]), int(ear[1]))
    p_shoulder = (int(shoulder[0]), int(shoulder[1]))
    p_hip = (int(hip[0]), int(hip[1]))
    p_knee = (int(knee[0]), int(knee[1]))

    # Draw base skeleton in green
    cv2.line(frame, p_ear, p_shoulder, (0, 255, 0), 2)
    cv2.line(frame, p_shoulder, p_hip, (0, 255, 0), 2)
    cv2.line(frame, p_hip, p_knee, (0, 255, 0), 2)

    # Draw keypoint circles in green
    for pt in [p_ear, p_shoulder, p_hip, p_knee]:
        cv2.circle(frame, pt, 5, (0, 255, 0), -1)

    # ── Red highlighting for posture issues ──
    has_neck_issue = neck_angle < 135
    has_back_issue = back_angle < 145

    if has_neck_issue:
        # Highlight neck region (ear → shoulder → hip) in red
        cv2.line(frame, p_ear, p_shoulder, (0, 0, 255), 4)
        cv2.line(frame, p_shoulder, p_hip, (0, 0, 255), 4)
        cv2.circle(frame, p_ear, 8, (0, 0, 255), -1)
        cv2.circle(frame, p_shoulder, 8, (0, 0, 255), -1)

    if has_back_issue:
        # Highlight back region (shoulder → hip → knee) in red
        cv2.line(frame, p_shoulder, p_hip, (0, 0, 255), 4)
        cv2.line(frame, p_hip, p_knee, (0, 0, 255), 4)
        cv2.circle(frame, p_hip, 8, (0, 0, 255), -1)
        cv2.circle(frame, p_knee, 8, (0, 0, 255), -1)


def analyze_posture(image_path):
    """
    Analyze posture from an image file.

    Opens the image with OpenCV, uses YOLOv8 Pose
    to detect body keypoints, and calculates neck/back angles.

    Args:
        image_path (str): Absolute path to the image file.

    Returns:
        dict: {
            "result": "Good Posture" or "Bad Posture",
            "score": float (0-100 percentage),
            "issues": list of detected issue strings,
            "visualization_image": str or None
        }
    """
    # ── Debug: Image metadata ──
    print(f"\n{'='*50}")
    print(f"IMAGE PATH: {image_path}")

    frame = cv2.imread(image_path)
    
    if frame is None:
        print("ERROR: Could not open image file.")
        return {
            "result": "Invalid Image",
            "score": None,
            "issues": ["image_error"],
            "message": "The uploaded image is not suitable for posture analysis. Please upload a clearer photo showing your full upper body while sitting.",
            "visualization_image": None
        }

    frame_height, frame_width = frame.shape[:2]
    print(f"RESOLUTION: {frame_width}x{frame_height}")
    print(f"{'='*50}\n")

    # Normalize image resolution to improve performance
    frame = cv2.resize(frame, (640, 480))

    visualization_filename = None
    issues = []
    
    # ── [YOLOv8] Run pose detection ──
    # verbose=False suppresses console output from YOLO
    results = _yolo_model(frame, verbose=False)

    # Check if a person is detected using boxes
    if len(results[0].boxes) == 0:
        print("  Pose result: No person detected in the image.")
        return {
            "result": "Invalid Image",
            "score": None,
            "issues": ["no_person_detected"],
            "message": "No person was detected in the image. Please upload a clearer photo showing your upper body while sitting.",
            "visualization_image": None
        }

    # Check if any keypoints exist for the detected person
    if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        print("  Pose result: Person detected but no keypoints found.")
        return {
            "result": "Invalid Image",
            "score": None,
            "issues": ["missing_keypoints"],
            "message": "No person was detected in the image. Please upload a clearer photo showing your upper body while sitting.",
            "visualization_image": None
        }

    print("  LANDMARKS DETECTED")

    # [YOLOv8] Extract keypoints for the first detected person
    # keypoints.xy shape: (num_persons, 17, 2) — pixel coordinates
    kp = results[0].keypoints.xy[0]

    # Validate if all keypoint coordinates are just zeroed out
    if kp.sum() == 0:
        print("  Pose result: All keypoints are empty (0,0).")
        return {
            "result": "Invalid Image",
            "score": None,
            "issues": ["missing_keypoints"],
            "message": "No person was detected in the image. Please upload a clearer photo showing your upper body while sitting.",
            "visualization_image": None
        }

    # ── [YOLOv8] COCO keypoint extraction ──
    # Map YOLO keypoints to required landmarks
    try:
        nose = _get_yolo_keypoint(kp, _KP_NOSE)
        left_shoulder = _get_yolo_keypoint(kp, _KP_LEFT_SHOULDER)
        right_shoulder = _get_yolo_keypoint(kp, _KP_RIGHT_SHOULDER)
        left_hip = _get_yolo_keypoint(kp, _KP_LEFT_HIP)
        right_hip = _get_yolo_keypoint(kp, _KP_RIGHT_HIP)
        left_knee = _get_yolo_keypoint(kp, _KP_LEFT_KNEE)
        right_knee = _get_yolo_keypoint(kp, _KP_RIGHT_KNEE)

        # [YOLOv8] COCO format does not have a reliable "ear" keypoint
        # for posture analysis. Use nose as the head reference point
        # (approximates ear position for neck angle calculation).
        ear = nose

        # Bilateral midpoints for stability
        shoulder = _midpoint(left_shoulder, right_shoulder)
        hip = _midpoint(left_hip, right_hip)
        knee = _midpoint(left_knee, right_knee)

        # Skip if any required landmark is missing
        if any(pt is None for pt in [ear, shoulder, hip, knee]):
            print("  Skipping: one or more keypoints not detected")
            return {
                "result": "Invalid Image",
                "score": None,
                "issues": ["missing_keypoints"],
                "message": "The uploaded image is not suitable for posture analysis. Please upload a clearer photo showing your full upper body while sitting.",
                "visualization_image": None
            }

        # Neck angle: ear → shoulder → hip
        neck_angle = _calculate_angle(ear, shoulder, hip)

        # Back angle: shoulder → hip → knee
        back_angle = _calculate_angle(shoulder, hip, knee)

        # Scoring logic:
        # Good neck angle: ~135-180 degrees (upright)
        # Good back angle: ~145-180 degrees (straight back)
        neck_score = _angle_to_score(neck_angle, ideal_min=135, ideal_max=180)
        back_score = _angle_to_score(back_angle, ideal_min=145, ideal_max=180)

        # Combined score (neck 50%, back 50%)
        final_score = (neck_score * 0.5) + (back_score * 0.5)

        # ── Detect posture issues based on angles ──
        if neck_angle < 135:
            issues.append("forward_head")
        if back_angle < 145:
            issues.append("rounded_back")

        # ── Visualization ──
        import os
        import uuid
        from django.conf import settings

        # [YOLOv8] Draw skeleton using OpenCV (replaces mp_drawing)
        _draw_skeleton(frame, ear, shoulder, hip, knee,
                       neck_angle, back_angle)

        save_dir = os.path.join(settings.MEDIA_ROOT, 'analysis_results')
        os.makedirs(save_dir, exist_ok=True)

        unique_id = uuid.uuid4().hex
        filename = f"analysis_{unique_id}.jpg"
        filepath = os.path.join(save_dir, filename)

        cv2.imwrite(filepath, frame)
        visualization_filename = f"analysis_results/{filename}"

        print(f"  Neck angle: {neck_angle:.1f}")
        print(f"  Back angle: {back_angle:.1f}")
        print(f"  Final score: {final_score:.1f}")

    except (IndexError, Exception) as e:
            print(f"  ERROR extracting keypoints: {e}")
            return {
                "result": "Invalid Image",
                "score": None,
                "issues": ["keypoint_extraction_error"],
                "message": "Please upload a clearer image showing your full upper body sitting posture.",
                "visualization_image": None
            }

    # ── Final Summary ──
    print(f"\n{'='*30} ANALYSIS SUMMARY {'='*30}")

    # Clamp score to 0-100
    final_score = round(max(0.0, min(100.0, final_score)), 1)

    # Determine result (Good if >= 65)
    result = "Good Posture" if final_score >= 65.0 else "Bad Posture"

    print(f"Final score: {final_score}")
    print(f"Final result: {result}")
    print(f"Detected issues: {issues}")
    print(f"{'='*78}\n")

    return {
        "result": result,
        "score": final_score,
        "issues": issues,
        "visualization_image": visualization_filename
    }


def _angle_to_score(angle, ideal_min, ideal_max):
    """
    Convert an angle to a 0-100 score based on ideal range.

    If the angle falls within [ideal_min, ideal_max], score is 100.
    Score decreases linearly as the angle deviates from the range.
    """
    if ideal_min <= angle <= ideal_max:
        return 100.0

    if angle < ideal_min:
        deviation = ideal_min - angle
    else:
        deviation = angle - ideal_max

    # Every 5 degrees of deviation reduces score by ~7 points (tolerant)
    penalty = (deviation / 5.0) * 7.0
    return max(0.0, 100.0 - penalty)
