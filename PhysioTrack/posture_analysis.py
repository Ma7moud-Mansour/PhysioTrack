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


def analyze_posture(video_path):
    """
    Analyze posture from a video file.

    Opens the video with OpenCV, samples every 5th frame, uses YOLOv8 Pose
    to detect body keypoints, and calculates neck/back angles.

    Args:
        video_path (str): Absolute path to the video file.

    Returns:
        dict: {
            "result": "Good Posture" or "Bad Posture",
            "score": float (0-100 percentage),
            "issues": list of detected issue strings,
            "visualization_image": str or None
        }
    """
    # ── Debug: Video metadata ──
    print(f"\n{'='*50}")
    print(f"VIDEO PATH: {video_path}")

    cap = cv2.VideoCapture(video_path)
    print(f"VIDEO OPENED: {cap.isOpened()}")

    if not cap.isOpened():
        print("ERROR: Could not open video file.")
        return {"result": "Bad Posture", "score": 0.0}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"FPS: {fps}")
    print(f"FRAME COUNT: {total_frame_count}")
    print(f"RESOLUTION: {frame_width}x{frame_height}")
    print(f"{'='*50}\n")

    # ── Counters & Trackers ──
    total_frames_read = 0
    frames_processed = 0
    frames_with_landmarks = 0
    frame_scores = []
    neck_angles = []
    back_angles = []
    frame_index = 0
    saved_visualization = False
    visualization_filename = None

    # Temporal smoothing: store last 5 frames of landmark positions
    ear_history = deque(maxlen=5)
    shoulder_history = deque(maxlen=5)
    hip_history = deque(maxlen=5)
    knee_history = deque(maxlen=5)

    # ── Video Processing Loop ──
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print(f"Failed to read frame at index {frame_index}. Exiting loop.")
            break

        total_frames_read += 1

        # Process every 5th frame for performance
        if frame_index % 5 != 0:
            frame_index += 1
            continue

        frame_index += 1
        frames_processed += 1

        # Normalize frame resolution before pose detection
        frame = cv2.resize(frame, (640, 480))

        # ── [YOLOv8] Run pose detection ──
        # verbose=False suppresses per-frame console output from YOLO
        results = _yolo_model(frame, verbose=False)

        print(f"Frame index: {frame_index}")

        # Check if any person was detected with keypoints
        has_keypoints = (
            results[0].keypoints is not None
            and len(results[0].keypoints.xy) > 0
        )
        print(f"  Pose result: {has_keypoints}")

        if has_keypoints:
            print("  LANDMARKS DETECTED")
            frames_with_landmarks += 1

            # [YOLOv8] Extract keypoints for the first detected person
            # keypoints.xy shape: (num_persons, 17, 2) — pixel coordinates
            kp = results[0].keypoints.xy[0]

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

                # Skip frame if any required landmark is missing
                if any(pt is None for pt in [ear, shoulder, hip, knee]):
                    print("  Skipping frame: one or more keypoints not detected")
                    continue

                # Temporal smoothing: accumulate and average
                ear_history.append(ear)
                shoulder_history.append(shoulder)
                hip_history.append(hip)
                knee_history.append(knee)

                ear = _smooth_point(ear_history)
                shoulder = _smooth_point(shoulder_history)
                hip = _smooth_point(hip_history)
                knee = _smooth_point(knee_history)

                # Neck angle: ear → shoulder → hip
                neck_angle = _calculate_angle(ear, shoulder, hip)

                # Back angle: shoulder → hip → knee
                back_angle = _calculate_angle(shoulder, hip, knee)

                # Scoring logic (relaxed thresholds for video-based detection):
                # Good neck angle: ~135-180 degrees (upright)
                # Good back angle: ~145-180 degrees (straight back)
                neck_score = _angle_to_score(neck_angle, ideal_min=135, ideal_max=180)
                back_score = _angle_to_score(back_angle, ideal_min=145, ideal_max=180)

                # Combined frame score (neck 50%, back 50%)
                frame_score = (neck_score * 0.5) + (back_score * 0.5)
                frame_scores.append(frame_score)
                neck_angles.append(neck_angle)
                back_angles.append(back_angle)

                # ── Visualization ──
                if not saved_visualization:
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
                    saved_visualization = True

                print(f"  Neck angle: {neck_angle:.1f}")
                print(f"  Back angle: {back_angle:.1f}")
                print(f"  Frame score: {frame_score:.1f}")

            except (IndexError, Exception) as e:
                print(f"  ERROR extracting keypoints: {e}")
                continue
        else:
            print("  No landmarks detected in this frame.")

    cap.release()

    # ── Final Summary ──
    print(f"\n{'='*30} ANALYSIS SUMMARY {'='*30}")
    print(f"Total frames read: {total_frames_read}")
    print(f"Frames processed (every 5th): {frames_processed}")
    print(f"Frames with landmarks: {frames_with_landmarks}")

    if not frame_scores:
        print("WARNING: No pose landmarks detected in the entire video.")
        print(f"{'='*78}\n")
        return {"result": "Bad Posture", "score": 0.0, "issues": ["forward_head", "rounded_back"]}

    avg_score = sum(frame_scores) / len(frame_scores)

    # Clamp score to 0-100
    final_score = round(max(0.0, min(100.0, avg_score)), 1)

    # Determine result (Good if >= 65)
    result = "Good Posture" if final_score >= 65.0 else "Bad Posture"

    print(f"Average score: {final_score}")
    print(f"Final result: {result}")
    # ── Detect posture issues based on average angles ──
    avg_neck = sum(neck_angles) / len(neck_angles)
    avg_back = sum(back_angles) / len(back_angles)
    issues = []
    if avg_neck < 135:
        issues.append("forward_head")
    if avg_back < 145:
        issues.append("rounded_back")

    print(f"Avg neck angle: {avg_neck:.1f}")
    print(f"Avg back angle: {avg_back:.1f}")
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
