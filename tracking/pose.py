import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils

def draw_filtered_landmarks(mp_pose, frame, pose_landmarks):
    landmarks = pose_landmarks.landmark
    points = []

    filtered_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP
    ]

    for idx in filtered_landmarks:
        x = int(landmarks[idx].x * frame.shape[1])
        y = int(landmarks[idx].y * frame.shape[0])
        points.append((x, y))

    for point in points:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)

    if len(points) >= 6:
        cv2.line(frame, points[0], points[1], (255, 0, 0), 2)  # left shoulder -> left elbow
        cv2.line(frame, points[1], points[2], (255, 0, 0), 2)  # left elbow -> left wrist
        cv2.line(frame, points[3], points[4], (255, 0, 0), 2)  # right shoulder -> right elbow
        cv2.line(frame, points[4], points[5], (255, 0, 0), 2)  # right elbow -> right wrist
        cv2.line(frame, points[0], points[3], (0, 255, 0), 2)  # left shoulder -> right shoulder
        cv2.line(frame, points[6], points[7], (0, 0, 255), 2) # left hip -> right hip
        cv2.line(frame, points[0], points[6], (0, 0, 255), 2) # left shoulder -> left hip
        cv2.line(frame, points[3], points[7], (0, 0, 255), 2) # right shoulder -> right hip

    return frame

def detect_pose(mp_pose, pose, frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks
        frame = draw_filtered_landmarks(mp_pose, frame, pose_landmarks)
    return frame