import cv2
import os
import mediapipe as mp
import numpy as np
from SimSwap.inference import simswap_inference

VIDEO_PATH = '../../CONTAR.mp4'
OUTPUT_VIDEO_PATH = 'output.mp4'
FRAMES_DIR = 'frames/'
SWAPPED_FRAMES_DIR = 'swapped_frames/'
FPS = 30
SYNTHETIC_FACE_PATH = '../assets/synthetic_face.jpg'

os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(SWAPPED_FRAMES_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_idx:04d}.jpg")
    cv2.imwrite(frame_path, frame)
    frame_idx += 1
cap.release()

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

frame_files = sorted(os.listdir(FRAMES_DIR))

for frame_file in frame_files:
    frame_path = os.path.join(FRAMES_DIR, frame_file)
    img = cv2.imread(frame_path)
    height, width, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)

    if not results.detections:
        output_img = img
    else:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        x_min = int(bbox.xmin * width)
        y_min = int(bbox.ymin * height)
        w_box = int(bbox.width * width)
        h_box = int(bbox.height * height)

        top = max(0, y_min)
        left = max(0, x_min)
        bottom = min(height, y_min + h_box)
        right = min(width, x_min + w_box)

        output_img = simswap_inference(
            source_img_path=SYNTHETIC_FACE_PATH,
            target_img=img,
            face_rect=(top, right, bottom, left)
        )

    swapped_path = os.path.join(SWAPPED_FRAMES_DIR, frame_file)
    cv2.imwrite(swapped_path, output_img)

swapped_frame_files = sorted(os.listdir(SWAPPED_FRAMES_DIR))
first_frame = cv2.imread(os.path.join(SWAPPED_FRAMES_DIR, swapped_frame_files[0]))
height, width, _ = first_frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (width, height))

for swapped_file in swapped_frame_files:
    frame_path = os.path.join(SWAPPED_FRAMES_DIR, swapped_file)
    frame = cv2.imread(frame_path)
    out.write(frame)

out.release()
print("Anonymized video saved in:", OUTPUT_VIDEO_PATH)
