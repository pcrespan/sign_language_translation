import cv2
import os

def standardize_video(input_path, output_path, size, fps):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, size)
        out.write(frame_resized)
    cap.release()
    out.release()