import cv2
import os

def images_to_video(image_dir, output_path, size, fps):
    images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    for img in images:
        frame = cv2.imread(os.path.join(image_dir, img))
        frame_resized = cv2.resize(frame, size)
        out.write(frame_resized)
    out.release()