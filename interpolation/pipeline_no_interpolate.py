import os
import subprocess
from standardize_video import standardize_video

video_list = ["IGREJA.mp4", "UNIR.mp4", "MOTO.mp4"]
input_dir = os.path.expanduser("~/Videos/cut_with_frame_classifier/")
standardized_dir = "./assets/standardized"
output_dir = "./assets/final_output"
target_size = (640, 480)
target_fps = 30

os.makedirs(standardized_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

standardized_videos = []
for name in video_list:
    in_path = os.path.join(input_dir, name)
    out_path = os.path.join(standardized_dir, name)
    standardize_video(in_path, out_path, target_size, target_fps)
    standardized_videos.append(out_path)

with open("list.txt", "w") as f:
    for video in standardized_videos:
        f.write(f"file '{os.path.abspath(video)}'\n")

final_output = os.path.join(output_dir, "final_video.mp4")
subprocess.run([
    "ffmpeg", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", final_output
])