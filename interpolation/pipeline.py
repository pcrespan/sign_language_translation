import os
import subprocess
from standardize_video import standardize_video
from transition_frame_extractor import extract_transition_frames
from run_rife import run_rife_between_frames
from make_video import images_to_video

video_list = ["UNIR.mp4", "IGREJA.mp4"]

base_dir = os.path.dirname(__file__)
input_dir = "../input_videos"
standardized_dir = "/home/pcrespan/ic/assets/standardized"
transition_dir = "/home/pcrespan/ic/assets/transitions"
output_dir = "/home/pcrespan/ic/assets/final_output/"

target_size = (640, 480)
target_fps = 30
num_interp_frames = 5

os.makedirs(standardized_dir, exist_ok=True)
os.makedirs(transition_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

standardized_videos = []
for name in video_list:
    in_path = os.path.join(input_dir, name)
    out_path = os.path.join(standardized_dir, name)
    standardize_video(in_path, out_path, target_size, target_fps)
    standardized_videos.append(out_path)

transition_videos = []
for i in range(len(standardized_videos) - 1):
    va = standardized_videos[i]
    vb = standardized_videos[i + 1]
    base_name = f"t{i}_{i+1}"
    t_dir = os.path.join(transition_dir, base_name)
    os.makedirs(t_dir, exist_ok=True)

    fa = os.path.join(t_dir, "frame_a.png")
    fb = os.path.join(t_dir, "frame_b.png")
    extract_transition_frames(va, vb, fa, fb)

    run_rife_between_frames(fa, fb, t_dir, num_interp_frames)

    interpolated_dir = t_dir
    out_video = os.path.join(t_dir, f"{base_name}.mp4")
    images_to_video(interpolated_dir, out_video, target_size, target_fps)
    transition_videos.append(out_video)

list_path = os.path.join(base_dir, "list.txt")
with open(list_path, "w") as f:
    for i, video in enumerate(standardized_videos):
        f.write(f"file '{os.path.abspath(video)}'\n")
        if i < len(transition_videos):
            f.write(f"file '{os.path.abspath(transition_videos[i])}'\n")

final_output = os.path.join(output_dir, "final_video.mp4")
subprocess.run([
    "ffmpeg", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", final_output
])