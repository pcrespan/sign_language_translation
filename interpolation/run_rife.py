import os
import shutil
import subprocess
import math

def run_rife_between_frames(frame_a_path, frame_b_path, output_dir, num_frames):
    rife_dir = os.path.join(os.path.dirname(__file__), "rife")
    rife_output_dir = os.path.join(rife_dir, "output")

    if os.path.exists(rife_output_dir):
        shutil.rmtree(rife_output_dir)
    os.makedirs(rife_output_dir)

    exp = math.ceil(math.log2(num_frames + 1))

    subprocess.run([
        "python3", "inference_img.py",
        "--img", frame_a_path, frame_b_path,
        "--exp", str(exp)
    ], cwd=rife_dir)

    os.makedirs(output_dir, exist_ok=True)

    for f in os.listdir(rife_output_dir):
        src = os.path.join(rife_output_dir, f)
        dst = os.path.join(output_dir, f)
        shutil.move(src, dst)