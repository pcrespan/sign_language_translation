import cv2
import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

def load_valid_videos():
    path = "../dataset/valid_videos.txt"
    valid_videos = []

    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            valid_videos.append(line.strip())
    return valid_videos

def save_invalid_frames(valid_videos_invalid_frames,
                        path_to_save="../dataset/invalid",
                        base_path_to_videos=f"/home/{os.getenv('USER')}/Videos/",
                        seconds_to_consider_start=0.45,
                        seconds_to_consider_end=0.45):

    for input_video in valid_videos_invalid_frames:
        frame_counter = 1
        video_path = f"{base_path_to_videos}/{input_video}.mp4"
        print(video_path)
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames_to_get_start = int(fps * seconds_to_consider_start)
        frames_to_get_end = int(fps * seconds_to_consider_end)

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index <= frames_to_get_start or frame_index >= (total_frames - frames_to_get_end):
                filename = f"{path_to_save}/{input_video}_{frame_counter}.png"
                cv2.imwrite(filename, frame)
                print(f"Frame {frame_counter} saved")
                frame_counter += 1
            frame_index += 1

        cap.release()

def get_valid_videos_path(valid_videos):
    base_path = f"/home/{os.getenv('USER')}/Videos"
    video_paths = []

    for valid_video in valid_videos:
        print(valid_video)
        path = f"{base_path}/cut/{valid_video}.mp4"
        if os.path.exists(path):
            video_paths.append(path)

    return video_paths

def save_valid_frames(valid_videos, path_to_save="../dataset/valid"):
    valid_video_paths = get_valid_videos_path(valid_videos)

    for input_video, video_name in zip(valid_video_paths, valid_videos):
        frame_counter = 1
        cap = cv2.VideoCapture(input_video)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            filename = f"{path_to_save}/{video_name}_{frame_counter}.png"
            cv2.imwrite(filename, frame)
            frame_counter += 1
            print(f"Frame {frame_counter} saved")
        cap.release()

def split_dataset_by_video(
    src_root="../dataset",
    dst_root="../dataset_split",
    train_ratio=0.8,
    seed=42
):
    random.seed(seed)

    categories = ["valid", "invalid"]
    video_to_files = defaultdict(lambda: {"valid": [], "invalid": []})

    for category in categories:
        src_dir = Path(src_root) / category
        for file in src_dir.glob("*.png"):
            filename = file.name
            if "_" not in filename:
                continue
            video_id = filename.split("_")[0]
            video_to_files[video_id][category].append(file)

    all_videos = list(video_to_files.keys())
    random.shuffle(all_videos)
    split_index = int(len(all_videos) * train_ratio)
    train_videos = set(all_videos[:split_index])
    val_videos = set(all_videos[split_index:])

    print(f"Total videos: {len(all_videos)}")
    print(f"Train: {len(train_videos)} | Val: {len(val_videos)}")

    for video_id, files in video_to_files.items():
        split = "train" if video_id in train_videos else "val"
        for category in categories:
            for src_path in files[category]:
                dst_dir = Path(dst_root) / split / category
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_path = dst_dir / src_path.name
                shutil.copy(src_path, dst_path)

    print("Split finished.")

if __name__ == "__main__":
    split_dataset_by_video()

def main():
    #valid_videos = load_valid_videos()
    #save_valid_frames(valid_videos)
    #save_invalid_frames(valid_videos)
    split_dataset_by_video()

if __name__ == "__main__":
    main()