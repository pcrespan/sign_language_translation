import json
import numpy as np
import cv2

def load_coordinates(file):
    with open(file, 'r') as f:
        return json.load(f)

def get_video_frame_number():
    with open("../dataset/frames_per_video.json", "r") as f:
        data = json.load(f)
    return data

def extract_features(coordinates_dict):
    frames = []
    frame_sizes = []

    for interpreter, sign_dict in coordinates_dict.items():
        for sign, frame_list in sign_dict.items():
            for frame in frame_list:
                frame_coordinates = []

                for body_part, coordinates in frame.items():
                    if isinstance(coordinates, list):
                        flat_list = [item for sublist in coordinates for item in (sublist if isinstance(sublist, list) else [sublist])]
                        frame_coordinates.extend(flat_list)

                frame_sizes.append(len(frame_coordinates))
                frames.append(frame_coordinates)

    max_features = max(frame_sizes)
    print(f"Max detected feature size: {max_features}")

    for i in range(len(frames)):
        if len(frames[i]) < max_features:
            frames[i].extend([0.0] * (max_features - len(frames[i])))
        elif len(frames[i]) > max_features:
            frames[i] = frames[i][:max_features]

    if any(not isinstance(frame, list) or any(not isinstance(x, (int, float)) for x in frame) for frame in frames):
        print("Error: Nested or non-numeric elements detected!")
        for i, frame in enumerate(frames):
            if not isinstance(frame, list) or any(not isinstance(x, (int, float)) for x in frame):
                print(f"Invalid structure in frame {i}:", frame)
        return None

    return np.array(frames, dtype=np.float32)

def extract_true_removed_frame_indices(removed_frame_indices):
    number_frames = get_video_frame_number()
    frames_per_video = [[] for _ in range(len(number_frames))]

    for index in removed_frame_indices:
        cumulative_frames = 0
        for video_index, frames_in_video in enumerate(number_frames):
            if cumulative_frames <= index < cumulative_frames + frames_in_video:
                adjusted_index = index - cumulative_frames
                frames_per_video[video_index].append(adjusted_index)
                break
            cumulative_frames += frames_in_video

    return frames_per_video

def fill_frame_gaps(frame_list, total_frames, max_gap_size=5):
    if not frame_list:
        return []

    frame_list = sorted(set(frame_list))
    filled = []

    for i in range(len(frame_list) - 1):
        current = frame_list[i]
        next_frame = frame_list[i + 1]
        filled.append(current)

        gap = next_frame - current
        if 1 < gap <= max_gap_size:
            filled.extend(range(current + 1, next_frame))

    filled.append(frame_list[-1])

    start_block = []
    for i in range(len(filled)):
        if i == 0 or filled[i] == filled[i - 1] + 1:
            start_block.append(filled[i])
        else:
            break
    if start_block and start_block[0] > 0:
        start_block = list(range(0, start_block[-1] + 1))

    end_block = []
    for i in range(len(filled) - 1, -1, -1):
        if i == len(filled) - 1 or filled[i] == filled[i + 1] - 1:
            end_block.insert(0, filled[i])
        else:
            break
    if end_block:
        end_block = list(range(end_block[0], total_frames))

    final_frames = sorted(set(start_block + filled + end_block))
    return final_frames

def filter_rest_frames(frames, threshold=0.01):
    initial_frame = frames[0]
    variances = np.var(frames - initial_frame, axis=1)

    rest_frames = variances < threshold
    removed_indices = np.where(rest_frames)[0]
    filtered_frames = frames[~rest_frames]
    print(removed_indices)
    removed_indices = extract_true_removed_frame_indices(removed_indices)

    frames_per_video = get_video_frame_number()
    removed_frames_per_video = []

    for removed_frames, num_frames in zip(removed_indices, frames_per_video):
        removed_frames_per_video.append(fill_frame_gaps(removed_frames, num_frames, 50))

    return filtered_frames, variances, removed_frames_per_video

def cut_video_frames(input_video, frames_to_cut, counter):
    output_video = f"/home/enid/Videos/interpreter1/cut/{counter}.mp4"
    counter += 1
    cap = cv2.VideoCapture(input_video)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        if frame_index not in frames_to_cut:
            out.write(frame)

        frame_index += 1

    print(f"Video #{counter} saved.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()