import cv2
import os

def cut_video_frames(input_video, seconds_to_cut_start=0.45, seconds_to_cut_end=0.45):
    filename = os.path.basename(input_video)
    output_video = f"/home/{os.getenv('USER')}/Videos/cut/{filename}"
    cap = cv2.VideoCapture(input_video)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_to_cut_start = int(fps * seconds_to_cut_start)
    frames_to_cut_end = int(fps * seconds_to_cut_end)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index >= frames_to_cut_start and frame_index < (total_frames - frames_to_cut_end):
            out.write(frame)

        frame_index += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video cut and saved to {output_video}")