import cv2

def extract_transition_frames(video_a, video_b, frame_a_path, frame_b_path):
    cap_a = cv2.VideoCapture(video_a)
    cap_b = cv2.VideoCapture(video_b)
    total_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_a.set(cv2.CAP_PROP_POS_FRAMES, total_a - 1)
    ret_a, frame_a = cap_a.read()
    cv2.imwrite(frame_a_path, frame_a)
    cap_b.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret_b, frame_b = cap_b.read()
    cv2.imwrite(frame_b_path, frame_b)
    cap_a.release()
    cap_b.release()