import json

import cv2
import mediapipe as mp
import os

from tracking.coordinates_utils import extract_landmarks
from tracking.face import detect_face
from tracking.hands import detect_hands
from tracking.pose import detect_pose

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.4,
               min_tracking_confidence=0.4,
               model_complexity=1)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.4,
                    min_tracking_confidence=0.4)

def main():
    base_path = f"/home/{os.getenv('USER')}/Videos"
    dataset_base_path = "../dataset/skeleton/"
    interpreter_dirs = ["interpreter1", "interpreter2", "interpreter3"]
    coordinates_per_frame = []

    with open("../dataset/frames_per_video.json", "w") as f:
        sign_video = []

        for interpreter_dir in interpreter_dirs:
            interpreter_path = os.path.join(base_path, interpreter_dir)

            for video_file in os.listdir(interpreter_path):
                path = os.path.join(interpreter_path, video_file)
                if video_file.endswith(".mp4"):
                    sign = os.path.splitext(video_file)[0]

                    capture = cv2.VideoCapture(path)
                    print(f"Processing sign {sign}...")

                    frame_idx = 0

                    while True:
                        ret, frame = capture.read()
                        if not ret:
                            print("Error capturing frame")
                            break

                        print(frame_idx)

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results_face = face_mesh.process(frame_rgb)
                        results_hands = hands.process(frame_rgb)
                        results_pose = pose.process(frame_rgb)
                        extract_landmarks(coordinates_per_frame, results_face, results_hands, results_pose, frame_idx)

                        #frame = detect_face(mp_face, face_mesh, frame)
                        #frame = detect_pose(mp_pose, pose, frame)
                        #frame = detect_hands(mp_hands, hands, frame)

                        cv2.imshow("Body Landmarks", frame)

                        frame_idx += 1

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    with open(dataset_base_path + interpreter_dir + "/" + sign + ".json", "w") as sign_coordinates:
                        sign_coordinates.write(json.dumps(coordinates_per_frame))

                    sign_video.append(frame_idx)

        f.write(json.dumps(sign_video))

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()