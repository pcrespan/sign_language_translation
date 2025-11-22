import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils

class FingerLandmarks:
    def __init__(self, handedness="Unknown"):
        self.handedness = handedness
        self.thumb = []
        self.index = []
        self.middle = []
        self.ring = []
        self.pinky = []

    def add_landmarks(self, finger, landmarks):
        if finger == 'thumb':
            self.thumb = landmarks
        elif finger == 'index':
            self.index = landmarks
        elif finger == 'middle':
            self.middle = landmarks
        elif finger == 'ring':
            self.ring = landmarks
        elif finger == 'pinky':
            self.pinky = landmarks

    def __repr__(self):
        return (f"Handedness: {self.handedness}\n"
                f"Thumb: {self.thumb}\n"
                f"Index: {self.index}\n"
                f"Middle: {self.middle}\n"
                f"Ring: {self.ring}\n"
                f"Pinky: {self.pinky}")

def draw_finger_contours(frame, landmarks, handedness):
    finger_connections = [
        [4, 3, 2, 1],  # polegar
        [8, 7, 6, 5],  # indicador
        [12, 11, 10, 9],  # médio
        [16, 15, 14, 13],  # anelar
        [20, 19, 18, 17]  # mínimo
    ]

    finger_landmarks = FingerLandmarks(handedness)

    for finger, finger_name in zip(finger_connections, ['thumb', 'index', 'middle', 'ring', 'pinky']):
        coordinates = []
        for i in range(len(finger) - 1):
            x1, y1 = int(landmarks[finger[i]].x * frame.shape[1]), int(landmarks[finger[i]].y * frame.shape[0])
            x2, y2 = int(landmarks[finger[i + 1]].x * frame.shape[1]), int(landmarks[finger[i + 1]].y * frame.shape[0])
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            coordinates.append((x1, y1))
            coordinates.append((x2, y2))
        finger_landmarks.add_landmarks(finger_name, coordinates)
    return frame, finger_landmarks

def invert_hand_label(label):
    return "Right" if label == "Left" else "Left"

def detect_hands(mp_hands, hands, frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = invert_hand_label(handedness.classification[0].label)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            frame, finger_landmarks = draw_finger_contours(frame, hand_landmarks.landmark, hand_label)
            #print(finger_landmarks)
    return frame