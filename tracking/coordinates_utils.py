def extract_landmarks(coordinates, results_face, results_hands, results_pose, frame_idx):

    frame_coordinates = {}
    frame_coordinates["frame"] = frame_idx

    if not frame_coordinates.get("face"):
        frame_coordinates["face"] = []

    if not frame_coordinates.get("pose"):
        frame_coordinates["pose"] = []

    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            for i, lm in enumerate(face_landmarks.landmark):
                frame_coordinates["face"].append((lm.x, lm.y, lm.z))

    if results_hands.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
            hand_label = "right_hand" if handedness.classification[0].label == "Right" else "left_hand"

            if not frame_coordinates.get(hand_label):
                frame_coordinates[hand_label] = []

            for i, lm in enumerate(hand_landmarks.landmark):
                frame_coordinates[hand_label].append((lm.x, lm.y, lm.z))

    if results_pose.pose_landmarks:
        for i, lm in enumerate(results_pose.pose_landmarks.landmark):
            frame_coordinates["pose"].append((lm.x, lm.y, lm.z))

    coordinates.append(frame_coordinates)