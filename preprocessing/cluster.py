import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

NUM_CLUSTERS = 3

def load_coordinates(file):
    with open(file, 'r') as f:
        return json.load(f)

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

def cluster_frames(frames):
    scaler = StandardScaler()
    frames_normalized = scaler.fit_transform(frames)

    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
    labels = kmeans.fit_predict(frames_normalized)

    return labels, kmeans, frames_normalized

def filter_resting_frames(frames, labels):
    rest_clusters = [0]
    filtered_frames = [frame for i, frame in enumerate(frames) if labels[i] not in rest_clusters]
    return np.array(filtered_frames, dtype=np.float32)

def main():
    coordinates_dict = load_coordinates("../dataset/skeleton_dataset.json")
    features = extract_features(coordinates_dict)
    if features is not None:
        print(f"Final dataset shape: {features.shape}")

    labels, kmeans, normalized_frames = cluster_frames(features)
    print(f"Labels: {labels}")

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)

    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.5)
    plt.colorbar()
    plt.show()

    cluster_variances = [np.var(normalized_frames[labels == i]) for i in range(NUM_CLUSTERS)]
    rest_clusters = np.argsort(cluster_variances)[:1]

    print(f"Clusters detected as rest frames: {rest_clusters}")

    filtered_features = filter_resting_frames(features, labels)
    print(f"Format after removing rest frames: {filtered_features.shape}")

if __name__ == '__main__':
    main()
