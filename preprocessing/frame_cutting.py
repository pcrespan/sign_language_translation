import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from variance import load_coordinates, extract_features, filter_rest_frames, get_video_frame_number
import time_slicing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = f"/home/{os.getenv('USER')}/Videos"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def list_videos(path):
    videos = []

    for video_file in os.listdir(path):
        if video_file.endswith(".mp4"):
            videos.append(video_file)

    return videos

def cut_by_time(videos):

    for video_name in videos:
        time_slicing.cut_video_frames(f"{base_path}/{video_name}")

def cut_with_frame_classifier(model, invalid_videos):
    model.eval()
    counter = 1

    for invalid_video in invalid_videos:
        output_video = f"{base_path}/cut_with_frame_classifier/{invalid_video}"
        input_video = f"{base_path}/{invalid_video}"
        cap = cv2.VideoCapture(input_video)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = transform(img).unsqueeze(0).to(device) # specify that one image is being passed

            with torch.no_grad():
                output = model(input_tensor)
                predicted = torch.argmax(output, dim=1).item()

            print("Predicted:", predicted)

            if predicted == 1:
                out.write(frame)
                counter += 1

        print(f"Video {invalid_video} cut.")

        cap.release()
        out.release()
        cv2.destroyAllWindows()

def load_model():
    model_path = "../models/frame_classifier_v5.pth"
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 2)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def main():
    model = load_model()
    videos = list_videos(f"{base_path}") # cutting videos that were previously cut using time slicing
    cut_with_frame_classifier(model, videos)


if __name__ == '__main__':
    main()