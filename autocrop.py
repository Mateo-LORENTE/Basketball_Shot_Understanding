
import cv2
import random
import numpy as np
from ultralytics import YOLO

def autocroping(video_path, num_samples=300, band=40):
    """
    Automatically detects the cropping region using YOLO-based ball and pose detections.

    Args:
        video_path (str): Path to the video file.
        num_samples (int): Number of frames to sample.
        band (int): Extra margin around the detected region.

    Returns:
        tuple: (x1, y1, x2, y2) crop coordinates.
    """
    # Loading YOLO models
    model_ball = YOLO("Yolo/big.pt")
    model_pose = YOLO("Yolo/yolo11n-pose.pt")

    # open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la vidéo : {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Image sampling
    sample_indices = random.sample(range(total_frames), min(num_samples, total_frames))
    sample_frames = []
    valid_indices = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            sample_frames.append(frame)
            valid_indices.append(idx)

    cap.release()

    tops = []      # Minimum ball height (y1)
    bottoms = []   # Maximum foot height (y2)
    lefts = []     # Minimum left position of people (x1)
    rights = []    # Maximum right position of people (x2)

    for frame in sample_frames:
        res_ball = model_ball(frame, verbose=False)
        res_pose = model_pose(frame, verbose=False)

        # Ball detections
        for box in res_ball[0].boxes:
            x1_ball, y1_ball, x2_ball, y2_ball = map(int, box.xyxy[0])
            tops.append(y1_ball)

        # KP detections
        if res_pose[0].keypoints is not None:
            kps = res_pose[0].keypoints.xy.cpu().numpy()
            persons = []

            for person in kps:
                valid = person[(person[:, 0] > 0) & (person[:, 1] > 0)]
                if len(valid) > 0:
                    x_min = valid[:, 0].min()
                    x_max = valid[:, 0].max()
                    y_max = valid[:, 1].max()  # Pieds
                    persons.append((x_min, x_max, y_max))

            if persons:
                lefts.append(min(p[0] for p in persons))
                rights.append(max(p[1] for p in persons))
                bottoms.append(max(p[2] for p in persons))

    if not tops or not bottoms or not lefts or not rights:
        raise ValueError("Échec de la détection : listes vides")


    top = max(0, min(tops) - band)
    bottom = min(height, max(bottoms) + band)
    left = max(0, min(lefts) - band)
    right = min(width, max(rights) + band)


    left, top, right, bottom = map(int, (left, top, right, bottom))

    if left >= right or top >= bottom:
        raise ValueError("Coordonnées de recadrage invalides : left doit être < right et top doit être < bottom")

    print("RECADRAGE FINAL :")
    print(f"top={top}, bottom={bottom}, left={left}, right={right}")

    return left, top, right, bottom