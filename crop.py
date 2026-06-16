import cv2

def crop_video(input_path, output_path, x1, y1, x2, y2):
    """
    Crops a video and saves the result.

    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to save the cropped video.
        x1, y1 (int): Top-left corner of the crop region.
        x2, y2 (int): Bottom-right corner of the crop region.
    """
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

    if x1 >= x2 or y1 >= y2:
        raise ValueError("Coordonnées de recadrage invalides : x1 doit être < x2 et y1 doit être < y2")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la vidéo : {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    crop_width = x2 - x1
    crop_height = y2 - y1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None:
                continue
            cropped = frame[y1:y2, x1:x2]
            out.write(cropped)
    finally:
        cap.release()
        out.release()
    print(f"Cropped video saved to : {output_path}")