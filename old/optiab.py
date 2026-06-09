import cv2
import numpy as np
import random
import itertools
from ultralytics import YOLO

# --- Fonction pour appliquer brighten ---
def brighten(frame, alpha, beta):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

# --- Fonction pour calculer la somme des confiances ---
def sum_confidences(model_ball, frame, device):
    brightened = brighten(frame, alpha=1.0, beta=0)  # alpha/beta seront remplacés dans la boucle
    results_ball = model_ball(brightened, device=device)

    total_conf = 0.0
    for box in results_ball[0].boxes:
        conf = float(box.conf[0])
        total_conf += conf
    return total_conf

# --- Fonction principale pour trouver les meilleurs alpha/beta ---
def find_optimal_alpha_beta(video_path, model_ball, device, num_frames=100):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Impossible d'ouvrir la vidéo.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        num_frames = total_frames

    # Sélectionner 50 frames aléatoires (les mêmes pour tous les couples)
    frame_indices = random.sample(range(total_frames), num_frames)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    # Valeurs à tester
    alpha_values = [1.0, 1.2, 1.5, 1.7, 2.0]
    beta_values = [0, 15, 30, 45, 60]

    best_alpha, best_beta = 1.0, 0
    best_total_conf = -1

    # Tester toutes les combinaisons
    for alpha, beta in itertools.product(alpha_values, beta_values):
        total_conf = 0.0
        for frame in frames:
            brightened = brighten(frame, alpha=alpha, beta=beta)
            results_ball = model_ball(brightened, device=device)

            # Calculer la somme des confiances pour cette frame
            frame_conf = 0.0
            for box in results_ball[0].boxes:
                conf = float(box.conf[0])
                frame_conf += conf
            total_conf += frame_conf

        # Mettre à jour les meilleurs paramètres
        if total_conf > best_total_conf:
            best_total_conf = total_conf
            best_alpha, best_beta = alpha, beta

        print(f"alpha={alpha}, beta={beta} -> Total conf: {total_conf:.2f}")

    print(f"\nMeilleurs paramètres : alpha={best_alpha}, beta={best_beta} (Total conf: {best_total_conf:.2f})")
    return best_alpha, best_beta

# --- Exemple d'utilisation ---
if __name__ == "__main__":
    model_ball = YOLO("Yolo/big.pt")
    video_path = "video/output2.mp4"
    # Remplace `model_ball` et `device` par ton modèle et ton device (ex: "cuda" ou "cpu")
    best_alpha, best_beta = find_optimal_alpha_beta(video_path, model_ball, device="cpu")