import cv2
import numpy as np
import random

def brighten(frame, alpha=1.7, beta=60):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

# Chemin vers ta vidéo
video_path = "video/output2.mp4"

# Ouvre la vidéo
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

# Récupère une frame aléatoire
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
random_frame_index = random.randint(0, total_frames - 1)
cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)
ret, frame = cap.read()

if not ret:
    print("Erreur : Impossible de lire la frame.")
    exit()

# Demande à l'utilisateur de choisir alpha
alpha = float(input("Entrez la valeur d'alpha (ex: 1.7) : "))

# Liste des valeurs de beta à tester
beta_values = [0, 15, 30, 45, 60]

# Crée une image combinée pour afficher toutes les versions
combined = np.hstack([brighten(frame, alpha=alpha, beta=beta) for beta in beta_values])

# Affiche la frame originale et les versions modifiées
cv2.imshow("Frame originale", frame)
cv2.imshow(f"Frame avec alpha={alpha} et beta=[0, 15, 30, 45, 60]", combined)

# Attend une touche pour fermer
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()