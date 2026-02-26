
import cv2

#fonction
def lire_video(chemin):
    """Lit une vidéo et affiche chaque frame."""
    cap = cv2.VideoCapture(chemin)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Chemin relatif depuis le fichier Python dans Basketball/
    lire_video("video/ma_video.mp4")

