import cv2

# Ouvrir la vidéo
cap = cv2.VideoCapture("video/video2.mp4")

# Récupérer les propriétés
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Définir le crop
#x1, y1 = 70, 200
#x2, y2 = width - 70, height - 250
x1, y1 = 0, 100
x2, y2 = width - 0, height - 250

print(x2)
print(y2)

crop_width = x2 - x1
crop_height = y2 - y1

# Writer pour sauvegarder
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("video/output2.mp4", fourcc, fps, (crop_width, crop_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop (IMPORTANT : [y1:y2, x1:x2])
    cropped = frame[y1:y2, x1:x2]

    # Écrire la frame cropée
    out.write(cropped)

# Libérer les ressources
cap.release()
out.release()

# Message une fois la vidéo entière traitée
print("Done")


cap = cv2.VideoCapture("video/output2.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("frame", frame)

    if cv2.waitKey(10)== ord('q'):
        break

cap.release()
cv2.destroyAllWindows()