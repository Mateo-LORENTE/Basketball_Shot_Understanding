import cv2

cap = cv2.VideoCapture("video/video2.mp4.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("frame", frame)

    if cv2.waitKey(50)== ord('q'):
        break

cap.release()
cv2.destroyAllWindows()