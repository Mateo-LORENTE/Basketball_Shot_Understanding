from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2


def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
  lw = max(round(sum(image.shape) / 2 * 0.003), 2)
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
  if label:
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(image,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)
    
def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=.001):
  #Define COCO Labels
  if labels == []:
    labels = {0: 'person', 32: 'sports ball'}
  #Define colors
  if colors == []:
    colors = {
        0: (89,161,197),
        32: (67,161,255)
    }
  
  #plot each boxes
  for box in boxes:
    #add score in label if score=True
    if score :
      label = labels[int(box[-1])] + " " + str(round(100 * float(box[-2]),1)) + "%"
    else :
      label = labels[int(box[-1])]
    #filter every box under conf threshold if conf threshold setted
    if conf :
      if box[-2] > conf:
        color = colors[int(box[-1])]
        box_label(image, box, label, color)
    else:
      color = colors[int(box[-1])]
      box_label(image, box, label, color)

  #show image
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  cv2.imshow("YOLO Detection", image)
  cv2.waitKey(0)  # attendre que tu appuies sur une touche
  cv2.destroyAllWindows()




model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("video/ma_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = np.asarray(frame).copy()

    results = model.predict(image, classes=[0,32])

    plot_bboxes(image, results[0].boxes.data, score=True)
    if cv2.waitKey(0.1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

