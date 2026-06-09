from ultralytics import YOLO
import cv2
from old.utils import get_device

# Couleur par classe
CLASS_COLORS = {
    0: (255, 0, 0),    # ball  — bleu
    1: (0, 200, 0),    # rim   — vert
    2: (0,0,180),
}
DEFAULT_COLOR = (0, 200, 255)  # autres classes — jaune

class ShotDetector:
    def __init__(self):
        self.model = YOLO("Yolo/big.pt")
        self.device = get_device()

        self.cap = cv2.VideoCapture("video/output2.mp4")

        self.frames = []
        self.detections = {}

        self.load_video_and_detect()
        self.display()

    # -------------------------------
    # 🟦 YOLO detection
    # -------------------------------
    def load_video_and_detect(self):
        frame_id = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frames.append(frame)
            results = self.model(frame, device=self.device)

            bboxes = []
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf > 0.3:
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    cls_name = self.model.names.get(cls, str(cls))
                    bboxes.append((x1, y1, w, h, conf, cls, cls_name))

            self.detections[frame_id] = bboxes
            frame_id += 1

        self.cap.release()

    # -------------------------------
    # 🟥 DISPLAY
    # -------------------------------
    def display(self):
        i = 0
        total_frames = len(self.frames)

        while True:
            frame = self.frames[i].copy()

            bboxes = self.detections.get(i, [])

            if bboxes:
                for x, y, w, h, conf, cls, cls_name in bboxes:
                    color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{cls_name} {conf:.2f}", (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.putText(frame, f"{len(bboxes)} detection(s)", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "No detection", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, f"Frame {i}/{total_frames}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)

            cv2.imshow("YOLO Detections", frame)

            key = cv2.waitKey(0) & 0xFF

            if key == 27:
                break
            elif key == ord('d'):
                i = min(i + 1, total_frames - 1)
            elif key == ord('q'):
                i = max(i - 1, 0)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    ShotDetector()