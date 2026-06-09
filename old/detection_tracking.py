from ultralytics import YOLO
import cv2
import numpy as np
from old.utils import get_device

class ShotDetector:
    def __init__(self):
        self.model = YOLO("Yolo/big.pt")
        self.device = get_device()

        self.cap = cv2.VideoCapture("video/output2.mp4")

        self.frames = []
        self.detections = {}

        self.forward_track = {}
        self.backward_track = {}
        self.final_track = {}

        self.load_video_and_detect()
        self.forward_tracking()
        self.backward_tracking()
        self.display()

    # -------------------------------
    # 🟦 YOLO detection
    # -------------------------------
    def brighten(self, frame, alpha=1.3, beta=60):
        """
        alpha : contraste (1.0 = inchangé, >1 = plus de contraste)
        beta  : luminosité ajoutée (0 = inchangé, 30-60 = gain notable)
        """
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    def load_video_and_detect(self):
        frame_id = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frames.append(frame)
            #rame = cv2.resize(frame, (640, 360))
            #brightened = self.brighten(frame)
            results = self.model(frame, device=self.device)

            bbox = None
            for box in results[0].boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.3:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    conf = float(box.conf[0])
                    bbox = (x1, y1, w, h, conf)
                    break

            self.detections[frame_id] = bbox
            frame_id += 1

        self.cap.release()
        self.remove_static_detections()

    def remove_static_detections(self, min_occurrences=10, move_threshold=4):
        """
        Regroupe les détections par position approchée.
        Si un même endroit est détecté min_occurrences fois sans bouger
        de plus de move_threshold pixels → toutes ces détections sont supprimées.
        """

        # Collecter tous les centres détectés
        detected_frames = [
            (fid, det) for fid, det in self.detections.items() if det is not None
        ]

        # Construire des clusters de positions statiques
        # Chaque cluster = liste de frame_ids proches en position
        clusters = []  # [(center, [frame_ids])]

        for fid, det in detected_frames:
            x, y, w, h, _ = det
            cx, cy = x + w / 2, y + h / 2

            matched = False
            for cluster in clusters:
                ccx, ccy = cluster['center']
                dist = np.linalg.norm(np.array([cx, cy]) - np.array([ccx, ccy]))
                if dist < move_threshold:
                    cluster['frames'].append(fid)
                    matched = True
                    break

            if not matched:
                clusters.append({'center': (cx, cy), 'frames': [fid]})

        # Supprimer les détections appartenant à un cluster trop grand (= objet fixe)
        removed = 0
        for cluster in clusters:
            if len(cluster['frames']) >= min_occurrences:
                for fid in cluster['frames']:
                    self.detections[fid] = None
                    removed += 1
                print(f"  [STATIC] Cluster en {cluster['center']} — "
                    f"{len(cluster['frames'])} détections supprimées")

        print(f"[STATIC FILTER] {removed} détections supprimées au total")

    # -------------------------------
    # 🟩 Forward tracking
    # -------------------------------
    def forward_tracking(self):
        tracker = None
        tracking = False

        for i in range(len(self.frames)):
            frame = self.frames[i]
            det = self.detections[i]

            if det is not None:
                x, y, w, h, _ = det
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x, y, w, h))
                tracking = True
                self.forward_track[i] = (x, y, w, h)

            elif tracking:
                success, bbox = tracker.update(frame)

                if success:
                    self.forward_track[i] = bbox
                else:
                    tracking = False
                    self.forward_track[i] = None
            else:
                self.forward_track[i] = None

    # -------------------------------
    # 🟪 Backward tracking
    # -------------------------------
    def backward_tracking(self):
        tracker = None
        tracking = False

        for i in reversed(range(len(self.frames))):
            frame = self.frames[i]
            det = self.detections[i]

            if det is not None:
                x, y, w, h, _ = det
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x, y, w, h))
                tracking = True
                self.backward_track[i] = (x, y, w, h)

            elif tracking:
                success, bbox = tracker.update(frame)

                if success:
                    self.backward_track[i] = bbox
                else:
                    tracking = False
                    self.backward_track[i] = None
            else:
                self.backward_track[i] = None

 

    # -------------------------------
    # 🔎 helper YOLO nearest
    # -------------------------------
    def find_nearest_yolo(self, index):
        prev_idx = None
        next_idx = None

        for i in range(index, -1, -1):
            if self.detections.get(i) is not None:
                prev_idx = i
                break

        for i in range(index, len(self.frames)):
            if self.detections.get(i) is not None:
                next_idx = i
                break

        return prev_idx, next_idx

    # -------------------------------
    # 🟥 DISPLAY
    # -------------------------------
    def display(self):
        i = 0
        total_frames = len(self.frames)

        while True:
            frame = self.frames[i].copy()

            det = self.detections.get(i)
            f = self.forward_track.get(i)
            b = self.backward_track.get(i)

            # --- YOLO (bleu + confidence)
            if det is not None:
                x, y, w, h, conf = det
                x, y, w, h = int(x), int(y), int(w), int(h)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
                cv2.putText(frame, f"YOLO conf: {conf:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
            """
            # --- Forward (vert)
            if f is not None:
                x, y, w, h = map(int, f)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, "FORWARD", (x, y-25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # --- Backward (violet)
            if b is not None:
                x, y, w, h = map(int, b)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,255), 2)
                cv2.putText(frame, "BACKWARD", (x, y-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
            """
            # -------------------------------
            # 🎯 DECISION LOGIC
            # -------------------------------
            prev_idx, next_idx = self.find_nearest_yolo(i)

            decision_bbox = None

            if prev_idx is not None and next_idx is not None:
                if abs(next_idx - i) < abs(i - prev_idx):
                    decision_bbox = self.backward_track.get(i)
                else:
                    decision_bbox = self.forward_track.get(i)

            elif prev_idx is not None:
                decision_bbox = self.forward_track.get(i)

            elif next_idx is not None:
                decision_bbox = self.backward_track.get(i)

            # --- Decision (orange)
            if decision_bbox is not None:
                x, y, w, h = map(int, decision_bbox)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,128,255), 2)
                cv2.putText(frame, "DECISION", (x, y-70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,255), 2)

            # --- frame info
            cv2.putText(frame, f"Frame {i}/{total_frames}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255,255,255), 2)

            cv2.imshow("Tracking Debug", frame)

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