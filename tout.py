from ultralytics import YOLO
import cv2
import numpy as np
from utils import get_device


KEYPOINT_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class ShotDetector:
    def __init__(self):
        self.model_ball = YOLO("Yolo/b.pt")
        self.model_hoop = YOLO("Yolo/best.pt")  # 🔥 modèle panier
        self.model_pose = YOLO("yolo11n-pose.pt")

        self.device = get_device()
        self.cap = cv2.VideoCapture("video/output.mp4")

        self.frames = []
        self.detections = {}
        self.hoops = {}
        self.poses = {}

        self.forward_track = {}
        self.backward_track = {}

        self.load_video_and_detect()
        self.forward_tracking()
        self.backward_tracking()
        self.display()

    def brighten(self, frame, alpha=1.3, beta=60):
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # -------------------------------
    # 🔥 DETECTION SÉPARÉE
    # -------------------------------
    def load_video_and_detect(self):
        frame_id = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frames.append(frame)
            brightened = self.brighten(frame)

            # ---------------- 🏀 BALLE
            results_ball = self.model_ball(brightened, device=self.device)
            ball_bbox = None

            for box in results_ball[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == 0 and conf > 0.3:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    ball_bbox = (x1, y1, x2 - x1, y2 - y1, conf)

            # ---------------- 🏀 HOOP (best.pt)
            results_hoop = self.model_hoop(brightened, device=self.device)

            hoop_bbox = None
            best_conf = 0

            for box in results_hoop[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # 🔥 LABEL 1 = HOOP
                if cls == 1 and conf > 0.4 and conf > best_conf:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    hoop_bbox = (x1, y1, x2 - x1, y2 - y1, conf)
                    best_conf = conf

            self.detections[frame_id] = ball_bbox
            self.hoops[frame_id] = hoop_bbox

            # ---------------- 🦴 POSE
            results_pose = self.model_pose(brightened, device=self.device)
            persons = []

            if results_pose[0].keypoints is not None:
                kps = results_pose[0].keypoints.xy.cpu().numpy()
                confs = results_pose[0].keypoints.conf

                if confs is not None:
                    confs = confs.cpu().numpy()

                for p_idx in range(len(kps)):
                    persons.append((kps[p_idx], confs[p_idx]))

            self.poses[frame_id] = persons
            frame_id += 1

        self.cap.release()

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
                self.forward_track[i] = bbox if success else None
                if not success:
                    tracking = False
            else:
                self.forward_track[i] = None

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
                self.backward_track[i] = bbox if success else None
                if not success:
                    tracking = False
            else:
                self.backward_track[i] = None

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

    def draw_pose(self, frame, keypoints, confs, kp_threshold=0.4):
        for idx, (x, y) in enumerate(keypoints):
            if x == 0 and y == 0:
                continue
            if confs is not None and confs[idx] < kp_threshold:
                continue
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)

        for a, b in KEYPOINT_EDGES:
            xa, ya = keypoints[a]
            xb, yb = keypoints[b]

            if xa == 0 and ya == 0:
                continue
            if xb == 0 and yb == 0:
                continue

            if confs is not None:
                if confs[a] < kp_threshold or confs[b] < kp_threshold:
                    continue

            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)),
                     (0, 200, 255), 2)

    # -------------------------------
    def display(self):
        i = 0
        total_frames = len(self.frames)

        while True:
            frame = self.frames[i].copy()

            # 🏀 HOOP
            hoop = self.hoops.get(i)
            if hoop is not None:
                hx, hy, hw, hh, conf = hoop
                cv2.rectangle(frame, (hx, hy), (hx + hw, hy + hh), (0, 215, 255), 2)
                cv2.putText(frame, f"HOOP {conf:.2f}", (hx, hy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

            # 🦴 POSE
            for (kps, confs) in self.poses.get(i, []):
                self.draw_pose(frame, kps, confs)

            # 🔵 YOLO balle (raw)
            det = self.detections.get(i)
            if det is not None:
                x, y, w, h, conf = det
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # =====================================================
            # 🔥 🎯 DECISION (TRACKING FUSION)
            # =====================================================
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

            # 🔥 affichage FINAL balle (fusion tracking)
            if decision_bbox is not None:
                x, y, w, h = map(int, decision_bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 128, 255), 2)
                cv2.putText(frame, "BALL", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)

            # ---------------- UI ----------------
            bar_h = 50
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame.shape[0] - bar_h),
                        (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            nb_persons = len(self.poses.get(i, []))
            ball_conf = f"balle conf:{det[4]:.2f}" if det else "no ball"

            cv2.putText(frame,
                        f"Frame {i}/{total_frames}   {ball_conf}   joueurs:{nb_persons}",
                        (15, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.imshow("ShotDetector", frame)

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