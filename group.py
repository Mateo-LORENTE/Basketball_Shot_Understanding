from ultralytics import YOLO
import cv2
import numpy as np
from utils import get_device

GROUP_COLORS = [
    (0, 255, 128),
    (0, 128, 255),
    (255, 0, 128),
    (255, 255, 0),
    (0, 255, 255),
    (128, 0, 255),
    (0, 200, 255),
    (255, 128, 0),
]

def get_group_color(group_id):
    return GROUP_COLORS[group_id % len(GROUP_COLORS)]

def bbox_center(bbox):
    x, y, w, h = bbox[:4]
    return (x + w / 2, y + h / 2)

def euclidean(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))


class ShotDetector:
    def __init__(self, group_threshold=40, group_timeout=20):
        self.model = YOLO("Yolo/b.pt")
        self.device = get_device()
        self.cap = cv2.VideoCapture("video/output.mp4")

        self.group_threshold = group_threshold  # distance max en pixels
        self.group_timeout = group_timeout      # nb de détections sans alimentation avant fermeture

        self.frames = []
        self.detections = {}
        self.forward_track = {}
        self.backward_track = {}
        self.groups = []
        self.frame_to_group = {}

        self.load_video_and_detect()
        self.forward_tracking()
        self.backward_tracking()
        self.build_groups_on_decision()
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
            bbox = None
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 0 and conf > 0.3:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bbox = (x1, y1, x2 - x1, y2 - y1, conf)
                    break
            self.detections[frame_id] = bbox
            frame_id += 1
        self.cap.release()

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
                self.forward_track[i] = bbox if success else None
                if not success:
                    tracking = False
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
                self.backward_track[i] = bbox if success else None
                if not success:
                    tracking = False
            else:
                self.backward_track[i] = None

    # -------------------------------
    # 🔎 Helper YOLO nearest
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
    # 🎯 Decision bbox pour une frame
    # -------------------------------
    def get_decision_bbox(self, i):
        prev_idx, next_idx = self.find_nearest_yolo(i)
        if prev_idx is not None and next_idx is not None:
            if abs(next_idx - i) < abs(i - prev_idx):
                return self.backward_track.get(i)
            else:
                return self.forward_track.get(i)
        elif prev_idx is not None:
            return self.forward_track.get(i)
        elif next_idx is not None:
            return self.backward_track.get(i)
        return None

    # -------------------------------
    # 🟩 Groupement sur la décision
    # -------------------------------
    def build_groups_on_decision(self):
        """
        Groupe les frames où la décision est non nulle.
        Règles :
          - Représentant du groupe = centre de la dernière frame ajoutée
          - Un groupe se ferme après group_timeout détections sans alimentation
          - Si une frame est à moins de group_threshold px du représentant → même groupe
          - Sinon → nouveau groupe
        """
        groups = []
        open_groups = []
        detection_count = 0

        for frame_id in range(len(self.frames)):
            bbox = self.get_decision_bbox(frame_id)
            if bbox is None:
                continue

            detection_count += 1
            center = bbox_center(bbox)

            # Fermer les groupes inactifs
            for g in open_groups[:]:
                if detection_count - g['last_det_count'] > self.group_timeout:
                    g['open'] = False
                    open_groups.remove(g)

            # Chercher le groupe ouvert le plus proche
            best_group = None
            best_dist = float('inf')
            for g in open_groups:
                d = euclidean(center, g['last_center'])
                if d < self.group_threshold and d < best_dist:
                    best_dist = d
                    best_group = g

            if best_group is not None:
                best_group['frames'].append(frame_id)
                best_group['last_center'] = center
                best_group['last_det_count'] = detection_count
            else:
                new_group = {
                    'id': len(groups),
                    'frames': [frame_id],
                    'last_center': center,
                    'last_det_count': detection_count,
                    'open': True
                }
                groups.append(new_group)
                open_groups.append(new_group)

        self.groups = groups

        # Index frame -> group_id
        self.frame_to_group = {}
        for g in groups:
            for fid in g['frames']:
                # En cas de conflit (rare) → groupe le plus grand
                if fid in self.frame_to_group:
                    existing_gid = self.frame_to_group[fid]
                    if len(g['frames']) > len(groups[existing_gid]['frames']):
                        self.frame_to_group[fid] = g['id']
                else:
                    self.frame_to_group[fid] = g['id']

        print(f"[GROUPS] {len(groups)} groupes formés sur la décision")
        for g in groups:
            print(f"  Groupe {g['id']} : {len(g['frames'])} frames, "
                  f"{g['frames'][0]}→{g['frames'][-1]}")

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

            # --- YOLO (bleu)
            if det is not None:
                x, y, w, h, conf = det
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)

            # --- Forward (vert)
            if f is not None:
                x, y, w, h = map(int, f)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

            # --- Backward (violet)
            if b is not None:
                x, y, w, h = map(int, b)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 1)

            # --- Décision colorée par groupe
            decision_bbox = self.get_decision_bbox(i)
            group_id = self.frame_to_group.get(i)

            if decision_bbox is not None and group_id is not None:
                x, y, w, h = map(int, decision_bbox[:4])
                color = get_group_color(group_id)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(frame, f"G{group_id}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            # --- Légende groupes
            for idx, g in enumerate(self.groups):
                c = get_group_color(g['id'])
                cv2.rectangle(frame, (10, 50 + idx * 22), (22, 62 + idx * 22), c, -1)
                cv2.putText(frame, f"G{g['id']} — {len(g['frames'])} frames",
                            (28, 62 + idx * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

            # --- Bandeau bas
            bar_h = 40
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame.shape[0] - bar_h),
                          (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            group_label = f"Groupe {group_id}" if group_id is not None else "Hors groupe"
            cv2.putText(frame, f"Frame {i}/{total_frames}   {group_label}",
                        (15, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.imshow("Decision Groups", frame)

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