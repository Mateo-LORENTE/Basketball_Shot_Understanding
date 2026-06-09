from ultralytics import YOLO
import cv2
import numpy as np
import torch
from collections import defaultdict
import pandas as pd
import os
import itertools
import random
from pathlib import Path

def euclidean(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))



def bbox_center(bbox):
    x, y, w, h = bbox[:4]
    return (x + w / 2, y + h / 2)


def get_device():
    """Automatically select devices -> mps Mac -> cpu"""
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device



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



KEYPOINT_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# video_dir = Path("video")
# mp4_files = list(video_dir.glob("*.mp4"))
# if not mp4_files:
#     raise FileNotFoundError("Aucun fichier .mp4 trouvé dans le dossier video")
# VIDEO_PATH = str(mp4_files[0])

class ShotDetector:
    def __init__(self):
        self.model_ball = YOLO("Yolo/big.pt")
        self.model_pose = YOLO("Yolo/yolo11n-pose.pt")

        self.device = get_device()
        self.cap = cv2.VideoCapture("video/output2.mp4")

        self.frames = []
        self.detections = {}
        self.hoops = {}
        self.poses = {}

        self.group_timeout = 15
        self.group_threshold = 20
        self.JUMP_THRESHOLD = 35
        #$
        #$

        self.forward_track = {}
        self.backward_track = {}

        self.load_video_and_detect()
        self.forward_tracking()
        self.backward_tracking()
        self.build_groups_on_decision()
        self._filter_person_groups() 
        self.export_data()
        self.display()

    def _ball_center(self, bbox):
        x, y, w, h, _ = bbox
        return (x + w / 2, y + h / 2)
    
    def _dist(self, c1, c2):
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

    def _best_ball(self, candidates, prev_bbox, next_bbox):
        if len(candidates) == 1:
            return candidates[0]
        refs = []
        if prev_bbox is not None:
            refs.append(self._ball_center(prev_bbox))
        if next_bbox is not None:
            refs.append(self._ball_center(next_bbox))
        if not refs:
            return max(candidates, key=lambda b: b[4])
        ref_cx = sum(r[0] for r in refs) / len(refs)
        ref_cy = sum(r[1] for r in refs) / len(refs)
        return min(candidates, key=lambda b: self._dist(self._ball_center(b), (ref_cx, ref_cy)))
    
    



    #$
    # -----------------------------------------------------------------------
    def brighten(self, frame, alpha, beta):
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    def sum_confidences(self,model_ball, frame, device):
        brightened = self.brighten(frame, alpha=1.0, beta=0)  
        results_ball = model_ball(brightened, device=device)

        total_conf = 0.0
        for box in results_ball[0].boxes:
            conf = float(box.conf[0])
            total_conf += conf
        return total_conf
    
    def find_optimal_alpha_beta(self,cap, model_ball, device, num_frames=100):
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

        # Valeurs à tester
        alpha_values = [1.0, 1.2, 1.5, 1.7, 2.0]
        beta_values = [0, 15, 30, 45, 60]

        best_alpha, best_beta = 1.0, 0
        best_total_conf = -1

        # Tester toutes les combinaisons
        for alpha, beta in itertools.product(alpha_values, beta_values):
            total_conf = 0.0
            for frame in frames:
                brightened = self.brighten(frame, alpha=alpha, beta=beta)
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

    # -----------------------------------------------------------------------
    def load_video_and_detect(self):
        frame_id = 0
        raw_balls = {}
        alpha_best , beta_best = self.find_optimal_alpha_beta(self.cap,self.model_ball,self.device)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frames.append(frame)
            
            brightened = self.brighten(frame,alpha_best,beta_best)

            results_ball = self.model_ball(brightened, device=self.device)

            # --- HOOP règle 1 : conf max
            hoop_bbox = None
            best_conf = 0
            for box in results_ball[0].boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 2 and conf > 0.4 and conf > best_conf:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    hoop_bbox = (x1, y1, x2 - x1, y2 - y1, conf)
                    best_conf = conf
            self.hoops[frame_id] = hoop_bbox

            # --- BALLE : collecter tous les candidats
            ball_candidates = []
            for box in results_ball[0].boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 0 and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    ball_candidates.append((x1, y1, x2 - x1, y2 - y1, conf))
            raw_balls[frame_id] = ball_candidates

            # --- POSE
            results_pose = self.model_pose(brightened, device=self.device)
            persons = []
            if results_pose[0].keypoints is not None:
                kps   = results_pose[0].keypoints.xy.cpu().numpy()
                confs = results_pose[0].keypoints.conf
                if confs is not None:
                    confs = confs.cpu().numpy()
                for p_idx in range(len(kps)):
                    persons.append((kps[p_idx], confs[p_idx]))
            self.poses[frame_id] = persons

            

            frame_id += 1

        self.cap.release()
        total = frame_id

        # --- Règle 2 : résolution multi-balles (prev / next)
        last_confirmed = None
        next_raw = {}
        nxt = None
        for fid in range(total - 1, -1, -1):
            if raw_balls.get(fid):
                nxt = raw_balls[fid][0]
            next_raw[fid] = nxt

        for fid in range(total):
            candidates = raw_balls.get(fid, [])
            if not candidates:
                self.detections[fid] = None
                continue
            chosen = self._best_ball(candidates, last_confirmed, next_raw.get(fid))
            self.detections[fid] = chosen
            last_confirmed = chosen


    # -----------------------------------------------------------------------
    def forward_tracking(self):
        tracker = None
        tracking = False
        for i in range(len(self.frames)):
            frame = self.frames[i]
            det   = self.detections[i]
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
            det   = self.detections[i]
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
        prev_idx = next_idx = None
        for i in range(index, -1, -1):
            if self.detections.get(i) is not None:
                prev_idx = i
                break
        for i in range(index, len(self.frames)):
            if self.detections.get(i) is not None:
                next_idx = i
                break
        return prev_idx, next_idx
    
   

    def _is_track_moving(self, track_dict, i, window=3):
        """
        Vérifie si le tracking bouge autour de la frame i.
        Regarde les `window` frames avant et après, si aucun déplacement
        > MOVE_THRESHOLD px → considéré immobile.
        """

        MOVE_THRESHOLD = 3 
        center_i = None
        bbox_i = track_dict.get(i)
        if bbox_i is None:
            return False
        center_i = bbox_center(bbox_i)

        for j in range(max(0, i - window), min(len(self.frames), i + window + 1)):
            if j == i:
                continue
            bbox_j = track_dict.get(j)
            if bbox_j is None:
                continue
            if self._dist(center_i, bbox_center(bbox_j)) > MOVE_THRESHOLD:
                return True

        return False

    def get_decision_bbox(self, i):
        prev_idx, next_idx = self.find_nearest_yolo(i)

        forward  = self.forward_track.get(i)
        backward = self.backward_track.get(i)

        forward_moving  = self._is_track_moving(self.forward_track, i)
        backward_moving = self._is_track_moving(self.backward_track, i)

        if prev_idx is not None and next_idx is not None:
            closer_to_next = abs(next_idx - i) < abs(i - prev_idx)
            preferred  = backward if closer_to_next else forward
            preferred_moving  = backward_moving if closer_to_next else forward_moving
            fallback   = forward  if closer_to_next else backward
            fallback_moving   = forward_moving  if closer_to_next else backward_moving

            if preferred_moving:
                return preferred
            elif fallback_moving:
                return fallback
            else:
                return None  # les deux sont immobiles → pas de décision

        elif prev_idx is not None:
            return forward if forward_moving else None
        elif next_idx is not None:
            return backward if backward_moving else None

        return None

    # -------------------------------
    # Groupement sur la décision
    # -------------------------------
    def build_groups_on_decision(self):
        """
        Règles :
        - Mouvement continu (< JUMP_THRESHOLD) → reste dans le groupe courant
        - Saut détecté → cherche un groupe existant encore ouvert dont le
            dernier centre est proche (< group_threshold) → le rejoindre
            sinon → nouveau groupe
        - Un groupe se ferme après group_timeout détections sans alimentation
        """
        groups = []
        current_group = None
        last_center = None
        last_det_count = 0
        detection_count = 0
        

        for frame_id in range(len(self.frames)):
            bbox = self.get_decision_bbox(frame_id)
            if bbox is None:
                continue

            detection_count += 1
            center = bbox_center(bbox)

            # Fermer les groupes inactifs (uniquement pour libérer la mémoire
            # des groupes trop anciens — ils ne peuvent plus être rejoints)
            for g in groups:
                if g['open'] and detection_count - g['last_det_count'] > self.group_timeout:
                    g['open'] = False

            if last_center is None:
                is_jump = False
            else:
                gap = detection_count - last_det_count - 1
                if gap > self.group_timeout:
                    is_jump = True
                else:
                    is_jump = euclidean(center, last_center) > self.JUMP_THRESHOLD

            if not is_jump and current_group is not None:
                # Mouvement continu → même groupe
                current_group['frames'].append(frame_id)
                current_group['last_center'] = center
                current_group['last_det_count'] = detection_count

            else:
                # Saut (ou première frame) → chercher un groupe ouvert compatible
                best_group = None
                best_dist = float('inf')
                for g in groups:
                    if not g['open']:
                        continue
                    d = euclidean(center, g['last_center'])
                    if d < self.group_threshold and d < best_dist:
                        best_dist = d
                        best_group = g

                if best_group is not None:
                    # Rejoindre le groupe existant (ex: retour en A)
                    best_group['frames'].append(frame_id)
                    best_group['last_center'] = center
                    best_group['last_det_count'] = detection_count
                    current_group = best_group
                else:
                    # Nouveau territoire → nouveau groupe
                    current_group = {
                        'id': len(groups),
                        'frames': [frame_id],
                        'last_center': center,
                        'last_det_count': detection_count,
                        'open': True
                    }
                    groups.append(current_group)

            last_center = center
            last_det_count = detection_count

        self.groups = groups

        # Index frame -> group_id
        self.frame_to_group = {}
        for g in groups:
            for fid in g['frames']:
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
            if xa == 0 and ya == 0: continue
            if xb == 0 and yb == 0: continue
            if confs is not None:
                if confs[a] < kp_threshold or confs[b] < kp_threshold:
                    continue
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (0, 200, 255), 2)

    # -----------------------------------------------------------------------
    # Suppression des groupes collés à un keypoint personne
    # -----------------------------------------------------------------------
    PERSON_KP_INDICES = [0, 1, 2, 3, 4, 9, 10, 15, 16]   # tête(nez), cheville gauche, cheville droite
    PERSON_KP_TOLERANCE = 15          # px — "correspond à" un keypoint
    PERSON_GROUP_RATIO  = 0.60        # >50% des frames du groupe → éliminé
    #$

    def _frame_ball_on_keypoint(self, frame_id, bbox):
        """
        Retourne le (person_idx, kp_idx) si le centre de bbox coïncide
        avec un keypoint, None sinon.
        """
        cx, cy = bbox_center(bbox)
        for p_idx, (kps, confs) in enumerate(self.poses.get(frame_id, [])):
            for kp_idx in self.PERSON_KP_INDICES:
                if kp_idx >= len(kps):
                    continue
                x, y = kps[kp_idx]
                if x == 0 and y == 0:
                    continue
                if confs is not None and confs[kp_idx] < 0.4:
                    continue
                if self._dist((cx, cy), (x, y)) <= self.PERSON_KP_TOLERANCE:
                    return (p_idx, kp_idx)
        return None

    def _filter_person_groups(self):
        eliminated_ids = set()

        for g in self.groups:
            # Compter les hits par (person_idx, kp_idx)
            hits_per_kp = defaultdict(int)

            for fid in g['frames']:
                bbox = self.get_decision_bbox(fid)
                if bbox is None:
                    continue
                match = self._frame_ball_on_keypoint(fid, bbox)
                if match is not None:
                    hits_per_kp[match] += 1

            # Vérifier si UN SEUL kp dépasse le ratio
            total = len(g['frames'])
            for (p_idx, kp_idx), count in hits_per_kp.items():
                ratio = count / total
                if ratio > self.PERSON_GROUP_RATIO:
                    eliminated_ids.add(g['id'])
                    print(f"[FILTER] Groupe {g['id']} éliminé — "
                        f"personne {p_idx} kp {kp_idx} : "
                        f"{count}/{total} frames (ratio={ratio:.2f})")
                    break  # un seul kp suffit à éliminer le groupe

        if not eliminated_ids:
            print("[FILTER] Aucun groupe éliminé.")
            return

        for g in self.groups:
            if g['id'] not in eliminated_ids:
                continue
            for fid in g['frames']:
                self.detections[fid] = None

        self.forward_track  = {}
        self.backward_track = {}
        self.forward_tracking()
        self.backward_tracking()
        self.build_groups_on_decision()

    KEYPOINT_NAMES = [
        "nose", "eye_l", "eye_r", "ear_l", "ear_r",
        "shoulder_l", "shoulder_r",
        "elbow_l", "elbow_r",
        "wrist_l", "wrist_r",
        "hip_l", "hip_r",
        "knee_l", "knee_r",
        "ankle_l", "ankle_r",
    ]

    def export_data(self):
        os.makedirs("DATA", exist_ok=True)

        # BALLE
        ball_rows = []
        for fid in range(len(self.frames)):
            bbox = self.get_decision_bbox(fid)
            det  = self.detections.get(fid)
            gid  = self.frame_to_group.get(fid)
            if bbox is not None:
                cx, cy = bbox_center(bbox)
                ball_rows.append({
                    "frame":     fid,
                    "cx":        round(cx, 2),
                    "cy":        round(cy, 2),
                    "x":         round(bbox[0], 2),
                    "y":         round(bbox[1], 2),
                    "w":         round(bbox[2], 2),
                    "h":         round(bbox[3], 2),
                    "yolo_conf": round(det[4], 3) if det is not None else None,
                    "group_id":  gid,
                })
            else:
                ball_rows.append({
                    "frame":     fid,
                    "cx":        None,
                    "cy":        None,
                    "x":         None,
                    "y":         None,
                    "w":         None,
                    "h":         None,
                    "yolo_conf": None,
                    "group_id":  None,
                })
        pd.DataFrame(ball_rows).to_csv("DATA/ball.csv", index=False)
        print(f"[EXPORT] ball.csv → {len(ball_rows)} frames")

        hoop_rows = []
        for fid in range(len(self.frames)):   # ← comme display, toutes les frames dans l'ordre
            hoop = self.hoops.get(fid)
            if hoop is not None:
                x, y, w, h, conf = hoop
                hoop_rows.append({
                    "frame": fid,
                    "cx":    round(x + w / 2, 2),
                    "cy":    round(y + h / 2, 2),
                    "x": x, "y": y, "w": w, "h": h,
                    "conf":  round(conf, 3),
                })
            else:
                hoop_rows.append({
                    "frame": fid,
                    "cx": None, "cy": None,
                    "x": None, "y": None,
                    "w": None, "h": None,
                    "conf": None,
                })
        pd.DataFrame(hoop_rows).to_csv("DATA/hoop.csv", index=False)

        # POSE
        pose_rows = []
        for fid in range(len(self.frames)):
            persons = self.poses.get(fid, [])
            if not persons:
                # Frame sans personne → ligne vide
                row = {"frame": fid, "person_id": None}
                for name in self.KEYPOINT_NAMES:
                    row[f"{name}_x"]    = None
                    row[f"{name}_y"]    = None
                    row[f"{name}_conf"] = None
                pose_rows.append(row)
            else:
                for p_idx, (kps, confs) in enumerate(persons):
                    row = {"frame": fid, "person_id": p_idx}
                    for kp_idx, name in enumerate(self.KEYPOINT_NAMES):
                        if kp_idx < len(kps):
                            x, y = kps[kp_idx]
                            conf = float(confs[kp_idx]) if confs is not None else None
                            row[f"{name}_x"]    = round(float(x), 2)
                            row[f"{name}_y"]    = round(float(y), 2)
                            row[f"{name}_conf"] = round(conf, 3) if conf is not None else None
                        else:
                            row[f"{name}_x"]    = None
                            row[f"{name}_y"]    = None
                            row[f"{name}_conf"] = None
                    pose_rows.append(row)
        pd.DataFrame(pose_rows).to_csv("DATA/poses.csv", index=False)
        print(f"[EXPORT] poses.csv → {len(pose_rows)} lignes")

        # GROUPES
        group_rows = []
        for g in self.groups:
            group_rows.append({
                "group_id":    g["id"],
                "nb_frames":   len(g["frames"]),
                "frame_start": g["frames"][0],
                "frame_end":   g["frames"][-1],
                "last_cx":     round(g["last_center"][0], 2),
                "last_cy":     round(g["last_center"][1], 2),
            })
        pd.DataFrame(group_rows).to_csv("DATA/groups.csv", index=False)
        print(f"[EXPORT] groups.csv → {len(group_rows)} groupes")

        print("[EXPORT] ✅ Terminé → dossier DATA/")

    def display(self):
        i = 0
        total_frames = len(self.frames)

        while True:
            frame = self.frames[i].copy()

            det = self.detections.get(i)
            f = self.forward_track.get(i)
            b = self.backward_track.get(i)
            hoop = self.hoops.get(i)

            # ---------------------------
            # HOOP (panier)
            # ---------------------------
            if hoop is not None:
                x, y, w, h, conf = hoop
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 215, 255), 2)
                cv2.putText(frame, f"HOOP {conf:.2f}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

            # ---------------------------
            # POSES (points + squelette)
            # ---------------------------
            for (kps, confs) in self.poses.get(i, []):
                self.draw_pose(frame, kps, confs)

            # ---------------------------
            # YOLO (détection brute)
            # ---------------------------
            if det is not None:
                x, y, w, h, conf = det
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)

            # ---------------------------
            # Forward tracking
            # ---------------------------
            if f is not None:
                x, y, w, h = map(int, f)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

            # ---------------------------
            # Backward tracking
            # ---------------------------
            if b is not None:
                x, y, w, h = map(int, b)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 1)

            # ---------------------------
            # Décision + GROUPES
            # ---------------------------
            decision_bbox = self.get_decision_bbox(i)
            group_id = self.frame_to_group.get(i)

            if decision_bbox is not None:
                x, y, w, h = map(int, decision_bbox[:4])

                if group_id is not None:
                    color = get_group_color(group_id)
                    label = f"G{group_id} ({len(self.groups[group_id]['frames'])})"
                else:
                    color = (0, 128, 255)
                    label = "BALL"

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(frame, label, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            # ---------------------------
            # Légende groupes
            # ---------------------------
            for idx, g in enumerate(self.groups):
                c = get_group_color(g['id'])
                cv2.rectangle(frame, (10, 60 + idx * 22), (22, 72 + idx * 22), c, -1)
                cv2.putText(frame,
                            f"G{g['id']} — {len(g['frames'])} frames",
                            (28, 72 + idx * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

            # ---------------------------
            # Bandeau bas
            # ---------------------------
            bar_h = 50
            overlay = frame.copy()
            cv2.rectangle(overlay,
                        (0, frame.shape[0] - bar_h),
                        (frame.shape[1], frame.shape[0]),
                        (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            nb_persons = len(self.poses.get(i, []))
            ball_conf = f"conf:{det[4]:.2f}" if det else "no ball"
            group_label = f"G{group_id}" if group_id is not None else "no group"

            cv2.putText(frame,
                        f"Frame {i}/{total_frames} | {ball_conf} | joueurs:{nb_persons} | {group_label}",
                        (15, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # ---------------------------
            # SHOW
            # ---------------------------
            cv2.imshow("ShotDetector - Full Debug", frame)

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