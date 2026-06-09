from ultralytics import YOLO
import cv2
import numpy as np
import torch
from collections import defaultdict
import pandas as pd
import os
from pathlib import Path


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
# video_dir = Path("video")
# mp4_files = list(video_dir.glob("*.mp4"))
# if not mp4_files:
#     raise FileNotFoundError("Aucun fichier .mp4 trouvé dans le dossier video")
# VIDEO_PATH = str(mp4_files[0])
VIDEO_PATH      = "video/output2.mp4"
MODEL_BALL_PATH = "Yolo/big.pt"
MODEL_POSE_PATH = "Yolo/yolo11n-pose.pt"
GROUP_TIMEOUT   = 15
GROUP_THRESHOLD = 30
JUMP_THRESHOLD  = 35
MOVE_THRESHOLD  = 3
PERSON_KP_INDICES  = [0, 1, 2, 3, 4, 9, 10, 15, 16]
PERSON_KP_TOLERANCE = 15
PERSON_GROUP_RATIO  = 0.60

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():   return 'cuda'
    if torch.backends.mps.is_available(): return 'mps'
    return 'cpu'

def euclidean(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

def bbox_center(bbox):
    x, y, w, h = bbox[:4]
    return (x + w / 2, y + h / 2)

def ball_center(bbox):
    x, y, w, h, _ = bbox
    return (x + w / 2, y + h / 2)

def dist(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

# -----------------------------------------------------------------------
# Détection
# -----------------------------------------------------------------------
def load_video_and_detect(model_ball, model_pose, device):
    cap = cv2.VideoCapture(VIDEO_PATH)
    frames, detections, hoops, poses = [], {}, {}, {}
    raw_balls = {}
    frame_id  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        results_ball = model_ball(frame, device=device)

        hoop_bbox, best_conf = None, 0
        ball_candidates      = []

        for box in results_ball[0].boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls == 2 and conf > 0.4 and conf > best_conf:
                hoop_bbox  = (x1, y1, x2 - x1, y2 - y1, conf)
                best_conf  = conf
            if cls == 0 and conf > 0.5:
                ball_candidates.append((x1, y1, x2 - x1, y2 - y1, conf))

        hoops[frame_id]     = hoop_bbox
        raw_balls[frame_id] = ball_candidates

        results_pose = model_pose(frame, device=device)
        persons      = []
        if results_pose[0].keypoints is not None:
            kps   = results_pose[0].keypoints.xy.cpu().numpy()
            confs = results_pose[0].keypoints.conf
            if confs is not None:
                confs = confs.cpu().numpy()
            for p_idx in range(len(kps)):
                persons.append((kps[p_idx], confs[p_idx]))
        poses[frame_id] = persons

        frame_id += 1

    cap.release()
    total = frame_id

    # Règle 2 : résolution multi-balles
    last_confirmed, next_raw, nxt = None, {}, None
    for fid in range(total - 1, -1, -1):
        if raw_balls.get(fid):
            nxt = raw_balls[fid][0]
        next_raw[fid] = nxt

    for fid in range(total):
        candidates = raw_balls.get(fid, [])
        if not candidates:
            detections[fid] = None
            continue
        chosen = best_ball(candidates, last_confirmed, next_raw.get(fid))
        detections[fid] = chosen
        last_confirmed  = chosen

    return frames, detections, hoops, poses

def best_ball(candidates, prev_bbox, next_bbox):
    if len(candidates) == 1:
        return candidates[0]
    refs = []
    if prev_bbox is not None: refs.append(ball_center(prev_bbox))
    if next_bbox is not None: refs.append(ball_center(next_bbox))
    if not refs:
        return max(candidates, key=lambda b: b[4])
    ref_cx = sum(r[0] for r in refs) / len(refs)
    ref_cy = sum(r[1] for r in refs) / len(refs)
    return min(candidates, key=lambda b: dist(ball_center(b), (ref_cx, ref_cy)))

# -----------------------------------------------------------------------
# Tracking
# -----------------------------------------------------------------------
def forward_tracking(frames, detections):
    forward_track = {}
    tracker, tracking = None, False
    for i, frame in enumerate(frames):
        det = detections[i]
        if det is not None:
            x, y, w, h, _ = det
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (x, y, w, h))
            tracking = True
            forward_track[i] = (x, y, w, h)
        elif tracking:
            success, bbox = tracker.update(frame)
            forward_track[i] = bbox if success else None
            if not success: tracking = False
        else:
            forward_track[i] = None
    return forward_track

def backward_tracking(frames, detections):
    backward_track = {}
    tracker, tracking = None, False
    for i in reversed(range(len(frames))):
        det = detections[i]
        if det is not None:
            x, y, w, h, _ = det
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frames[i], (x, y, w, h))
            tracking = True
            backward_track[i] = (x, y, w, h)
        elif tracking:
            success, bbox = tracker.update(frames[i])
            backward_track[i] = bbox if success else None
            if not success: tracking = False
        else:
            backward_track[i] = None
    return backward_track

def find_nearest_yolo(index, detections, total):
    prev_idx = next_idx = None
    for i in range(index, -1, -1):
        if detections.get(i) is not None:
            prev_idx = i; break
    for i in range(index, total):
        if detections.get(i) is not None:
            next_idx = i; break
    return prev_idx, next_idx

def is_track_moving(track_dict, i, total, window=3):
    bbox_i = track_dict.get(i)
    if bbox_i is None: return False
    center_i = bbox_center(bbox_i)
    for j in range(max(0, i - window), min(total, i + window + 1)):
        if j == i: continue
        bbox_j = track_dict.get(j)
        if bbox_j is None: continue
        if dist(center_i, bbox_center(bbox_j)) > MOVE_THRESHOLD:
            return True
    return False

def get_decision_bbox(i, detections, forward_track, backward_track, total):
    prev_idx, next_idx = find_nearest_yolo(i, detections, total)
    forward  = forward_track.get(i)
    backward = backward_track.get(i)
    fwd_mov  = is_track_moving(forward_track,  i, total)
    bwd_mov  = is_track_moving(backward_track, i, total)

    if prev_idx is not None and next_idx is not None:
        closer_to_next    = abs(next_idx - i) < abs(i - prev_idx)
        preferred, pref_mov = (backward, bwd_mov) if closer_to_next else (forward, fwd_mov)
        fallback,  fall_mov = (forward,  fwd_mov) if closer_to_next else (backward, bwd_mov)
        if pref_mov:   return preferred
        if fall_mov:   return fallback
        return None
    elif prev_idx is not None:
        return forward  if fwd_mov else None
    elif next_idx is not None:
        return backward if bwd_mov else None
    return None

# -----------------------------------------------------------------------
# Groupement
# -----------------------------------------------------------------------
def build_groups(frames, detections, forward_track, backward_track):
    total         = len(frames)
    groups        = []
    current_group = None
    last_center   = None
    last_det_count = detection_count = 0

    for frame_id in range(total):
        bbox = get_decision_bbox(frame_id, detections, forward_track, backward_track, total)
        if bbox is None:
            continue

        detection_count += 1
        center = bbox_center(bbox)

        for g in groups:
            if g['open'] and detection_count - g['last_det_count'] > GROUP_TIMEOUT:
                g['open'] = False

        if last_center is None:
            is_jump = False
        else:
            gap     = detection_count - last_det_count - 1
            is_jump = gap > GROUP_TIMEOUT or euclidean(center, last_center) > JUMP_THRESHOLD

        if not is_jump and current_group is not None:
            current_group['frames'].append(frame_id)
            current_group['last_center']    = center
            current_group['last_det_count'] = detection_count
        else:
            best_group, best_dist_val = None, float('inf')
            for g in groups:
                if not g['open']: continue
                d = euclidean(center, g['last_center'])
                if d < GROUP_THRESHOLD and d < best_dist_val:
                    best_dist_val = d; best_group = g

            if best_group is not None:
                best_group['frames'].append(frame_id)
                best_group['last_center']    = center
                best_group['last_det_count'] = detection_count
                current_group = best_group
            else:
                current_group = {
                    'id': len(groups), 'frames': [frame_id],
                    'last_center': center, 'last_det_count': detection_count, 'open': True
                }
                groups.append(current_group)

        last_center    = center
        last_det_count = detection_count

    frame_to_group = {}
    for g in groups:
        for fid in g['frames']:
            if fid in frame_to_group:
                if len(g['frames']) > len(groups[frame_to_group[fid]]['frames']):
                    frame_to_group[fid] = g['id']
            else:
                frame_to_group[fid] = g['id']

    return groups, frame_to_group

# -----------------------------------------------------------------------
# Filtre keypoints
# -----------------------------------------------------------------------
def frame_ball_on_keypoint(frame_id, bbox, poses):
    cx, cy = bbox_center(bbox)
    for p_idx, (kps, confs) in enumerate(poses.get(frame_id, [])):
        for kp_idx in PERSON_KP_INDICES:
            if kp_idx >= len(kps): continue
            x, y = kps[kp_idx]
            if x == 0 and y == 0: continue
            if confs is not None and confs[kp_idx] < 0.4: continue
            if dist((cx, cy), (x, y)) <= PERSON_KP_TOLERANCE:
                return (p_idx, kp_idx)
    return None

def filter_person_groups(groups, detections, frames, poses):
    eliminated_ids = set()

    for g in groups:
        hits_per_kp = defaultdict(int)
        for fid in g['frames']:
            bbox = get_decision_bbox(fid, detections,
                                     forward_tracking(frames, detections),
                                     backward_tracking(frames, detections),
                                     len(frames))
            if bbox is None: continue
            match = frame_ball_on_keypoint(fid, bbox, poses)
            if match is not None:
                hits_per_kp[match] += 1

        total = len(g['frames'])
        for (p_idx, kp_idx), count in hits_per_kp.items():
            if count / total > PERSON_GROUP_RATIO:
                eliminated_ids.add(g['id'])
                print(f"[FILTER] Groupe {g['id']} éliminé — kp {kp_idx} ratio={count/total:.2f}")
                break

    for g in groups:
        if g['id'] not in eliminated_ids: continue
        for fid in g['frames']:
            detections[fid] = None

    fwd  = forward_tracking(frames, detections)
    bwd  = backward_tracking(frames, detections)
    return build_groups(frames, detections, fwd, bwd), fwd, bwd


if __name__ == "__main__":
    device     = get_device()
    model_ball = YOLO(MODEL_BALL_PATH)
    model_pose = YOLO(MODEL_POSE_PATH)

    frames, detections, hoops, poses = load_video_and_detect(model_ball, model_pose, device)
    fwd   = forward_tracking(frames, detections)
    bwd   = backward_tracking(frames, detections)
    groups, frame_to_group = build_groups(frames, detections, fwd, bwd)
    (groups, frame_to_group), fwd, bwd = filter_person_groups(groups, detections, frames, poses)