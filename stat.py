import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import cv2 as cv2
ball   = pd.read_csv("DATA/ball.csv")
hoop   = pd.read_csv("DATA/hoop.csv")
poses  = pd.read_csv("DATA/poses.csv")

FPS = 30
SHOOT_DURATION_FRAMES = FPS * 3  # 90 frames
FREEZE_THRESHOLD = 3  # px — en dessous = considéré immobile


def plot_shoot(traj):
    plt.figure(figsize=(8, 6))
    plt.plot(traj["cx"], traj["cy"], marker="o", markersize=2, label="balle")
    plt.scatter(HOOP_CX, HOOP_CY, color="red", s=100, zorder=5, label="panier")
    plt.gca().invert_yaxis()  # y inversé car coordonnées image
    plt.legend()
    plt.show()

def reconstruction(debut):
    traj = ball[(ball["frame"] >= debut) & (ball["frame"] <= debut + SHOOT_DURATION_FRAMES)].copy()
    traj = traj.reset_index(drop=True)
    
    # Détecter les plages freeze
    freeze_mask = (
    (traj["cx"].diff().abs() < FREEZE_THRESHOLD) &
    (traj["cy"].diff().abs() < FREEZE_THRESHOLD) &
    (traj["cx"].diff(-1).abs() > FREEZE_THRESHOLD) |
    (traj["cy"].diff(-1).abs() > FREEZE_THRESHOLD)
)
    
    # Mettre à None les frames freezées pour interpolation
    traj.loc[freeze_mask, ["cx", "cy"]] = None
    
    # Interpoler linéairement
    traj[["cx", "cy"]] = traj[["cx", "cy"]].interpolate(method="polynomial", order=2)
    

    return traj

def is_airball(traj):
    near_hoop = traj[traj["cx"].sub(HOOP_CX).abs() < 20]
    if near_hoop.empty:
        return False
    
    # Vérifier que cy est monotone (pas de rupture) autour du panier
    cy_near = near_hoop["cy"].values
    diffs   = np.diff(cy_near)
    
    # Monotone = tous les diffs du même signe → trajectoire continue
    return bool(np.all(diffs > 0) or np.all(diffs < 0))

def is_shot_made(traj):
    near_hoop = traj[traj["cx"].sub(HOOP_CX).abs() < 8]  # rayon du panier ~30px
    if near_hoop.empty:
        return False
    above = near_hoop[near_hoop["cy"] < HOOP_CY]
    below = near_hoop[near_hoop["cy"] > HOOP_CY]
    # La balle doit passer au-dessus puis en dessous
    if above.empty or below.empty:
        return False
    return above.index[-1] < below.index[0]


def display_shoots(frames_video, shoot_frames, ball, poses, HOOP_CX, HOOP_CY, hoop_w=30, hoop_h=30, fps=30):
    made  = 0
    total = 0
    KEYPOINT_NAMES = [
    "nose", "eye_l", "eye_r", "ear_l", "ear_r",
    "shoulder_l", "shoulder_r",
    "elbow_l", "elbow_r",
    "wrist_l", "wrist_r",
    "hip_l", "hip_r",
    "knee_l", "knee_r",
    "ankle_l", "ankle_r",
    ]
    i = 0
    near_hoop = [s + 50 for s in shoot_frames]
    decision = [s + 50 + j for s in shoot_frames for j in range(60)]
    shoot_display = {}
    for s in shoot_frames:
        traj   = reconstruction(s)
        result = not is_airball(traj) and is_shot_made(traj)
        for f in range(s, s + 130):
            shoot_display[f] = result
    while i < len(frames_video):
        frame = frames_video[i].copy()

        # Panier
        hx = int(HOOP_CX - hoop_w / 2)
        hy = int(HOOP_CY - hoop_h / 2)
        cv2.rectangle(frame, (hx, hy), (hx + hoop_w, hy + hoop_h), (0, 215, 255), 2)

        # Balle
        ball_row = ball[ball["frame"] == i]
        if not ball_row.empty and pd.notna(ball_row.iloc[0]["cx"]):
            bx = int(ball_row.iloc[0]["x"])
            by = int(ball_row.iloc[0]["y"])
            bw = int(ball_row.iloc[0]["w"])
            bh = int(ball_row.iloc[0]["h"])
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 128, 255), 2)

        # Pose
        pose_rows = poses[(poses["frame"] == i) & (poses["person_id"] == 0)]
        if not pose_rows.empty:
            row = pose_rows.iloc[0]
            for name in KEYPOINT_NAMES:
                x = row.get(f"{name}_x")
                y = row.get(f"{name}_y")
                if pd.notna(x) and pd.notna(y) and x > 0 and y > 0:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)

        # Bandeau haut
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(frame, f"{made}/{total}", (frame.shape[1] - 120, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        if i in shoot_frame:
            total +=1
        if i in near_hoop:
            if i in shoot_display:
                if shoot_display[i]:
                    made+=1
        


        if i in shoot_display:
            result = shoot_display[i]
            color  = (0, 255, 0) if result else (0, 0, 255)
            gray = (100, 100, 100)

            near = False
            if not near:
                cv2.putText(frame, f"SHOOT", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, gray, 2)
            if i in decision:
                near = True
                cv2.putText(frame, f"SHOOT — {'MADE' if result else 'MISSED'}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            

        cv2.imshow("Shoots", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('d'):
            i = min(i + 1, len(frames_video) - 1)
        elif key == ord('q'):
            i = max(i - 1, 0)
        else:
            i += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    hoop = pd.read_csv("DATA/hoop.csv")

    hoop_detected = hoop.dropna(subset=["cx", "cy"]).copy()

    RADIUS = 20
    clusters = []

    for _, row in hoop_detected.iterrows():
        assigned = False
        for cl in clusters:
            dist = ((row["cx"] - cl["cx"]) ** 2 + (row["cy"] - cl["cy"]) ** 2) ** 0.5
            if dist <= RADIUS:
                cl["cx"] = (cl["cx"] * cl["count"] + row["cx"]) / (cl["count"] + 1)
                cl["cy"] = (cl["cy"] * cl["count"] + row["cy"]) / (cl["count"] + 1)
                cl["count"] += 1
                assigned = True
                break
        if not assigned:
            clusters.append({"cx": row["cx"], "cy": row["cy"], "count": 1})

    
    dominant = max(clusters, key=lambda c: c["count"])
    HOOP_CX = dominant["cx"]
    HOOP_CY = dominant["cy"]
    head_kps = ["nose_y", "eye_l_y", "eye_r_y", "ear_l_y", "ear_r_y"]

    p0 = poses[poses["person_id"] == 0][["frame"] + head_kps].copy()

    p0["head_y"] = p0[head_kps].min(axis=1)  # le plus haut = y minimal

    merged = ball.merge(p0[["frame", "head_y"]], on="frame", how="left")

    ball_above_head = merged[merged["cy"] < merged["head_y"] +25]
  
    # Regrouper par proximité temporelle (saut de 2 frames toléré)
    frames = sorted(ball_above_head["frame"].tolist())

    groups = []
    current = [frames[0]]

    for f in frames[1:]:
        if f - current[-1] <= 2:
            current.append(f)
        else:
            groups.append(current)
            current = [f]
    groups.append(current)

    # Filtre < 15 frames
    groups = [g for g in groups if len(g) >= 15]
    shoot_frame = []

    for i, g in enumerate(groups):
        #print(f"\nGroupe {i} | frames {g[0]}→{g[-1]} | durée {g[-1]-g[0]} frames")

        group_frames = ball_above_head[ball_above_head["frame"].isin(g)][["frame", "cx", "cy"]].copy()

        group_frames["dist_hoop"] = (
            (group_frames["cx"] - HOOP_CX) ** 2 +
              (group_frames["cy"] - HOOP_CY) ** 2
            ) ** 0.5

        print(group_frames[["frame", "cx", "cy", "dist_hoop"]].to_string(index=False))
            
        dists     = group_frames["dist_hoop"].tolist()
        min_dist  = min(dists)
         # Décroissance : la distance moyenne de la 2e moitié < 1ère moitié
        #mid       = len(dists) // 2
        # first_half_mean  = sum(dists[:mid]) / mid
        # second_half_mean = sum(dists[mid:]) / (len(dists) - mid)
        # decreasing = second_half_mean < first_half_mean

        is_shot = min_dist < 20
        if is_shot:
            shoot_frame.append(g[0])
        
    print(shoot_frame)
    for i in range(len(shoot_frame)):
        traj = reconstruction(shoot_frame[i])
        if is_airball(traj):
            print(False)
        else:
            print(is_shot_made(traj))


    cap = cv2.VideoCapture("video/output2.mp4")
    frames_video = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_video.append(frame)
    cap.release()

    display_shoots(frames_video, shoot_frame, ball, poses, HOOP_CX, HOOP_CY)



       
