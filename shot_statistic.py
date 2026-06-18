# scripts/shot_statistics.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

class ShotStatistics:
    def __init__(self, ball_csv="DATA/ball3.csv", hoop_csv="DATA/hoop3.csv", poses_csv="DATA/poses3.csv", video_path="VID_IN/out.mp4", OFFSET_X=0, OFFSET_Y=0):
        """
        Initializes shot analysis using CSV data and video input.

        Args:
            ball_csv (str): Path to the ball detection CSV file.
            hoop_csv (str): Path to the hoop detection CSV file.
            poses_csv (str): Path to the pose detection CSV file.
            video_path (str): Path to the video to be analyzed.
        """
        self.ball = pd.read_csv(ball_csv)
        self.hoop = pd.read_csv(hoop_csv)
        self.poses = pd.read_csv(poses_csv)
        self.OFFSET_X = OFFSET_X
        self.OFFSET_Y = OFFSET_Y
        self.video_path = video_path
        self.FPS = self._get_video_fps()
        self.SHOOT_DURATION_FRAMES = int(self.FPS * 3)  # 3 seconds of trajectory correpond to a shot
        self.FREEZE_THRESHOLD = 3 # Movement threshold to detect a freeze
        self.HOOP_CX, self.HOOP_CY = self._detect_hoop_position()
        self.shoot_frames = self._detect_shoot_frames()
        self.frames_video = self._load_video_frames()

    def _get_video_fps(self):
        """Get the video's FPS."""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    def _load_video_frames(self):
        """Loads all video frames into a list."""
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def _detect_hoop_position(self):
        """Detects the hoop position using clustering of detections."""
        hoop_detected = self.hoop.dropna(subset=["cx", "cy"]).copy()
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

        if not clusters:
            return 0, 0  

        dominant = max(clusters, key=lambda c: c["count"])
        return dominant["cx"], dominant["cy"]

    def _detect_shoot_frames(self):
        """
        Detects frames where a shot is potentially taken.
        Based on when wrist is higher than head and ball higher than hoop
        """
        head_kps = ["nose_y", "eye_l_y", "eye_r_y", "ear_l_y", "ear_r_y"]
        wrist_kps = ["wrist_l_y","wrist_r_y"]
        p0 = self.poses[self.poses["person_id"] == 0][["frame"] + head_kps + wrist_kps].copy()
        p0["head_y"] = p0[head_kps].min(axis=1)  # le plus haut = y minimal
        p0["wrist_above_head"] = (p0["wrist_l_y"] < p0["head_y"]) | (p0["wrist_r_y"] < p0["head_y"])
        merged = self.ball.merge(p0[["frame", "head_y"]], on="frame", how="left")
        ball_above_head = merged[merged["cy"] < merged["head_y"] + 25]

        frames = sorted(ball_above_head["frame"].tolist())
        groups = []
        if not frames:
            return []

        current = [frames[0]]
        for f in frames[1:]:
            if f - current[-1] <= 2:
                current.append(f)
            else:
                groups.append(current)
                current = [f]
        groups.append(current)

        groups = [g for g in groups if len(g) >= 15]
        shoot_frames = []

        for g in groups:
            group_frames = ball_above_head[ball_above_head["frame"].isin(g)][["frame", "cx", "cy"]].copy()
            group_frames["dist_hoop"] = (
                (group_frames["cx"] - self.HOOP_CX) ** 2 +
                (group_frames["cy"] - self.HOOP_CY) ** 2
            ) ** 0.5
            min_dist = min(group_frames["dist_hoop"].tolist())

            if min_dist < 20:
                wrist_in_group = p0[(p0["frame"].isin(g)) & (p0["wrist_above_head"])]
                if not wrist_in_group.empty:
                    wrist_start = wrist_in_group["frame"].min()
                    shoot_frames.append(wrist_start)
        
        return shoot_frames

    def reconstruct_trajectory(self, start_frame):
        """Reconstruit la trajectoire de la balle à partir d'une frame de départ."""
        traj = self.ball[(self.ball["frame"] >= start_frame) &
                         (self.ball["frame"] <= start_frame + self.SHOOT_DURATION_FRAMES)].copy()
        traj = traj.reset_index(drop=True)

        freeze_mask = (
            (traj["cx"].diff().abs() < self.FREEZE_THRESHOLD) &
            (traj["cy"].diff().abs() < self.FREEZE_THRESHOLD) &
            ((traj["cx"].diff(-1).abs() > self.FREEZE_THRESHOLD) |
             (traj["cy"].diff(-1).abs() > self.FREEZE_THRESHOLD))
        )

        traj.loc[freeze_mask, ["cx", "cy"]] = None
        traj[["cx", "cy"]] = traj[["cx", "cy"]].interpolate(method="polynomial", order=2)
        return traj

    def is_airball(self, traj):
        """Checks whether the shot is an airball (the ball does not touch the hoop)."""
        near_hoop = traj[traj["cx"].sub(self.HOOP_CX).abs() < 20]
        if near_hoop.empty:
            return False
        cy_near = near_hoop["cy"].values
        diffs = np.diff(cy_near)
        return bool(np.all(diffs > 0) or np.all(diffs < 0))

    def is_shot_made(self,traj, hoop_w=30, hoop_h=30):
        """
        Determines whether a shot is made by checking if the ball 
        passes through the hoop region from entry to exit (top to bottom trajectory)
        """
        hx_min = self.HOOP_CX - hoop_w/2
        hx_max = self.HOOP_CX + hoop_w/2
        hy_min = self.HOOP_CY - hoop_h/2
        hy_max = self.HOOP_CY + hoop_h/2

        # points à l'intérieur
        inside = (
            (traj["cx"] > hx_min) &
            (traj["cx"] < hx_max) &
            (traj["cy"] > hy_min) &
            (traj["cy"] < hy_max)
        )

        idx_inside = traj[inside].index

        if len(idx_inside) == 0:
            return False

        first_idx = idx_inside[0]
        last_idx = idx_inside[-1]

        # impossible si on n'a pas un point avant et après
        if first_idx == 0 or last_idx == len(traj)-1:
            return False

        before = traj.iloc[first_idx-1]
        first = traj.iloc[first_idx]

        last = traj.iloc[last_idx]
        after = traj.iloc[last_idx+1]

        x_before, y_before = before["cx"], before["cy"]
        x_first, y_first = first["cx"], first["cy"]

        x_last, y_last = last["cx"], last["cy"]
        x_after, y_after = after["cx"], after["cy"]

        # Déterminer par où ça entre
        if y_before < hy_min:
            entry = "top"
        elif y_before > hy_max:
            entry = "bottom"
        elif x_before < hx_min:
            entry = "left"
        elif x_before > hx_max:
            entry = "right"
        else:
            return False

        # Déterminer par où ça sort
        if y_after < hy_min:
            exit = "top"
        elif y_after > hy_max:
            exit = "bottom"
        elif x_after < hx_min:
            exit = "left"
        elif x_after > hx_max:
            exit = "right"
        else:
            return False

        return entry == "top" and exit == "bottom"



    def plot_shoot(self,traj, hoop_w=30, hoop_h=30):
        plt.figure(figsize=(8, 6))
        plt.plot(traj["cx"], traj["cy"], marker="o", markersize=2, label="balle")
        plt.scatter(self.HOOP_CX, self.HOOP_CY, color="red", s=100, zorder=5, label="panier")

        hx = self.HOOP_CX - hoop_w / 2
        hy = self.HOOP_CY - hoop_h / 2
        plt.gca().add_patch(
            plt.Rectangle((hx, hy), hoop_w, hoop_h, fill=False, edgecolor="red", linewidth=2)
        )

        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()


    def analyze_shoots(self):
        """Analyzes all detected shots and displays the results."""
        shoot_results = {}
        made = 0
        total = len(self.shoot_frames)

        for shoot_start in self.shoot_frames:
            traj = self.reconstruct_trajectory(shoot_start)
            airball = self.is_airball(traj)
            shot_made = self.is_shot_made(traj) if not airball else False
            shoot_results[shoot_start] = shot_made
            if shot_made:
                made += 1
            print(f"[SHOT STATISTICS] Shot at frame {shoot_start} : {'Scored' if shot_made else 'Missed'}")
            self.plot_shoot(traj, shoot_start)

        print(f"[SHOT STATISTICS] Resultats : {made}/{total} shot made.")
        return shoot_results

    def display_shoots(self):
        """Displays the video with shot annotations."""
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
        near_hoop_frames = [s + 50 for s in self.shoot_frames]
        decision_frames = [s + 50 + j for s in self.shoot_frames for j in range(60)]
        shoot_display = {}

        for s in self.shoot_frames:
            traj = self.reconstruct_trajectory(s)
            result = not self.is_airball(traj) and self.is_shot_made(traj)
            for f in range(s, s + 130):
                shoot_display[f] = result
        made = 0
        total = 0
        os.makedirs("VID_OUT", exist_ok=True)
        height, width = self.frames_video[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("VID_OUT/output_annotated.mp4", fourcc, self.FPS, (width, height))

        for i in range(len(self.frames_video)):
            frame = self.frames_video[i].copy()

            # Draw the hoop
            hoop_w, hoop_h = 30, 30
            hx = int(self.HOOP_CX - hoop_w / 2) + self.OFFSET_X
            hy = int(self.HOOP_CY - hoop_h / 2) + self.OFFSET_Y
            cv2.rectangle(frame, (hx, hy), (hx + hoop_w, hy + hoop_h), (0, 215, 255), 2)

            # Draw the abll
            ball_row = self.ball[self.ball["frame"] == i]
            if not ball_row.empty and pd.notna(ball_row.iloc[0]["cx"]):
                bx = int(ball_row.iloc[0]["x"]) + self.OFFSET_X
                by = int(ball_row.iloc[0]["y"]) + self.OFFSET_Y
                bw = int(ball_row.iloc[0]["w"])
                bh = int(ball_row.iloc[0]["h"])
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 128, 255), 2)

            # DDraw Keypoints
            pose_rows = self.poses[(self.poses["frame"] == i) & (self.poses["person_id"] == 0)]
            if not pose_rows.empty:
                row = pose_rows.iloc[0]
                for name in KEYPOINT_NAMES:
                    x = row.get(f"{name}_x")
                    y = row.get(f"{name}_y")
                    if pd.notna(x) and pd.notna(y) and x > 0 and y > 0:
                        cv2.circle(frame, (int(x) + self.OFFSET_X, int(y) + self.OFFSET_Y), 4, (0, 255, 255), -1)

            #Band
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
            cv2.putText(frame, f"{made}/{total}", (frame.shape[1] - 120, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Display shot annotations
            if i in self.shoot_frames:
                total += 1
            if i in near_hoop_frames:
                if i in shoot_display:
                    if shoot_display[i]:
                        made += 1

            if i in shoot_display:
                result = shoot_display[i]
                color = (0, 255, 0) if result else (0, 0, 255)
                gray = (100, 100, 100)

                near = False
                if not near:
                    cv2.putText(frame, f"SHOT", (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, gray, 2)
                if i in decision_frames:
                    near = True
                    cv2.putText(frame, f"SHOT  {'Made' if result else 'Missed'}", (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            out.write(frame)

        out.release()
        print(f"[VIDEO] Saved → VID_OUT/output_annotated.mp4")