import pandas as pd

ball   = pd.read_csv("DATA/ball.csv")
hoop   = pd.read_csv("DATA/hoop.csv")
poses  = pd.read_csv("DATA/poses.csv")

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

       
