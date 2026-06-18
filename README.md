
# Basketball Shot Detector

An end-to-end computer vision pipeline that takes raw basketball footage and automatically detects, tracks, and classifies every shot attempt as made or missed.

<img width="480" height="848" alt="exemple" src="https://github.com/user-attachments/assets/854c63ec-430c-4234-be89-5bfaa8c895a6" />

## Usage

```bash
git clone https://github.com/yourusername/basketball-shot-detector.git
```

Drop your video into the `VID_IN/` folder, run the pipeline, and find the annotated result in `VID_OUT/`.


## How it works

**1. Auto-crop**
The pipeline first analyzes a sample of frames to detect where the action actually happens (ball, players, hoop), then automatically crops out unused borders of the footage to improve detection accuracy and speed up processing.

**2. Detection (YOLO)**
Two YOLO models run on each frame: one detects the ball and the hoop, the other estimates player pose (keypoints for head, shoulders, wrists, hips, ankles, etc.).

**3. Tracking**
Since YOLO detections can be noisy or missing on some frames, a CSRT tracker runs both forward and backward through the video to fill gaps and produce a smooth, continuous trajectory for the ball.

**4. Data extraction**
All detections are exported to structured CSV files: ball position per frame, hoop position, and player keypoints. This gives a clean dataset describing exactly what's happening on court at every moment.

**5. Shot detection**
Using the extracted data, the pipeline identifies the start of a shot attempt by looking for the moment a player's wrist rises above their head with the ball nearby, then reconstructs the ball's trajectory for the following few seconds.

**6. Classification**
Each detected trajectory is analyzed geometrically relative to the hoop position to classify the shot as made or missed, filtering out airballs and other edge cases.

The final output is the original video annotated with the ball, hoop, player pose, and a live shot counter (made/total) overlaid in real time.


