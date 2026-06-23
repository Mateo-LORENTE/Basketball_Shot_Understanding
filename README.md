
# Basketball Shot Detector 
![Python](https://img.shields.io/badge/Python-3.10-blue) ![YOLOv8](https://img.shields.io/badge/YOLO-v8-darkgreen) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)


An end-to-end computer vision pipeline that takes raw basketball footage and automatically detects, tracks, and classifies every shot attempt as made or missed. 

<img width="480" height="848" alt="exemple" src="https://github.com/user-attachments/assets/854c63ec-430c-4234-be89-5bfaa8c895a6" />

## Usage

```bash
git clone https://github.com/Mateo-LORENTE/Basketball_Shot_Understanding
```
```bash
cd Basketball_Shot_Understanding
pip install -r requirements.txt
```


Drop your video into the `VID_IN/` folder, run 
```bash
python shot_analysis.py 
```
in terminal, and find the annotated result in `VID_OUT/`.


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

## Technologies

- **YOLOv8** — object detection model used for ball detection, fine-tuned on a custom basketball dataset sourced from Roboflow and trained.
- **YOLOv11n-pose** — lightweight pose estimation model used to extract body keypoints (shoulders, elbows, wrists, hips, knees) for shot mechanics analysis
- **OpenCV** — handles video I/O, frame preprocessing (brightness adjustment, drawing overlays), and output video rendering
- **TrackerCSRT** — OpenCV's CSRT tracker used to maintain ball tracking between frames when YOLO detection confidence is low or the ball is temporarily occluded

## Limitations

- **Processing speed:** Execution time is approximately 10× slower than real-time (1 minute of video takes ~10 minutes to process) on a CPU-only setup (6 threads, no GPU), making long videos computationally intensive. Performance will vary depending on hardware.
- **Ball tracking:** Detection is not perfect and may occasionally miss shot attempts
- **Single player:** The current pipeline is designed for one player at a time
- **Fixed camera:** The camera must remain stationary throughout the recording

## Future Work

- **Multi-player detection** — extend the pipeline to track and analyze multiple players simultaneously
- **Court calibration** — automatically detect and map the court geometry to locate the shot
- **Personalized shot feedback** — provide detailed analysis on shot quality, ball arc height, body balance, and release mechanics
- **Ultimate goal: data-driven shot optimization** — store session data over time to identify shooting patterns and tendencies, then leverage extracted pose keypoints with machine learning to help players maximize their shooting percentage through personalized, data-backed recommendations

