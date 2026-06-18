import glob
from autocrop import autocroping 
from crop import crop_video
from extract import ShotDetector
from shot_statistic import ShotStatistics
import glob


if __name__ == "__main__":
    # List all video files in the "video/" directory
    video_files = glob.glob("VID_IN/*.mp4")
    if not video_files:
        raise FileNotFoundError("No video files found in the 'video/' directory.")

    # Use the first video by default
    first_video = video_files[0]
    print(f"Loading video: {first_video}...")
    print(f"Analysing Video: {first_video}...")
    
    #Find the best crop for better detection
    x1, y1, x2, y2 = autocroping(first_video)
    
    # Crop the video
    output_video = "VID_IN/out.mp4"
    crop_video(first_video, output_video, x1, y1, x2, y2)
    print("Features Detection and Tracking")

    # Run ShotDetector on cropped video
    detector = ShotDetector(video_path="VID_IN/out.mp4")

    #Analyse trajectory of the ball and player movement to understand the video
    statistics = ShotStatistics(
        ball_csv="DATA_extracted/ball.csv",
        hoop_csv="DATA_extracted/hoop.csv",
        poses_csv="DATA_extracted/poses.csv",
        video_path=first_video,
        OFFSET_X=x1,
        OFFSET_Y=y1
    )
    print("Analysing data...")
    statistics.analyze_shoots()
    print("Annotating video...")
    statistics.display_shoots()


