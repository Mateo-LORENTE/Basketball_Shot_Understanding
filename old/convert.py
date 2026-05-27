import subprocess

subprocess.run([
    "ffmpeg", "-i", "video/IMG_7171.MOV",
    "-vf", "scale=1280:-1",   # réduit à 1280px de large
    "-crf", "23",              # compression (18=haute qualité, 28=plus compressé)
    "video/IMG_7171.mp4"
])