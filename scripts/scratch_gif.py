import cv2
from PIL import Image
import os
import sys

def convert_video_to_gif(video_path, gif_path, max_frames=50, skip_frames=3, resize_width=640):
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        sys.exit(1)
        
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or len(frames) >= max_frames:
            break
        if count % skip_frames == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize logic for GIF optimization
            h, w = frame_rgb.shape[:2]
            new_h = int(h * (resize_width / w))
            frame_resized = cv2.resize(frame_rgb, (resize_width, new_h))
            
            frames.append(Image.fromarray(frame_resized))
        count += 1
    
    # original FPS is probably ~30. with skip_frames=3, it's ~10fps.
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None: fps = 30.0
    duration_ms = int(1000 / (fps / skip_frames))
    cap.release()
    
    if frames:
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], optimize=True, duration=duration_ms, loop=0)
        print(f"Saved {gif_path} with {len(frames)} frames.")
    else:
        print("No frames extracted.")

if __name__ == "__main__":
    convert_video_to_gif('data/output/result.mp4', 'demo.gif')
