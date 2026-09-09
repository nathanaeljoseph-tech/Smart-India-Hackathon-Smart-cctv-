import os
from pathlib import Path

def list_videos():
    video_dir = Path("data/virat/videos")
    videos = list(video_dir.glob("*.mp4"))
    
    print(f"Found {len(videos)} videos in {video_dir}:")
    for v in videos:
        print(f"  {v}")
        
    out_file = Path("data/virat/virat_video_list.txt")
    with open(out_file, "w") as f:
        for v in videos:
            f.write(f"{v}\n")
    print(f"\nWrote video list to {out_file}")

if __name__ == "__main__":
    list_videos()
