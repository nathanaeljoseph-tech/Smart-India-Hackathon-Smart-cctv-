import zipfile
import collections
import re

z = zipfile.ZipFile('archive.zip')

# Group files:
# Format: UCF-Crime with Fence Climbing/{split}/{class_name}/{frame_filename}
dataset = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))

for name in z.namelist():
    if name.endswith('/'):
        continue
    parts = name.split('/')
    if len(parts) >= 4 and parts[0] == 'UCF-Crime with Fence Climbing':
        split = parts[1]
        cls_name = parts[2]
        filename = parts[3]
        
        # Determine video sequence name
        # Examples:
        # 'Fence Climbing 1_frame_000445.jpg' -> 'Fence Climbing 1'
        # 'Suspicious10_frame_000000.jpg' -> 'Suspicious10'
        # 'Fighting002_x264_0.png' -> 'Fighting002_x264'
        # 'Normal_Videos015_x264_120.png' -> 'Normal_Videos015_x264'
        # 'Robbery001_x264_10.png' -> 'Robbery001_x264'
        if '_frame_' in filename:
            video_name = filename.split('_frame_')[0]
        elif '_x264_' in filename:
            video_name = filename.rsplit('_', 1)[0]
        elif '_' in filename:
            # check if last part is number with extension
            prefix, last = filename.rsplit('_', 1)
            video_name = prefix
        else:
            video_name = filename
            
        dataset[split][cls_name][video_name].append(filename)

print("DATASET OVERVIEW:")
total_frames = 0
total_videos = 0

all_classes = collections.defaultdict(lambda: {"videos": set(), "frames": 0, "examples": []})

for split in sorted(dataset.keys()):
    print(f"\n--- Split: {split} ---")
    for cls_name in sorted(dataset[split].keys()):
        vids = dataset[split][cls_name]
        n_frames = sum(len(f) for f in vids.values())
        print(f"  Class: {cls_name:<16} | Unique Videos: {len(vids):>4} | Total Frames: {n_frames:>6}")
        all_classes[cls_name]["frames"] += n_frames
        for v in vids:
            all_classes[cls_name]["videos"].add(f"{split}/{v}")
        if not all_classes[cls_name]["examples"]:
            all_classes[cls_name]["examples"] = list(vids.keys())[:5]

print("\nCOMBINED SUMMARY BY CLASS:")
for cls_name, info in sorted(all_classes.items()):
    print(f"Class: {cls_name:<16} | Total Videos: {len(info['videos']):>4} | Total Frames: {info['frames']:>6} | Example Vids: {info['examples'][:3]}")
