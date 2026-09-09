import sys
import os
import argparse
from pathlib import Path
import yaml
import cv2

# Add parent directory to path so we can import from main project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import YOLODetector
from tracker import FallbackIOUTracker
from data_exporter import DataExporter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=5, help="Number of videos to process")
    parser.add_argument("--headless", action="store_true", help="Run without showing video window")
    args = parser.parse_args()

    # Load config
    config_path = Path("config/datasets.yaml")
    if not config_path.exists():
        print(f"Config not found at {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    print(f"Loaded config: {config}")

    video_dir = Path(config.get("virat_root", "data/virat")) / "videos"
    video_files = list(video_dir.glob("*.mp4"))
    
    subset = video_files[:args.subset]
    print(f"Processing {len(subset)} videos: {[v.name for v in subset]}")

    detector = YOLODetector(
        model_path="yolo11n.pt", 
        input_size=config.get("image_size", 640),
        device=config.get("device", "auto")
    )
    detector.load_model()
    
    output_dir = Path("output/virat")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for video_path in subset:
        print(f"\n--- Processing {video_path.name} ---")
        cap = cv2.VideoCapture(str(video_path))
        
        tracker = FallbackIOUTracker(iou_threshold=0.3, max_lost_frames=20)
        exporter = DataExporter(output_dir=str(output_dir), batch_size=config.get("batch_size", 8))
        
        frame_id = 0
        window_name = f"Processing {video_path.name}"
        if not args.headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_id += 1
            dets = detector.detect(frame)
            
            # Since the user requested "a lot of objects and guns", we pass all detections to the exporter
            # but tracking is mostly for people in the default pipeline.
            # Let's track everything that is a person or vehicle/bag.
            trackable_dets = [d for d in dets if d.get("class_id", -1) in [0, 2, 3, 5, 7]] # person, car, motorcycle, bus, truck
            
            tracks = tracker.update(trackable_dets)
            exporter.add_frame(frame_id, dets, tracks)
            
            if not args.headless:
                # Basic drawing
                for d in dets:
                    bbox = d["bbox"]
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), d.get("color", (0, 255, 0)), 2)
                    label = f"{d['class']} {d['confidence']:.2f}"
                    cv2.putText(frame, label, (bbox[0], max(0, bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, d.get("color", (0, 255, 0)), 2)
                
                for t in tracks:
                    bbox = t["bbox"]
                    cv2.putText(frame, f"ID:{t['track_id']}", (bbox[0], max(0, bbox[1]-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting early...")
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)
                    
        exporter.flush()
        cap.release()
        if not args.headless:
            cv2.destroyWindow(window_name)
            
    print(f"\nDone! Output saved to {output_dir}")

if __name__ == "__main__":
    main()
