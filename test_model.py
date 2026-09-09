from ultralytics import YOLO

# Use the standard YOLO11 Small model ('yolo11s.pt') for better accuracy on household items.
# (If it's not downloaded yet, ultralytics will auto-download it).
# NOTE: The COCO dataset knows 80 objects (cell phone, knife, scissors, etc.)
# but it does NOT know "pen", which is why it often guesses "toothbrush".
model = YOLO('yolo11s.pt')

# Run prediction using your WEBCAM (source=0) so you can test it live!
# Press 'q' in the video window to stop it.
results = model.predict(source=0, conf=0.25, show=True)
print("Inference completed!")
