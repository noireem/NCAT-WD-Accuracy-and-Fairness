import cv2
import os
from pathlib import Path
from ultralytics import YOLO

def run_baseline_audit(image_dir, output_dir, weights_path, confidence_threshold=0.70):
    print(f"Loading custom YOLOv8 model from {weights_path}...")
    model = YOLO(weights_path)
    
    # Grab all PNG files (and JPGs just in case)
    print("Scanning directory for frames...")
    image_paths = sorted(list(Path(image_dir).rglob("*.png")))
    image_paths.extend(sorted(list(Path(image_dir).rglob("*.jpg"))))
    
    print(f"Found {len(image_paths)} individual frames to process.")

    os.makedirs(output_dir, exist_ok=True)

    for i, img_path in enumerate(image_paths):
        # We still want to skip frames so this doesn't take 5 days to run.
        # Processing every 15th frame is a good balance.
        if i % 15 != 0:
            continue

        # Run YOLOv8 Inference directly on the image file
        results = model.predict(str(img_path), conf=confidence_threshold, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                for box in boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    
                    # Catch various custom weapon class names
                    if class_name.lower() in ['weapon', 'gun', 'pistol', 'handgun', 'firearm', 'rifle', 'item']:

                        
                        # Use the parent folder name + frame name so files don't overwrite each other
                        parent_folder = img_path.parent.name
                        frame_name = f"{parent_folder}_{img_path.stem}_{class_name}_conf{conf:.2f}.jpg"
                        save_path = os.path.join(output_dir, frame_name)
                        
                        # Force tiny text and thin lines so we can actually see the image
                        annotated_frame = result.plot(line_width=1, font_size=1)
                        cv2.imwrite(save_path, annotated_frame)
                        
                        print(f" Flagged: {frame_name}")

    print("Ingestion and filtering complete. Ready for Step 3.")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    
    # Input: Import data from Azure Blob Storage (custom Dataset)
    IMAGE_DIRECTORY = PROJECT_ROOT / "data" / "raw" / ""
    
    # Output: Where the flagged images will go
    OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "custom_dataset_flagged_frames"
    
    # Model: The custom weights from Kushal
    WEIGHTS_FILE = PROJECT_ROOT / "models" / "weights" / "yolo_v8_baseline.pt"
    
    run_baseline_audit(
        image_dir=IMAGE_DIRECTORY, 
        output_dir=OUTPUT_DIRECTORY, 
        weights_path=WEIGHTS_FILE, 
        confidence_threshold=0.70
    )