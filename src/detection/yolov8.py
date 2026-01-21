import os
import sys
from pathlib import Path

def main():
    # Set project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    print(f"Working directory: {os.getcwd()}")
    
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    
    print(f"Training: epochs={epochs}, batch={batch_size}")
    
    try:
        from ultralytics import YOLO
        
        # Load pre-trained YOLOv8 model
        model = YOLO('yolov8n.pt')  # small model for faster training
        
        # Train the model
        model.train(
            data=os.path.join(project_root, 'model/cr_data.yaml'),  # dataset config
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            device=0,  # GPU
            patience=10,
            project='model',
            name='runs',
            pretrained=True,
            workers=4,
            plots=True,
            save=True,
            exist_ok=True,
        )
        
        print("Training complete!")
        print("Best weights: model/runs/runs/weights/best.pt")
        
    except ImportError:
        print("Ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
