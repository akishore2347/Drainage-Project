"""
DrainSight AI - YOLOv8 Model Training Script
Trains a custom YOLOv8 model on the drainage blockage dataset configured in data.yaml.
"""

import os
import sys
import argparse
from config import Config

def train_yolo(epochs=50, img_size=640, batch_size=8, device='cpu', pretrained_model='yolov8n.pt'):
    print("=" * 60)
    print("DrainSight AI - YOLOv8 Custom Model Training")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed.")
        print("Please install it with: pip install ultralytics")
        sys.exit(1)
        
    data_yaml_path = os.path.join(Config.BASE_DIR, 'data.yaml')
    if not os.path.exists(data_yaml_path):
        print(f"[ERROR] data.yaml not found at {data_yaml_path}")
        sys.exit(1)
        
    output_dir = os.path.join(Config.BASE_DIR, 'models')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[*] Base model: {pretrained_model}")
    print(f"[*] Dataset config: {data_yaml_path}")
    print(f"[*] Target epochs: {epochs}")
    print(f"[*] Batch size: {batch_size}")
    print(f"[*] Device: {device}")
    print(f"[*] Image size: {img_size}")
    print("-" * 60)
    
    try:
        model = YOLO(pretrained_model)
        
        # Train model
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            project=os.path.join(Config.BASE_DIR, 'runs'),
            name='drainsight_yolov8',
            save=True,
            val=True
        )
        
        print("\n[SUCCESS] Model training completed successfully!")
        
        # Look for the best weights file
        best_weights = os.path.join(Config.BASE_DIR, 'runs', 'drainsight_yolov8', 'weights', 'best.pt')
        target_model_path = Config.MODEL_PATH
        
        if os.path.exists(best_weights):
            import shutil
            shutil.copy(best_weights, target_model_path)
            print(f"[+] Exported best model weights to: {target_model_path}")
        else:
            print(f"[!] Warning: Best weights not found at {best_weights}. Check runs folder.")
            
    except Exception as e:
        print(f"[ERROR] Training failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv8 on DrainSight Dataset')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs (default: 30)')
    parser.add_argument('--batch', type=int, default=8, help='Batch size (default: 8)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size (default: 640)')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or 0 for CUDA (default: cpu)')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Pretrained model (default: yolov8n.pt)')
    
    args = parser.parse_args()
    train_yolo(epochs=args.epochs, img_size=args.imgsz, batch_size=args.batch, device=args.device, pretrained_model=args.model)
